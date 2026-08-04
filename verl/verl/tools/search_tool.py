# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import threading
import time
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor

from verl.tools.utils.search_r1_like_utils import perform_single_search_batch
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class SearchExecutionWorker:
    """Worker for executing search operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    # TODO we should make this available to the tool caller
                    logger.warning(f"Error when executing search: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_search_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize search execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(SearchExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class SearchTool(BaseTool):
    """Search tool for retrieving information using external retrieval services.

    This tool provides search functionality with rate limiting and concurrent execution
    support through Ray. It integrates with external retrieval services to perform
    semantic search operations.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Searches for relevant information based on queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search queries"
                            }
                        },
                        "required": ["query_list"]
                    }
                }
            }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.timeout = config.get("timeout", 30)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        # Retrieval service configuration
        self.retrieval_service_url = config.get("retrieval_service_url")
        assert self.retrieval_service_url, "Configuration must include 'retrieval_service_url'"
        self.topk = config.get("topk", 3)
        self.static_cp_enable = config.get("static_cp_enable", False)
        self.lambda_fixed = config.get("lambda_fixed", None)
        lambda_fixed_path = config.get("lambda_fixed_path", None)
        self.cp_k_max = int(config.get("cp_k_max", config.get("k_max", 10)))
        # backward compatibility for downstream subclasses relying on `self.k_max`
        self.k_max = self.cp_k_max

        # CP 过滤模式：
        # - aps: 自适应预测集（推荐）
        # - static: 旧版固定阈值（s_doc <= lambda_fixed）
        # - off: 不做后过滤
        if "cp_filter_mode" in config:
            self.cp_filter_mode = str(config.get("cp_filter_mode", "aps")).lower()
        else:
            # 兼容旧配置：只有 static_cp_enable 时走 static，否则关闭过滤
            self.cp_filter_mode = "static" if self.static_cp_enable else "off"

        if self.cp_filter_mode not in {"aps", "static", "off"}:
            raise ValueError(f"Unsupported cp_filter_mode: {self.cp_filter_mode}")

        # APS 超参数（可配置）
        self.aps_temperature = float(config.get("aps_temperature", 0.01))
        self.aps_q_hat = config.get("aps_q_hat", None)
        if self.aps_q_hat is not None:
            self.aps_q_hat = float(self.aps_q_hat)
        self.aps_min_docs = int(config.get("aps_min_docs", 2))
        self.aps_max_docs = int(config.get("aps_max_docs", 5))

        if self.lambda_fixed is None and lambda_fixed_path:
            with open(lambda_fixed_path, "r", encoding="utf-8") as f:
                lambda_cfg = json.load(f)
            if "LAMBDA_FIXED" in lambda_cfg:
                self.lambda_fixed = float(lambda_cfg["LAMBDA_FIXED"])
            if self.aps_q_hat is None:
                # 兼容多种键名
                if "APS_Q_HAT" in lambda_cfg:
                    self.aps_q_hat = float(lambda_cfg["APS_Q_HAT"])
                elif "aps_q_hat" in lambda_cfg:
                    self.aps_q_hat = float(lambda_cfg["aps_q_hat"])
                elif "q_hat" in lambda_cfg:
                    self.aps_q_hat = float(lambda_cfg["q_hat"])

        # static 模式需要阈值，APS/off 模式不需要
        if self.cp_filter_mode == "static" and self.lambda_fixed is None:
            raise ValueError(
                "cp_filter_mode=static requires `lambda_fixed` or `lambda_fixed_path` in tool config."
            )
        if self.cp_filter_mode == "aps" and self.aps_q_hat is None:
            raise ValueError(
                "cp_filter_mode=aps requires `aps_q_hat` (or `APS_Q_HAT` in lambda_fixed_path json)."
            )

        if self.retrieval_service_url == "":
            raise ValueError("retrieval_service_url is not set")

        logger.info(f"Initialized SearchTool with config: {config}")

    def _append_debug_trace(self, record: dict[str, Any]) -> None:
        """将检索中间信息写入 JSONL，便于排查 APS 与返回条目数。"""
        trace_dir = os.getenv("SEARCHR1_DEBUG_TRACE_DIR", "").strip()
        if not trace_dir:
            return
        try:
            os.makedirs(trace_dir, exist_ok=True)
            trace_path = os.path.join(trace_dir, f"search_tool_pid_{os.getpid()}.jsonl")
            payload = {
                "ts": time.time(),
                "pid": os.getpid(),
                **record,
            }
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write search debug trace: {e}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating the instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()

    def execute_search(self, instance_id: str, query_list: list, retrieval_service_url: str, topk: int, timeout: int):
        """Execute search operation using retrieval service.

        Args:
            instance_id: Tool instance ID
            query_list: List of search queries
            retrieval_service_url: URL of the retrieval service
            topk: Number of top results to return
            timeout: Request timeout in seconds

        Returns:
            Tuple of (result_text, metadata)
        """
        result_text, metadata = perform_single_search_batch(
            retrieval_service_url=retrieval_service_url,
            query_list=query_list,
            topk=topk,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
            lambda_fixed=self.lambda_fixed if self.cp_filter_mode == "static" else None,
            cp_k_max=self.cp_k_max,
            cp_filter_mode=self.cp_filter_mode,
            aps_temperature=self.aps_temperature,
            aps_q_hat=self.aps_q_hat,
            aps_min_docs=self.aps_min_docs,
            aps_max_docs=self.aps_max_docs,
        )
        logger.debug(f"Search result for instance {instance_id}: {result_text}")
        return result_text, metadata

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query_list and optional timeout

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        timeout = self.timeout
        query_list_from_params = parameters.get("query_list")

        if not query_list_from_params or not isinstance(query_list_from_params, list):
            error_msg = "Error: 'query_list' is missing, empty, or not a list in parameters."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}

        # Execute search using Ray execution pool
        try:
            result_text, metadata = await self.execution_pool.execute.remote(
                self.execute_search, instance_id, query_list_from_params, self.retrieval_service_url, self.topk, timeout
            )

            # Store results in instance dictionary
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            # Convert metadata to metrics
            cp_filter_meta = metadata.get("cp_filter", {}) if isinstance(metadata.get("cp_filter", {}), dict) else {}
            per_query_meta = cp_filter_meta.get("per_query", []) if isinstance(cp_filter_meta, dict) else []
            aps_final_ks = []
            if isinstance(per_query_meta, list):
                for item in per_query_meta:
                    if isinstance(item, dict) and item.get("mode") == "aps" and "final_k" in item:
                        try:
                            aps_final_ks.append(float(item["final_k"]))
                        except (TypeError, ValueError):
                            pass

            metrics = {
                "query_count": metadata.get("query_count", 0),
                "status": metadata.get("status", "unknown"),
                "total_results": metadata.get("total_results", 0),
                "api_request_error": metadata.get("api_request_error"),
                # APS 专用可观测指标：用于验证每次检索到底返回了多少条文档
                "aps_docs_returned_total": float(sum(aps_final_ks)) if aps_final_ks else 0.0,
                "aps_docs_returned_avg_per_query": float(sum(aps_final_ks) / len(aps_final_ks)) if aps_final_ks else 0.0,
                "aps_docs_returned_min_per_query": float(min(aps_final_ks)) if aps_final_ks else 0.0,
                "aps_docs_returned_max_per_query": float(max(aps_final_ks)) if aps_final_ks else 0.0,
            }

            # 记录调试信息：检索参数、返回文本、APS每query截断信息等
            self._append_debug_trace(
                {
                    "type": "search_execute",
                    "instance_id": instance_id,
                    "query_list": query_list_from_params,
                    "topk": self.topk,
                    "cp_filter_mode": self.cp_filter_mode,
                    "aps_q_hat": self.aps_q_hat,
                    "aps_temperature": self.aps_temperature,
                    "metadata": metadata,
                    "metrics": metrics,
                    "result_text": result_text,
                }
            )

            return ToolResponse(text=result_text), 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error(f"[SearchTool] Execution failed: {e}")
            self._append_debug_trace(
                {
                    "type": "search_execute_error",
                    "instance_id": instance_id,
                    "query_list": query_list_from_params,
                    "error": str(e),
                }
            )
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
