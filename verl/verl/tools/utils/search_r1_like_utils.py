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
import threading
import time
import traceback
import uuid
from typing import Any, Optional

import numpy as np
import requests

DEFAULT_TIMEOUT = 30  # Default search request timeout
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10

logger = logging.getLogger(__name__)


def _apply_static_cp_filter(
    retrieval: list[dict[str, Any]],
    lambda_fixed: Optional[float],
    cp_k_max: int,
) -> list[dict[str, Any]]:
    """Apply static conformal filtering on retrieval docs using s_doc = -score."""
    if not isinstance(retrieval, list):
        return []

    if lambda_fixed is None:
        return retrieval

    filtered: list[dict[str, Any]] = []
    for item in retrieval:
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            # No usable score -> skip under static CP mode.
            continue

        s_doc = -score
        if s_doc <= lambda_fixed:
            filtered.append(item)

    if cp_k_max > 0:
        filtered = filtered[:cp_k_max]
    return filtered


def _apply_aps_cp_filter(
    retrieval: list[dict[str, Any]],
    aps_temperature: float,
    aps_q_hat: float,
    aps_min_docs: int,
    aps_max_docs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """使用 APS(Adaptive Prediction Sets) 进行自适应过滤。

    核心流程：
    1) 按原始正向 score 降序排序；
    2) temperature softmax 转概率；
    3) 按累计概率达到 q_hat 的最小 k 截断；
    4) 用 min/max docs 做安全边界。
    """
    if not isinstance(retrieval, list):
        return [], {"mode": "aps", "reason": "invalid_retrieval_type"}

    scored_docs: list[tuple[dict[str, Any], float]] = []
    for item in retrieval:
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            # APS 需要可用分数；缺分数样本直接跳过
            continue
        scored_docs.append((item, score))

    if not scored_docs:
        return [], {"mode": "aps", "reason": "no_valid_scores"}

    # 1) 使用正向相似度分数降序排列
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    sorted_docs = [x[0] for x in scored_docs]
    scores = np.asarray([x[1] for x in scored_docs], dtype=np.float64)

    # 参数兜底，防止非法配置导致数值问题
    safe_temp = max(float(aps_temperature), 1e-8)
    safe_q_hat = float(min(1.0, max(0.0, aps_q_hat)))
    safe_min_docs = max(int(aps_min_docs), 1)
    safe_max_docs = max(int(aps_max_docs), safe_min_docs)

    # 2) 温度缩放 + 数值稳定 softmax（先减 max 再 exp）
    # 等价于 (scores - max(scores)) / temperature
    scaled_scores = (scores - float(np.max(scores))) / safe_temp
    exp_scores = np.exp(scaled_scores)
    exp_sum = float(np.sum(exp_scores))
    if not np.isfinite(exp_sum) or exp_sum <= 0:
        probs = np.full_like(exp_scores, 1.0 / len(exp_scores), dtype=np.float64)
    else:
        probs = exp_scores / exp_sum

    # 3) 累积概率，找到第一个达到 q_hat 的位置
    cum_probs = np.cumsum(probs)
    k = int(np.searchsorted(cum_probs, safe_q_hat, side="left")) + 1

    # 4) 应用安全边界，保证不饿死也不撑爆
    final_k = max(safe_min_docs, min(k, safe_max_docs, len(sorted_docs)))
    filtered = sorted_docs[:final_k]

    debug_info = {
        "mode": "aps",
        "temperature": safe_temp,
        "q_hat": safe_q_hat,
        "min_docs": safe_min_docs,
        "max_docs": safe_max_docs,
        "k_by_mass": k,
        "final_k": final_k,
        "candidate_docs": len(sorted_docs),
        "top_score": float(scores[0]),
        "top_prob": float(probs[0]),
        "cum_prob_at_final_k": float(cum_probs[final_k - 1]),
    }
    return filtered, debug_info


def call_search_api(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Calls the remote search API to perform retrieval with retry logic for various errors,
    using increasing delay between retries. Logs internal calls with a unique ID.

    Args:
        retrieval_service_url: The URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        return_scores: Whether to return scores.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (response_json, error_message).
        If successful, response_json is the API's returned JSON object, error_message is None.
        If failed after retries, response_json is None, error_message contains the error information.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "

    payload = {"queries": query_list, "topk": topk, "return_scores": return_scores}

    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling search API at {retrieval_service_url}"
            )
            response = requests.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                last_error = (
                    f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt "
                    f"{attempt + 1}/{MAX_RETRIES}"
                )
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            # Check for other HTTP errors (e.g., 4xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(f"{log_prefix}Search API call successful on attempt {attempt + 1}")
            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break  # Exit retry loop on other request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}Search API call failed. Last error: {last_error}")
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


def _passages2string(retrieval_result):
    """Convert retrieval results to formatted string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return format_reference.strip()


def perform_single_search_batch(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    timeout: int = DEFAULT_TIMEOUT,
    lambda_fixed: Optional[float] = None,
    cp_k_max: int = 10,
    cp_filter_mode: str = "static",
    aps_temperature: float = 0.01,
    aps_q_hat: Optional[float] = None,
    aps_min_docs: int = 2,
    aps_max_docs: int = 5,
    k_max: Optional[int] = None,
) -> tuple[str, dict[str, Any]]:
    """
    Performs a single batch search for multiple queries (original search tool behavior).

    Args:
        retrieval_service_url: The URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        concurrent_semaphore: Optional semaphore for concurrency control.
        timeout: Request timeout in seconds.
        cp_k_max: Max documents kept after static CP filtering.
        k_max: Deprecated alias of cp_k_max.

    Returns:
        A tuple (result_text, metadata).
        result_text: The search result JSON string.
        metadata: Metadata dictionary for the batch search.
    """
    logger.info(f"Starting batch search for {len(query_list)} queries.")
    if k_max is not None:
        cp_k_max = int(k_max)
    cp_filter_mode = str(cp_filter_mode).lower()

    api_response = None
    error_msg = None

    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg = call_search_api(
                    retrieval_service_url=retrieval_service_url,
                    query_list=query_list,
                    topk=topk,
                    return_scores=True,
                    timeout=timeout,
                )
        else:
            api_response, error_msg = call_search_api(
                retrieval_service_url=retrieval_service_url,
                query_list=query_list,
                topk=topk,
                return_scores=True,
                timeout=timeout,
            )
    except Exception as e:
        error_msg = f"API Request Exception during batch search: {e}"
        logger.error(f"Batch search: {error_msg}")
        traceback.print_exc()

    metadata = {
        "query_count": len(query_list),
        "queries": query_list,
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "total_results": 0,
        "formatted_result": None,
    }

    result_text = json.dumps({"result": "Search request failed or timed out after retries."}, ensure_ascii=False)

    if error_msg:
        metadata["status"] = "api_error"
        result_text = json.dumps({"result": f"Search error: {error_msg}"}, ensure_ascii=False)
        logger.error(f"Batch search: API error occurred: {error_msg}")
    elif api_response:
        logger.debug(f"Batch search: API Response: {api_response}")
        metadata["api_response"] = api_response

        try:
            raw_results = api_response.get("result", [])
            if raw_results:
                pretty_results = []
                total_results = 0
                cp_debug_list: list[dict[str, Any]] = []

                for retrieval in raw_results:
                    if cp_filter_mode == "aps":
                        if aps_q_hat is None:
                            raise ValueError("cp_filter_mode=aps requires non-null aps_q_hat.")
                        filtered_retrieval, cp_debug = _apply_aps_cp_filter(
                            retrieval=retrieval,
                            aps_temperature=aps_temperature,
                            aps_q_hat=aps_q_hat,
                            aps_min_docs=aps_min_docs,
                            aps_max_docs=aps_max_docs,
                        )
                        logger.info(
                            "APS filter applied: candidate_docs=%s, final_k=%s, q_hat=%.6f, temp=%.5f",
                            cp_debug.get("candidate_docs", 0),
                            cp_debug.get("final_k", 0),
                            float(cp_debug.get("q_hat", 0.0)),
                            float(cp_debug.get("temperature", aps_temperature)),
                        )
                    elif cp_filter_mode == "off":
                        filtered_retrieval = retrieval if isinstance(retrieval, list) else []
                        cp_debug = {"mode": "off", "returned_docs": len(filtered_retrieval)}
                    else:
                        # 兼容旧版 static 逻辑：s_doc = -score, 保留 s_doc <= lambda_fixed
                        filtered_retrieval = _apply_static_cp_filter(
                            retrieval=retrieval,
                            lambda_fixed=lambda_fixed,
                            cp_k_max=cp_k_max,
                        )
                        cp_debug = {
                            "mode": "static",
                            "enabled": lambda_fixed is not None,
                            "lambda_fixed": lambda_fixed,
                            "cp_k_max": cp_k_max,
                        }

                    formatted = _passages2string(filtered_retrieval) if filtered_retrieval else ""
                    pretty_results.append(formatted)
                    total_results += len(filtered_retrieval)
                    cp_debug_list.append(cp_debug)

                final_result = "\n---\n".join(pretty_results)
                result_text = json.dumps({"result": final_result}, ensure_ascii=False)
                metadata["status"] = "success"
                metadata["total_results"] = total_results
                metadata["formatted_result"] = final_result
                metadata["cp_filter"] = {
                    "mode": cp_filter_mode,
                    "lambda_fixed": lambda_fixed,
                    "cp_k_max": cp_k_max,
                    "aps_temperature": aps_temperature,
                    "aps_q_hat": aps_q_hat,
                    "aps_min_docs": aps_min_docs,
                    "aps_max_docs": aps_max_docs,
                    "per_query": cp_debug_list,
                }
                logger.info(f"Batch search: Successful, got {total_results} total results")
            else:
                result_text = json.dumps({"result": "No search results found."}, ensure_ascii=False)
                metadata["status"] = "no_results"
                metadata["total_results"] = 0
                logger.info("Batch search: No results found")
        except Exception as e:
            error_msg = f"Error processing search results: {e}"
            result_text = json.dumps({"result": error_msg}, ensure_ascii=False)
            metadata["status"] = "processing_error"
            logger.error(f"Batch search: {error_msg}")
    else:
        metadata["status"] = "unknown_api_state"
        result_text = json.dumps(
            {"result": "Unknown API state (no response and no error message)."}, ensure_ascii=False
        )
        logger.error("Batch search: Unknown API state.")

    return result_text, metadata
