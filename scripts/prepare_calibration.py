#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_MAX_RETRIES = 5
LLM_RETRY_DELAY = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_messages(prompt_obj: Any) -> list[dict[str, str]]:
    if isinstance(prompt_obj, np.ndarray):
        prompt_obj = prompt_obj.tolist()
    if isinstance(prompt_obj, pd.Series):
        prompt_obj = prompt_obj.tolist()
    if isinstance(prompt_obj, tuple):
        prompt_obj = list(prompt_obj)

    if isinstance(prompt_obj, list):
        return prompt_obj
    if isinstance(prompt_obj, str):
        maybe_list = json.loads(prompt_obj)
        if isinstance(maybe_list, np.ndarray):
            maybe_list = maybe_list.tolist()
        if isinstance(maybe_list, tuple):
            maybe_list = list(maybe_list)
        if isinstance(maybe_list, list):
            return maybe_list
    raise ValueError(f"Unsupported prompt format: {type(prompt_obj)}")


def normalize_answer_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def to_answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [str(value)]


def get_ground_truth_answers(row: pd.Series) -> list[str]:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, str):
        try:
            reward_model = json.loads(reward_model)
        except json.JSONDecodeError:
            reward_model = None
    if isinstance(reward_model, dict):
        gt = reward_model.get("ground_truth", {})
        if isinstance(gt, dict):
            return to_answer_list(gt.get("target", []))
    return []


def get_data_source(row: pd.Series) -> str:
    """Best-effort extraction of sample data source tag."""
    direct = row.get("data_source")
    if isinstance(direct, str) and direct.strip():
        return direct.strip().lower()

    extra_info = _maybe_parse_json(row.get("extra_info"))
    if isinstance(extra_info, dict):
        tools_kwargs = extra_info.get("tools_kwargs", {})
        if isinstance(tools_kwargs, dict):
            search_kwargs = tools_kwargs.get("search", {})
            if isinstance(search_kwargs, dict):
                create_kwargs = search_kwargs.get("create_kwargs", {})
                if isinstance(create_kwargs, dict):
                    source = create_kwargs.get("data_source")
                    if isinstance(source, str) and source.strip():
                        return source.strip().lower()
    return ""


def _maybe_parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2:
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines).strip()
    return s


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    s = _strip_code_fence(text)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    left = s.find("{")
    right = s.rfind("}")
    if left >= 0 and right > left:
        snippet = s[left : right + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def _safe_json_dumps(value: Any) -> str:
    def _default(o: Any):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, (pd.Series, pd.Index)):
            return o.tolist()
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        return str(o)

    return json.dumps(value, ensure_ascii=False, default=_default)


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(_safe_json_dumps(row) + "\n")


def _call_teacher_llm_json(llm_cfg: dict[str, Any], user_content: str, system_content: str) -> tuple[dict[str, Any] | None, str]:
    payload = {
        "model": llm_cfg["model"],
        "temperature": llm_cfg.get("temperature", 0.0),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
    }

    api_base = str(llm_cfg["api_base"]).rstrip("/")
    if api_base.endswith("/chat/completions"):
        url = api_base
    else:
        url = api_base + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {llm_cfg['api_key']}",
        "Content-Type": "application/json",
    }
    last_err: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=llm_cfg.get("timeout", 60))
            resp.raise_for_status()
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            obj = _extract_json_from_text(content)
            return obj, content
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(LLM_RETRY_DELAY * (attempt + 1))
                continue
            raise
        except Exception as e:
            last_err = e
            raise

    raise RuntimeError(f"Teacher LLM call failed after retries: {last_err}")


def _llm_extract_pairs_from_row_with_trace(
    row: pd.Series,
    llm_cfg: dict[str, Any],
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    question = get_question_text(row).strip()
    gt_answers = get_ground_truth_answers(row)
    if not question:
        return [], {
            "trace_type": "hop_decomposition",
            "status": "empty_question",
            "orig_row_id": row.get("orig_row_id"),
            "calibration_id": row.get("calibration_id"),
            "data_source": get_data_source(row),
            "question": "",
            "ground_truth_answers": gt_answers,
            "pairs": [],
            "llm_raw_content": "",
        }

    metadata = _maybe_parse_json(row.get("metadata"))
    extra_info = _maybe_parse_json(row.get("extra_info"))

    user_content = (
        "Given a QA sample, split it into single-hop query-answer pairs.\n"
        "Rules:\n"
        "1) If single-hop, return one pair.\n"
        "2) If multi-hop, return one pair per hop.\n"
        "3) answer should be concise and factual.\n"
        "4) Output JSON only in schema: "
        '{"pairs":[{"query":"...","answer":"..."}]}'
        "\n\n"
        f"question: {question}\n"
        f"ground_truth_answers: {_safe_json_dumps(gt_answers)}\n"
        f"metadata: {_safe_json_dumps(metadata) if isinstance(metadata, (dict, list)) else 'null'}\n"
        f"extra_info: {_safe_json_dumps(extra_info) if isinstance(extra_info, (dict, list)) else 'null'}"
    )
    system_content = "You decompose QA tasks into hop-level search query-answer pairs. Return strict JSON."

    obj, content = _call_teacher_llm_json(
        llm_cfg=llm_cfg,
        user_content=user_content,
        system_content=system_content,
    )
    if not obj:
        return [], {
            "trace_type": "hop_decomposition",
            "status": "invalid_json",
            "orig_row_id": row.get("orig_row_id"),
            "calibration_id": row.get("calibration_id"),
            "data_source": get_data_source(row),
            "question": question,
            "ground_truth_answers": gt_answers,
            "pairs": [],
            "llm_raw_content": content,
        }

    pairs: list[tuple[str, str]] = []
    for item in _to_list(obj.get("pairs")):
        if not isinstance(item, dict):
            continue
        q = str(item.get("query", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            pairs.append((q, a))
    trace = {
        "trace_type": "hop_decomposition",
        "status": "ok",
        "orig_row_id": row.get("orig_row_id"),
        "calibration_id": row.get("calibration_id"),
        "data_source": get_data_source(row),
        "question": question,
        "ground_truth_answers": gt_answers,
        "pairs": [{"query": q, "answer": a} for q, a in pairs],
        "llm_raw_content": content,
    }
    return pairs, trace


def _llm_extract_pairs_from_row(row: pd.Series, llm_cfg: dict[str, Any]) -> list[tuple[str, str]]:
    pairs, _ = _llm_extract_pairs_from_row_with_trace(row=row, llm_cfg=llm_cfg)
    return pairs


def _extract_query_from_step(step: Any) -> list[str]:
    step = _maybe_parse_json(step)
    if not isinstance(step, dict):
        return []

    query_candidates: list[str] = []

    single_keys = ["query", "search_query", "sub_query", "question"]
    multi_keys = ["query_list", "queries", "search_queries", "sub_queries"]

    for key in single_keys:
        val = step.get(key)
        if isinstance(val, str) and val.strip():
            query_candidates.append(val.strip())

    for key in multi_keys:
        val = step.get(key)
        for item in _to_list(val):
            if isinstance(item, str) and item.strip():
                query_candidates.append(item.strip())

    # 去重并保持顺序
    dedup: list[str] = []
    seen = set()
    for q in query_candidates:
        if q not in seen:
            dedup.append(q)
            seen.add(q)
    return dedup


def _extract_answers_from_step(step: Any) -> list[str]:
    step = _maybe_parse_json(step)
    if not isinstance(step, dict):
        return []

    answer_candidates: list[str] = []
    answer_keys = [
        "answer",
        "answers",
        "target",
        "targets",
        "intermediate_answer",
        "intermediate_answers",
        "final_answer",
        "final_answers",
        "golden_answer",
        "golden_answers",
    ]

    for key in answer_keys:
        val = step.get(key)
        for item in _to_list(val):
            if isinstance(item, str) and item.strip():
                answer_candidates.append(item.strip())

    dedup: list[str] = []
    seen = set()
    for a in answer_candidates:
        if a not in seen:
            dedup.append(a)
            seen.add(a)
    return dedup


def flatten_query_answer_pairs_from_row(
    row: pd.Series,
    llm_cfg: dict[str, Any] | None = None,
    llm_only_when_no_trajectory: bool = True,
    teacher_hop_trace_collector: list[dict[str, Any]] | None = None,
) -> list[tuple[str, str]]:
    """
    从单条样本中拍平提取 (query, answer) 对：
    - 单跳：返回 1 对
    - 多跳：返回多对（每一跳至少一对）
    """
    gt_answers = get_ground_truth_answers(row)

    # 收集所有可能的“轨迹容器”
    candidates: list[Any] = []
    for field in [
        "search_trajectory",
        "ground_truth_trajectory",
        "trajectory",
        "search_path",
        "hops",
        "steps",
    ]:
        candidates.append(row.get(field))

    for field in ["metadata", "extra_info"]:
        blob = _maybe_parse_json(row.get(field))
        if isinstance(blob, dict):
            for key in [
                "search_trajectory",
                "ground_truth_trajectory",
                "trajectory",
                "search_path",
                "hops",
                "steps",
            ]:
                candidates.append(blob.get(key))
        elif isinstance(blob, list):
            candidates.append(blob)

    # 归一化到 step 列表
    step_list: list[Any] = []
    for c in candidates:
        c = _maybe_parse_json(c)
        if c is None:
            continue
        if isinstance(c, dict):
            # dict 可能是单跳 step，也可能是包裹结构
            nested = None
            for key in ["steps", "hops", "trajectory", "search_trajectory"]:
                if key in c:
                    nested = c[key]
                    break
            if nested is not None:
                step_list.extend(_to_list(_maybe_parse_json(nested)))
            else:
                step_list.append(c)
        elif isinstance(c, list):
            step_list.extend([_maybe_parse_json(x) for x in c])

    flattened_pairs: list[tuple[str, str]] = []
    for step in step_list:
        queries = _extract_query_from_step(step)
        answers = _extract_answers_from_step(step)
        if not answers:
            answers = gt_answers
        for q in queries:
            for a in answers:
                if q and a:
                    flattened_pairs.append((q, a))

    should_try_llm = llm_cfg is not None and ((not llm_only_when_no_trajectory) or (len(flattened_pairs) == 0))
    if should_try_llm:
        try:
            llm_pairs, llm_trace = _llm_extract_pairs_from_row_with_trace(row=row, llm_cfg=llm_cfg)
            if teacher_hop_trace_collector is not None:
                teacher_hop_trace_collector.append(llm_trace)
        except Exception as e:
            if teacher_hop_trace_collector is not None:
                teacher_hop_trace_collector.append(
                    {
                        "trace_type": "hop_decomposition",
                        "status": "exception",
                        "orig_row_id": row.get("orig_row_id"),
                        "calibration_id": row.get("calibration_id"),
                        "data_source": get_data_source(row),
                        "question": get_question_text(row).strip(),
                        "ground_truth_answers": get_ground_truth_answers(row),
                        "pairs": [],
                        "llm_raw_content": "",
                        "error": str(e),
                    }
                )
            llm_pairs = []
        if llm_pairs:
            return llm_pairs

    # 没有显式轨迹且 LLM 未提供结果时，回退到 (question, gt_answer)
    if not flattened_pairs:
        question = get_question_text(row).strip()
        if question:
            for a in gt_answers:
                if a:
                    flattened_pairs.append((question, a))

    return flattened_pairs


def _extract_nq_direct_pairs_from_row(row: pd.Series) -> list[tuple[str, str]]:
    """
    For NQ samples, directly use (question, answer) without hop decomposition.
    """
    question = get_question_text(row).strip()
    answers = [a.strip() for a in get_ground_truth_answers(row) if isinstance(a, str) and a.strip()]
    if not question or not answers:
        return []
    # Keep one canonical pair per NQ sample.
    return [(question, answers[0])]


def extract_pairs_for_calibration_row(
    row: pd.Series,
    llm_cfg: dict[str, Any] | None = None,
    llm_only_when_no_trajectory: bool = True,
    teacher_hop_trace_collector: list[dict[str, Any]] | None = None,
) -> list[tuple[str, str]]:
    """
    Data-source aware extraction:
    - nq: direct (question, answer)
    - hotpotqa: mandatory teacher-LLM hop decomposition
    - others: fallback to trajectory flattening logic
    """
    source = get_data_source(row)
    source_norm = source.lower()

    if source_norm.startswith("nq"):
        return _extract_nq_direct_pairs_from_row(row)

    if "hotpotqa" in source_norm or source_norm.startswith("hotpot"):
        if llm_cfg is None:
            raise ValueError(
                "HotpotQA calibration requires teacher-LLM extraction, but llm_cfg is missing. "
                "Please pass --enable_llm_hop_extraction and valid LLM credentials."
            )
        pairs, llm_trace = _llm_extract_pairs_from_row_with_trace(row=row, llm_cfg=llm_cfg)
        if teacher_hop_trace_collector is not None:
            teacher_hop_trace_collector.append(llm_trace)
        if not pairs:
            raise ValueError(
                f"Teacher-LLM returned no hop pairs for hotpotqa sample "
                f"(orig_row_id={row.get('orig_row_id')}, calibration_id={row.get('calibration_id')})."
            )
        return pairs

    return flatten_query_answer_pairs_from_row(
        row=row,
        llm_cfg=llm_cfg,
        llm_only_when_no_trajectory=llm_only_when_no_trajectory,
        teacher_hop_trace_collector=teacher_hop_trace_collector,
    )


def get_question_text(row: pd.Series) -> str:
    extra_info = row.get("extra_info")
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            extra_info = None
    if isinstance(extra_info, dict) and isinstance(extra_info.get("question"), str):
        return extra_info["question"]

    prompt = ensure_messages(row["prompt"])
    for message in prompt:
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def call_retrieval(retrieval_service_url: str, query: str, topk: int, timeout: int) -> list[dict[str, Any]]:
    payload = {"queries": [query], "topk": topk, "return_scores": True}
    resp = requests.post(retrieval_service_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    raw_result = body.get("result", [])
    if not raw_result:
        return []
    first = raw_result[0]
    return first if isinstance(first, list) else []


def doc_to_text(doc_item: dict[str, Any]) -> str:
    doc = doc_item.get("document", {})
    if isinstance(doc, dict):
        if "contents" in doc and isinstance(doc["contents"], str):
            return doc["contents"]
        title = str(doc.get("title", ""))
        text = str(doc.get("text", ""))
        return f"{title}\n{text}".strip()
    return str(doc_item)


def _as_int_list(value: Any) -> list[int]:
    out: list[int] = []
    for item in _to_list(value):
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _llm_select_golden_docs_from_retrieval(
    query: str,
    answer: str,
    retrieved_docs: list[dict[str, Any]],
    llm_cfg: dict[str, Any],
    pair_meta: dict[str, Any] | None = None,
) -> tuple[int, list[int], dict[str, Any]]:
    """让 teacher-LLM 从检索条目中判断可用于推理出答案的文档。"""
    docs_for_llm: list[dict[str, Any]] = []
    for idx, doc_item in enumerate(retrieved_docs):
        score_val = doc_item.get("score", None)
        try:
            score_val = float(score_val) if score_val is not None else None
        except (TypeError, ValueError):
            score_val = None
        docs_for_llm.append(
            {
                "index": idx,
                "score": score_val,
                # 控制单条文本长度，避免 teacher 调用超长
                "contents": doc_to_text(doc_item)[:4000],
            }
        )

    user_content = (
        "You are given a retrieval query, its target answer, and retrieved documents.\n"
        "Decide which document(s) can support reasoning to the target answer.\n"
        "Return strict JSON with schema:\n"
        '{"golden_doc_indices":[0,1], "best_golden_doc_index":0, "reason":"..."}\n'
        "Rules:\n"
        "1) If none can support the answer, return empty golden_doc_indices and -1 as best index.\n"
        "2) best_golden_doc_index must be one item in golden_doc_indices, or -1.\n"
        "3) Never output markdown.\n\n"
        f"query: {query}\n"
        f"answer: {answer}\n"
        f"retrieved_docs: {_safe_json_dumps(docs_for_llm)}"
    )
    system_content = "You are a precise retrieval relevance judge for QA reasoning."

    obj, raw_content = _call_teacher_llm_json(
        llm_cfg=llm_cfg,
        user_content=user_content,
        system_content=system_content,
    )

    valid_indices: list[int] = []
    best_index = -1
    reason = ""
    status = "ok"
    if not obj:
        status = "invalid_json"
    else:
        valid_raw = _as_int_list(obj.get("golden_doc_indices", []))
        seen = set()
        for idx in valid_raw:
            if 0 <= idx < len(retrieved_docs) and idx not in seen:
                valid_indices.append(idx)
                seen.add(idx)

        try:
            best_index = int(obj.get("best_golden_doc_index", -1))
        except (TypeError, ValueError):
            best_index = -1

        if best_index not in valid_indices:
            best_index = valid_indices[0] if valid_indices else -1
        reason = str(obj.get("reason", "")).strip()

    trace: dict[str, Any] = {
        "trace_type": "golden_doc_judgement",
        "status": status,
        "query": query,
        "answer": answer,
        "num_retrieved_docs": len(retrieved_docs),
        "golden_doc_indices": valid_indices,
        "best_golden_doc_index": best_index,
        "reason": reason,
        "llm_raw_content": raw_content,
    }
    if pair_meta:
        trace.update(pair_meta)
    return best_index, valid_indices, trace


def contains_any_answer(doc_text: str, answer_list: list[str]) -> bool:
    norm_doc = normalize_answer_text(doc_text)
    for ans in answer_list:
        norm_ans = normalize_answer_text(ans)
        if norm_ans and norm_ans in norm_doc:
            return True
    return False


def _stable_softmax_with_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    """带温度系数且数值稳定的 softmax。"""
    safe_temp = max(float(temperature), 1e-8)
    shifted = (scores - float(np.max(scores))) / safe_temp
    exp_scores = np.exp(shifted)
    denom = float(np.sum(exp_scores))
    if not np.isfinite(denom) or denom <= 0:
        return np.full_like(exp_scores, 1.0 / len(exp_scores), dtype=np.float64)
    return exp_scores / denom


def calibrate_aps(
    calibration_items: list[dict[str, Any]],
    aps_temperature: float = 0.01,
    aps_alpha: float = 0.10,
) -> tuple[float, np.ndarray]:
    """离线 APS 校准，返回 q_hat 与每条样本的非遵循分数 E_i。

    每个 calibration_item 至少包含：
    - scores: 检索分数列表（优先认为是正向相似度分数）
    - golden_index: 黄金文档在原始列表中的索引，若不存在用 -1
    可选：
    - scores_are_s_doc: 若为 True，则 scores 代表 s_doc=-retrieval_score，需要取负转回正向分数
    """
    if not calibration_items:
        raise ValueError("calibration_items is empty; cannot calibrate APS q_hat.")

    e_values: list[float] = []
    quantile = 1.0 - float(aps_alpha)

    for item in calibration_items:
        raw_scores = np.asarray(item.get("scores", []), dtype=np.float64).reshape(-1)
        golden_index = int(item.get("golden_index", -1))
        scores_are_s_doc = bool(item.get("scores_are_s_doc", False))

        if raw_scores.size == 0:
            e_values.append(1.0)
            continue

        # 1) 输入处理：转为正向分数并降序
        scores = -raw_scores if scores_are_s_doc else raw_scores.copy()
        # 若出现负值（例如混入了 s_doc 风格数据），做绝对值兜底转正。
        if np.any(scores < 0):
            scores = np.abs(scores)

        order = np.argsort(-scores)  # 降序
        sorted_scores = scores[order]

        # 黄金文档不在 top-k 或索引非法，按定义 E_i=1.0
        if golden_index < 0 or golden_index >= len(scores):
            e_values.append(1.0)
            continue
        ranked_pos_arr = np.where(order == golden_index)[0]
        if ranked_pos_arr.size == 0:
            e_values.append(1.0)
            continue
        ranked_pos = int(ranked_pos_arr[0])

        # 2) 温度缩放 + softmax（减 max 保证数值稳定）
        probs = _stable_softmax_with_temperature(sorted_scores, temperature=aps_temperature)

        # 3) 非遵循分数 E_i：累计到黄金文档所在位置（包含该文档）
        cum_probs = np.cumsum(probs)
        e_i = float(cum_probs[ranked_pos])
        e_values.append(float(min(1.0, max(0.0, e_i))))

    e_np = np.asarray(e_values, dtype=np.float64)
    try:
        q_hat = float(np.quantile(e_np, quantile, method="higher"))
    except TypeError:
        q_hat = float(np.quantile(e_np, quantile, interpolation="higher"))

    return q_hat, e_np


def build_tag_patterns(tokenizer, tag: str) -> list[list[int]]:
    variants = [tag, f" {tag}", f"\n{tag}", f"\n\n{tag}"]
    patterns: list[list[int]] = []
    for variant in variants:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids and ids not in patterns:
            patterns.append(ids)
    return patterns


def find_subsequence(seq: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern:
        return -1
    max_start = len(seq) - len(pattern)
    for idx in range(start, max_start + 1):
        if seq[idx : idx + len(pattern)] == pattern:
            return idx
    return -1


def find_first_pattern(seq: list[int], patterns: list[list[int]], start: int = 0) -> tuple[int, int]:
    best_idx = -1
    best_len = 0
    for pattern in patterns:
        idx = find_subsequence(seq, pattern, start=start)
        if idx >= 0 and (best_idx < 0 or idx < best_idx):
            best_idx = idx
            best_len = len(pattern)
    return best_idx, best_len


def build_prompt_text(messages: list[dict[str, str]], tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    chunks = []
    for item in messages:
        role = item.get("role", "user")
        content = item.get("content", "")
        chunks.append(f"{role}: {content}")
    chunks.append("assistant:")
    return "\n".join(chunks)


@dataclass
class CalibrationOutputs:
    calibration_path: str
    train_rest_path: str
    lambda_path: str | None
    scores_path: str | None


@dataclass
class DCRGRPOCalibConfig:
    """Centralized hyperparameters for offline DCR-GRPO calibration."""

    calib_sample_size: int = 150
    cp_alpha: float = 0.20
    cp_k_max: int = 10
    aps_temperature: float = 0.01
    aps_alpha: float = 0.10
    aps_min_docs: int = 2
    aps_max_docs: int = 5

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DCRGRPOCalibConfig":
        return cls(
            calib_sample_size=int(args.calib_sample_size),
            cp_alpha=float(args.cp_alpha),
            cp_k_max=int(args.cp_k_max),
            aps_temperature=float(args.aps_temperature),
            aps_alpha=float(args.aps_alpha),
            aps_min_docs=int(args.aps_min_docs),
            aps_max_docs=int(args.aps_max_docs),
        )


def run(args) -> CalibrationOutputs:
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    dcr_cfg = DCRGRPOCalibConfig.from_args(args)

    df = pd.read_parquet(args.train_parquet)
    if dcr_cfg.calib_sample_size > len(df):
        raise ValueError(f"calib_sample_size={dcr_cfg.calib_sample_size} > dataset_size={len(df)}")

    sampled_idx = np.random.choice(len(df), size=dcr_cfg.calib_sample_size, replace=False)
    sampled_idx_set = set(sampled_idx.tolist())

    calib_df = df.iloc[sampled_idx].copy().reset_index(drop=False).rename(columns={"index": "orig_row_id"})
    train_rest_df = (
        df.iloc[[i for i in range(len(df)) if i not in sampled_idx_set]]
        .copy()
        .reset_index(drop=False)
        .rename(columns={"index": "orig_row_id"})
    )

    calib_df["calibration_id"] = np.arange(len(calib_df))
    calib_df["question"] = calib_df.apply(get_question_text, axis=1)
    calib_df["split_role"] = "calibration"
    train_rest_df["split_role"] = "rl_train"

    # Hard guarantee: calibration rows are strictly excluded from RL training rows.
    calib_ids = set(calib_df["orig_row_id"].tolist())
    train_ids = set(train_rest_df["orig_row_id"].tolist())
    overlap = calib_ids.intersection(train_ids)
    if overlap:
        raise ValueError(
            f"Calibration/train split overlap detected: {len(overlap)} rows. "
            "Calibration rows must be isolated and excluded from RL training."
        )
    if len(calib_df) + len(train_rest_df) != len(df):
        raise ValueError("Split size mismatch: calibration + rl_train does not equal original dataset size.")

    calibration_path = os.path.join(args.output_dir, "calibration_dataset.parquet")
    train_rest_path = os.path.join(args.output_dir, "rl_train_dataset.parquet")
    calib_df.to_parquet(calibration_path, index=False)
    train_rest_df.to_parquet(train_rest_path, index=False)
    excluded_ids_path = os.path.join(args.output_dir, "calibration_row_ids.json")
    with open(excluded_ids_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_calibration_rows": len(calib_ids),
                "orig_row_ids": sorted(calib_ids),
                "guarantee": "These rows are excluded from rl_train_dataset.parquet.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    lambda_path = None
    if not args.skip_retrieval_calibration:
        llm_cfg = None
        if args.enable_llm_hop_extraction:
            api_key = args.llm_api_key or os.getenv(args.llm_api_key_env)
            if not api_key:
                raise ValueError(
                    "LLM hop extraction enabled but API key is missing. "
                    f"Set --llm_api_key or env `{args.llm_api_key_env}`."
                )
            llm_cfg = {
                "api_base": args.llm_api_base,
                "api_key": api_key,
                "model": args.llm_model,
                "timeout": args.llm_timeout,
                "temperature": args.llm_temperature,
            }

        source_counts = calib_df.apply(get_data_source, axis=1).value_counts(dropna=False).to_dict()
        print(f"[Calibration Source Mix] {source_counts}")
        has_hotpotqa = any(("hotpotqa" in str(src)) or str(src).startswith("hotpot") for src in source_counts.keys())
        if has_hotpotqa and llm_cfg is None:
            raise ValueError(
                "Calibration set contains hotpotqa, which requires teacher-LLM hop extraction. "
                "Please set --enable_llm_hop_extraction with valid llm_api_key/api_base/model."
            )

        flattened_pairs: list[dict[str, Any]] = []
        teacher_hop_trace_records: list[dict[str, Any]] = []
        for _, row in tqdm(calib_df.iterrows(), total=len(calib_df), desc="Flatten trajectories"):
            row_pairs = extract_pairs_for_calibration_row(
                row=row,
                llm_cfg=llm_cfg,
                llm_only_when_no_trajectory=args.llm_only_when_no_trajectory,
                teacher_hop_trace_collector=teacher_hop_trace_records,
            )
            for hop_idx, (query, answer) in enumerate(row_pairs):
                flattened_pairs.append(
                    {
                        "query": query,
                        "answer": answer,
                        "hop_idx": hop_idx,
                        "orig_row_id": row.get("orig_row_id"),
                        "calibration_id": row.get("calibration_id"),
                        "data_source": get_data_source(row),
                    }
                )

        if not flattened_pairs:
            raise ValueError("No (query, answer) pairs extracted from calibration trajectories.")

        # 保存 teacher-LLM 的多跳拆分结果，便于人工检查“如何分割多跳问题”
        if teacher_hop_trace_records:
            hop_trace_path = os.path.join(args.output_dir, "teacher_llm_hop_decomposition.jsonl")
            _write_jsonl(hop_trace_path, teacher_hop_trace_records)
            print(f"[Teacher LLM] hop decomposition traces -> {hop_trace_path}")

        positive_scores: list[float] = []
        aps_calibration_items: list[dict[str, Any]] = []
        matched_docs = 0
        total_docs = 0
        teacher_doc_judge_records: list[dict[str, Any]] = []

        for pair_item in tqdm(flattened_pairs, total=len(flattened_pairs), desc="Trajectory retrieval calibration"):
            query = str(pair_item.get("query", ""))
            answer = str(pair_item.get("answer", ""))
            retrieved = call_retrieval(
                retrieval_service_url=args.retrieval_service_url,
                query=query,
                topk=args.retrieval_topk,
                timeout=args.retrieval_timeout,
            )
            total_docs += len(retrieved)

            # 先筛出分数可用的条目，再做 teacher 判定与 APS 样本构造
            valid_docs: list[dict[str, Any]] = []
            valid_scores: list[float] = []
            for doc_item in retrieved:
                try:
                    score = float(doc_item["score"])
                except (TypeError, ValueError, KeyError):
                    continue
                valid_docs.append(doc_item)
                valid_scores.append(score)

            golden_index: int = -1
            golden_indices: list[int] = []

            if llm_cfg is not None and valid_docs:
                try:
                    golden_index, golden_indices, judge_trace = _llm_select_golden_docs_from_retrieval(
                        query=query,
                        answer=answer,
                        retrieved_docs=valid_docs,
                        llm_cfg=llm_cfg,
                        pair_meta={
                            "orig_row_id": pair_item.get("orig_row_id"),
                            "calibration_id": pair_item.get("calibration_id"),
                            "data_source": pair_item.get("data_source"),
                            "hop_idx": pair_item.get("hop_idx"),
                        },
                    )
                    teacher_doc_judge_records.append(judge_trace)
                except Exception as e:
                    teacher_doc_judge_records.append(
                        {
                            "trace_type": "golden_doc_judgement",
                            "status": "exception",
                            "query": query,
                            "answer": answer,
                            "orig_row_id": pair_item.get("orig_row_id"),
                            "calibration_id": pair_item.get("calibration_id"),
                            "data_source": pair_item.get("data_source"),
                            "hop_idx": pair_item.get("hop_idx"),
                            "num_retrieved_docs": len(valid_docs),
                            "golden_doc_indices": [],
                            "best_golden_doc_index": -1,
                            "error": str(e),
                        }
                    )
                    golden_index = -1
                    golden_indices = []

            # 若 teacher 结果为空，回退到旧逻辑，避免全量样本被置为未命中
            if not golden_indices and valid_docs:
                for local_idx, doc_item in enumerate(valid_docs):
                    doc_text = doc_to_text(doc_item)
                    if contains_any_answer(doc_text, [answer]):
                        golden_indices.append(local_idx)
                if golden_indices:
                    golden_index = golden_indices[0]

            for idx in golden_indices:
                if 0 <= idx < len(valid_scores):
                    # 兼容旧 static CP 校准（LAMBDA_FIXED）
                    positive_scores.append(-valid_scores[idx])
                    matched_docs += 1

            aps_calibration_items.append(
                {
                    "scores": valid_scores,
                    "golden_index": golden_index,
                    "scores_are_s_doc": False,
                }
            )

        if teacher_doc_judge_records:
            judge_trace_path = os.path.join(args.output_dir, "teacher_llm_golden_doc_judgement.jsonl")
            _write_jsonl(judge_trace_path, teacher_doc_judge_records)
            print(f"[Teacher LLM] golden doc judgement traces -> {judge_trace_path}")

        if not aps_calibration_items:
            raise ValueError("No retrieval calibration items found; cannot estimate APS q_hat.")

        aps_q_hat, aps_e_values = calibrate_aps(
            calibration_items=aps_calibration_items,
            aps_temperature=dcr_cfg.aps_temperature,
            aps_alpha=dcr_cfg.aps_alpha,
        )

        lambda_fixed: float | None = None
        if positive_scores:
            quantile = 1.0 - float(dcr_cfg.cp_alpha)
            positive_np = np.array(positive_scores, dtype=np.float32)
            try:
                lambda_fixed = float(np.quantile(positive_np, quantile, method="higher"))
            except TypeError:
                lambda_fixed = float(np.quantile(positive_np, quantile, interpolation="higher"))
        else:
            print("[WARN] No positive retrieval samples for static CP; LAMBDA_FIXED will be null.")

        lambda_path = os.path.join(args.output_dir, "lambda_fixed.json")
        with open(lambda_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "LAMBDA_FIXED": lambda_fixed,
                    "cp_alpha": float(dcr_cfg.cp_alpha),
                    "target_coverage": 1.0 - float(dcr_cfg.cp_alpha),
                    "cp_k_max": int(dcr_cfg.cp_k_max),
                    "APS_Q_HAT": float(aps_q_hat),
                    "aps_q_hat": float(aps_q_hat),
                    "aps_alpha": float(dcr_cfg.aps_alpha),
                    "aps_target_coverage": 1.0 - float(dcr_cfg.aps_alpha),
                    "aps_temperature": float(dcr_cfg.aps_temperature),
                    "aps_min_docs": int(dcr_cfg.aps_min_docs),
                    "aps_max_docs": int(dcr_cfg.aps_max_docs),
                    "num_flattened_pairs": len(flattened_pairs),
                    "num_positive_scores": len(positive_scores),
                    "num_matched_docs": matched_docs,
                    "num_total_docs": total_docs,
                    "num_aps_items": len(aps_calibration_items),
                    "aps_e_mean": float(aps_e_values.mean()) if aps_e_values.size > 0 else None,
                    "aps_e_std": float(aps_e_values.std()) if aps_e_values.size > 0 else None,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(
            "[APS Calibration] "
            f"q_hat={aps_q_hat:.6f} (aps_alpha={dcr_cfg.aps_alpha}, temperature={dcr_cfg.aps_temperature}) "
            f"-> {lambda_path}"
        )
        print(
            "[Trajectory Calibration] "
            f"LAMBDA_FIXED={lambda_fixed if lambda_fixed is not None else 'null'} "
            f"(cp_alpha={dcr_cfg.cp_alpha}, cp_k_max={dcr_cfg.cp_k_max}) "
            f"-> {lambda_path}"
        )

    scores_path = None
    if not args.skip_base_scoring:
        if not args.base_model:
            raise ValueError("base_model is required unless --skip_base_scoring is set.")

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[args.torch_dtype]

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=args.device_map,
        )
        model.eval()
        model_device = next(model.parameters()).device

        open_patterns = build_tag_patterns(tokenizer, "<answer>")
        close_patterns = build_tag_patterns(tokenizer, "</answer>")
        if not open_patterns or not close_patterns:
            raise ValueError("Failed to tokenize <answer> tags for base-model scoring.")

        all_scores = []
        for _, row in tqdm(calib_df.iterrows(), total=len(calib_df), desc="Base model scoring"):
            messages = ensure_messages(row["prompt"])
            prompt_text = build_prompt_text(messages, tokenizer)
            model_inputs = tokenizer(prompt_text, return_tensors="pt")
            model_inputs = {k: v.to(model_device) for k, v in model_inputs.items()}

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "return_dict_in_generate": True,
                "output_scores": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.temperature > 0:
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = args.top_p

            with torch.no_grad():
                output = model.generate(**model_inputs, **gen_kwargs)

            generated_ids = output.sequences[0, model_inputs["input_ids"].shape[1] :]
            if len(output.scores) == 0 or generated_ids.numel() == 0:
                all_scores.append(torch.tensor(20.0))
                continue

            token_log_probs = []
            for step_scores, token_id in zip(output.scores, generated_ids):
                log_probs = torch.log_softmax(step_scores[0], dim=-1)
                token_log_probs.append(log_probs[token_id].detach().cpu())
            token_log_probs = torch.stack(token_log_probs).float()
            gen_token_ids = generated_ids.detach().cpu().tolist()

            open_start, open_len = find_first_pattern(gen_token_ids, open_patterns, start=0)
            if open_start >= 0:
                answer_start = open_start + open_len
                close_start, _ = find_first_pattern(gen_token_ids, close_patterns, start=answer_start)
            else:
                answer_start, close_start = -1, -1

            answer_mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
            if answer_start >= 0 and close_start > answer_start:
                answer_mask[answer_start:close_start] = True
            else:
                answer_mask[:] = True

            nll = -token_log_probs[answer_mask].mean()
            all_scores.append(nll.detach().cpu())

        initial_scores = torch.stack(all_scores).float()
        scores_path = os.path.join(args.output_dir, "initial_scores.pt")
        torch.save(initial_scores, scores_path)
        print(
            "[Base Scoring] initial_scores stats: "
            f"mean={initial_scores.mean().item():.6f}, std={initial_scores.std().item():.6f}, "
            f"num={initial_scores.numel()} -> {scores_path}"
        )

    return CalibrationOutputs(
        calibration_path=calibration_path,
        train_rest_path=train_rest_path,
        lambda_path=lambda_path,
        scores_path=scores_path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare DCR-GRPO calibration artifacts.")
    parser.add_argument("--train_parquet", type=str, required=True, help="Path to original RL train parquet.")
    parser.add_argument("--output_dir", type=str, default="data/calibration", help="Output directory.")
    parser.add_argument(
        "--calib_sample_size",
        type=int,
        default=DCRGRPOCalibConfig.calib_sample_size,
        help="Number of samples used for offline calibration.",
    )
    # backward compatibility alias
    parser.add_argument("--calibration_size", dest="calib_sample_size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--retrieval_service_url", type=str, default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--retrieval_topk", type=int, default=20)
    parser.add_argument("--retrieval_timeout", type=int, default=30)
    parser.add_argument(
        "--cp_alpha",
        type=float,
        default=DCRGRPOCalibConfig.cp_alpha,
        help="Miscoverage alpha for static CP threshold (target coverage = 1-alpha).",
    )
    # backward compatibility alias
    parser.add_argument("--alpha", dest="cp_alpha", type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--cp_k_max",
        type=int,
        default=DCRGRPOCalibConfig.cp_k_max,
        help="Recommended max doc cap after static CP filtering (saved into lambda_fixed.json).",
    )
    parser.add_argument(
        "--aps_temperature",
        type=float,
        default=DCRGRPOCalibConfig.aps_temperature,
        help="Temperature used by APS softmax scaling.",
    )
    parser.add_argument(
        "--aps_alpha",
        type=float,
        default=DCRGRPOCalibConfig.aps_alpha,
        help="APS miscoverage alpha; q_hat quantile is computed at (1 - aps_alpha).",
    )
    parser.add_argument(
        "--aps_min_docs",
        type=int,
        default=DCRGRPOCalibConfig.aps_min_docs,
        help="Recommended APS lower bound for online truncation.",
    )
    parser.add_argument(
        "--aps_max_docs",
        type=int,
        default=DCRGRPOCalibConfig.aps_max_docs,
        help="Recommended APS upper bound for online truncation.",
    )
    parser.add_argument("--skip_retrieval_calibration", action="store_true")
    # backward compatibility alias
    parser.add_argument("--skip_teacher_calibration", dest="skip_retrieval_calibration", action="store_true")
    parser.add_argument("--enable_llm_hop_extraction", action="store_true")
    parser.add_argument("--llm_only_when_no_trajectory", dest="llm_only_when_no_trajectory", action="store_true")
    parser.add_argument("--llm_force_for_all", dest="llm_only_when_no_trajectory", action="store_false")
    parser.set_defaults(llm_only_when_no_trajectory=True)
    parser.add_argument("--llm_api_base", type=str, default="https://api.deepseek.com")
    parser.add_argument("--llm_api_key", type=str, default=None)
    parser.add_argument("--llm_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--llm_model", type=str, default="deepseek-chat")
    parser.add_argument("--llm_timeout", type=int, default=60)
    parser.add_argument("--llm_temperature", type=float, default=0.0)

    parser.add_argument("--base_model", type=str, default=None, help="Base model name/path for initial scoring.")
    parser.add_argument("--device_map", type=str, default="auto", help="HuggingFace device_map.")
    parser.add_argument("--torch_dtype", type=str, choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--skip_base_scoring", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outputs = run(args)
    print(json.dumps(outputs.__dict__, indent=2, ensure_ascii=False))
