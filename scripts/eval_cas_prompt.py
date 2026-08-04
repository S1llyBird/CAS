#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# VERL retains the legacy Search-R1 utility module names as its protocol API.
from cas.reward_score.cas_format import compute_score_em, extract_solution
from verl.tools.utils.search_r1_like_utils import perform_single_search_batch
from verl.tools.utils.search_r1_postprocess import extract_first_search_query, truncate_after_first_search_request


ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)


def maybe_parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def ensure_messages(prompt_obj: Any) -> list[dict[str, str]]:
    if isinstance(prompt_obj, np.ndarray):
        prompt_obj = prompt_obj.tolist()
    if isinstance(prompt_obj, pd.Series):
        prompt_obj = prompt_obj.tolist()
    if isinstance(prompt_obj, tuple):
        prompt_obj = list(prompt_obj)
    if isinstance(prompt_obj, str):
        prompt_obj = json.loads(prompt_obj)
    if not isinstance(prompt_obj, list):
        raise ValueError(f"Unsupported prompt format: {type(prompt_obj)}")
    return prompt_obj


def safe_json_dumps(value: Any) -> str:
    def default(obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return str(obj)

    return json.dumps(value, ensure_ascii=False, default=default)


def get_question(row: pd.Series) -> str:
    extra_info = maybe_parse_json(row.get("extra_info"))
    if isinstance(extra_info, dict) and isinstance(extra_info.get("question"), str):
        return extra_info["question"]
    for message in ensure_messages(row["prompt"]):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def get_ground_truth(row: pd.Series) -> dict[str, Any]:
    reward_model = maybe_parse_json(row.get("reward_model"))
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth", {})
        if isinstance(ground_truth, dict):
            return ground_truth
    return {"target": []}


def build_prompt_text(messages: list[dict[str, str]], tokenizer: Any) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chunks = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
    chunks.append("assistant:")
    return "\n".join(chunks)


def get_torch_dtype(name: str):
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def get_input_device(model: Any) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for device in hf_device_map.values():
            if device not in ("cpu", "disk", None):
                return torch.device(device)
    return next(model.parameters()).device


def load_aps_settings(path: str, fallback_temperature: float, fallback_min_docs: int, fallback_max_docs: int) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    q_hat = None
    for key in ("APS_Q_HAT", "aps_q_hat", "q_hat"):
        if cfg.get(key) is not None:
            q_hat = float(cfg[key])
            break
    if q_hat is None:
        raise ValueError(f"No APS q_hat found in {path}. Expected APS_Q_HAT, aps_q_hat, or q_hat.")

    return {
        "aps_q_hat": q_hat,
        "aps_temperature": float(cfg.get("aps_temperature", fallback_temperature)),
        "aps_min_docs": int(cfg.get("aps_min_docs", fallback_min_docs)),
        "aps_max_docs": int(cfg.get("aps_max_docs", fallback_max_docs)),
    }


def search_once(query: str, args: argparse.Namespace, aps_settings: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    if args.enable_aps:
        assert aps_settings is not None
        return perform_single_search_batch(
            retrieval_service_url=args.retrieval_service_url,
            query_list=[query],
            topk=args.topk_aps,
            timeout=args.retrieval_timeout,
            cp_filter_mode="aps",
            aps_q_hat=aps_settings["aps_q_hat"],
            aps_temperature=aps_settings["aps_temperature"],
            aps_min_docs=aps_settings["aps_min_docs"],
            aps_max_docs=aps_settings["aps_max_docs"],
        )

    return perform_single_search_batch(
        retrieval_service_url=args.retrieval_service_url,
        query_list=[query],
        topk=args.topk_no_aps,
        timeout=args.retrieval_timeout,
        cp_filter_mode="off",
    )


def compact_search_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_count": metadata.get("query_count"),
        "status": metadata.get("status"),
        "total_results": metadata.get("total_results"),
        "api_request_error": metadata.get("api_request_error"),
        "cp_filter": metadata.get("cp_filter"),
    }


def truncate_left_to_context(tokenizer: Any, text: str, max_context_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_context_tokens:
        return text
    ids = ids[-max_context_tokens:]
    return tokenizer.decode(ids, skip_special_tokens=False)


def generate_chunk_hf(model: Any, tokenizer: Any, prompt_text: str, args: argparse.Namespace) -> str:
    prompt_text = truncate_left_to_context(
        tokenizer=tokenizer,
        text=prompt_text,
        max_context_tokens=max(1, args.max_context_tokens - args.max_new_tokens_per_turn),
    )
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_device = get_input_device(model)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens_per_turn,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    generated_ids = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=False)


def post_json(url: str, payload: dict[str, Any], timeout: int, api_key: str | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible request failed: HTTP {e.code}: {body}") from e


def generate_chunk_openai(tokenizer: Any, prompt_text: str, args: argparse.Namespace) -> str:
    prompt_text = truncate_left_to_context(
        tokenizer=tokenizer,
        text=prompt_text,
        max_context_tokens=max(1, args.max_context_tokens - args.max_new_tokens_per_turn),
    )
    payload = {
        "model": args.openai_model or args.model_path,
        "prompt": prompt_text,
        "max_tokens": args.max_new_tokens_per_turn,
        "temperature": max(0.0, args.temperature),
        "top_p": args.top_p,
    }
    result = post_json(
        url=f"{args.openai_base_url.rstrip('/')}/v1/completions",
        payload=payload,
        timeout=args.openai_timeout,
        api_key=args.openai_api_key,
    )
    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenAI-compatible response has no choices: {result}")
    return choices[0].get("text", "")


def generate_chunk(model: Any | None, tokenizer: Any, prompt_text: str, args: argparse.Namespace) -> str:
    if args.generation_backend == "openai":
        return generate_chunk_openai(tokenizer=tokenizer, prompt_text=prompt_text, args=args)
    if model is None:
        raise ValueError("HF generation backend requires a loaded model.")
    return generate_chunk_hf(model=model, tokenizer=tokenizer, prompt_text=prompt_text, args=args)


def run_one(row: pd.Series, model: Any | None, tokenizer: Any, args: argparse.Namespace, aps_settings: dict[str, Any] | None):
    messages = ensure_messages(row["prompt"])
    initial_prompt = build_prompt_text(messages, tokenizer)
    full_text = initial_prompt
    response_parts: list[str] = []
    search_traces: list[dict[str, Any]] = []

    for turn_idx in range(args.max_assistant_turns):
        chunk = generate_chunk(model=model, tokenizer=tokenizer, prompt_text=full_text, args=args)
        if not chunk:
            break

        chunk = truncate_after_first_search_request(chunk)
        full_text += chunk
        response_parts.append(chunk)

        response_text = "".join(response_parts)
        if ANSWER_RE.search(response_text):
            break

        query = extract_first_search_query(chunk)
        if not query:
            break

        search_text, metadata = search_once(query=query, args=args, aps_settings=aps_settings)
        information = f"<information>{search_text}</information>\n"
        full_text += information
        response_parts.append(information)
        search_traces.append(
            {
                "turn": turn_idx,
                "query": query,
                "metadata": compact_search_metadata(metadata),
            }
        )

    return "".join(response_parts), search_traces


def evaluate_row(
    row_id: int,
    row: pd.Series,
    model: Any | None,
    tokenizer: Any,
    args: argparse.Namespace,
    aps_settings: dict[str, Any] | None,
) -> tuple[dict[str, Any], str, float, float]:
    generated, search_traces = run_one(row=row, model=model, tokenizer=tokenizer, args=args, aps_settings=aps_settings)
    ground_truth = get_ground_truth(row)
    data_source = str(row.get("data_source", "unknown"))
    extra_info = maybe_parse_json(row.get("extra_info"))

    reward_result = compute_score_em(
        solution_str=generated,
        ground_truth=ground_truth,
        data_source=data_source,
        extra_info=extra_info,
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0,
        format_score=0,
        score=1.0,
    )
    reward = float(reward_result["score"])
    acc = float(reward_result.get("acc", 0.0))
    record = {
        "row_id": int(row_id),
        "data_source": data_source,
        "question": get_question(row),
        "ground_truth": ground_truth,
        "prediction": extract_solution(generated),
        "generated": generated,
        "reward": reward,
        "metrics": {k: v for k, v in reward_result.items() if k != "score"},
        "search_traces": search_traces,
        "aps_enabled": bool(args.enable_aps),
    }
    return record, data_source, reward, acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt-only CAS evaluation.")
    parser.add_argument("--generation_backend", default="hf", choices=["hf", "openai"])
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--openai_base_url", default="http://127.0.0.1:30000")
    parser.add_argument("--openai_model", default=None)
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--openai_timeout", type=int, default=120)
    parser.add_argument("--data_path", default="data/cas_subset_test.parquet")
    parser.add_argument("--output_dir", default="outputs/prompt_eval/cas_qwen2.5-3b-instruct")
    parser.add_argument("--output_name", default=None)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--retrieval_service_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--retrieval_timeout", type=int, default=30)
    parser.add_argument("--enable_aps", action="store_true")
    parser.add_argument("--lambda_fixed_path", default="/home/zixizhu/un_rag/CAS/data/calibration/lambda_fixed.json")
    parser.add_argument("--topk_no_aps", type=int, default=3)
    parser.add_argument("--topk_aps", type=int, default=5)
    parser.add_argument("--aps_temperature", type=float, default=0.01)
    parser.add_argument("--aps_min_docs", type=int, default=2)
    parser.add_argument("--aps_max_docs", type=int, default=5)

    parser.add_argument("--max_assistant_turns", type=int, default=4)
    parser.add_argument("--max_new_tokens_per_turn", type=int, default=768)
    parser.add_argument("--max_context_tokens", type=int, default=15000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--max_memory",
        default=None,
        help='Optional Transformers max_memory value for each visible GPU, e.g. "22GiB".',
    )
    parser.add_argument("--local_files_only", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    mode_name = "aps" if args.enable_aps else "topk3"
    output_name = args.output_name or f"predictions_{mode_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    jsonl_path = os.path.join(args.output_dir, output_name + ".jsonl")
    summary_path = os.path.join(args.output_dir, output_name + "_summary.json")

    aps_settings = None
    if args.enable_aps:
        aps_settings = load_aps_settings(
            path=args.lambda_fixed_path,
            fallback_temperature=args.aps_temperature,
            fallback_min_docs=args.aps_min_docs,
            fallback_max_docs=args.aps_max_docs,
        )
        print(f"[APS] enabled with settings: {aps_settings}")
    else:
        print(f"[APS] disabled; using topk={args.topk_no_aps} without post-filtering.")

    df = pd.read_parquet(args.data_path)
    if args.max_samples > 0:
        df = df.head(args.max_samples)
    print(f"[Data] loaded {len(df)} rows from {args.data_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=bool(args.local_files_only))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = None
    if args.generation_backend == "hf":
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": get_torch_dtype(args.torch_dtype),
            "device_map": args.device_map,
            "local_files_only": bool(args.local_files_only),
        }
        if args.max_memory:
            visible_gpu_count = torch.cuda.device_count()
            model_kwargs["max_memory"] = {idx: args.max_memory for idx in range(visible_gpu_count)}

        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
        model.eval()
    else:
        print(f"[Generation] using OpenAI-compatible backend at {args.openai_base_url}")

    metric_by_source: dict[str, list[float]] = defaultdict(list)
    acc_by_source: dict[str, list[float]] = defaultdict(list)
    rows_written = 0

    with open(jsonl_path, "w", encoding="utf-8") as f:
        if args.generation_backend == "openai" and args.num_workers > 1:
            print(f"[Eval] using {args.num_workers} concurrent workers.")
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [
                    executor.submit(
                        evaluate_row,
                        int(i),
                        row,
                        model,
                        tokenizer,
                        args,
                        aps_settings,
                    )
                    for i, row in df.iterrows()
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Prompt eval"):
                    record, data_source, reward, acc = future.result()
                    metric_by_source[data_source].append(reward)
                    acc_by_source[data_source].append(acc)
                    f.write(safe_json_dumps(record) + "\n")
                    rows_written += 1
        else:
            if args.num_workers > 1:
                print("[Eval] num_workers is only used with --generation_backend openai; running serially.")
            for i, row in tqdm(df.iterrows(), total=len(df), desc="Prompt eval"):
                record, data_source, reward, acc = evaluate_row(
                    row_id=int(i),
                    row=row,
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                    aps_settings=aps_settings,
                )
                metric_by_source[data_source].append(reward)
                acc_by_source[data_source].append(acc)
                f.write(safe_json_dumps(record) + "\n")
                rows_written += 1

    summary = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "rows": rows_written,
        "aps_enabled": bool(args.enable_aps),
        "retrieval_service_url": args.retrieval_service_url,
        "topk": args.topk_aps if args.enable_aps else args.topk_no_aps,
        "aps_settings": aps_settings,
        "reward_mean": float(np.mean([x for values in metric_by_source.values() for x in values])) if rows_written else None,
        "acc_mean": float(np.mean([x for values in acc_by_source.values() for x in values])) if rows_written else None,
        "reward_by_source": {k: float(np.mean(v)) for k, v in metric_by_source.items()},
        "acc_by_source": {k: float(np.mean(v)) for k, v in acc_by_source.items()},
        "jsonl_path": jsonl_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(safe_json_dumps(summary) + "\n")

    print(f"[Done] predictions -> {jsonl_path}")
    print(f"[Done] summary -> {summary_path}")
    print(safe_json_dumps(summary))


if __name__ == "__main__":
    main()
