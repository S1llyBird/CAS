#!/usr/bin/env python3
"""
CP 检索诊断脚本
==============
目标：分析静态 CP 过滤为什么只把 Doc 1 返回给模型。

支持直接传入 VERL/FSDP checkpoint 路径，脚本会自动合并成 HF 格式后加载。

运行方式（使用默认路径）：
    CUDA_VISIBLE_DEVICES=0,1 python scripts/debug_cp_retrieval.py

或自定义参数：
    CUDA_VISIBLE_DEVICES=0,1 python scripts/debug_cp_retrieval.py \\
        --step  700 \\
        --n_samples  5 \\
        --topk  10

注意：检索服务必须在运行中。
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def call_retrieval_raw(url: str, queries: list[str], topk: int, timeout: int = 30) -> dict:
    """调用检索服务，返回原始 JSON（含 scores）。"""
    payload = {"queries": queries, "topk": topk, "return_scores": True}
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def apply_cp_filter(
    docs_with_scores: list[dict],
    lambda_fixed: float | None,
    cp_k_max: int,
) -> tuple[list[dict], list[dict]]:
    """
    模拟 _apply_static_cp_filter 的逻辑：
      s_doc = -score，保留 s_doc <= lambda_fixed 的文档。
    返回 (kept, filtered_out)。
    """
    if lambda_fixed is None:
        return docs_with_scores, []

    kept, rejected = [], []
    for item in docs_with_scores:
        try:
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            rejected.append(item)
            continue
        s_doc = -score
        if s_doc <= lambda_fixed:
            kept.append(item)
        else:
            rejected.append(item)

    if cp_k_max > 0:
        kept = kept[:cp_k_max]
    return kept, rejected


def ensure_messages(prompt_obj: Any) -> list[dict]:
    if isinstance(prompt_obj, (np.ndarray,)):
        prompt_obj = prompt_obj.tolist()
    if isinstance(prompt_obj, list):
        return prompt_obj
    if isinstance(prompt_obj, str):
        parsed = json.loads(prompt_obj)
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"Cannot parse prompt: {type(prompt_obj)}")


def format_doc(doc_item: dict, idx: int) -> str:
    """给单个文档格式化用于打印。"""
    score = doc_item.get("score", "N/A")
    s_doc = -float(score) if isinstance(score, (int, float)) else "N/A"
    doc = doc_item.get("document", {})
    contents = doc.get("contents", "") if isinstance(doc, dict) else str(doc)
    title_line = contents.split("\n")[0] if contents else "(no title)"
    snippet = contents[:200].replace("\n", " ") if contents else ""
    score_str = f"{score:.6f}" if isinstance(score, (int, float)) else str(score)
    s_doc_str = f"{s_doc:.6f}" if isinstance(s_doc, float) else str(s_doc)
    return (
        f"  [{idx}] score={score_str}  s_doc(-score)={s_doc_str}\n"
        f"       title: {title_line}\n"
        f"       snippet: {snippet}..."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint 解析（支持 FSDP 自动合并）
# ──────────────────────────────────────────────────────────────────────────────

# 真实训练产出的 checkpoint 根目录
_CKPT_ROOT = "/home/zixizhu/un_rag/CAS/checkpoints/CAS/CAS_qwen2.5-3b-instruct_grpo"


def resolve_hf_checkpoint(ckpt_path: str, step: int | None) -> str:
    """
    将 checkpoint 路径解析为 HuggingFace 格式目录。

    处理逻辑：
    1. 如果 ckpt_path 已包含 config.json → 直接返回（已是 HF 格式）
    2. 如果是 FSDP 分片目录（actor/ 子目录） → 自动调用 verl.model_merger 合并
    3. 如果 ckpt_path 是 global_step_N 上层目录且 step 非空 → 拼上 /global_step_N/actor
    """
    import subprocess
    import sys

    # 若传入的就是 HF 格式目录，直接用
    if os.path.isfile(os.path.join(ckpt_path, "config.json")):
        print(f"[INFO] checkpoint 已是 HF 格式，直接加载: {ckpt_path}")
        return ckpt_path

    # 尝试拼接 global_step_N/actor
    fsdp_actor_path = ckpt_path
    if step is not None and not ckpt_path.rstrip("/").endswith("actor"):
        fsdp_actor_path = os.path.join(ckpt_path, f"global_step_{step}", "actor")

    if not os.path.isdir(fsdp_actor_path):
        raise FileNotFoundError(
            f"找不到 FSDP checkpoint 目录: {fsdp_actor_path}\n"
            f"请检查 --checkpoint 或 --step 是否正确。"
        )

    # 合并目标放到临时目录，避免污染原始 ckpt
    merge_target = f"/tmp/cas_debug_hf/step{step if step is not None else 'x'}"
    if os.path.isfile(os.path.join(merge_target, "config.json")):
        print(f"[INFO] 检测到已合并的 HF checkpoint，跳过合并: {merge_target}")
        return merge_target

    print(f"[INFO] 检测到 FSDP 分片格式，开始合并 → {merge_target}")
    print(f"       fsdp_actor_path = {fsdp_actor_path}")
    os.makedirs(merge_target, exist_ok=True)
    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", fsdp_actor_path,
        "--target_dir", merge_target,
    ]
    print(f"[INFO] 执行: {' '.join(cmd)}")
    ret = subprocess.run(cmd, check=True)
    if ret.returncode != 0:
        raise RuntimeError(f"model_merger 合并失败，返回码: {ret.returncode}")
    print(f"[INFO] 合并完成 → {merge_target}")
    return merge_target


# ──────────────────────────────────────────────────────────────────────────────
# 主体：加载模型，对每个样本做搜索，并分析 CP
# ──────────────────────────────────────────────────────────────────────────────

def extract_queries_from_generation(generated_text: str) -> list[list[str]]:
    """从模型生成文本中提取所有 <search> 里的查询。"""
    import re
    searches = re.findall(r"<search>(.*?)</search>", generated_text, re.DOTALL)
    return [[q.strip()] for q in searches if q.strip()]


def run_diagnosis(args):
    # ── 0. 加载 lambda_fixed ───────────────────────────────────────────────
    lambda_fixed: float | None = None
    cp_k_max: int = args.cp_k_max

    if args.lambda_fixed_path and os.path.exists(args.lambda_fixed_path):
        with open(args.lambda_fixed_path, "r") as f:
            ld = json.load(f)
        lambda_fixed = float(ld["LAMBDA_FIXED"])
        # 如果文件里记录了 cp_k_max，也读出来作为参考
        if "cp_k_max" in ld:
            print(f"[INFO] lambda_fixed.json 中记录的 cp_k_max = {ld['cp_k_max']}")
        print(f"\n{'='*70}")
        print(f"  LAMBDA_FIXED = {lambda_fixed:.8f}  (来自 {args.lambda_fixed_path})")
        print(f"  cp_alpha     = {ld.get('cp_alpha', 'N/A')}")
        print(f"  target_coverage = {ld.get('target_coverage', 'N/A')}")
        print(f"  num_positive_scores = {ld.get('num_positive_scores', 'N/A')}")
        print(f"  cp_k_max (本次使用) = {cp_k_max}")
        print(f"{'='*70}\n")
    elif args.lambda_fixed is not None:
        lambda_fixed = float(args.lambda_fixed)
        print(f"[INFO] 使用命令行传入的 lambda_fixed = {lambda_fixed}")
    else:
        print("[WARN] 未提供 lambda_fixed，将跳过 CP 过滤分析（显示原始检索结果）")

    # ── 1. 加载数据集 ──────────────────────────────────────────────────────
    df = pd.read_parquet(args.val_parquet)
    sample_df = df.sample(n=min(args.n_samples, len(df)), random_state=42)
    print(f"[INFO] 从 {args.val_parquet} 中随机抽取 {len(sample_df)} 条样本\n")

    # ── 2. 解析 checkpoint（自动合并 FSDP） ──────────────────────────────────
    ckpt_path = resolve_hf_checkpoint(
        ckpt_path=os.path.abspath(args.checkpoint),
        step=args.step,
    )

    print(f"[INFO] 加载 tokenizer from: {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[INFO] 加载模型 from: {ckpt_path}")
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",   # 自动分配到可用 GPU（CUDA_VISIBLE_DEVICES=0,1）
    )
    model.eval()
    print(f"[INFO] 模型加载完成，设备: {next(model.parameters()).device}\n")

    # ── 3. 对每个样本做诊断 ────────────────────────────────────────────────
    for sample_idx, (_, row) in enumerate(sample_df.iterrows()):
        messages = ensure_messages(row["prompt"])
        question_text = ""
        for m in messages:
            if m.get("role") == "user":
                question_text = m.get("content", "")
                break

        print(f"\n{'#'*70}")
        print(f"  样本 {sample_idx + 1}/{len(sample_df)}")
        print(f"{'#'*70}")
        print(f"  问题: {textwrap.shorten(question_text, width=120)}")

        # ── 3a. 模型生成（单轮，不带 tool loop） ────────────────────────────
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        print(f"\n  --- 模型生成文本（前 500 字符）---")
        print(textwrap.indent(generated_text[:500], "  "))

        # ── 3b. 提取模型生成的查询 ────────────────────────────────────────
        query_batches = extract_queries_from_generation(generated_text)
        if not query_batches:
            # 从 user content 中提取问题：去掉 system prompt 前缀，只取 "Question: " 之后的内容
            import re as _re
            q_match = _re.search(r"Question:\s*(.+)", question_text, _re.DOTALL)
            raw_question = q_match.group(1).strip() if q_match else question_text
            raw_question = raw_question[:300]
            print(f"\n  [!] 模型未生成任何 <search>，用原始问题作为查询: {raw_question[:80]}...")
            query_batches = [[raw_question]]

        print(f"\n  --- 模型生成的查询 ({len(query_batches)} 次 <search>) ---")
        for bi, ql in enumerate(query_batches):
            print(f"  <search>[{bi}]: {ql}")

        # ── 3c. 对每次查询调用检索并分析 CP ─────────────────────────────────
        for bi, query_list in enumerate(query_batches):
            print(f"\n  ━━━ <search>[{bi}] 检索分析 ━━━")
            print(f"  queries = {query_list}")

            try:
                raw = call_retrieval_raw(
                    url=args.retrieval_url,
                    queries=query_list,
                    topk=args.topk,
                    timeout=30,
                )
            except Exception as e:
                print(f"  [ERROR] 检索服务调用失败: {e}")
                continue

            per_query_results = raw.get("result", [])

            for qi, query in enumerate(query_list):
                print(f"\n  [Query {qi}] \"{query}\"")
                docs = per_query_results[qi] if qi < len(per_query_results) else []

                if not docs:
                    print("    (无检索结果)")
                    continue

                print(f"  原始检索到 {len(docs)} 个文档（topk={args.topk}）：")
                for di, doc_item in enumerate(docs):
                    print(format_doc(doc_item, di + 1))

                # CP 过滤
                kept, rejected = apply_cp_filter(docs, lambda_fixed, cp_k_max)

                print(f"\n  CP 过滤结果（lambda_fixed={lambda_fixed}, cp_k_max={cp_k_max}）：")
                print(f"    保留: {len(kept)} 个文档")
                print(f"    过滤掉: {len(rejected)} 个文档")

                if lambda_fixed is not None:
                    scores = []
                    for item in docs:
                        try:
                            scores.append(float(item["score"]))
                        except Exception:
                            pass
                    if scores:
                        s_docs = [-s for s in scores]
                        print(f"\n  分数统计 (s_doc = -retrieval_score)：")
                        print(f"    s_doc 范围: [{min(s_docs):.6f}, {max(s_docs):.6f}]")
                        print(f"    LAMBDA_FIXED (阈值): {lambda_fixed:.6f}")
                        print(f"    保留条件: s_doc <= LAMBDA_FIXED")
                        below = [s for s in s_docs if s <= lambda_fixed]
                        above = [s for s in s_docs if s > lambda_fixed]
                        print(f"    满足条件的文档数: {len(below)} → s_docs: {[f'{s:.4f}' for s in below]}")
                        print(f"    被过滤文档数: {len(above)} → s_docs: {[f'{s:.4f}' for s in above]}")

                        # 诊断提示
                        if len(kept) == 1:
                            print(f"\n  ⚠️  诊断: 只有 1 个文档通过 CP 过滤！")
                            print(f"      可能原因：LAMBDA_FIXED={lambda_fixed:.6f} 过于严格（太大的负数），")
                            print(f"      导致只有分数最高（s_doc 最小）的 Doc 1 通过。")
                            print(f"      建议：检查校准集质量、cp_alpha 设置，或适当增大 lambda_fixed。")
                        elif len(kept) == 0:
                            print(f"\n  ⚠️  诊断: 没有文档通过 CP 过滤！全部被过滤。")

        print()  # 样本间空行

    print("\n" + "="*70)
    print("  诊断完成")
    print("="*70)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    # 硬编码的真实默认路径，直接运行无需传参
    _BASE = "/home/zixizhu/un_rag/CAS"
    _DEFAULT_STEP = 700
    _DEFAULT_CKPT = _CKPT_ROOT   # 只需传根目录，--step 指定步数
    _DEFAULT_VAL  = f"{_BASE}/data/cas_test.parquet"
    _DEFAULT_LAMBDA = f"{_BASE}/data/calibration/lambda_fixed.json"

    p = argparse.ArgumentParser(
        description="CP 检索诊断：分析为何只返回 Doc 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, default=_DEFAULT_CKPT,
                   help="checkpoint 根目录（FSDP 或 HF 格式均可，脚本自动判断）")
    p.add_argument("--step", type=int, default=_DEFAULT_STEP,
                   help="训练步数，用于定位 global_step_N/actor 子目录")
    p.add_argument("--val_parquet", type=str, default=_DEFAULT_VAL,
                   help="验证集 parquet 路径")
    p.add_argument("--retrieval_url", type=str, default="http://127.0.0.1:8000/retrieve",
                   help="检索服务 URL")
    p.add_argument("--lambda_fixed_path", type=str, default=_DEFAULT_LAMBDA,
                   help="lambda_fixed.json 路径")
    p.add_argument("--lambda_fixed", type=float, default=None,
                   help="直接指定 lambda_fixed 值（覆盖 lambda_fixed_path）")
    p.add_argument("--cp_k_max", type=int, default=10,
                   help="CP 过滤后保留的最大文档数")
    p.add_argument("--topk", type=int, default=10,
                   help="向检索服务请求的文档数")
    p.add_argument("--n_samples", type=int, default=5,
                   help="从 val_parquet 中随机选取的样本数")
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="模型最大生成 token 数")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_diagnosis(args)
