#!/usr/bin/env python3
"""
CP retrieval diagnostic
=======================
Purpose: determine why static CP filtering returns only Doc 1 to the model.

The script accepts a VERL/FSDP checkpoint path and automatically merges it
into Hugging Face format before loading it.

Run with the default paths:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/debug_cp_retrieval.py

Or provide custom arguments:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/debug_cp_retrieval.py \\
        --step  700 \\
        --n_samples  5 \\
        --topk  10

The retrieval service must already be running.
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
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def call_retrieval_raw(url: str, queries: list[str], topk: int, timeout: int = 30) -> dict:
    """Call the retrieval service and return its raw JSON, including scores."""
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
    Reproduce the logic of _apply_static_cp_filter:
      s_doc = -score; keep documents where s_doc <= lambda_fixed.
    Return (kept, filtered_out).
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
    """Format one document for display."""
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
# Checkpoint resolution with automatic FSDP merging
# ──────────────────────────────────────────────────────────────────────────────

# Root directory of the checkpoint produced by training.
_CKPT_ROOT = "/home/zixizhu/un_rag/CAS/checkpoints/CAS/CAS_qwen2.5-3b-instruct_grpo"


def resolve_hf_checkpoint(ckpt_path: str, step: int | None) -> str:
    """
    Resolve a checkpoint path to a Hugging Face-format directory.

    Resolution order:
    1. Return ckpt_path directly when it contains config.json.
    2. Merge an FSDP shard directory (the actor/ subdirectory) with verl.model_merger.
    3. Append /global_step_N/actor when ckpt_path is the parent directory and step is set.
    """
    import subprocess
    import sys

    # Use an existing Hugging Face-format directory directly.
    if os.path.isfile(os.path.join(ckpt_path, "config.json")):
        print(f"[INFO] Checkpoint is already in Hugging Face format: {ckpt_path}")
        return ckpt_path

    # Resolve the global_step_N/actor directory when needed.
    fsdp_actor_path = ckpt_path
    if step is not None and not ckpt_path.rstrip("/").endswith("actor"):
        fsdp_actor_path = os.path.join(ckpt_path, f"global_step_{step}", "actor")

    if not os.path.isdir(fsdp_actor_path):
        raise FileNotFoundError(
            f"FSDP checkpoint directory not found: {fsdp_actor_path}\n"
            f"Check whether --checkpoint and --step are correct."
        )

    # Write the merged checkpoint to a temporary directory to preserve the source.
    merge_target = f"/tmp/cas_debug_hf/step{step if step is not None else 'x'}"
    if os.path.isfile(os.path.join(merge_target, "config.json")):
        print(f"[INFO] Found an existing merged Hugging Face checkpoint: {merge_target}")
        return merge_target

    print(f"[INFO] Detected FSDP shards; merging into {merge_target}")
    print(f"       fsdp_actor_path = {fsdp_actor_path}")
    os.makedirs(merge_target, exist_ok=True)
    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", fsdp_actor_path,
        "--target_dir", merge_target,
    ]
    print(f"[INFO] Running: {' '.join(cmd)}")
    ret = subprocess.run(cmd, check=True)
    if ret.returncode != 0:
        raise RuntimeError(f"model_merger failed with exit code {ret.returncode}")
    print(f"[INFO] Merge complete: {merge_target}")
    return merge_target


# ──────────────────────────────────────────────────────────────────────────────
# Main workflow: load the model, search for each sample, and analyze CP
# ──────────────────────────────────────────────────────────────────────────────

def extract_queries_from_generation(generated_text: str) -> list[list[str]]:
    """Extract queries from all <search> blocks in generated text."""
    import re
    searches = re.findall(r"<search>(.*?)</search>", generated_text, re.DOTALL)
    return [[q.strip()] for q in searches if q.strip()]


def run_diagnosis(args):
    # ── 0. Load lambda_fixed ───────────────────────────────────────────────
    lambda_fixed: float | None = None
    cp_k_max: int = args.cp_k_max

    if args.lambda_fixed_path and os.path.exists(args.lambda_fixed_path):
        with open(args.lambda_fixed_path, "r") as f:
            ld = json.load(f)
        lambda_fixed = float(ld["LAMBDA_FIXED"])
        # Report cp_k_max from the file when available for comparison.
        if "cp_k_max" in ld:
            print(f"[INFO] cp_k_max recorded in lambda_fixed.json = {ld['cp_k_max']}")
        print(f"\n{'='*70}")
        print(f"  LAMBDA_FIXED = {lambda_fixed:.8f}  (from {args.lambda_fixed_path})")
        print(f"  cp_alpha     = {ld.get('cp_alpha', 'N/A')}")
        print(f"  target_coverage = {ld.get('target_coverage', 'N/A')}")
        print(f"  num_positive_scores = {ld.get('num_positive_scores', 'N/A')}")
        print(f"  cp_k_max (current run) = {cp_k_max}")
        print(f"{'='*70}\n")
    elif args.lambda_fixed is not None:
        lambda_fixed = float(args.lambda_fixed)
        print(f"[INFO] Using lambda_fixed from the command line: {lambda_fixed}")
    else:
        print("[WARN] lambda_fixed was not provided; skipping CP filtering and showing raw results")

    # ── 1. Load the dataset ────────────────────────────────────────────────
    df = pd.read_parquet(args.val_parquet)
    sample_df = df.sample(n=min(args.n_samples, len(df)), random_state=42)
    print(f"[INFO] Sampled {len(sample_df)} rows from {args.val_parquet}\n")

    # ── 2. Resolve the checkpoint and merge FSDP shards if needed ─────────
    ckpt_path = resolve_hf_checkpoint(
        ckpt_path=os.path.abspath(args.checkpoint),
        step=args.step,
    )

    print(f"[INFO] Loading tokenizer from: {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[INFO] Loading model from: {ckpt_path}")
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",   # Automatically allocate across visible GPUs.
    )
    model.eval()
    print(f"[INFO] Model loaded on: {next(model.parameters()).device}\n")

    # ── 3. Diagnose each sample ────────────────────────────────────────────
    for sample_idx, (_, row) in enumerate(sample_df.iterrows()):
        messages = ensure_messages(row["prompt"])
        question_text = ""
        for m in messages:
            if m.get("role") == "user":
                question_text = m.get("content", "")
                break

        print(f"\n{'#'*70}")
        print(f"  Sample {sample_idx + 1}/{len(sample_df)}")
        print(f"{'#'*70}")
        print(f"  Question: {textwrap.shorten(question_text, width=120)}")

        # ── 3a. Generate once without a tool loop ──────────────────────────
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

        print("\n  --- Generated text (first 500 characters) ---")
        print(textwrap.indent(generated_text[:500], "  "))

        # ── 3b. Extract generated queries ──────────────────────────────────
        query_batches = extract_queries_from_generation(generated_text)
        if not query_batches:
            # Fall back to the content following "Question:" in the user message.
            import re as _re
            q_match = _re.search(r"Question:\s*(.+)", question_text, _re.DOTALL)
            raw_question = q_match.group(1).strip() if q_match else question_text
            raw_question = raw_question[:300]
            print(f"\n  [!] No <search> block was generated; using the original question: {raw_question[:80]}...")
            query_batches = [[raw_question]]

        print(f"\n  --- Generated queries ({len(query_batches)} <search> blocks) ---")
        for bi, ql in enumerate(query_batches):
            print(f"  <search>[{bi}]: {ql}")

        # ── 3c. Retrieve and analyze CP for each query ─────────────────────
        for bi, query_list in enumerate(query_batches):
            print(f"\n  === <search>[{bi}] retrieval analysis ===")
            print(f"  queries = {query_list}")

            try:
                raw = call_retrieval_raw(
                    url=args.retrieval_url,
                    queries=query_list,
                    topk=args.topk,
                    timeout=30,
                )
            except Exception as e:
                print(f"  [ERROR] Retrieval service call failed: {e}")
                continue

            per_query_results = raw.get("result", [])

            for qi, query in enumerate(query_list):
                print(f"\n  [Query {qi}] \"{query}\"")
                docs = per_query_results[qi] if qi < len(per_query_results) else []

                if not docs:
                    print("    (no retrieval results)")
                    continue

                print(f"  Retrieved {len(docs)} raw documents (topk={args.topk}):")
                for di, doc_item in enumerate(docs):
                    print(format_doc(doc_item, di + 1))

                # Apply CP filtering.
                kept, rejected = apply_cp_filter(docs, lambda_fixed, cp_k_max)

                print(f"\n  CP filter results (lambda_fixed={lambda_fixed}, cp_k_max={cp_k_max}):")
                print(f"    Kept: {len(kept)} documents")
                print(f"    Rejected: {len(rejected)} documents")

                if lambda_fixed is not None:
                    scores = []
                    for item in docs:
                        try:
                            scores.append(float(item["score"]))
                        except Exception:
                            pass
                    if scores:
                        s_docs = [-s for s in scores]
                        print("\n  Score statistics (s_doc = -retrieval_score):")
                        print(f"    s_doc range: [{min(s_docs):.6f}, {max(s_docs):.6f}]")
                        print(f"    LAMBDA_FIXED (threshold): {lambda_fixed:.6f}")
                        print("    Keep condition: s_doc <= LAMBDA_FIXED")
                        below = [s for s in s_docs if s <= lambda_fixed]
                        above = [s for s in s_docs if s > lambda_fixed]
                        print(f"    Documents satisfying the condition: {len(below)} -> s_docs: {[f'{s:.4f}' for s in below]}")
                        print(f"    Rejected documents: {len(above)} -> s_docs: {[f'{s:.4f}' for s in above]}")

                        # Print targeted diagnostic guidance.
                        if len(kept) == 1:
                            print("\n  WARNING: Only one document passed CP filtering.")
                            print(f"      LAMBDA_FIXED={lambda_fixed:.6f} may be too strict (too negative),")
                            print("      so only the highest-scoring document (the smallest s_doc) passed.")
                            print("      Check calibration quality and cp_alpha, or increase lambda_fixed.")
                        elif len(kept) == 0:
                            print("\n  WARNING: No documents passed CP filtering.")

        print()  # Separate samples with a blank line.

    print("\n" + "="*70)
    print("  Diagnosis complete")
    print("="*70)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    # Default paths allow the script to run without arguments.
    _BASE = "/home/zixizhu/un_rag/CAS"
    _DEFAULT_STEP = 700
    _DEFAULT_CKPT = _CKPT_ROOT   # --step selects a checkpoint beneath this root.
    _DEFAULT_VAL  = f"{_BASE}/data/cas_test.parquet"
    _DEFAULT_LAMBDA = f"{_BASE}/data/calibration/lambda_fixed.json"

    p = argparse.ArgumentParser(
        description="Diagnose why CP retrieval returns only Doc 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, default=_DEFAULT_CKPT,
                   help="Checkpoint root in FSDP or Hugging Face format; detected automatically")
    p.add_argument("--step", type=int, default=_DEFAULT_STEP,
                   help="Training step used to locate the global_step_N/actor directory")
    p.add_argument("--val_parquet", type=str, default=_DEFAULT_VAL,
                   help="Path to the validation parquet file")
    p.add_argument("--retrieval_url", type=str, default="http://127.0.0.1:8000/retrieve",
                   help="Retrieval service URL")
    p.add_argument("--lambda_fixed_path", type=str, default=_DEFAULT_LAMBDA,
                   help="Path to lambda_fixed.json")
    p.add_argument("--lambda_fixed", type=float, default=None,
                   help="Explicit lambda_fixed value; overrides lambda_fixed_path")
    p.add_argument("--cp_k_max", type=int, default=10,
                   help="Maximum number of documents kept after CP filtering")
    p.add_argument("--topk", type=int, default=10,
                   help="Number of documents requested from the retrieval service")
    p.add_argument("--n_samples", type=int, default=5,
                   help="Number of samples drawn from val_parquet")
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="Maximum number of tokens generated by the model")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_diagnosis(args)
