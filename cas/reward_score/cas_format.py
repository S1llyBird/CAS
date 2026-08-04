import re
import string
import random
import os
import json
import time

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence(text):
    # Remove role tags (user/assistant) that may appear between structured tags
    content = text

    # First pass: Simply remove lines that contain only role tags
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Only skip lines that are exactly "user" or "assistant"
        if stripped not in ['user', 'assistant']:
            cleaned_lines.append(line)

    # Rejoin the lines
    content = '\n'.join(cleaned_lines)


    # Check for balanced tags - now using think, search, information, answer
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    # Now check for proper sequence pattern and no extraneous content

    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)

    # 2. Keep track of the current position in the expected sequence
    # start -> [<think> -> <search> -> <information>]* -> <think> -> <answer> -> end
    state = "start"

    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue

        # Check if this is a tag
        if re.match(r"</?(?:think|search|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "after_information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "after_information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "after_information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"

    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def extract_solution(solution_str):
    """Extract the answer from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are no matches, return None
    if len(matches) == 0:
        return None

    # Return the last answer tag content
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    """Extract information from <information> tags."""
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def _append_reward_debug_trace(record: dict):
    """记录 reward 侧中间信息（模型输出、抽取答案、格式校验等）。"""
    trace_dir = os.getenv("CAS_DEBUG_TRACE_DIR", "").strip()
    if not trace_dir:
        return
    try:
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, f"reward_pid_{os.getpid()}.jsonl")
        payload = {"ts": time.time(), "pid": os.getpid(), **record}
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # 调试日志失败不影响训练主流程
        pass


def compute_score_em(
    solution_str, ground_truth, data_source, extra_info, 
    structure_format_score=0, final_format_score=0, retrieval_score=0, format_score=0, score=1.,
    *args, **kwargs):
    """The scoring function for exact match (EM) with detailed metrics tracking.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        data_source: the data source
        extra_info: extra information
        structure_format_score: score for valid structure format
        final_format_score: score for partial format
        retrieval_score: score for correct retrieval
        format_score: deprecated format score parameter
        score: the score for the correct answer

    Returns:
        dict with 'reward_tensor' and 'reward_extra_info' containing detailed metrics
    """
    is_valid_format, error_msg = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])

    answer = extract_solution(solution_str=solution_str)
    answer_correct = False
    if answer is not None:
        answer_correct = em_check(answer, ground_truth['target'])

    # Count tool calls (information retrieval attempts)
    num_tool_calls = len(extract_information_blocks(solution_str))

    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    # Calculate final reward (original logic)
    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                final_reward = structure_format_score + retrieval_score # 0.3
            else:
                final_reward = structure_format_score # 0.2
        else:
            final_reward = 0
    else:
        if answer_correct:
            if is_valid_format:
                final_reward = score # 1
            else:
                final_reward = score - structure_format_score # 0.8
        elif is_valid_format:
            if retrieval_correct:
                final_reward = structure_format_score + retrieval_score # 0.3
            else:
                final_reward = structure_format_score # 0.2
        else:
            final_reward = final_format_score # 0.1

    # Return reward with detailed metrics for tracking
    # Note: The NaiveRewardManager wrapper expects a dict with "score" key
    # and automatically collects all keys as reward_extra_info
    _append_reward_debug_trace(
        {
            "type": "reward_eval",
            "data_source": data_source,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "format_valid": bool(is_valid_format),
            "format_error": error_msg,
            "retrieval_correct": bool(retrieval_correct),
            "answer_extracted": answer,
            "answer_correct": bool(answer_correct),
            "num_tool_calls": int(num_tool_calls),
            "reward_final": float(final_reward),
            "solution_str": solution_str,
        }
    )

    return {
        "score": final_reward,  # Required: the actual reward value
        # Additional metrics (automatically collected by the wrapper)
        "format_valid": 1.0 if is_valid_format else 0.0,
        "has_answer": 1.0 if answer is not None else 0.0,
        "acc": 1.0 if answer_correct else 0.0,
        "num_tool_calls": float(num_tool_calls),
        "reward_final": final_reward,
    }
