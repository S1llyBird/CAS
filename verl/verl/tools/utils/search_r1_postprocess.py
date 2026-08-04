# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
import re
from typing import Optional

_SEARCH_BLOCK_RE = re.compile(r"<search(?:\s+[^>]*)?>(.*?)</search>", re.DOTALL)
_SEARCH_SELFCLOSE_RE = re.compile(r"<search\b[^>]*?/>", re.DOTALL)
_SEARCH_QUERY_ATTR_RE = re.compile(r"""\bquery\s*=\s*(["'])(.*?)\1""", re.DOTALL)


def truncate_after_first_search_request(text: str) -> str:
    """
    SearchR1 standard post-processing:
    keep only the first search request and everything before it;
    drop all content after that request.
    """
    search_open_idx = text.find("<search")
    if search_open_idx == -1:
        return text

    candidates: list[tuple[int, str]] = []

    close_idx = text.find("</search>", search_open_idx)
    if close_idx != -1:
        candidates.append((close_idx, "close"))

    open_tag_end_idx = text.find(">", search_open_idx)
    if open_tag_end_idx != -1:
        self_close_idx = text.find("/>", search_open_idx, open_tag_end_idx + 1)
        if self_close_idx != -1:
            candidates.append((self_close_idx, "self_close"))

    partial_close_idx = text.find("</search", search_open_idx)
    if partial_close_idx != -1 and close_idx != partial_close_idx:
        candidates.append((partial_close_idx, "partial_close"))

    if not candidates:
        return text

    first_idx, first_kind = min(candidates, key=lambda x: x[0])
    if first_kind == "close":
        return text[: first_idx + len("</search>")]
    if first_kind == "self_close":
        return text[: first_idx + 2]

    return text[:first_idx] + "</search>"


def extract_first_search_query(text: str) -> Optional[str]:
    """
    Extract query from the first search request.
    Supports:
    1) <search>query</search>
    2) <search query="query" />
    """
    block_match = _SEARCH_BLOCK_RE.search(text)
    if block_match:
        query = block_match.group(1).strip()
        return query or None

    self_close_match = _SEARCH_SELFCLOSE_RE.search(text)
    if self_close_match:
        tag_text = self_close_match.group(0)
        attr_match = _SEARCH_QUERY_ATTR_RE.search(tag_text)
        if attr_match:
            query = attr_match.group(2).strip()
            return query or None

    return None
