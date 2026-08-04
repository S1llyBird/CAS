"""
CAS template.

This is the default template used in cas/data/preprocess.py with detailed
user instructions for reasoning and search tool usage.
"""

from . import PromptTemplate, register_template

TEMPLATE_NAME = "cas"

DESCRIPTION = "CAS template with detailed reasoning and adaptive search instructions"

SYSTEM_CONTENT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

USER_CONTENT_PREFIX = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search>query</search> and it will return the top searched results between <information> and </information>. \
You can search as many times as you want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: """




# Register this template
template = PromptTemplate(
    name=TEMPLATE_NAME, description=DESCRIPTION, system_content=SYSTEM_CONTENT, user_content_prefix=USER_CONTENT_PREFIX
)
register_template(template)
