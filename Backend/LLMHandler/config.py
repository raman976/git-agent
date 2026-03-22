import os
from typing import List, Optional

from dotenv import load_dotenv


DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_PLANNING_MODEL = "llama-3.3-70b-versatile"
DEFAULT_FINAL_MODEL = DEFAULT_MODEL
MAX_TOOL_CALLS = 14
FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2-instruct-0905",
]
PLANNING_FALLBACK_MODELS = [
    "moonshotai/kimi-k2-instruct-0905",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
]
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_API_KEY_VAR = "llm_api_key"
DEFAULT_GITHUB_TOKEN_VAR = "GITHUB_TOKEN"

# Repository fetch strategy
GITHUB_API_TIMEOUT_SECONDS = 15
GITHUB_MAX_SELECTED_FILES = 600
GITHUB_MAX_TOTAL_BYTES = 25_000_000
GITHUB_MAX_FILE_BYTES = 250_000

# Model-rate guardrails (character budgets for final synthesis payload)
FINAL_PROMPT_EVIDENCE_CHAR_BUDGET = 10_000
FINAL_PROMPT_TOOL_TRACE_CHAR_BUDGET = 3_000
FINAL_PROMPT_HISTORY_CHAR_BUDGET = 1_200
FEATURE_RANKING_EVIDENCE_CHAR_BUDGET = 5_000


def load_api_key(api_key_var: str = DEFAULT_API_KEY_VAR) -> str:
    load_dotenv()
    api_key = os.getenv(api_key_var)
    if not api_key:
        raise ValueError(f"Missing API key in environment variable '{api_key_var}'.")
    return api_key


def resolve_api_key(api_key: Optional[str], api_key_var: str = DEFAULT_API_KEY_VAR) -> str:
    return api_key or load_api_key(api_key_var=api_key_var)


def build_model_candidates(primary_model: str) -> List[str]:
    candidates = [primary_model, *FALLBACK_MODELS]
    deduped: List[str] = []
    seen = set()
    for model_name in candidates:
        if model_name in seen:
            continue
        seen.add(model_name)
        deduped.append(model_name)
    return deduped


def build_planning_model_candidates(primary_model: str) -> List[str]:
    candidates = [primary_model, *PLANNING_FALLBACK_MODELS]
    deduped: List[str] = []
    seen = set()
    for model_name in candidates:
        if model_name in seen:
            continue
        seen.add(model_name)
        deduped.append(model_name)
    return deduped
