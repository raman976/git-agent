import json
import os
import importlib.util
import sys
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from groq import Groq


DEFAULT_MODEL = "moonshotai/kimi-k2-instruct-0905"
DEFAULT_SYSTEM_PROMPT = (
    "You are a senior codebase assistant. Use only the provided context to answer. "
    "If context is insufficient, explicitly say what is missing."
)


CURRENT_DIR = os.path.dirname(__file__)
FUNCTION_HANDLER_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "FunctionHandler"))


def load_function_tool_module() -> Any:
    function_tool_path = os.path.join(FUNCTION_HANDLER_DIR, "functionTOtext.py")
    if FUNCTION_HANDLER_DIR not in sys.path:
        sys.path.insert(0, FUNCTION_HANDLER_DIR)

    spec = importlib.util.spec_from_file_location("functionTOtext", function_tool_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load functionTOtext from {function_tool_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_api_key(api_key_var: str = "llm_api_key") -> str:
    load_dotenv()
    api_key = os.getenv(api_key_var)
    if not api_key:
        raise ValueError(
            f"Missing API key in environment variable '{api_key_var}'."
        )
    return api_key


def get_client(api_key: Optional[str] = None, api_key_var: str = "llm_api_key") -> Groq:
    resolved_key = api_key or load_api_key(api_key_var=api_key_var)
    return Groq(api_key=resolved_key)


def _safe_get(item: Dict[str, Any], keys: Iterable[str], default: str = "") -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default


def format_context_items(results: List[Dict[str, Any]], top_k: int = 3) -> str:
    selected = results[:top_k]
    if not selected:
        return "No retrieved context provided."

    blocks = []
    for idx, item in enumerate(selected, 1):
        score = _safe_get(item, ["score", "retrieval_score"], 0)
        name = _safe_get(item, ["name", "function_name"], "unknown")
        language = _safe_get(item, ["language"], "unknown")
        file_path = _safe_get(item, ["file", "file_path"], "unknown")
        start_line = _safe_get(item, ["start_line"], "?")
        end_line = _safe_get(item, ["end_line"], "?")
        code = _safe_get(item, ["code", "text"], "")

        blocks.append(
            "\n".join(
                [
                    f"Result {idx}:",
                    f"Score: {score}",
                    f"Function: {name}",
                    f"Language: {language}",
                    f"File: {file_path}:{start_line}-{end_line}",
                    "Code:",
                    code,
                ]
            )
        )

    return "\n\n".join(blocks)


def build_messages(
    question: str,
    results: List[Dict[str, Any]],
    top_k: int = 3,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    context_block = format_context_items(results, top_k=top_k)

    user_payload = {
        "question": question,
        "required_output_format": {
            "summary": "Concise answer to the question.",
            "reasoning": "Step-by-step reasoning grounded in evidence.",
            "evidence": [
                {
                    "file": "path",
                    "function": "name",
                    "why_relevant": "short reason",
                }
            ],
            "gaps": "What information is missing or uncertain.",
        },
        "retrieved_context": context_block,
    }

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, indent=2)},
    ]


def generate_response(
    question: str,
    results: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_completion_tokens: int = 4096,
    top_p: float = 1.0,
    top_k_results: int = 3,
    stream: bool = True,
    api_key: Optional[str] = None,
    api_key_var: str = "llm_api_key",
) -> str:
    client = get_client(api_key=api_key, api_key_var=api_key_var)
    messages = build_messages(question=question, results=results, top_k=top_k_results)

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        stream=stream,
        stop=None,
    )

    if stream:
        chunks: List[str] = []
        for chunk in completion:
            text = chunk.choices[0].delta.content or ""
            if text:
                print(text, end="", flush=True)
                chunks.append(text)
        print()
        return "".join(chunks)

    return completion.choices[0].message.content or ""


def generate_response_from_query(
    question: str,
    query_top_k: int = 3,
    context_top_k: int = 3,
    repo_path: Optional[str] = None,
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    llm_model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_completion_tokens: int = 4096,
    top_p: float = 1.0,
    stream: bool = True,
    api_key: Optional[str] = None,
    api_key_var: str = "llm_api_key",
) -> Dict[str, Any]:
    function_tool = load_function_tool_module()
    retrieval = function_tool.query_top_functions(
        query=question,
        top_k=query_top_k,
        repo_path=repo_path,
        model_name=embedding_model_name,
    )
    results = retrieval.get("results", [])

    answer = generate_response(
        question=question,
        results=results,
        model=llm_model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        top_k_results=context_top_k,
        stream=stream,
        api_key=api_key,
        api_key_var=api_key_var,
    )

    return {
        "question": question,
        "retrieval": retrieval,
        "answer": answer,
    }


if __name__ == "__main__":
    output = generate_response_from_query(
        question="How does authentication work?",
        query_top_k=3,
        context_top_k=3,
        stream=True,
    )
    print("Top results used:", len(output["retrieval"].get("results", [])))
