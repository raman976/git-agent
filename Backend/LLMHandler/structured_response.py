import json
from typing import Any, Dict, Iterable, List, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from .agent_graph import run_multi_reasoning_agent
from .config import DEFAULT_API_KEY_VAR, DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL, resolve_api_key
from .prompts import DEFAULT_SYSTEM_PROMPT
from .runtime import load_function_tool_module


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
    api_key_var: str = DEFAULT_API_KEY_VAR,
) -> str:
    resolved_key = resolve_api_key(api_key=api_key, api_key_var=api_key_var)
    messages = build_messages(question=question, results=results, top_k=top_k_results)

    llm = ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_completion_tokens,
        top_p=top_p,
        groq_api_key=resolved_key,
        streaming=stream,
    )

    prompt_messages = [
        SystemMessage(content=messages[0]["content"]),
        HumanMessage(content=messages[1]["content"]),
    ]

    if stream:
        chunks: List[str] = []
        for chunk in llm.stream(prompt_messages):
            text = chunk.content or ""
            if text:
                print(text, end="", flush=True)
                chunks.append(text)
        print()
        return "".join(chunks)

    response = llm.invoke(prompt_messages)
    if isinstance(response.content, str):
        return response.content
    return str(response.content)


def generate_response_from_query(
    question: str,
    query_top_k: int = 3,
    context_top_k: int = 3,
    repo_path: Optional[str] = None,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    llm_model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_completion_tokens: int = 4096,
    top_p: float = 1.0,
    stream: bool = True,
    api_key: Optional[str] = None,
    api_key_var: str = DEFAULT_API_KEY_VAR,
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
    output = run_multi_reasoning_agent(
        question="How does authentication work in this repository?",
        max_reasoning_steps=6,
    )
    print("Tool calls:", len(output.get("tool_calls_executed", [])))
    print("Final answer size:", len(output.get("final_answer", "")))
