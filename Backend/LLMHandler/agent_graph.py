import json
import re
import time
from typing import Any, Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from .config import (
    DEFAULT_API_KEY_VAR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FINAL_MODEL,
    FEATURE_RANKING_EVIDENCE_CHAR_BUDGET,
    FINAL_PROMPT_EVIDENCE_CHAR_BUDGET,
    FINAL_PROMPT_HISTORY_CHAR_BUDGET,
    FINAL_PROMPT_TOOL_TRACE_CHAR_BUDGET,
    DEFAULT_MODEL,
    DEFAULT_PLANNING_MODEL,
    MAX_TOOL_CALLS,
    build_model_candidates,
    build_planning_model_candidates,
    resolve_api_key,
)
from .prompts import (
    FINAL_OUTPUT_INSTRUCTION,
    MULTI_REASONING_SYSTEM_PROMPT,
    STRICT_REPO_GROUNDED_FLOW_INSTRUCTION,
)
from .runtime import FunctionContextRuntime
from .tools import build_langchain_tools

RETRY_MAX_ATTEMPTS = 4
RETRY_INITIAL_DELAY = 1.0
RETRY_BACKOFF_MULTIPLIER = 2.0

RETRYABLE_ERROR_PATTERNS = {
    "503", "over capacity", "service unavailable",
    "429", "rate limit", "tokens per minute",
    "413", "request too large", "timeout"
}

ERROR_PARSING_KEYWORDS = {
    "tool_use_failed", "failed to call a function",
    "output_parse_failed", "parsing failed", "failed_generation",
    "rate_limit_exceeded"
}


def _is_retryable_error(error: Exception) -> bool:
    error_text = str(error).lower()
    has_retryable_pattern = any(pattern in error_text for pattern in RETRYABLE_ERROR_PATTERNS)
    has_parsing_issue = any(keyword in error_text for keyword in ERROR_PARSING_KEYWORDS)
    return has_retryable_pattern or has_parsing_issue


def _invoke_graph_with_retry(
    graph: Any,
    payload: Dict[str, Any],
    config: Dict[str, Any],
    max_attempts: int = RETRY_MAX_ATTEMPTS,
    initial_delay: float = RETRY_INITIAL_DELAY,
) -> Dict[str, Any]:
    delay = initial_delay
    for attempt in range(max_attempts):
        try:
            return graph.invoke(payload, config=config)
        except Exception as error:
            if not _is_retryable_error(error) or attempt == max_attempts - 1:
                raise
            time.sleep(delay)
            delay *= RETRY_BACKOFF_MULTIPLIER

    raise RuntimeError("Graph invocation failed after retries")


def _invoke_planning_llm(
    messages: List[BaseMessage],
    planning_model: str,
    groq_api_key: str,
    max_tokens: int,
) -> str:
    for model_candidate in build_planning_model_candidates(planning_model):
        try:
            llm = ChatGroq(
                model=model_candidate,
                temperature=0.0,
                max_tokens=max_tokens,
                model_kwargs={"top_p": 1.0},
                groq_api_key=groq_api_key,
            )
            response = llm.invoke(messages)
            text = _extract_message_text(response).strip()
            if text:
                return text
        except Exception as error:
            if not _is_retryable_error(error):
                raise
            continue

    return ""


def _extract_message_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    
    if isinstance(message.content, list):
        text_parts = [
            str(item["text"]) for item in message.content
            if isinstance(item, dict) and "text" in item
        ]
        return "\n".join(text_parts)
    
    return str(message.content)


def _extract_tool_calls_from_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    trace: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            trace.append({
                "name": call.get("name"),
                "arguments": call.get("args", {}),
            })
    return trace


def _parse_json_from_content(content: Any) -> Optional[Dict[str, Any]]:
    if isinstance(content, str):
        text = content.strip()
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return None

    if isinstance(content, list):
        text_parts = [
            str(item["text"]) for item in content
            if isinstance(item, dict) and "text" in item
        ]
        if text_parts:
            try:
                parsed = json.loads("\n".join(text_parts))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

    return None


def _parse_safe_json_dict(raw_text: str) -> Optional[Dict[str, Any]]:
    text = raw_text.strip()
    
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


def _extract_tool_evidence(
    messages: List[BaseMessage],
    max_entries: int = 16,
    max_snippet_chars: int = 260,
) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    seen = set()

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        payload = _parse_json_from_content(message.content)
        if not payload:
            continue

        # search_code/search_functions/get_function_details shape
        for item in payload.get("results", []) if isinstance(payload.get("results"), list) else []:
            file_path = str(item.get("file", ""))
            name = str(item.get("name", ""))
            start_line = item.get("start_line")
            end_line = item.get("end_line")
            text = str(item.get("text", ""))
            signature = ("result", file_path, name, start_line, end_line)
            if not file_path or signature in seen:
                continue
            seen.add(signature)
            evidence.append(
                {
                    "source": "search_result",
                    "file": file_path,
                    "name": name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "snippet": text[:max_snippet_chars],
                }
            )
            if len(evidence) >= max_entries:
                return evidence

        # read_file shape
        if isinstance(payload.get("content"), str) and payload.get("file"):
            file_path = str(payload.get("file", ""))
            signature = ("file", file_path)
            if signature not in seen:
                seen.add(signature)
                evidence.append(
                    {
                        "source": "file_read",
                        "file": file_path,
                        "snippet": str(payload.get("content", ""))[:max_snippet_chars],
                    }
                )
                if len(evidence) >= max_entries:
                    return evidence

        # get_dependencies shape
        if payload.get("file") and isinstance(payload.get("dependencies"), list):
            file_path = str(payload.get("file", ""))
            deps = [str(dep) for dep in payload.get("dependencies", [])[:8]]
            signature = ("deps", file_path, tuple(deps))
            if signature not in seen:
                seen.add(signature)
                evidence.append(
                    {
                        "source": "dependencies",
                        "file": file_path,
                        "dependencies": deps,
                    }
                )
                if len(evidence) >= max_entries:
                    return evidence

    return evidence


def _has_tool_call(trace: List[Dict[str, Any]], tool_name: str) -> bool:
    return any(item.get("name") == tool_name for item in trace)


def _has_redundant_calls(trace: List[Dict[str, Any]]) -> bool:
    seen = set()
    duplicate_count = 0
    for item in trace:
        signature = (
            str(item.get("name", "")),
            json.dumps(item.get("arguments", {}), sort_keys=True),
        )
        if signature in seen:
            duplicate_count += 1
        else:
            seen.add(signature)

    return duplicate_count >= 2


def _needs_dependency_trace(question: str) -> bool:
    q = question.lower()
    keywords = ["flow", "how", "path", "pipeline", "authentication", "auth", "request"]
    return any(word in q for word in keywords)


def _needs_flow_reconstruction(question: str) -> bool:
    q = question.lower()
    markers = [
        "how does",
        "complete flow",
        "full flow",
        "step by step",
        "request flow",
        "pipeline",
        "lifecycle",
        "authentication flow",
        "auth flow",
    ]
    return any(marker in q for marker in markers)


def _ground_query_concept(
    question: str,
    planning_model: str,
    groq_api_key: str,
) -> Dict[str, Any]:
    # Generic fallback — used only if LLM call fails entirely
    fallback = {
        "concept": "",
        "search_terms": [],
        "anti_signals": [],
        "needs_dependency_trace": _needs_dependency_trace(question),
        "flow_steps": [],
    }

    prompt = (
        "Return ONLY valid JSON with keys: concept, search_terms, anti_signals, needs_dependency_trace, flow_steps. "
        "concept: one concise software concept label (e.g. 'authentication', 'data processing', 'routing'). "
        "search_terms: 8-12 SHORT CODE-LEVEL TOKENS that would appear in actual source code implementing this concept. "
        "Use function names, class names, variable patterns — NOT filenames or file extensions. "
        "Example for 'authentication': ['login', 'session', 'token', 'verify', 'password', 'oauth', 'jwt', 'middleware']. "
        "Example for 'database': ['query', 'insert', 'connection', 'schema', 'migration', 'model', 'repository']. "
        "anti_signals: tokens that are related but wrong (e.g. for auth: ['permission', 'role', 'feature_flag']). "
        "needs_dependency_trace: boolean. "
        "flow_steps: ordered array of step labels for this concept.\n\n"
        f"Query: {question}"
    )

    try:
        text = _invoke_planning_llm(
            messages=[HumanMessage(content=prompt)],
            planning_model=planning_model,
            groq_api_key=groq_api_key,
            max_tokens=300,
        )
        parsed = _parse_safe_json_dict(text)
        if not parsed:
            return fallback

        concept = str(parsed.get("concept", "")).strip()
        search_terms = [
            str(x).strip().lower()
            for x in parsed.get("search_terms", [])
            if str(x).strip() and "." not in str(x)  # strip anything with a dot (filenames/extensions)
        ][:12]
        anti_signals = [str(x).strip().lower() for x in parsed.get("anti_signals", []) if str(x).strip()][:10]
        flow_steps = [str(x).strip().lower() for x in parsed.get("flow_steps", []) if str(x).strip()][:8]

        return {
            "concept": concept,
            "search_terms": search_terms,
            "anti_signals": anti_signals,
            "needs_dependency_trace": bool(parsed.get("needs_dependency_trace", fallback["needs_dependency_trace"])),
            "flow_steps": flow_steps,
        }
    except Exception:
        return fallback


def _reconstruct_concept_flow(
    question: str,
    concept_grounding: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    planning_model: str,
    groq_api_key: str,
) -> Dict[str, Any]:
    fallback = {
        "entry_point": "Not clearly identified from current evidence.",
        "processing_steps": [],
        "storage_validation": "Not clearly identified from current evidence.",
        "additional_layers": [],
        "conclusion": "Insufficient evidence to reconstruct a complete repository-specific flow.",
        "missing_steps": ["entry_point", "verification", "session_or_token_handling", "request_validation"],
    }

    compact_evidence: List[Dict[str, Any]] = []
    for item in evidence[:14]:
        compact_evidence.append(
            {
                "source": item.get("source"),
                "file": item.get("file"),
                "name": item.get("name"),
                "snippet": str(item.get("snippet", ""))[:500],
            }
        )

    prompt = (
        "Return ONLY valid JSON with keys: entry_point, processing_steps, storage_validation, additional_layers, conclusion, missing_steps. "
        "processing_steps must be an ordered array of concrete repository-specific steps. "
        "Do NOT produce generic textbook flow. If evidence is missing, explicitly list missing_steps and keep claims conservative.\n\n"
        f"Question: {question}\n"
        f"Concept grounding: {json.dumps(concept_grounding)}\n"
        f"Evidence: {json.dumps(compact_evidence)}"
    )

    try:
        text = _invoke_planning_llm(
            messages=[HumanMessage(content=prompt)],
            planning_model=planning_model,
            groq_api_key=groq_api_key,
            max_tokens=360,
        )
        parsed = _parse_safe_json_dict(text)
        if not parsed:
            return fallback

        processing_steps = [str(x).strip() for x in parsed.get("processing_steps", []) if str(x).strip()][:8]
        additional_layers = [str(x).strip() for x in parsed.get("additional_layers", []) if str(x).strip()][:6]
        missing_steps = [str(x).strip() for x in parsed.get("missing_steps", []) if str(x).strip()][:8]

        return {
            "entry_point": str(parsed.get("entry_point", fallback["entry_point"])).strip() or fallback["entry_point"],
            "processing_steps": processing_steps,
            "storage_validation": str(parsed.get("storage_validation", fallback["storage_validation"])).strip()
            or fallback["storage_validation"],
            "additional_layers": additional_layers,
            "conclusion": str(parsed.get("conclusion", fallback["conclusion"])).strip() or fallback["conclusion"],
            "missing_steps": missing_steps,
        }
    except Exception:
        return fallback


def _build_evidence_only_flow_answer(
    question: str,
    concept_grounding: Dict[str, Any],
    flow_reconstruction: Dict[str, Any],
    evidence: List[Dict[str, Any]],
) -> str:
    concept = str(concept_grounding.get("concept", "")).strip() or "system"
    entry_point = str(flow_reconstruction.get("entry_point", "")).strip() or "Not clearly identified from current evidence."
    processing_steps = [
        str(x).strip()
        for x in flow_reconstruction.get("processing_steps", [])
        if str(x).strip()
    ]
    storage_validation = (
        str(flow_reconstruction.get("storage_validation", "")).strip()
        or "Not clearly identified from current evidence."
    )
    additional_layers = [
        str(x).strip()
        for x in flow_reconstruction.get("additional_layers", [])
        if str(x).strip()
    ]
    missing_steps = [
        str(x).strip()
        for x in flow_reconstruction.get("missing_steps", [])
        if str(x).strip()
    ]
    conclusion = (
        str(flow_reconstruction.get("conclusion", "")).strip()
        or "Insufficient evidence to reconstruct a complete repository-specific flow."
    )

    observed_components: List[Dict[str, str]] = []
    graph_edges: List[Dict[str, str]] = []
    seen = set()
    for item in evidence:
        file_path = str(item.get("file", "")).strip()
        if not file_path:
            continue
        source = str(item.get("source", "")).strip()
        if source in {"concept_grounding", "cluster_coverage"}:
            continue
        if source == "symbol_graph":
            raw_edges = item.get("graph_edges", [])
            if isinstance(raw_edges, list):
                for edge in raw_edges:
                    if isinstance(edge, dict):
                        graph_edges.append(
                            {
                                "from": str(edge.get("from", "")).strip(),
                                "to": str(edge.get("to", "")).strip(),
                                "via": str(edge.get("via", "")).strip(),
                                "kind": str(edge.get("kind", "")).strip(),
                            }
                        )
            continue
        name = str(item.get("name", "")).strip()
        snippet = " ".join(str(item.get("snippet", "")).split())[:180]
        key = (file_path.lower(), name.lower())
        if key in seen:
            continue
        seen.add(key)
        observed_components.append(
            {
                "file": file_path,
                "name": name,
                "snippet": snippet,
            }
        )
        if len(observed_components) >= 6:
            break

    anchor_terms = set()
    for comp in observed_components:
        file_name = comp["file"].replace("\\", "/").split("/")[-1].lower()
        if file_name:
            anchor_terms.add(file_name)
        if comp["name"]:
            anchor_terms.add(comp["name"].lower())

    def _is_anchored(text: str) -> bool:
        lowered = text.lower()
        if not lowered:
            return False
        return any(term and term in lowered for term in anchor_terms)

    def _is_speculative(text: str) -> bool:
        lowered = text.lower()
        speculative_markers = [
            "possibly",
            "potentially",
            "might",
            "may be",
            "could",
            "likely",
            "probably",
            "not explicitly found",
        ]
        return any(marker in lowered for marker in speculative_markers)

    if not observed_components:
        entry_point = "Not clearly identified from current evidence."
        processing_steps = []
        storage_validation = "Not clearly identified from current evidence."
        additional_layers = []
        conclusion = "Insufficient evidence to reconstruct a complete repository-specific flow."
        missing_steps = [
            "entry_point",
            "processing_steps",
            "storage_or_validation",
            "supporting_layers",
        ]
    else:
        def _component_text(comp: Dict[str, str]) -> str:
            return " ".join(
                [
                    comp.get("file", ""),
                    comp.get("name", ""),
                    comp.get("snippet", ""),
                ]
            ).lower()

        step_tokens: Dict[str, List[str]] = {}
        for step in concept_grounding.get("flow_steps", []) if isinstance(concept_grounding.get("flow_steps", []), list) else []:
            label = str(step).strip().lower()
            if not label:
                continue
            tokens = [tok for tok in re.split(r"[^a-z0-9]+", label) if len(tok) >= 4]
            if tokens:
                step_tokens[label] = tokens

        search_terms = [
            str(term).strip().lower()
            for term in concept_grounding.get("search_terms", [])
            if str(term).strip()
        ]

        mapped_by_step: Dict[str, List[Dict[str, str]]] = {key: [] for key in step_tokens.keys()}
        for comp in observed_components:
            text = _component_text(comp)
            for label, tokens in step_tokens.items():
                if any(tok in text for tok in tokens):
                    mapped_by_step[label].append(comp)

        def _format_step_from_comp(step_label: str, comp: Dict[str, str]) -> str:
            short_file = comp.get("file", "").replace("\\", "/").split("/")[-1]
            name = comp.get("name", "")
            anchor = f"{short_file}::{name}" if name else short_file
            pretty = step_label.replace("_", " ")
            return f"Candidate {pretty} component: {anchor}."

        if (not _is_anchored(entry_point)) or _is_speculative(entry_point):
            entry_candidates: List[Dict[str, str]] = []
            for comp in observed_components:
                text = _component_text(comp)
                if any(tok in text for tok in ["entry", "login", "signin", "auth", "nextauth", "oauth", "callback", "controller", "route", "endpoint"]):
                    entry_candidates.append(comp)
            if entry_candidates:
                entry_point = _format_step_from_comp("entry_point", entry_candidates[0])
            else:
                entry_point = "Not clearly identified from current evidence."

        anchored_steps = [step for step in processing_steps if _is_anchored(step) and not _is_speculative(step)]
        processing_steps = anchored_steps
        if not processing_steps:
            derived_steps: List[str] = []
            used_keys = set()
            for label, comps in mapped_by_step.items():
                if not comps:
                    continue
                comp = comps[0]
                key = (comp.get("file", ""), comp.get("name", ""))
                if key in used_keys:
                    continue
                used_keys.add(key)
                derived_steps.append(_format_step_from_comp(label, comp))
            if not derived_steps:
                for comp in observed_components[:3]:
                    text = _component_text(comp)
                    if any(tok in text for tok in search_terms[:8]):
                        derived_steps.append(_format_step_from_comp("observed_step", comp))
            processing_steps = derived_steps[:6]

        if (not _is_anchored(storage_validation)) or _is_speculative(storage_validation):
            storage_candidates = []
            for comp in observed_components:
                text = _component_text(comp)
                if any(tok in text for tok in ["session", "token", "cookie", "jwt", "store", "persist", "validate", "verify", "middleware"]):
                    storage_candidates.append(comp)
            if storage_candidates:
                storage_validation = _format_step_from_comp("storage_validation", storage_candidates[0])
            else:
                storage_validation = "Not clearly identified from current evidence."

        anchored_layers = [layer for layer in additional_layers if _is_anchored(layer) and not _is_speculative(layer)]
        additional_layers = anchored_layers
        if not additional_layers:
            layer_candidates = []
            for comp in observed_components:
                text = _component_text(comp)
                if any(tok in text for tok in ["2fa", "two-factor", "mfa", "saml", "oidc", "signout", "logout", "email", "verification", "role"]):
                    layer_candidates.append(comp)
            additional_layers = [_format_step_from_comp("additional_layer", comp) for comp in layer_candidates[:3]]

        if step_tokens:
            derived_missing = []
            for label, comps in mapped_by_step.items():
                if not comps:
                    derived_missing.append(label)
            if derived_missing:
                missing_steps = derived_missing
            elif not missing_steps:
                missing_steps = ["none"]

        if (not _is_anchored(conclusion)) or _is_speculative(conclusion):
            conclusion = "Only partial flow is confirmed from directly observed repository evidence."

    lines: List[str] = []
    lines.append("### 1. Observed Flow (from code only)")
    lines.append(f"- Query: {question}")
    lines.append(f"- Entry point: {entry_point if entry_point else 'not found'}")
    connected_chain_steps: List[str] = []
    for edge in graph_edges[:5]:
        src = edge.get("from", "").replace("\\", "/").split("/")[-1]
        dst = edge.get("to", "").replace("\\", "/").split("/")[-1]
        via = edge.get("via", "")
        kind = edge.get("kind", "relation")
        if src and dst:
            connected_chain_steps.append(f"{src} -> {dst} via {kind}:{via}")

    merged_steps = processing_steps
    seen_steps = set()
    unique_steps: List[str] = []
    for step in merged_steps:
        key = step.strip().lower()
        if not key or key in seen_steps:
            continue
        seen_steps.add(key)
        unique_steps.append(step)

    if unique_steps:
        for step in unique_steps[:7]:
            lines.append(f"- {step}")
    elif processing_steps:
        for step in processing_steps:
            lines.append(f"- {step}")
    else:
        lines.append("- observed processing steps: not found")
    lines.append(f"- storage/validation: {storage_validation if storage_validation else 'not found'}")
    lines.append("- additional observed layers:")
    if additional_layers:
        for layer in additional_layers:
            lines.append(f"- {layer}")
    else:
        lines.append("- not found")

    lines.append("")
    lines.append("### 2. Flow Connection")
    if connected_chain_steps:
        for step in connected_chain_steps:
            lines.append(f"- {step}")
    else:
        lines.append("- connection chain from imports/usages: not found")

    lines.append("")
    lines.append("### 3. Missing Parts")
    if missing_steps:
        for missing in missing_steps:
            lines.append(f"- {missing}: not found")
    else:
        lines.append("- missing critical parts: not found")

    lines.append("")
    lines.append("### 4. Conclusion")
    lines.append(conclusion)
    lines.append("")
    lines.append("Evidence anchors:")
    if observed_components:
        for comp in observed_components:
            name_part = f"::{comp['name']}" if comp["name"] else ""
            snippet_part = f" | {comp['snippet']}" if comp["snippet"] else ""
            lines.append(f"- {comp['file']}{name_part}{snippet_part}")
    else:
        lines.append("- not found")

    return "\n".join(lines)


def _build_dynamic_answer_instruction(
    question: str,
    evidence: List[Dict[str, Any]],
    plan_steps: List[str],
    planning_model: str,
    groq_api_key: str,
) -> str:
    """Use a small LLM pass to generate query-specific answer guidance."""
    try:
        prompt = (
            "Create short answer-writing instructions for a code QA assistant. "
            "Focus only on this user question and available evidence. "
            "Prefer concrete technology names when present (e.g., database names). "
            "Return plain text under 80 words.\n\n"
            f"Question: {question}\n"
            f"Plan: {json.dumps(plan_steps)}\n"
            f"Evidence sample: {json.dumps(evidence[:4])}"
        )
        text = _invoke_planning_llm(
            messages=[HumanMessage(content=prompt)],
            planning_model=planning_model,
            groq_api_key=groq_api_key,
            max_tokens=140,
        )
        if text:
            return text[:500]
    except Exception:
        pass

    return (
        "Answer the exact user question first in one clear sentence, then justify with concrete file/function evidence. "
        "If a specific technology is present in evidence, name it explicitly."
    )


def _normalize_query_scope(raw_scope: str) -> str:
    lowered = raw_scope.strip().lower()
    if lowered in {"general", "repo_specific", "hybrid"}:
        return lowered
    return "repo_specific"


def classify_query_scope(
    question: str,
    planning_model: str = DEFAULT_PLANNING_MODEL,
    api_key: Optional[str] = None,
    api_key_var: str = DEFAULT_API_KEY_VAR,
) -> str:
    q = question.strip()
    if not q:
        return "repo_specific"

    heuristic_repo_hints = [
        "this repo",
        "this repository",
        "in this codebase",
        "in this project",
        "in the repo",
        "file",
        "function",
        "class",
        "module",
        "endpoint",
    ]
    lowered = q.lower()
    if any(token in lowered for token in heuristic_repo_hints):
        return "repo_specific"

    resolved_key = resolve_api_key(api_key=api_key, api_key_var=api_key_var)
    try:
        prompt = (
            "Classify the user query as exactly one token from this set: "
            "repo_specific, general, hybrid. "
            "repo_specific: asks about repository internals/implementation. "
            "general: broad knowledge not tied to repository context. "
            "hybrid: needs both broad explanation and repository grounding. "
            "Return only one token.\n\n"
            f"Query: {q}"
        )
        text = _invoke_planning_llm(
            messages=[HumanMessage(content=prompt)],
            planning_model=planning_model,
            groq_api_key=resolved_key,
            max_tokens=16,
        )
        return _normalize_query_scope(text)
    except Exception:
        return "repo_specific"


def _answer_general_question(
    question: str,
    history_tail: List[Dict[str, str]],
    llm_model: str,
    planning_model: str,
    temperature: float,
    max_completion_tokens: int,
    top_p: float,
    resolved_key: str,
) -> Dict[str, Any]:
    system = (
        "You are a senior software assistant. "
        "Answer with concise, accurate technical guidance. "
        "Do not assume repository context unless explicitly asked."
    )
    prompt = (
        f"Recent conversation context: {json.dumps(history_tail)}\n\n"
        f"User question: {question}\n\n"
        "Answer directly with practical detail. If uncertain, say so briefly."
    )

    final_answer = ""
    selected_model = llm_model
    for candidate_model in build_model_candidates(llm_model):
        try:
            llm = ChatGroq(
                model=candidate_model,
                temperature=temperature,
                max_tokens=max_completion_tokens,
                model_kwargs={"top_p": top_p},
                groq_api_key=resolved_key,
            )
            response = llm.invoke(
                [
                    SystemMessage(content=system),
                    HumanMessage(content=prompt),
                ]
            )
            final_answer = _extract_message_text(response).strip()
            if final_answer:
                selected_model = candidate_model
                break
        except Exception as exc:
            if not _is_retryable_error(exc):
                raise
            continue

    if not final_answer:
        final_answer = "I could not generate a reliable general answer in this run."

    return {
        "question": question,
        "tool_calls_executed": [],
        "final_answer": final_answer,
        "context_stats": {},
        "model_used": selected_model,
        "planning_model_used": planning_model,
        "plan": ["general_knowledge"],
        "max_tool_calls": MAX_TOOL_CALLS,
    }


def _pack_json_by_char_budget(items: List[Dict[str, Any]], max_chars: int) -> str:
    if not items:
        return "[]"

    packed: List[Dict[str, Any]] = []
    for item in items:
        packed.append(item)
        candidate = json.dumps(packed)
        if len(candidate) > max_chars:
            packed.pop()
            break

    if not packed:
        first = json.dumps(items[0])
        return first[:max_chars]

    return json.dumps(packed)


LOW_SIGNAL_PATH_MARKERS = {
    "__typetests__",
    "__tests__",
    "tests",
    "test",
    "typetests",
    "example",
    "examples",
    "demo",
    "fixture",
    "fixtures",
    "mock",
    "mocks",
    "benchmark",
    "benchmarks",
}

PRODUCTION_SIGNAL_MARKERS = {
    "src",
    "app",
    "backend",
    "frontend",
    "api",
    "services",
    "service",
    "controllers",
    "controller",
    "models",
    "model",
    "core",
    "lib",
    "android",
    "ios",
}

TOOLING_PATH_MARKERS = {
    "scripts",
    "script",
    "eslint",
    "lints",
    "lint",
    "spec",
    "specs",
    "tester",
    "private",
    "fixtures",
    "mock",
    "mocks",
}


def _path_parts(path: str) -> List[str]:
    return [part for part in path.replace("\\", "/").split("/") if part]


def _is_low_signal_path(path: str) -> bool:
    parts = {part.lower() for part in _path_parts(path)}
    return bool(parts.intersection(LOW_SIGNAL_PATH_MARKERS))


def _signal_tier_for_path(path: str) -> str:
    lowered = path.lower()
    if _is_low_signal_path(lowered):
        return "low"

    parts = {part.lower() for part in _path_parts(lowered)}
    if parts.intersection(PRODUCTION_SIGNAL_MARKERS):
        return "high"

    return "medium"


def _is_tooling_path(path: str) -> bool:
    parts = {part.lower() for part in _path_parts(path)}
    return bool(parts.intersection(TOOLING_PATH_MARKERS))


def _is_repository_overview_question(question: str) -> bool:
    lowered = question.lower()
    overview_markers = [
        "what does this repo",
        "what does the repo",
        "what is this repository",
        "main features",
        "primary features",
        "core features",
        "key features",
        "main capabilities",
        "primary capabilities",
        "applications",
        "overview",
        "architecture",
        "high level",
        "repo do",
    ]
    if any(marker in lowered for marker in overview_markers):
        return True

    asks_about_repo = "repo" in lowered or "repository" in lowered or "codebase" in lowered
    overview_intent = any(
        marker in lowered
        for marker in [
            "feature",
            "features",
            "capabilit",
            "provide",
            "provides",
            "provided by",
            "used for",
            "use it",
            "what can",
            "applications",
            "high level",
        ]
    )
    if asks_about_repo and overview_intent:
        return True

    return bool(
        re.search(
            r"what\\s+.*\\b(repo|repository|codebase)\\b.*\\b(do|does|provide|provides|features|capabilities)\\b",
            lowered,
        )
    )


def _needs_feature_ranking(question: str) -> bool:
    lowered = question.lower()
    ranking_markers = [
        "main features",
        "primary features",
        "key features",
        "core features",
        "features it provides",
        "what features",
        "feature set",
        "capabilities",
        "what does this repo",
        "what does the repo",
        "applications",
    ]
    if any(marker in lowered for marker in ranking_markers):
        return True

    asks_about_repo = "repo" in lowered or "repository" in lowered or "codebase" in lowered
    ranking_intent = any(
        marker in lowered
        for marker in [
            "feature",
            "features",
            "capabilit",
            "provide",
            "provides",
            "provided by",
            "used for",
            "use it",
            "applications",
            "what can",
        ]
    )
    if asks_about_repo and ranking_intent:
        return True

    return bool(
        re.search(
            r"features?\\s+provided\\s+by\\s+(this\\s+)?(repo|repository|codebase)",
            lowered,
        )
    )


def _rank_primary_features(
    question: str,
    evidence: List[Dict[str, Any]],
    ranking_model: str,
    planning_model: str,
    groq_api_key: str,
) -> Dict[str, Any]:
    def _derive_product_level_features() -> List[str]:
        readme_snippets = [
            str(item.get("snippet", ""))
            for item in evidence
            if str(item.get("file", "")).lower().endswith("readme.md")
        ]
        readme_text = "\n".join(readme_snippets)[:2600]
        cluster_notes = [
            str(item.get("snippet", ""))
            for item in evidence
            if str(item.get("source", "")) == "cluster_coverage"
        ]

        if not readme_text and not cluster_notes:
            return []

        prompt = (
            "Return ONLY valid JSON: {\"product_features\":[string,...]}. "
            "List 3-5 PRIMARY product capabilities of this repository. "
            "Ignore internal utilities, feature flags, lint/build scripts, and test harnesses unless explicitly user-facing. "
            "Focus on what end users or integrators can do.\n\n"
            f"Question: {question}\n"
            f"README context: {readme_text}\n"
            f"Cluster context: {' '.join(cluster_notes)[:800]}"
        )

        candidates = build_model_candidates(ranking_model) + [planning_model]
        seen = set()
        for model_name in candidates:
            if model_name in seen:
                continue
            seen.add(model_name)
            try:
                llm = ChatGroq(
                    model=model_name,
                    temperature=0.0,
                    max_tokens=220,
                    model_kwargs={"top_p": 1.0},
                    groq_api_key=groq_api_key,
                )
                response = llm.invoke([HumanMessage(content=prompt)])
                parsed = _parse_safe_json_dict(_extract_message_text(response))
                if not parsed:
                    continue
                features = parsed.get("product_features", [])
                if isinstance(features, list):
                    clean = [str(x).strip() for x in features if str(x).strip()]
                    if clean:
                        return clean[:5]
            except Exception:
                continue

        # Heuristic fallback from README/product vocabulary when abstraction model output is unavailable.
        keyword_map = [
            ("booking", "Booking and meeting scheduling"),
            ("availability", "Availability management"),
            ("calendar", "Calendar integrations and synchronization"),
            ("team", "Team scheduling and shared routing"),
            ("workflow", "Scheduling workflows and automations"),
            ("integrations", "Third-party integrations"),
        ]
        lowered = readme_text.lower()
        out: List[str] = []
        for key, label in keyword_map:
            if key in lowered and label not in out:
                out.append(label)
        return out[:5]

    def _readme_primary_candidates() -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen = set()
        for item in evidence:
            file_path = str(item.get("file", "")).lower()
            if not file_path.endswith("readme.md"):
                continue
            snippet = str(item.get("snippet", ""))
            for match in re.finditer(r"-\s*\*\*(.+?)\*\*", snippet):
                title = match.group(1).strip()
                lowered = title.lower()
                if not title or lowered in seen:
                    continue
                seen.add(lowered)
                candidates.append(
                    {
                        "feature": title,
                        "importance": len(candidates) + 1,
                        "why_core": "Declared as a top-level capability in the repository README.",
                        "evidence_files": [str(item.get("file", "README.md"))],
                    }
                )
                if len(candidates) >= 4:
                    return candidates
        return candidates

    def _apply_primary_feature_policy(ranked: Dict[str, Any]) -> Dict[str, Any]:
        primary = ranked.get("primary_features", [])
        secondary = ranked.get("secondary_or_tooling", [])
        notes = str(ranked.get("notes", "")).strip()
        product_level = _derive_product_level_features()
        broad_primary_query = _needs_feature_ranking(question) and not any(
            marker in question.lower()
            for marker in [
                "flag",
                "feature flag",
                "opt-in",
                "feature toggle",
                "ab test",
                "experiment",
            ]
        )

        if not isinstance(primary, list):
            primary = []
        if not isinstance(secondary, list):
            secondary = []

        demote_path_markers = {
            "__tests__",
            "tests",
            "typetests",
            "scripts",
            "eslint",
            "babel-preset",
            "private",
            "tester",
            "fixtures",
            "normalize-color",
        }
        demote_feature_markers = {
            "lint",
            "eslint",
            "test",
            "tester",
            "harness",
            "utility",
            "plugin",
            "preset",
        }

        kept: List[Dict[str, Any]] = []
        moved: List[Dict[str, Any]] = []
        for item in primary:
            feature_text = str(item.get("feature", "")).lower()
            evidence_files = [str(x).lower() for x in item.get("evidence_files", []) if str(x).strip()]
            should_demote = False
            if any(marker in feature_text for marker in demote_feature_markers):
                should_demote = True
            if any(any(marker in path for marker in demote_path_markers) for path in evidence_files):
                should_demote = True

            if should_demote:
                moved.append(item)
            else:
                kept.append(item)

        if kept:
            primary = kept[:5]
            secondary = (secondary + moved)[:8]
            if moved:
                notes = (notes + " " if notes else "") + "Policy demotion applied for tooling/test-harness/utility items."

        if len(primary) < 3:
            readme_candidates = _readme_primary_candidates()
            existing_primary = {str(item.get("feature", "")).strip().lower() for item in primary if isinstance(item, dict)}
            for candidate in readme_candidates:
                key = str(candidate.get("feature", "")).strip().lower()
                if not key or key in existing_primary:
                    continue
                candidate["importance"] = len(primary) + 1
                primary.append(candidate)
                existing_primary.add(key)
                if len(primary) >= 4:
                    break
            if readme_candidates:
                notes = (notes + " " if notes else "") + "README capability bullets used to enrich primary feature list."

        if product_level:
            existing_primary = {str(item.get("feature", "")).strip().lower() for item in primary if isinstance(item, dict)}
            inserted: List[Dict[str, Any]] = []
            readme_files = [
                str(item.get("file", ""))
                for item in evidence
                if str(item.get("file", "")).lower().endswith("readme.md")
            ]
            readme_refs = readme_files[:1] if readme_files else []
            for feature in product_level:
                key = feature.strip().lower()
                if not key or key in existing_primary:
                    continue
                inserted.append(
                    {
                        "feature": feature,
                        "importance": len(inserted) + 1,
                        "why_core": "Product-level capability inferred from repository overview context and README.",
                        "evidence_files": readme_refs,
                    }
                )
                existing_primary.add(key)
                if len(inserted) >= 3:
                    break

            if inserted:
                # Promote product-level capabilities ahead of local/internal module findings.
                merged = inserted + [item for item in primary if isinstance(item, dict)]
                deduped: List[Dict[str, Any]] = []
                seen_features = set()
                for item in merged:
                    name = str(item.get("feature", "")).strip().lower()
                    if not name or name in seen_features:
                        continue
                    seen_features.add(name)
                    deduped.append(item)
                    if len(deduped) >= 5:
                        break
                primary = deduped
                notes = (notes + " " if notes else "") + "Product-level abstraction pass applied."

        if broad_primary_query and primary:
            internal_feature_markers = {
                "feature flag",
                "feature-flag",
                "opt-in",
                "toggle",
                "core capability around",
            }
            internal_path_markers = {
                "/feature-opt-in/",
                "/features/flags/",
                "check-if-user-has-feature",
            }
            kept_primary: List[Dict[str, Any]] = []
            demoted_primary: List[Dict[str, Any]] = []
            for item in primary:
                if not isinstance(item, dict):
                    continue
                feature_text = str(item.get("feature", "")).lower()
                evidence_files = [str(x).lower() for x in item.get("evidence_files", []) if str(x).strip()]
                is_internal = any(marker in feature_text for marker in internal_feature_markers) or any(
                    any(marker in file_path for marker in internal_path_markers)
                    for file_path in evidence_files
                )
                if is_internal:
                    demoted_primary.append(item)
                else:
                    kept_primary.append(item)

            if kept_primary:
                primary = kept_primary[:5]
                if demoted_primary:
                    secondary = (secondary + demoted_primary)[:8]
                    notes = (notes + " " if notes else "") + "Broad-feature guardrail demoted internal flag/opt-in modules to secondary."

        return {
            "primary_features": primary,
            "secondary_or_tooling": secondary,
            "notes": notes[:260],
        }

    def _heuristic_rank() -> Dict[str, Any]:
        primary: List[Dict[str, Any]] = []
        secondary: List[Dict[str, Any]] = []
        seen_primary = set()

        for item in evidence[:18]:
            file_path = str(item.get("file", "")).strip()
            if not file_path:
                continue
            lowered = file_path.lower()
            signal_tier = str(item.get("signal_tier") or _signal_tier_for_path(file_path))

            is_secondary = (
                signal_tier == "low"
                or "/script" in lowered
                or "/scripts/" in lowered
                or "release" in lowered
                or "debug" in lowered
                or _is_low_signal_path(lowered)
            )

            if is_secondary:
                secondary.append(
                    {
                        "finding": _directory_bucket(file_path),
                        "reason": "tooling/test/secondary signal",
                        "evidence_file": file_path,
                    }
                )
                continue

            bucket = _directory_bucket(file_path)
            if bucket in seen_primary:
                continue
            seen_primary.add(bucket)
            primary.append(
                {
                    "feature": f"Core capability around {bucket}",
                    "importance": len(primary) + 1,
                    "why_core": "Located in production-path code and referenced by repository evidence.",
                    "evidence_files": [file_path],
                }
            )
            if len(primary) >= 5:
                break

        return {
            "primary_features": primary[:5],
            "secondary_or_tooling": secondary[:6],
            "notes": "Heuristic ranking used due parser/model output limits.",
        }

    fallback = {
        "primary_features": [],
        "secondary_or_tooling": [],
        "notes": "Feature ranking unavailable; answer should rely on strongest production evidence.",
    }

    if not evidence:
        return fallback

    ranked_input = []
    for item in evidence[:14]:
        ranked_input.append(
            {
                "source": item.get("source"),
                "file": item.get("file"),
                "name": item.get("name"),
                "signal_tier": item.get("signal_tier")
                or _signal_tier_for_path(str(item.get("file", ""))),
                "snippet": str(item.get("snippet", ""))[:220],
            }
        )

    packed_evidence = _pack_json_by_char_budget(
        ranked_input,
        FEATURE_RANKING_EVIDENCE_CHAR_BUDGET,
    )

    prompt = (
        "Return ONLY valid JSON with keys: primary_features, secondary_or_tooling, notes. "
        "primary_features must be a ranked array of 3 to 5 items. "
        "Each item must have: feature, importance (1 highest), why_core, evidence_files (array). "
        "secondary_or_tooling must list incidental/test/build/release/debug-only findings. "
        "Ignore low-signal evidence (tests/typetests/examples/fixtures/mocks) unless no better evidence exists. "
        "For broad repository-feature questions, prefer capabilities corroborated by multiple files or top-level metadata (README/package manifest). "
        "Do not treat one-off internal classes or lint/build-only checks as primary features unless strongly corroborated. "
        "Treat build orchestration, lint rules, test harnesses, and narrow utility packages as secondary by default unless the user explicitly asks about them. "
        "Prioritize user-facing and architecture-defining capabilities over scripts/tooling.\n\n"
        f"Question: {question}\n"
        f"Evidence: {packed_evidence}"
    )

    ranking_candidates = build_model_candidates(ranking_model) + [planning_model]
    seen_models = set()
    for model_name in ranking_candidates:
        if model_name in seen_models:
            continue
        seen_models.add(model_name)
        try:
            ranker = ChatGroq(
                model=model_name,
                temperature=0.0,
                max_tokens=420,
                model_kwargs={"top_p": 1.0},
                groq_api_key=groq_api_key,
            )
            response = ranker.invoke([HumanMessage(content=prompt)])
            parsed = _parse_safe_json_dict(_extract_message_text(response))
            if not parsed:
                continue

            primary = parsed.get("primary_features", [])
            secondary = parsed.get("secondary_or_tooling", [])
            notes = str(parsed.get("notes", "")).strip()

            if not isinstance(primary, list):
                primary = []
            if not isinstance(secondary, list):
                secondary = []

            if not primary:
                continue

            return _apply_primary_feature_policy({
                "primary_features": primary[:5],
                "secondary_or_tooling": secondary[:6],
                "notes": notes[:260],
            })
        except Exception:
            continue

    return _apply_primary_feature_policy(_heuristic_rank())


def _directory_bucket(path: str) -> str:
    parts = _path_parts(path)
    if not parts:
        return "root"

    priority_anchors = [
        "packages",
        "src",
        "reactandroid",
        "android",
        "ios",
        "backend",
        "frontend",
        "app",
    ]
    lowered_parts = [part.lower() for part in parts]
    for anchor in priority_anchors:
        if anchor in lowered_parts:
            idx = lowered_parts.index(anchor)
            next_idx = min(idx + 1, len(lowered_parts) - 1)
            if idx == next_idx:
                return lowered_parts[idx]
            return f"{lowered_parts[idx]}/{lowered_parts[next_idx]}"

    tail = lowered_parts[-2:] if len(lowered_parts) >= 2 else lowered_parts
    return "/".join(tail)


def _dedupe_results(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in items:
        key = (
            str(item.get("file", "")),
            str(item.get("name", "")),
            item.get("start_line"),
            item.get("end_line"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _diversify_results(
    items: List[Dict[str, Any]],
    max_items: int,
    max_per_bucket: int = 2,
) -> List[Dict[str, Any]]:
    bucket_counts: Dict[str, int] = {}
    selected: List[Dict[str, Any]] = []

    for item in items:
        bucket = _directory_bucket(str(item.get("file", "")))
        count = bucket_counts.get(bucket, 0)
        if count >= max_per_bucket:
            continue
        selected.append(item)
        bucket_counts[bucket] = count + 1
        if len(selected) >= max_items:
            return selected

    if len(selected) < max_items:
        selected_keys = {
            (
                str(item.get("file", "")),
                str(item.get("name", "")),
                item.get("start_line"),
                item.get("end_line"),
            )
            for item in selected
        }
        for item in items:
            key = (
                str(item.get("file", "")),
                str(item.get("name", "")),
                item.get("start_line"),
                item.get("end_line"),
            )
            if key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(key)
            if len(selected) >= max_items:
                break

    return selected


def _cluster_results_by_folder(
    items: List[Dict[str, Any]],
    depth: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        file_path = str(item.get("file", "")).strip().replace("\\", "/")
        if not file_path:
            continue
        parts = [part for part in file_path.split("/") if part]
        key = "/".join(parts[: max(1, min(depth, len(parts)))])
        clusters.setdefault(key, []).append(item)
    return clusters


def _select_cluster_representatives(
    items: List[Dict[str, Any]],
    max_clusters: int,
    max_per_cluster: int,
) -> List[Dict[str, Any]]:
    clusters = _cluster_results_by_folder(items)
    selected: List[Dict[str, Any]] = []

    ordered_clusters = sorted(clusters.items(), key=lambda pair: len(pair[1]), reverse=True)
    for _, cluster_items in ordered_clusters[:max_clusters]:
        selected.extend(cluster_items[:max_per_cluster])

    if not selected:
        return items

    return _dedupe_results(selected)


def run_multi_reasoning_agent(
    question: str,
    repo_path: Optional[str] = None,
    runtime: Optional[FunctionContextRuntime] = None,
    conversation_context: Optional[List[Dict[str, str]]] = None,
    query_scope: Optional[str] = None,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    llm_model: str = DEFAULT_FINAL_MODEL,
    planning_model: str = DEFAULT_PLANNING_MODEL,
    temperature: float = 0.2,
    max_completion_tokens: int = 4096,
    top_p: float = 1.0,
    max_reasoning_steps: int = 2,
    budget_mode: bool = True,
    batch_size: int = 32,
    api_key: Optional[str] = None,
    api_key_var: str = DEFAULT_API_KEY_VAR,
) -> Dict[str, Any]:
    # Hard strategy limits for API-cost and loop control.
    max_reasoning_steps = min(max_reasoning_steps, 2)
    max_completion_tokens = min(max_completion_tokens, 2400 if budget_mode else 3200)

    resolved_key = resolve_api_key(api_key=api_key, api_key_var=api_key_var)
    resolved_scope = _normalize_query_scope(
        query_scope or classify_query_scope(
            question=question,
            planning_model=planning_model,
            api_key=resolved_key,
            api_key_var=api_key_var,
        )
    )

    history_tail = (conversation_context or [])[-4:]
    if resolved_scope == "general":
        result = _answer_general_question(
            question=question,
            history_tail=history_tail,
            llm_model=llm_model,
            planning_model=planning_model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            resolved_key=resolved_key,
        )
        result["query_scope"] = resolved_scope
        return result

    if runtime is None:
        if not repo_path:
            raise ValueError("repo_path is required for repository-grounded queries")
        runtime = FunctionContextRuntime.create(
            repo_path=repo_path,
            embedding_model_name=embedding_model_name,
            batch_size=batch_size,
        )

    hybrid_general_context = ""
    if resolved_scope == "hybrid":
        hybrid_preview = _answer_general_question(
            question=question,
            history_tail=history_tail,
            llm_model=llm_model,
            planning_model=planning_model,
            temperature=0.1,
            max_completion_tokens=min(280, max_completion_tokens),
            top_p=top_p,
            resolved_key=resolved_key,
        )
        hybrid_general_context = str(hybrid_preview.get("final_answer", ""))[:1200]

    cache: Dict[str, Dict[str, Any]] = {}
    tool_trace: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []

    def _cached_tool_call(name: str, arguments: Dict[str, Any], fn: Any) -> Dict[str, Any]:
        if len(tool_trace) >= MAX_TOOL_CALLS:
            return {"stopped": True, "reason": f"tool call cap reached: {MAX_TOOL_CALLS}"}

        cache_key = f"{name}:{json.dumps(arguments, sort_keys=True)}"
        if cache_key in cache:
            cached_result = cache[cache_key]
            preview = str(cached_result)[:220]
            tool_trace.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "cached": True,
                    "result_preview": preview,
                }
            )
            return cache[cache_key]

        result = fn(**arguments)
        cache[cache_key] = result
        preview = str(result)[:220]
        tool_trace.append(
            {
                "name": name,
                "arguments": arguments,
                "cached": False,
                "result_preview": preview,
            }
        )
        return result

    def _safe_json_parse(raw: str) -> Optional[Dict[str, Any]]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    def _build_adaptive_plan(input_question: str) -> Dict[str, Any]:
        fallback = {
            "search_query": input_question,
            "need_repo_summary": True,
            "need_file_reads": True,
            "need_dependency_trace": _needs_dependency_trace(input_question),
            "focus_terms": [],
        }
        try:
            prompt = (
                "Return ONLY valid JSON with keys: "
                "search_query (string), need_repo_summary (boolean), need_file_reads (boolean), "
                "need_dependency_trace (boolean), focus_terms (array of strings max 6). "
                "Choose values to best answer the user question from code evidence.\n"
                f"Question: {input_question}"
            )
            text = _invoke_planning_llm(
                messages=[HumanMessage(content=prompt)],
                planning_model=planning_model,
                groq_api_key=resolved_key,
                max_tokens=220,
            )
            parsed = _safe_json_parse(text)
            if not parsed:
                return fallback

            search_query = str(parsed.get("search_query", input_question)).strip() or input_question
            return {
                "search_query": search_query,
                "need_repo_summary": bool(parsed.get("need_repo_summary", True)),
                "need_file_reads": bool(parsed.get("need_file_reads", True)),
                "need_dependency_trace": bool(
                    parsed.get("need_dependency_trace", _needs_dependency_trace(input_question))
                ),
                "focus_terms": [str(x).strip().lower() for x in parsed.get("focus_terms", []) if str(x).strip()][:6],
            }
        except Exception:
            return fallback

    plan = _build_adaptive_plan(question)
    concept_grounding = _ground_query_concept(
        question=question,
        planning_model=planning_model,
        groq_api_key=resolved_key,
    )

    if concept_grounding.get("needs_dependency_trace"):
        plan["need_dependency_trace"] = True

    planning_query = re.sub(r"[^A-Za-z0-9_\-\.\s]", " ", plan["search_query"]).strip()
    if not planning_query or len(planning_query.split()) > 14:
        planning_query = question

    grounded_terms = [str(term).strip().lower() for term in concept_grounding.get("search_terms", []) if str(term).strip()]
    if grounded_terms:
        merged_query_terms = [planning_query] + grounded_terms[:10]
        planning_query = " ".join(dict.fromkeys(merged_query_terms))[:260]

    is_overview_question = _is_repository_overview_question(question)
    # For overview questions, override search query to find structural files
    if is_overview_question:
        planning_query = "main entry service module controller route handler register"
    feature_focus_question = is_overview_question or _needs_feature_ranking(question)
    flow_focus_question = _needs_flow_reconstruction(question)
    search_top_k = 10 if feature_focus_question else 4
    diversification_target = 10 if feature_focus_question else 5
    file_reads_target = 6 if feature_focus_question else 2
    if flow_focus_question:
        file_reads_target = max(file_reads_target, 4)

    plan_steps: List[str] = ["search_code", "inspect_details"]
    if plan.get("need_repo_summary"):
        plan_steps.insert(0, "repo_summary")
    if plan.get("need_file_reads"):
        plan_steps.append("read_files")
    if plan.get("need_dependency_trace"):
        plan_steps.append("dependency_trace")
    if max_reasoning_steps <= 1:
        plan_steps = [step for step in plan_steps if step in {"search_code", "inspect_details"}]

    if flow_focus_question:
        if "read_files" not in plan_steps:
            plan_steps.append("read_files")
        if "dependency_trace" not in plan_steps:
            plan_steps.append("dependency_trace")

    if "repo_summary" in plan_steps and len(tool_trace) < MAX_TOOL_CALLS:
        repo_summary = _cached_tool_call("get_repo_summary", {}, runtime.get_repo_summary)
        for top_file in repo_summary.get("top_function_files", [])[:2]:
            evidence.append(
                {
                    "source": "repo_summary",
                    "file": top_file.get("file"),
                    "name": "top_function_file",
                    "snippet": json.dumps(top_file)[:280],
                }
            )

    if feature_focus_question and len(tool_trace) < MAX_TOOL_CALLS:
        for metadata_file in ["README.md", "package.json", "packages/react-native/package.json"]:
            if len(tool_trace) >= MAX_TOOL_CALLS:
                break
            metadata_result = _cached_tool_call(
                "read_file",
                {"file": metadata_file, "max_chars": 2200},
                runtime.read_file,
            )
            if isinstance(metadata_result, dict) and metadata_result.get("error"):
                continue
            evidence.append(
                {
                    "source": "overview_metadata",
                    "file": metadata_result.get("file", metadata_file),
                    "signal_tier": "high",
                    "snippet": str(metadata_result.get("content", ""))[:2800],
                }
            )

    search_result = _cached_tool_call(
        "search_code",
        {"query": planning_query, "top_k": search_top_k},
        runtime.search_code,
    )
    results = search_result.get("results", []) if isinstance(search_result, dict) else []

    blocked_path_tokens = ["/node_modules/", "\\node_modules\\", "/vendor/", "\\vendor\\", "/.git/"]
    focus_terms = [term for term in plan.get("focus_terms", []) if term]
    if not focus_terms:
        focus_terms = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", question)][:6]
    if grounded_terms:
        focus_terms = list(dict.fromkeys(focus_terms + grounded_terms[:8]))

    anti_terms = [
        str(term).strip().lower()
        for term in concept_grounding.get("anti_signals", [])
        if str(term).strip()
    ]

    def _score_item(item: Dict[str, Any]) -> int:
        haystack = " ".join(
            [
                str(item.get("name", "")),
                str(item.get("file", "")),
                str(item.get("text", ""))[:500],
            ]
        ).lower()
        signal_tier = _signal_tier_for_path(str(item.get("file", "")))
        is_tooling = _is_tooling_path(str(item.get("file", "")))
        score = sum(1 for term in focus_terms if term in haystack)
        score -= 2 * sum(1 for term in anti_terms if term in haystack)

        if signal_tier == "high":
            score += 4
        elif signal_tier == "medium":
            score += 1
        else:
            # Keep low-signal files as weak supporting evidence only.
            score -= 6

        if is_tooling:
            score -= 9 if is_overview_question else 3

        return score

    filtered = []
    for item in results:
        file_path = str(item.get("file", "")).lower()
        if any(token in file_path for token in blocked_path_tokens):
            continue
        filtered.append(item)

    relevant_results = sorted(filtered, key=_score_item, reverse=True)

    def _concept_relevance(item: Dict[str, Any]) -> int:
        haystack = " ".join(
            [
                str(item.get("name", "")),
                str(item.get("file", "")),
                str(item.get("text", ""))[:500],
            ]
        ).lower()
        positive = sum(1 for term in grounded_terms if term in haystack)
        negative = sum(1 for term in anti_terms if term in haystack)
        return positive - negative

    if grounded_terms and len(tool_trace) < MAX_TOOL_CALLS:
        top_relevance = max((_concept_relevance(item) for item in relevant_results[:4]), default=-1)
        if top_relevance < 2:
            correction_query = " ".join(grounded_terms[:12]).strip()
            if correction_query:
                correction_result = _cached_tool_call(
                    "search_code",
                    {"query": correction_query, "top_k": max(6, search_top_k)},
                    runtime.search_code,
                )
                correction_items = (
                    correction_result.get("results", [])
                    if isinstance(correction_result, dict)
                    else []
                )
                correction_filtered = [
                    item
                    for item in correction_items
                    if not any(token in str(item.get("file", "")).lower() for token in blocked_path_tokens)
                ]
                relevant_results = _dedupe_results(relevant_results + correction_filtered)
                relevant_results = sorted(relevant_results, key=_score_item, reverse=True)

    # Multi-pass expansion for better repository coverage on broad questions.
    did_expanded_search = False
    if len(tool_trace) < MAX_TOOL_CALLS:
        should_expand = feature_focus_question or len(relevant_results) < 3
        if should_expand:
            broad_query = (
                f"{planning_query} architecture core api service module flow database "
                f"entrypoint main booking calendar scheduling availability auth integration"
            )
            if grounded_terms:
                broad_query = f"{planning_query} {' '.join(grounded_terms[:8])} architecture core flow entrypoint"
            expanded_result = _cached_tool_call(
                "search_code",
                {"query": broad_query.strip(), "top_k": max(5, search_top_k - 1)},
                runtime.search_code,
            )
            expanded_items = (
                expanded_result.get("results", [])
                if isinstance(expanded_result, dict)
                else []
            )
            expanded_filtered = [
                item
                for item in expanded_items
                if not any(token in str(item.get("file", "")).lower() for token in blocked_path_tokens)
            ]
            relevant_results = _dedupe_results(relevant_results + expanded_filtered)
            relevant_results = sorted(relevant_results, key=_score_item, reverse=True)
            did_expanded_search = True

    if feature_focus_question and relevant_results:
        clustered = _select_cluster_representatives(
            relevant_results,
            max_clusters=6,
            max_per_cluster=2,
        )
        relevant_results = _dedupe_results(clustered + relevant_results)
        relevant_results = sorted(relevant_results, key=_score_item, reverse=True)

    if feature_focus_question and len(tool_trace) < MAX_TOOL_CALLS:
        primary_clusters = _cluster_results_by_folder(relevant_results, depth=3)
        if len(primary_clusters) < 2:
            subsystem_query = (
                f"{question} modules subsystems architecture scheduling calendar integration "
                f"api web app"
            )
            subsystem_result = _cached_tool_call(
                "search_code",
                {"query": subsystem_query.strip(), "top_k": max(8, search_top_k)},
                runtime.search_code,
            )
            subsystem_items = subsystem_result.get("results", []) if isinstance(subsystem_result, dict) else []
            subsystem_filtered = [
                item
                for item in subsystem_items
                if not any(token in str(item.get("file", "")).lower() for token in blocked_path_tokens)
            ]
            relevant_results = _dedupe_results(relevant_results + subsystem_filtered)
            relevant_results = sorted(relevant_results, key=_score_item, reverse=True)

    production_first = [
        item
        for item in relevant_results
        if _signal_tier_for_path(str(item.get("file", ""))) != "low"
        and (not feature_focus_question or not _is_tooling_path(str(item.get("file", ""))))
    ]
    low_signal_results = [
        item for item in relevant_results if _signal_tier_for_path(str(item.get("file", ""))) == "low"
    ]
    if not relevant_results and len(tool_trace) < MAX_TOOL_CALLS and not did_expanded_search:
        fallback_result = _cached_tool_call(
            "search_code",
            {"query": question, "top_k": 4},
            runtime.search_code,
        )
        fallback_items = fallback_result.get("results", []) if isinstance(fallback_result, dict) else []
        relevant_results = [
            item
            for item in fallback_items
            if not any(token in str(item.get("file", "")).lower() for token in blocked_path_tokens)
        ]
        production_first = [
            item for item in relevant_results if _signal_tier_for_path(str(item.get("file", ""))) != "low"
        ]
        low_signal_results = [
            item for item in relevant_results if _signal_tier_for_path(str(item.get("file", ""))) == "low"
        ]

    prioritized_results = _diversify_results(
        production_first + low_signal_results,
        max_items=diversification_target,
        max_per_bucket=2 if feature_focus_question else 1,
    )

    symbol_graph_items: List[Dict[str, Any]] = []
    symbol_graph_edges: List[Dict[str, str]] = []
    if flow_focus_question and prioritized_results and len(tool_trace) <= MAX_TOOL_CALLS - 3:
        # Prefer logic files over response/model files as the traversal seed
        _logic_signals = {"middleware", "service", "controller", "handler", "auth", "login", "guard", "verify", "route"}
        _skip_signals = {"response/model", "utopia/response", "/model/", "response\\model"}

        seed_item = prioritized_results[0]  # default
        for _candidate in prioritized_results[:6]:
            _path = str(_candidate.get("file", "")).lower().replace("\\", "/")
            _is_logic = any(s in _path for s in _logic_signals)
            _is_model = any(s in _path for s in _skip_signals)
            if _is_logic and not _is_model:
                seed_item = _candidate
                break

        seed_file = str(seed_item.get("file", "")).strip()
        seed_name = str(seed_item.get("name", "")).strip()
        if seed_file:
            deps_result = _cached_tool_call(
                "get_dependencies",
                {"file": seed_file, "max_items": 24},
                runtime.get_dependencies,
            )
            deps = [str(dep).strip() for dep in deps_result.get("dependencies", []) if str(dep).strip()]
            for dep in deps[:6]:
                if len(tool_trace) >= MAX_TOOL_CALLS:
                    break
                # Only follow deps that contain at least one grounded concept term (concept relevance gate)
                dep_lower = dep.lower().replace("\\", "/")
                concept_hit = any(term in dep_lower for term in grounded_terms[:8]) if grounded_terms else True
                if not concept_hit:
                    continue
                dep_query = dep.split("/")[-1].strip() or dep
                dep_search = _cached_tool_call(
                    "search_code",
                    {"query": dep_query, "top_k": 2},
                    runtime.search_code,
                )
                dep_items = dep_search.get("results", []) if isinstance(dep_search, dict) else []
                for match in dep_items[:2]:
                    match_file = str(match.get("file", "")).strip()
                    if not match_file or match_file == seed_file:
                        continue
                    match_haystack = " ".join(
                        [
                            str(match.get("name", "")),
                            match_file,
                            str(match.get("text", ""))[:240],
                        ]
                    ).lower()
                    positive_hits = sum(1 for term in grounded_terms[:10] if term and term in match_haystack)
                    negative_hits = sum(1 for term in anti_terms[:8] if term and term in match_haystack)
                    if positive_hits <= negative_hits and flow_focus_question:
                        continue
                    symbol_graph_edges.append({"from": seed_file, "to": match_file, "via": dep, "kind": "import"})
                    symbol_graph_items.append(match)

            if seed_name and seed_name not in {"__construct", "__destruct", "__init__", "constructor"} and len(seed_name) > 4 and len(tool_trace) <= MAX_TOOL_CALLS - 2:
                usage_search = _cached_tool_call(
                    "search_code",
                    {"query": seed_name, "top_k": 3},
                    runtime.search_code,
                )
                usage_items = usage_search.get("results", []) if isinstance(usage_search, dict) else []
                for match in usage_items[:2]:
                    match_file = str(match.get("file", "")).strip()
                    if not match_file or match_file == seed_file:
                        continue
                    usage_haystack = " ".join(
                        [
                            str(match.get("name", "")),
                            match_file,
                            str(match.get("text", ""))[:240],
                        ]
                    ).lower()
                    positive_hits = sum(1 for term in grounded_terms[:10] if term and term in usage_haystack)
                    negative_hits = sum(1 for term in anti_terms[:8] if term and term in usage_haystack)
                    if positive_hits <= negative_hits and flow_focus_question:
                        continue
                    symbol_graph_edges.append({"from": seed_file, "to": match_file, "via": seed_name, "kind": "usage"})
                    symbol_graph_items.append(match)

            if symbol_graph_edges:
                evidence.append(
                    {
                        "source": "symbol_graph",
                        "file": seed_file,
                        "graph_edges": symbol_graph_edges[:12],
                        "snippet": json.dumps(
                            {
                                "seed": seed_file,
                                "edge_count": len(symbol_graph_edges),
                                "edges": symbol_graph_edges[:8],
                            }
                        )[:600],
                    }
                )
                for item in symbol_graph_items[:2]:
                    evidence.append(
                        {
                            "source": "symbol_graph_node",
                            "file": item.get("file"),
                            "name": item.get("name"),
                            "snippet": str(item.get("text", ""))[:260],
                        }
                    )

            # Second hop — follow the callers/deps of graph nodes found in first hop
            if symbol_graph_items and len(tool_trace) <= MAX_TOOL_CALLS - 2:
                for _hop2_item in symbol_graph_items[:2]:
                    _hop2_file = str(_hop2_item.get("file", "")).strip()
                    _hop2_name = str(_hop2_item.get("name", "")).strip()
                    if not _hop2_file or _hop2_file == seed_file:
                        continue
                    if len(tool_trace) >= MAX_TOOL_CALLS:
                        break
                    _hop2_deps = _cached_tool_call(
                        "get_dependencies",
                        {"file": _hop2_file, "max_items": 12},
                        runtime.get_dependencies,
                    )
                    _hop2_dep_list = [str(d).strip() for d in _hop2_deps.get("dependencies", []) if str(d).strip()]
                    for _dep in _hop2_dep_list[:2]:
                        if len(tool_trace) >= MAX_TOOL_CALLS:
                            break
                        _dep_q = _dep.split("/")[-1].strip() or _dep
                        _dep_res = _cached_tool_call(
                            "search_code",
                            {"query": _dep_q, "top_k": 1},
                            runtime.search_code,
                        )
                        _dep_hits = _dep_res.get("results", []) if isinstance(_dep_res, dict) else []
                        for _hit in _dep_hits[:1]:
                            _hit_file = str(_hit.get("file", "")).strip()
                            if not _hit_file or _hit_file in {seed_file, _hop2_file}:
                                continue
                            evidence.append({
                                "source": "hop2_dependency",
                                "file": _hit_file,
                                "name": str(_hit.get("name", "")),
                                "snippet": str(_hit.get("text", ""))[:600],
                            })

    if feature_focus_question and prioritized_results:
        cluster_map = _cluster_results_by_folder(prioritized_results, depth=3)
        evidence.append(
            {
                "source": "cluster_coverage",
                "snippet": json.dumps(
                    {
                        "cluster_count": len(cluster_map),
                        "clusters": list(cluster_map.keys())[:8],
                    }
                )[:420],
            }
        )

    if concept_grounding.get("concept"):
        evidence.append(
            {
                "source": "concept_grounding",
                "snippet": json.dumps(
                    {
                        "concept": concept_grounding.get("concept"),
                        "search_terms": grounded_terms[:10],
                        "anti_signals": anti_terms[:8],
                        "flow_steps": concept_grounding.get("flow_steps", [])[:6],
                    }
                )[:420],
            }
        )

    for item in prioritized_results[:3]:
        evidence.append(
            {
                "source": "search_result",
                "file": item.get("file"),
                "name": item.get("name"),
                "start_line": item.get("start_line"),
                "end_line": item.get("end_line"),
                "signal_tier": _signal_tier_for_path(str(item.get("file", ""))),
                "snippet": str(item.get("text", ""))[:280],
            }
        )

    if prioritized_results and len(tool_trace) < MAX_TOOL_CALLS and (not feature_focus_question or flow_focus_question):
        first = prioritized_results[0]
        first_name = str(first.get("name", "")).strip()
        first_file = str(first.get("file", "")).strip()
        if first_name:
            detail_result = _cached_tool_call(
                "get_function_details",
                {"name": first_name, "file": first_file, "top_k": 1},
                runtime.get_function_details,
            )
            for detail_item in detail_result.get("results", [])[:1]:
                evidence.append(
                    {
                        "source": "function_detail",
                        "file": detail_item.get("file"),
                        "name": detail_item.get("name"),
                        "start_line": detail_item.get("start_line"),
                        "end_line": detail_item.get("end_line"),
                        "snippet": str(detail_item.get("text", ""))[:280],
                    }
                )

    if "read_files" in plan_steps and prioritized_results and len(tool_trace) < MAX_TOOL_CALLS:
        inspected_files = set()
        read_candidates = prioritized_results
        if flow_focus_question and symbol_graph_items:
            read_candidates = _dedupe_results(symbol_graph_items + read_candidates)
        if feature_focus_question:
            read_candidates = [
                item
                for item in read_candidates
                if _signal_tier_for_path(str(item.get("file", ""))) != "low"
                and "/scripts/" not in str(item.get("file", "")).lower()
                and not _is_tooling_path(str(item.get("file", "")))
            ]
            if not read_candidates:
                read_candidates = prioritized_results

        for item in read_candidates[:file_reads_target]:
            file_name = str(item.get("file", "")).strip()
            if not file_name or file_name in inspected_files:
                continue
            read_result = _cached_tool_call(
                "read_file",
                {"file": file_name, "max_chars": 4000},
                runtime.read_file,
            )
            inspected_files.add(file_name)
            evidence.append(
                {
                    "source": "file_read",
                    "file": read_result.get("file", file_name),
                    "snippet": str(read_result.get("content", "")),
                }
            )
            if len(tool_trace) >= MAX_TOOL_CALLS:
                break

    if "dependency_trace" in plan_steps and prioritized_results and len(tool_trace) < MAX_TOOL_CALLS:
        file_name = str(prioritized_results[0].get("file", "")).strip()
        if file_name:
            deps_result = _cached_tool_call(
                "get_dependencies",
                {"file": file_name, "max_items": 40},
                runtime.get_dependencies,
            )
            evidence.append(
                {
                    "source": "dependencies",
                    "file": deps_result.get("file", file_name),
                    "snippet": json.dumps(deps_result)[:420],
                }
            )

            if flow_focus_question and len(tool_trace) < MAX_TOOL_CALLS:
                deps = [str(dep).strip() for dep in deps_result.get("dependencies", []) if str(dep).strip()]
                for dep in deps[:6]:
                    if len(tool_trace) >= MAX_TOOL_CALLS:
                        break
                    # Only follow deps that contain at least one grounded concept term (concept relevance gate)
                    dep_lower = dep.lower().replace("\\", "/")
                    concept_hit = any(term in dep_lower for term in grounded_terms[:8]) if grounded_terms else True
                    if not concept_hit:
                        continue
                    dep_search = _cached_tool_call(
                        "search_code",
                        {"query": dep, "top_k": 2},
                        runtime.search_code,
                    )
                    dep_items = dep_search.get("results", []) if isinstance(dep_search, dict) else []
                    if not dep_items:
                        continue
                    dep_file = str(dep_items[0].get("file", "")).strip()
                    dep_name = str(dep_items[0].get("name", "")).strip()
                    evidence.append(
                        {
                            "source": "dependency_search",
                            "file": dep_file,
                            "name": dep_name,
                            "snippet": str(dep_items[0].get("text", "")),
                        }
                    )

    if not evidence and len(tool_trace) < MAX_TOOL_CALLS:
        file_summaries = runtime.get_file_summaries(top_k=3)
        for summary in file_summaries.get("summaries", [])[:2]:
            file_name = str(summary.get("file", ""))
            read_result = _cached_tool_call(
                "read_file",
                {"file": file_name, "max_chars": 1200},
                runtime.read_file,
            )
            evidence.append(
                {
                    "source": "file_read",
                    "file": read_result.get("file", file_name),
                    "snippet": str(read_result.get("content", "")),
                }
            )

    if flow_focus_question and not evidence and len(tool_trace) < MAX_TOOL_CALLS:
        repo_summary = _cached_tool_call("get_repo_summary", {}, runtime.get_repo_summary)
        for top_file in repo_summary.get("top_function_files", [])[:2]:
            if len(tool_trace) >= MAX_TOOL_CALLS:
                break
            file_name = str(top_file.get("file", "")).strip()
            if not file_name:
                continue
            read_result = _cached_tool_call(
                "read_file",
                {"file": file_name, "max_chars": 1800},
                runtime.read_file,
            )
            evidence.append(
                {
                    "source": "low_evidence_fallback",
                    "file": read_result.get("file", file_name),
                    "signal_tier": _signal_tier_for_path(file_name),
                    "snippet": str(read_result.get("content", "")),
                }
            )

    # For overview questions, force broader file-level coverage even when some evidence already exists.
    if is_overview_question and len(tool_trace) < MAX_TOOL_CALLS:
        summary_result = runtime.get_file_summaries(top_k=10)
        overview_candidates = summary_result.get("summaries", []) if isinstance(summary_result, dict) else []
        read_targets: List[str] = []
        for item in overview_candidates:
            file_name = str(item.get("file", "")).strip()
            if not file_name:
                continue
            if _signal_tier_for_path(file_name) == "low":
                continue
            read_targets.append(file_name)
            if len(read_targets) >= 3:
                break

        for file_name in read_targets:
            if len(tool_trace) >= MAX_TOOL_CALLS:
                break
            read_result = _cached_tool_call(
                "read_file",
                {"file": file_name, "max_chars": 2200},
                runtime.read_file,
            )
            evidence.append(
                {
                    "source": "overview_file_read",
                    "file": read_result.get("file", file_name),
                    "signal_tier": _signal_tier_for_path(file_name),
                    "snippet": str(read_result.get("content", "")),
                }
            )

    query_specific_rule = (
        "Answer exactly what the user asked. Prefer direct, explicit technology names when present in evidence. "
        "If evidence is partial, say what is known and what remains uncertain without changing topic."
    )

    if budget_mode:
        dynamic_answer_instruction = (
            "Answer directly with concrete file/function evidence and keep uncertainty explicit."
        )
    else:
        dynamic_answer_instruction = _build_dynamic_answer_instruction(
            question=question,
            evidence=evidence,
            plan_steps=plan_steps,
            planning_model=planning_model,
            groq_api_key=resolved_key,
        )

    general_context_block = (
        f"General knowledge context for hybrid question: {hybrid_general_context}\n\n"
        if hybrid_general_context
        else ""
    )

    feature_ranking_block = ""
    concept_grounding_block = ""
    flow_reconstruction_block = ""
    ranked_features: Dict[str, Any] = {
        "primary_features": [],
        "secondary_or_tooling": [],
        "notes": "",
    }
    flow_reconstruction: Dict[str, Any] = {}
    if is_overview_question or _needs_feature_ranking(question):
        ranked_features = _rank_primary_features(
            question=question,
            evidence=evidence,
            ranking_model=llm_model,
            planning_model=planning_model,
            groq_api_key=resolved_key,
        )
        feature_ranking_block = (
            "Feature saliency ranking (use as the priority guide for final answer): "
            f"{json.dumps(ranked_features)}\n\n"
        )

    if concept_grounding.get("concept"):
        concept_grounding_block = (
            "Concept grounding (semantic intent and retrieval constraints): "
            f"{json.dumps(concept_grounding)}\n\n"
        )

    if _needs_flow_reconstruction(question) and not is_overview_question:
        flow_reconstruction = _reconstruct_concept_flow(
            question=question,
            concept_grounding=concept_grounding,
            evidence=evidence,
            planning_model=planning_model,
            groq_api_key=resolved_key,
        )
        flow_reconstruction_block = (
            "Flow reconstruction draft (prioritize this for flow questions): "
            f"{json.dumps(flow_reconstruction)}\n\n"
        )

    flow_question = bool(_needs_flow_reconstruction(question) and not is_overview_question)
    strict_flow_block = (
        f"Strict flow synthesis policy: {STRICT_REPO_GROUNDED_FLOW_INSTRUCTION}\n\n"
        if flow_question
        else ""
    )

    final_prompt_messages: List[BaseMessage] = [
        SystemMessage(
            content=(
                "You are a senior code-analysis assistant. "
                "Do not call tools. Use only provided evidence."
            )
        ),
        HumanMessage(
            content=(
                f"Question: {question}\n\n"
                f"Plan used: {json.dumps(plan_steps)}\n"
                f"Tool calls executed: {_pack_json_by_char_budget(tool_trace, FINAL_PROMPT_TOOL_TRACE_CHAR_BUDGET)}\n\n"
                f"Recent conversation context: {_pack_json_by_char_budget(history_tail, FINAL_PROMPT_HISTORY_CHAR_BUDGET)}\n\n"
                f"Evidence: {_pack_json_by_char_budget(evidence, FINAL_PROMPT_EVIDENCE_CHAR_BUDGET)}\n\n"
                + feature_ranking_block
                + concept_grounding_block
                + flow_reconstruction_block
                + strict_flow_block
                + general_context_block
                +
                "Rules: mention exact file/function evidence; avoid meta statements about tool traces. "
                "Stay focused on the user question and do not force unrelated conclusions. "
                "Do not infer primary behavior from low-signal files (tests, typetests, fixtures, demos, examples). "
                "Treat low-signal files as supporting evidence only; prioritize production paths for conclusions. "
                "When concept grounding is present, require primary evidence to match grounded concept signals; related-but-different anti-signals should be demoted or called out as non-primary. "
                "For flow questions, output this exact section order: 1) Entry point 2) Processing steps 3) Storage/validation 4) Additional layers 5) Conclusion. "
                "If a step is not grounded in evidence, explicitly state not found for that step instead of guessing. "
                "When answering feature/capability questions, list PRIMARY features first and keep tooling/build scripts in a separate secondary section. "
                "If Feature saliency ranking is present and has primary_features, use those items as the main features in rank order. "
                "Do not promote entries from secondary_or_tooling into primary features. "
                "For broad primary-feature questions, primary features must be repository-level capabilities, not one-off utility modules. "
                "Treat lint/build/test/runtime-harness utilities as secondary unless explicitly requested by the user. "
                + query_specific_rule
                + " "
                + f"Dynamic guidance: {dynamic_answer_instruction} "
                f"{FINAL_OUTPUT_INSTRUCTION}"
            )
        ),
    ]

    final_answer = ""
    selected_model = llm_model
    for candidate_model in build_model_candidates(llm_model):
        model_kwargs = {"top_p": top_p}
        if "gpt-oss" in candidate_model:
            final_llm = ChatGroq(
                model=candidate_model,
                temperature=temperature,
                max_tokens=max_completion_tokens,
                model_kwargs=model_kwargs,
                reasoning_effort="medium",
                groq_api_key=resolved_key,
            )
        else:
            final_llm = ChatGroq(
                model=candidate_model,
                temperature=temperature,
                max_tokens=max_completion_tokens,
                model_kwargs=model_kwargs,
                groq_api_key=resolved_key,
            )
        try:
            final_response = final_llm.invoke(final_prompt_messages)
            final_answer = _extract_message_text(final_response).strip()
            if final_answer:
                selected_model = candidate_model
                break
        except Exception as exc:
            if not _is_retryable_error(exc):
                raise
            continue

    if flow_question and not final_answer:
        final_answer = _build_evidence_only_flow_answer(
            question=question,
            concept_grounding=concept_grounding,
            flow_reconstruction=flow_reconstruction,
            evidence=evidence,
        )

    if not final_answer:
        final_answer = (
            "Direct answer: Insufficient grounded evidence to conclude confidently.\n\n"
            "Why: The current capped evidence is not enough for a stronger claim.\n\n"
            "Evidence: No conclusive file/function evidence was retained in this run.\n\n"
            "Limits/Unknowns: Additional file inspection is needed."
        )

    return {
        "question": question,
        "tool_calls_executed": tool_trace,
        "final_answer": final_answer,
        "context_stats": runtime.list_context_stats(),
        "query_scope": resolved_scope,
        "model_used": selected_model,
        "planning_model_used": planning_model,
        "plan": plan_steps,
        "max_tool_calls": MAX_TOOL_CALLS,
        "feature_ranking": ranked_features,
        "flow_reconstruction": flow_reconstruction,
    }
