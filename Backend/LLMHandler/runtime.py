import importlib.util
import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import DEFAULT_EMBEDDING_MODEL


CURRENT_DIR = os.path.dirname(__file__)
FUNCTION_HANDLER_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "FunctionHandler"))
DEFAULT_TEXT_TRIM_LENGTH = 700
_FUNCTION_TOOL_MODULE: Optional[Any] = None
_FUNCTION_TOOL_MODULE_LOCK = threading.Lock()


def load_function_tool_module() -> Any:
    global _FUNCTION_TOOL_MODULE
    if _FUNCTION_TOOL_MODULE is not None:
        return _FUNCTION_TOOL_MODULE

    with _FUNCTION_TOOL_MODULE_LOCK:
        if _FUNCTION_TOOL_MODULE is not None:
            return _FUNCTION_TOOL_MODULE

    function_tool_file = os.path.join(FUNCTION_HANDLER_DIR, "functionTOtext.py")
    if FUNCTION_HANDLER_DIR not in sys.path:
        sys.path.insert(0, FUNCTION_HANDLER_DIR)

    spec = importlib.util.spec_from_file_location("functionTOtext", function_tool_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load functionTOtext from {function_tool_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _FUNCTION_TOOL_MODULE = module
    return _FUNCTION_TOOL_MODULE


def trim_result_text(item: Dict[str, Any], max_chars: int = DEFAULT_TEXT_TRIM_LENGTH) -> Dict[str, Any]:
    text = item.get("text", "")
    if isinstance(text, str) and len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"

    return {
        "score": item.get("score"),
        "name": item.get("name"),
        "language": item.get("language"),
        "file": item.get("file"),
        "start_line": item.get("start_line"),
        "end_line": item.get("end_line"),
        "text": text,
    }


@dataclass
class FunctionContextRuntime:
    function_tool: Any
    model: Any
    embedding_model_name: str
    repo_path: Optional[str]
    extraction: Dict[str, Any]
    embedded_items: List[Dict[str, Any]]

    tool_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    visited_files: List[str] = field(default_factory=list)
    _grouped_items: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    _cached_repo_summary: Dict[str, Any] = field(default_factory=dict)
    _cached_file_summaries: Dict[str, Any] = field(default_factory=dict)
    _repo_files_cache: List[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        repo_path: Optional[str] = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 32,
    ) -> "FunctionContextRuntime":
        function_tool = load_function_tool_module()
        model = function_tool.load_embedding_model(model_name=embedding_model_name)
        extraction = function_tool.extract_and_embed_functions(
            repo_path=repo_path,
            model_name=embedding_model_name,
            model=model,
            batch_size=batch_size,
        )
        embedded_items = extraction.get("items", [])

        runtime = cls(
            function_tool=function_tool,
            model=model,
            embedding_model_name=embedding_model_name,
            repo_path=repo_path,
            extraction=extraction,
            embedded_items=embedded_items,
        )
        runtime._grouped_items = runtime._build_grouped_items()
        runtime._cached_file_summaries = runtime._compute_file_summaries(top_k=100)
        runtime._cached_repo_summary = runtime._compute_repo_summary()
        return runtime

    def _cache_key(self, tool_name: str, payload: Dict[str, Any]) -> str:
        return f"{tool_name}:{json.dumps(payload, sort_keys=True)}"

    def _get_cached(self, tool_name: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = self._cache_key(tool_name, payload)
        cached = self.tool_cache.get(key)
        if not cached:
            return None

        # Return compact cached payloads for heavy tools to reduce token cost.
        if tool_name == "read_file":
            return {
                "cached": True,
                "file": cached.get("file"),
                "chars": cached.get("chars"),
                "returned_chars": cached.get("returned_chars"),
                "truncated": cached.get("truncated"),
                "already_visited": True,
                "note": "Repeated read avoided; reuse previously returned file content from earlier tool output.",
            }

        if tool_name == "get_dependencies":
            deps = cached.get("dependencies", [])
            return {
                "cached": True,
                "file": cached.get("file"),
                "dependency_count": cached.get("dependency_count", len(deps)),
                "dependencies": deps[:8],
                "note": "Repeated dependency lookup avoided; reuse earlier full dependency list if needed.",
            }

        if tool_name in {"search_code", "search_functions", "get_function_details"}:
            results = cached.get("results", [])
            return {
                "cached": True,
                "query": cached.get("query"),
                "name": cached.get("name"),
                "file": cached.get("file"),
                "result_count": len(results),
                "results": results[:2],
                "note": "Repeated lookup avoided; reuse earlier full result set if needed.",
            }

        copy_cached = dict(cached)
        copy_cached["cached"] = True
        return copy_cached

    def _set_cached(self, tool_name: str, payload: Dict[str, Any], value: Dict[str, Any]) -> None:
        key = self._cache_key(tool_name, payload)
        self.tool_cache[key] = dict(value)

    def _build_grouped_items(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in self.embedded_items:
            file_path = str(item.get("file", ""))
            if not file_path:
                continue
            grouped.setdefault(file_path, []).append(item)
        return grouped

    def _group_items_by_file(self) -> Dict[str, List[Dict[str, Any]]]:
        if self._grouped_items:
            return self._grouped_items
        self._grouped_items = self._build_grouped_items()
        return self._grouped_items

    def _all_files(self) -> List[str]:
        return sorted(self._group_items_by_file().keys())

    def _all_repo_files(self) -> List[str]:
        if self._repo_files_cache:
            return self._repo_files_cache

        root = self.repo_path
        if not root or not os.path.isdir(root):
            self._repo_files_cache = []
            return self._repo_files_cache

        files: List[str] = []
        for walk_root, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {".git", "node_modules", "dist", "build", ".next", "vendor"}]
            for name in names:
                if name.startswith("."):
                    continue
                files.append(os.path.join(walk_root, name))

        self._repo_files_cache = sorted(files)
        return self._repo_files_cache

    def _resolve_file_path(self, file: str) -> Optional[str]:
        target = file.strip()
        if not target:
            return None

        repo_root = self.repo_path
        normalized_target = target.replace("\\", "/").lstrip("/")
        if repo_root:
            direct_path = os.path.abspath(os.path.join(repo_root, normalized_target))
            if os.path.isfile(direct_path):
                return direct_path

        all_files = self._all_files()
        if target in all_files:
            return target

        suffix_matches = [
            path for path in all_files if path.replace("\\", "/").endswith(normalized_target)
        ]
        if suffix_matches:
            return suffix_matches[0]

        if repo_root:
            repo_files = self._all_repo_files()
            basename_target = os.path.basename(normalized_target).lower()
            basename_matches = [
                path for path in repo_files if os.path.basename(path).lower() == basename_target
            ]
            if basename_matches:
                if len(basename_matches) == 1:
                    return basename_matches[0]
                for candidate in basename_matches:
                    if candidate.replace("\\", "/").endswith(normalized_target):
                        return candidate
                return basename_matches[0]

        return None

    def search_functions(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        if not query.strip():
            return {"error": "query is required"}

        bounded_top_k = max(1, min(int(top_k), 12))
        cache_payload = {"query": query.strip().lower(), "top_k": bounded_top_k}
        cached = self._get_cached("search_functions", cache_payload)
        if cached:
            return cached

        scored = self.function_tool.score_functions_by_query(
            query=query,
            embedded_items=self.embedded_items,
            model=self.model,
            model_name=self.embedding_model_name,
        )
        result = {
            "query": query,
            "top_k": bounded_top_k,
            "results": [trim_result_text(item) for item in scored[:bounded_top_k]],
        }
        self._set_cached("search_functions", cache_payload, result)
        return result

    def search_code(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        bounded_top_k = max(1, min(int(top_k), 12))
        cache_payload = {"query": query.strip().lower(), "top_k": bounded_top_k}
        cached = self._get_cached("search_code", cache_payload)
        if cached:
            return cached

        result = self.search_functions(query=query, top_k=bounded_top_k)
        self._set_cached("search_code", cache_payload, result)
        return result

    def get_function_details(self, name: str, file: str = "", top_k: int = 5) -> Dict[str, Any]:
        target_name = name.strip()
        target_file = file.strip()
        if not target_name:
            return {"error": "name is required"}

        bounded_top_k = max(1, min(int(top_k), 3))
        cache_payload = {
            "name": target_name.lower(),
            "file": target_file,
            "top_k": bounded_top_k,
        }
        cached = self._get_cached("get_function_details", cache_payload)
        if cached:
            return cached

        name_lower = target_name.lower()
        candidates = []
        for item in self.embedded_items:
            item_name = str(item.get("name", ""))
            item_file = str(item.get("file", ""))
            if item_name.lower() != name_lower:
                continue
            if target_file and item_file != target_file:
                continue
            candidates.append(trim_result_text(item))

        result = {
            "name": target_name,
            "file": target_file or None,
            "results": candidates[:bounded_top_k],
        }
        self._set_cached("get_function_details", cache_payload, result)
        return result

    def read_file(self, file: str, max_chars: int = 3000) -> Dict[str, Any]:
        resolved = self._resolve_file_path(file)
        if not resolved:
            # Fallback: try to read directly from disk (for README, package.json, etc. with no extracted functions)
            if self.repo_path:
                import os
                candidates = [
                    os.path.join(self.repo_path, file),
                    os.path.join(self.repo_path, file.lower()),
                    os.path.join(self.repo_path, file.upper()),
                ]
                for candidate in candidates:
                    if os.path.isfile(candidate):
                        try:
                            with open(candidate, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read(max_chars)
                            return {
                                "file": candidate,
                                "content": content,
                                "chars": len(content),
                                "returned_chars": len(content),
                                "truncated": len(content) >= max_chars,
                                "already_visited": False,
                            }
                        except Exception:
                            pass
            return {"error": "file not found in indexed context", "file": file}

        bounded_max_chars = max(500, min(int(max_chars), 12000))
        cache_payload = {"file": resolved, "max_chars": bounded_max_chars}
        cached = self._get_cached("read_file", cache_payload)
        if cached:
            return cached

        try:
            with open(resolved, "r", encoding="utf-8", errors="ignore") as handle:
                content = handle.read()
        except Exception as exc:
            return {"error": f"failed to read file: {exc}", "file": resolved}

        truncated = content[:bounded_max_chars]
        was_truncated = len(content) > len(truncated)
        already_visited = resolved in self.visited_files
        if not already_visited:
            self.visited_files.append(resolved)

        result = {
            "file": resolved,
            "chars": len(content),
            "returned_chars": len(truncated),
            "truncated": was_truncated,
            "already_visited": already_visited,
            "content": truncated,
        }
        self._set_cached("read_file", cache_payload, result)
        return result

    def get_dependencies(self, file: str, max_items: int = 80) -> Dict[str, Any]:
        bounded_max = max(1, min(int(max_items), 200))
        cache_payload = {"file": file.strip(), "max_items": bounded_max}
        cached = self._get_cached("get_dependencies", cache_payload)
        if cached:
            return cached

        read_result = self.read_file(file=file, max_chars=12000)
        if "error" in read_result:
            return read_result

        file_path = str(read_result.get("file", ""))
        content = str(read_result.get("content", ""))
        ext = os.path.splitext(file_path)[1].lower()
        deps: List[str] = []

        if ext == ".py":
            deps += re.findall(r"^\s*import\s+([A-Za-z0-9_\.]+)", content, flags=re.MULTILINE)
            deps += re.findall(r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import\s+", content, flags=re.MULTILINE)
        elif ext in {".ts", ".tsx", ".js", ".jsx"}:
            deps += re.findall(r"from\s+[\"']([^\"']+)[\"']", content)
            deps += re.findall(r"require\(\s*[\"']([^\"']+)[\"']\s*\)", content)
            deps += re.findall(r"import\(\s*[\"']([^\"']+)[\"']\s*\)", content)
        elif ext == ".go":
            deps += re.findall(r"^\s*import\s+[\"']([^\"']+)[\"']", content, flags=re.MULTILINE)
            block = re.search(r"import\s*\((.*?)\)", content, flags=re.DOTALL)
            if block:
                deps += re.findall(r"[\"']([^\"']+)[\"']", block.group(1))
        elif ext in {".java", ".kt", ".kts"}:
            deps += re.findall(r"^\s*import\s+([A-Za-z0-9_\.\*]+)", content, flags=re.MULTILINE)
        elif ext == ".rs":
            deps += re.findall(r"^\s*use\s+([^;]+);", content, flags=re.MULTILINE)
        elif ext == ".php":
            deps += re.findall(r"^\s*use\s+([^;]+);", content, flags=re.MULTILINE)
            deps += re.findall(r"require(?:_once)?\s*\(?\s*[\"']([^\"']+)[\"']", content)

        clean: List[str] = []
        seen = set()
        for dep in deps:
            dep_clean = dep.strip()
            if not dep_clean or dep_clean in seen:
                continue
            seen.add(dep_clean)
            clean.append(dep_clean)

        result = {
            "file": file_path,
            "dependency_count": len(clean),
            "dependencies": clean[:bounded_max],
        }
        self._set_cached("get_dependencies", cache_payload, result)
        return result

    def _compute_file_summaries(self, top_k: int = 20) -> Dict[str, Any]:
        grouped = self._group_items_by_file()
        summaries: List[Dict[str, Any]] = []
        for file_path, items in grouped.items():
            language = str(items[0].get("language", "unknown")) if items else "unknown"
            function_names = [str(item.get("name", "")) for item in items if item.get("name")]
            summaries.append(
                {
                    "file": file_path,
                    "language": language,
                    "function_count": len(items),
                    "functions": function_names[:8],
                }
            )

        summaries.sort(key=lambda x: x["function_count"], reverse=True)
        bounded_top_k = max(1, min(int(top_k), 100))
        return {
            "total_files": len(summaries),
            "summaries": summaries[:bounded_top_k],
        }

    def get_file_summaries(self, top_k: int = 20) -> Dict[str, Any]:
        cached_summaries = self._cached_file_summaries.get("summaries", [])
        if not cached_summaries:
            self._cached_file_summaries = self._compute_file_summaries(top_k=100)
            cached_summaries = self._cached_file_summaries.get("summaries", [])

        bounded_top_k = max(1, min(int(top_k), 100))
        return {
            "total_files": self._cached_file_summaries.get("total_files", len(cached_summaries)),
            "summaries": cached_summaries[:bounded_top_k],
        }

    def _compute_repo_summary(self) -> Dict[str, Any]:
        grouped = self._group_items_by_file()
        language_counts: Dict[str, int] = {}
        for items in grouped.values():
            if not items:
                continue
            lang = str(items[0].get("language", "unknown"))
            language_counts[lang] = language_counts.get(lang, 0) + 1

        top_files = self.get_file_summaries(top_k=5).get("summaries", [])
        key_modules = []
        keywords = ("auth", "token", "session", "client", "login", "user")
        for summary in self.get_file_summaries(top_k=30).get("summaries", []):
            file_path = str(summary.get("file", "")).lower()
            fnames = [str(name).lower() for name in summary.get("functions", [])]
            haystack = " ".join([file_path] + fnames)
            if any(word in haystack for word in keywords):
                key_modules.append(summary)

        return {
            "model_name": self.extraction.get("model_name"),
            "total_filtered_files": self.extraction.get("filtered_file_count"),
            "total_scanned_files": self.extraction.get("scanned_file_count"),
            "total_functions": self.extraction.get("function_count"),
            "language_file_counts": language_counts,
            "top_function_files": top_files,
            "key_modules": key_modules[:10],
        }

    def get_repo_summary(self) -> Dict[str, Any]:
        if not self._cached_repo_summary:
            self._cached_repo_summary = self._compute_repo_summary()
        return dict(self._cached_repo_summary)

    def list_context_stats(self) -> Dict[str, Any]:
        return {
            "model_name": self.extraction.get("model_name"),
            "filtered_file_count": self.extraction.get("filtered_file_count"),
            "scanned_file_count": self.extraction.get("scanned_file_count"),
            "function_count": self.extraction.get("function_count"),
            "embedding_dimension": self.extraction.get("embedding_dimension"),
            "visited_files": len(self.visited_files),
            "cache_entries": len(self.tool_cache),
        }

    def to_json(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload)
