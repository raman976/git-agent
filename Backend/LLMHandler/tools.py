from typing import List

from langchain_core.tools import BaseTool, tool

from .runtime import FunctionContextRuntime


def build_langchain_tools(
    runtime: FunctionContextRuntime,
    include_stats_tool: bool = False,
    include_dependency_tool: bool = False,
) -> List[BaseTool]:
    @tool
    def search_code(query: str, top_k: int = 5) -> str:
        """Search relevant code snippets/functions for a query."""
        return runtime.to_json(runtime.search_code(query=query, top_k=top_k))

    @tool
    def read_file(file: str, max_chars: int = 3000) -> str:
        """Read file content from indexed repository context."""
        return runtime.to_json(runtime.read_file(file=file, max_chars=max_chars))

    @tool
    def get_dependencies(file: str, max_items: int = 80) -> str:
        """Extract file imports/dependencies for flow tracing."""
        return runtime.to_json(runtime.get_dependencies(file=file, max_items=max_items))

    @tool
    def get_repo_summary() -> str:
        """Get repository-level summary (files, functions, languages)."""
        return runtime.to_json(runtime.get_repo_summary())

    @tool
    def get_file_summaries(top_k: int = 20) -> str:
        """Get per-file summaries with languages and function names."""
        return runtime.to_json(runtime.get_file_summaries(top_k=top_k))

    @tool
    def search_functions(query: str, top_k: int = 3) -> str:
        """Semantic search over extracted functions and return top matches."""
        return runtime.to_json(runtime.search_functions(query=query, top_k=top_k))

    @tool
    def get_function_details(name: str, file: str = "", top_k: int = 5) -> str:
        """Fetch function entries by function name and optional file path."""
        return runtime.to_json(runtime.get_function_details(name=name, file=file, top_k=top_k))

    @tool
    def list_context_stats() -> str:
        """Return context/model metadata for loaded function index."""
        return runtime.to_json(runtime.list_context_stats())

    base_tools = [
        search_code,
        read_file,
        get_repo_summary,
        get_file_summaries,
        search_functions,
        get_function_details,
    ]

    if include_dependency_tool:
        base_tools.append(get_dependencies)

    if include_stats_tool:
        return base_tools + [list_context_stats]

    return base_tools
