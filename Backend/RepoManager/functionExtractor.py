import ast
import os
import re
from typing import Dict, List, Optional, Tuple

import repofilter


LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".sh": "shell",
}


BRACE_LANGUAGES = {
    "javascript",
    "typescript",
    "java",
    "go",
    "rust",
    "php",
    "csharp",
    "cpp",
    "c",
    "swift",
    "kotlin",
}


FUNCTION_SIGNATURE_PATTERNS = {
    "javascript": [
        re.compile(r"^\s*(?:export\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^\)]*\)\s*\{", re.MULTILINE),
        re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\([^\)]*\)\s*=>\s*\{", re.MULTILINE),
    ],
    "typescript": [
        re.compile(r"^\s*(?:export\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^\)]*\)\s*(?::\s*[^\{]+)?\{", re.MULTILINE),
        re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\([^\)]*\)\s*(?::\s*[^=]+)?=>\s*\{", re.MULTILINE),
    ],
    "java": [
        re.compile(r"^\s*(?:public|private|protected)?\s*(?:static\s+)?[A-Za-z0-9_<>,\[\]\.?]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;\)]*\)\s*\{", re.MULTILINE),
    ],
    "go": [
        re.compile(r"^\s*func\s+(?:\([^\)]*\)\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^\)]*\)\s*(?:\([^\)]*\)|[A-Za-z0-9_\*\[\]]+)?\s*\{", re.MULTILINE),
    ],
    "rust": [
        re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^\)]*\)\s*(?:->\s*[^\{]+)?\{", re.MULTILINE),
    ],
    "php": [
        re.compile(r"^\s*(?:public|private|protected)?\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^\)]*\)\s*\{", re.MULTILINE),
    ],
    "csharp": [
        re.compile(r"^\s*(?:public|private|protected|internal)?\s*(?:static\s+)?[A-Za-z0-9_<>,\[\]\.?]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;\)]*\)\s*\{", re.MULTILINE),
    ],
    "cpp": [
        re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_:<>\*&\s]*\s+([A-Za-z_][A-Za-z0-9_:]*)\s*\([^;\)]*\)\s*(?:const\s*)?\{", re.MULTILINE),
    ],
    "c": [
        re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_\s\*]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;\)]*\)\s*\{", re.MULTILINE),
    ],
    "swift": [
        re.compile(r"^\s*func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^\)]*\)\s*(?:->\s*[^\{]+)?\{", re.MULTILINE),
    ],
    "kotlin": [
        re.compile(r"^\s*fun\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^\)]*\)\s*(?::\s*[^\{=]+)?\s*(?:\{|=)", re.MULTILINE),
    ],
    "ruby": [
        re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_\.!?=]*)", re.MULTILINE),
    ],
    "shell": [
        re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{", re.MULTILINE),
    ],
}


def detect_language(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    return LANGUAGE_BY_EXTENSION.get(ext, "unknown")


def read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_python_functions(file_path: str) -> List[Dict[str, object]]:
    code = read_text_file(file_path)
    lines = code.split("\n")
    functions: List[Dict[str, object]] = []

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            start = node.lineno
            end = getattr(node, "end_lineno", node.lineno)
            function_code = "\n".join(lines[start - 1:end])

            functions.append(
                {
                    "name": node.name,
                    "code": function_code,
                    "file": file_path,
                    "language": "python",
                    "start_line": start,
                    "end_line": end,
                }
            )
    except Exception as e:
        print(f"Error in {file_path}: {e}")

    return functions


def find_matching_brace_end(code: str, brace_start_index: int) -> Optional[int]:
    depth = 0
    in_single = False
    in_double = False
    in_backtick = False
    escape = False

    for i in range(brace_start_index, len(code)):
        ch = code[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if not in_double and not in_backtick and ch == "'":
            in_single = not in_single
            continue

        if not in_single and not in_backtick and ch == '"':
            in_double = not in_double
            continue

        if not in_single and not in_double and ch == "`":
            in_backtick = not in_backtick
            continue

        if in_single or in_double or in_backtick:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i

    return None


def extract_brace_language_functions(file_path: str, language: str) -> List[Dict[str, object]]:
    code = read_text_file(file_path)
    patterns = FUNCTION_SIGNATURE_PATTERNS.get(language, [])
    lines = code.split("\n")
    functions: List[Dict[str, object]] = []
    seen: set[Tuple[str, int]] = set()

    for pattern in patterns:
        for match in pattern.finditer(code):
            name = match.group(1)
            start_index = match.start()
            start_line = code.count("\n", 0, start_index) + 1

            if (name, start_line) in seen:
                continue
            seen.add((name, start_line))

            brace_index = code.find("{", match.start(), match.end() + 2)
            if brace_index == -1:
                continue

            end_index = find_matching_brace_end(code, brace_index)
            if end_index is None:
                preview_end = min(start_line + 25, len(lines))
                function_code = "\n".join(lines[start_line - 1:preview_end])
                end_line = preview_end
            else:
                end_line = code.count("\n", 0, end_index) + 1
                function_code = "\n".join(lines[start_line - 1:end_line])

            functions.append(
                {
                    "name": name,
                    "code": function_code,
                    "file": file_path,
                    "language": language,
                    "start_line": start_line,
                    "end_line": end_line,
                }
            )

    return functions


def extract_line_pattern_functions(file_path: str, language: str) -> List[Dict[str, object]]:
    code = read_text_file(file_path)
    patterns = FUNCTION_SIGNATURE_PATTERNS.get(language, [])
    lines = code.split("\n")
    functions: List[Dict[str, object]] = []
    seen: set[Tuple[str, int]] = set()

    for pattern in patterns:
        for match in pattern.finditer(code):
            name = match.group(1)
            start_line = code.count("\n", 0, match.start()) + 1

            if (name, start_line) in seen:
                continue
            seen.add((name, start_line))

            preview_end = min(start_line + 20, len(lines))
            function_code = "\n".join(lines[start_line - 1:preview_end])

            functions.append(
                {
                    "name": name,
                    "code": function_code,
                    "file": file_path,
                    "language": language,
                    "start_line": start_line,
                    "end_line": preview_end,
                }
            )

    return functions


def extract_functions(file_path: str) -> List[Dict[str, object]]:
    language = detect_language(file_path)

    if language == "python":
        return extract_python_functions(file_path)

    if language in BRACE_LANGUAGES:
        return extract_brace_language_functions(file_path, language)

    return extract_line_pattern_functions(file_path, language)


def extract_functions_from_files(file_paths: List[str]) -> Tuple[List[Dict[str, object]], int]:
    all_functions: List[Dict[str, object]] = []
    scanned_files = 0

    for file_path in file_paths:
        scanned_files += 1
        all_functions.extend(extract_functions(file_path))

    return all_functions, scanned_files


def extract_functions_from_filtered_files(repo_path: Optional[str] = None) -> Tuple[List[Dict[str, object]], int, int]:
    if repo_path:
        filtered_files = repofilter.get_filtered_files(repo_path)
    else:
        filtered_files = repofilter.get_default_filtered_files()

    all_functions, scanned_file_count = extract_functions_from_files(filtered_files)
    return all_functions, scanned_file_count, len(filtered_files)


if __name__ == "__main__":
    functions, scanned_file_count, total_filtered_files = extract_functions_from_filtered_files()

    print(f"Filtered files: {total_filtered_files}")
    print(f"Files scanned for extraction: {scanned_file_count}")
    print(f"Functions extracted: {len(functions)}")

    if functions:
        print("Sample:", functions[0]["name"], "from", functions[0]["file"])
