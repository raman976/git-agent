import base64
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from Backend.RepoManager import repofilter
from Backend.LLMHandler.config import (
    DEFAULT_GITHUB_TOKEN_VAR,
    GITHUB_API_TIMEOUT_SECONDS,
    GITHUB_MAX_FILE_BYTES,
    GITHUB_MAX_SELECTED_FILES,
    GITHUB_MAX_TOTAL_BYTES,
)


GITHUB_REPO_URL_PATTERN = re.compile(
    r"https?://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)(?:\.git)?(?:/|$)",
    re.IGNORECASE,
)

API_BASE = "https://api.github.com"
LANGUAGE_HINTS = {
    "python": {".py"},
    "javascript": {".js", ".jsx"},
    "typescript": {".ts", ".tsx"},
    "java": {".java"},
    "go": {".go"},
    "rust": {".rs"},
    "csharp": {".cs"},
    "kotlin": {".kt", ".kts"},
}

KEY_DIRECTORIES = {
    "src",
    "app",
    "backend",
    "frontend",
    "api",
    "services",
    "service",
    "models",
    "model",
    "controllers",
    "controller",
    "routes",
    "query",
    "db",
    "database",
    "core",
    "lib",
}

LOW_SIGNAL_PATH_MARKERS = {
    "__tests__",
    "tests",
    "test",
    "__typetests__",
    "typetests",
    "example",
    "examples",
    "demo",
    "fixtures",
    "fixture",
    "mock",
    "mocks",
    "benchmarks",
    "benchmark",
}

TOOLING_PATH_MARKERS = {
    "scripts",
    "release",
    "debugger",
    "tools",
    "ci",
    "github",
    "workflows",
    "eslint",
    "spec",
    "specs",
    "tester",
    "private",
    "flow-typed",
}

PRIMARY_FILE_BASENAMES = {
    "readme.md",
    "package.json",
    "pyproject.toml",
    "go.mod",
    "cargo.toml",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
}

PRIMARY_SOURCE_BASENAMES = {
    "index.ts",
    "index.tsx",
    "index.js",
    "index.jsx",
    "main.ts",
    "main.js",
    "app.ts",
    "app.js",
}


@dataclass
class TreeEntry:
    path: str
    size: int


_TREE_CACHE: Dict[Tuple[str, str, str], List[TreeEntry]] = {}
_TREE_CACHE_LOCK = threading.Lock()
_CONTENT_CACHE: Dict[Tuple[str, str, str, str], str] = {}
_CONTENT_CACHE_LOCK = threading.Lock()


def is_github_url(repo_url: str) -> bool:
    if not repo_url:
        return False
    return GITHUB_REPO_URL_PATTERN.search(repo_url.strip()) is not None


def parse_github_repo(repo_url: str) -> Tuple[str, str]:
    match = GITHUB_REPO_URL_PATTERN.search(repo_url.strip())
    if not match:
        raise ValueError("Only canonical GitHub repository URLs are supported")

    owner = match.group("owner")
    repo = match.group("repo")
    return owner, repo


def _build_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Gh-Agent",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_json(url: str, headers: Dict[str, str], timeout: int) -> Dict[str, object]:
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except HTTPError as exc:
        reset_ts = exc.headers.get("x-ratelimit-reset") if exc.headers else None
        remaining = exc.headers.get("x-ratelimit-remaining") if exc.headers else None
        if exc.code == 403 and remaining == "0":
            reset_hint = ""
            if reset_ts and reset_ts.isdigit():
                wait_seconds = max(0, int(reset_ts) - int(time.time()))
                reset_hint = f" Retry after ~{wait_seconds}s."
            raise RuntimeError(f"GitHub API rate limit reached.{reset_hint}") from exc
        raise RuntimeError(f"GitHub API error {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"Unable to reach GitHub API: {exc.reason}") from exc


def _load_github_token(token: Optional[str] = None, token_var: str = DEFAULT_GITHUB_TOKEN_VAR) -> Optional[str]:
    if token:
        return token
    load_dotenv()
    return os.getenv(token_var)


def _repo_meta(owner: str, repo: str, headers: Dict[str, str], timeout: int) -> Dict[str, object]:
    return _request_json(f"{API_BASE}/repos/{owner}/{repo}", headers=headers, timeout=timeout)


def _branch_head_commit_sha(
    owner: str,
    repo: str,
    branch: str,
    headers: Dict[str, str],
    timeout: int,
) -> str:
    branch_payload = _request_json(
        f"{API_BASE}/repos/{owner}/{repo}/branches/{quote(branch)}",
        headers=headers,
        timeout=timeout,
    )
    commit = branch_payload.get("commit", {}) if isinstance(branch_payload, dict) else {}
    commit_sha = str(commit.get("sha", ""))
    if not commit_sha:
        raise RuntimeError("Unable to resolve branch head commit SHA from GitHub API")
    return commit_sha


def _commit_tree_sha(
    owner: str,
    repo: str,
    commit_sha: str,
    headers: Dict[str, str],
    timeout: int,
) -> str:
    commit_payload = _request_json(
        f"{API_BASE}/repos/{owner}/{repo}/git/commits/{quote(commit_sha)}",
        headers=headers,
        timeout=timeout,
    )
    tree = commit_payload.get("tree", {}) if isinstance(commit_payload, dict) else {}
    tree_sha = str(tree.get("sha", ""))
    if not tree_sha:
        raise RuntimeError("Unable to resolve commit tree SHA from GitHub API")
    return tree_sha


def _recursive_tree(
    owner: str,
    repo: str,
    branch: str,
    headers: Dict[str, str],
    timeout: int,
) -> Tuple[str, List[TreeEntry]]:
    commit_sha = _branch_head_commit_sha(
        owner=owner,
        repo=repo,
        branch=branch,
        headers=headers,
        timeout=timeout,
    )
    tree_sha = _commit_tree_sha(
        owner=owner,
        repo=repo,
        commit_sha=commit_sha,
        headers=headers,
        timeout=timeout,
    )

    tree_payload = _request_json(
        f"{API_BASE}/repos/{owner}/{repo}/git/trees/{quote(tree_sha)}?recursive=1",
        headers=headers,
        timeout=timeout,
    )

    if tree_payload.get("truncated"):
        raise RuntimeError("Repository tree is too large/truncated by GitHub API for recursive listing")

    entries = []
    for item in tree_payload.get("tree", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "blob":
            continue
        path = str(item.get("path", ""))
        size = int(item.get("size") or 0)
        if not repofilter.is_valid_repo_path(path, size):
            continue
        entries.append(TreeEntry(path=path, size=size))

    return commit_sha, entries


def _tokenize(text: str) -> Set[str]:
    return {token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{1,}", text.lower())}


def _guess_extensions_from_query(query: str) -> Set[str]:
    lowered = query.lower()
    extensions: Set[str] = set()
    for lang, ext_set in LANGUAGE_HINTS.items():
        if lang in lowered:
            extensions.update(ext_set)
    if "react" in lowered:
        extensions.update({".tsx", ".jsx"})
    if "api" in lowered or "endpoint" in lowered:
        extensions.update({".py", ".ts", ".js", ".go", ".java"})
    return extensions


def _is_overview_query(query: str) -> bool:
    lowered = query.lower()
    markers = [
        "main features",
        "primary features",
        "key features",
        "what does this repo",
        "what does the repo",
        "applications",
        "overview",
        "architecture",
        "high level",
    ]
    return any(marker in lowered for marker in markers)


def _entry_bucket(path: str) -> str:
    parts = [part for part in path.lower().split("/") if part]
    if not parts:
        return "root"
    anchors = ["packages", "src", "app", "backend", "frontend", "reactandroid", "android", "ios"]
    for anchor in anchors:
        if anchor in parts:
            idx = parts.index(anchor)
            next_idx = min(idx + 1, len(parts) - 1)
            if idx == next_idx:
                return parts[idx]
            return f"{parts[idx]}/{parts[next_idx]}"
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]}/{parts[1]}"


def _diversify_entries(
    entries: Sequence[TreeEntry],
    max_items: int,
    max_per_bucket: int,
) -> List[TreeEntry]:
    selected: List[TreeEntry] = []
    bucket_counts: Dict[str, int] = {}
    seen_paths: Set[str] = set()

    for entry in entries:
        bucket = _entry_bucket(entry.path)
        count = bucket_counts.get(bucket, 0)
        if count >= max_per_bucket:
            continue
        if entry.path in seen_paths:
            continue
        selected.append(entry)
        seen_paths.add(entry.path)
        bucket_counts[bucket] = count + 1
        if len(selected) >= max_items:
            return selected

    if len(selected) < max_items:
        for entry in entries:
            if entry.path in seen_paths:
                continue
            selected.append(entry)
            seen_paths.add(entry.path)
            if len(selected) >= max_items:
                break

    return selected


def _rank_entries(entries: Sequence[TreeEntry], query: str) -> List[TreeEntry]:
    query_tokens = _tokenize(query)
    preferred_exts = _guess_extensions_from_query(query)
    overview_mode = _is_overview_query(query)

    def score(entry: TreeEntry) -> Tuple[int, int]:
        path_lower = entry.path.lower()
        base_name = os.path.basename(path_lower)
        name_tokens = _tokenize(base_name)
        path_tokens = _tokenize(path_lower)
        parts = set(path_lower.split("/"))

        score_value = 0
        if query_tokens:
            score_value += 12 * len(name_tokens.intersection(query_tokens))
            score_value += 4 * len(path_tokens.intersection(query_tokens))

        ext = os.path.splitext(path_lower)[1]
        if preferred_exts and ext in preferred_exts:
            score_value += 8

        score_value += 3 * len(parts.intersection(KEY_DIRECTORIES))

        if base_name in PRIMARY_FILE_BASENAMES:
            score_value += 14
        if base_name in PRIMARY_SOURCE_BASENAMES:
            score_value += 10

        if parts.intersection(LOW_SIGNAL_PATH_MARKERS):
            score_value -= 24

        if parts.intersection(TOOLING_PATH_MARKERS):
            score_value -= 9 if overview_mode else 4

        if overview_mode and parts.intersection({"src", "app", "backend", "frontend", "reactandroid", "android", "ios", "packages"}):
            score_value += 10

        # Slight preference for moderately sized files over very tiny or huge ones.
        if 300 <= entry.size <= 60_000:
            score_value += 2

        return score_value, -entry.size

    return sorted(entries, key=score, reverse=True)


def _decode_content_payload(payload: Dict[str, object]) -> str:
    if payload.get("type") != "file":
        return ""
    if str(payload.get("encoding", "")).lower() != "base64":
        return ""

    encoded = str(payload.get("content", "")).replace("\n", "")
    if not encoded:
        return ""
    raw = base64.b64decode(encoded)
    return raw.decode("utf-8", errors="ignore")


def _fetch_file_content(
    owner: str,
    repo: str,
    commit_sha: str,
    path: str,
    headers: Dict[str, str],
    timeout: int,
) -> str:
    key = (owner, repo, commit_sha, path)
    with _CONTENT_CACHE_LOCK:
        cached = _CONTENT_CACHE.get(key)
    if cached is not None:
        return cached

    encoded_path = quote(path)
    payload = _request_json(
        f"{API_BASE}/repos/{owner}/{repo}/contents/{encoded_path}?ref={quote(commit_sha)}",
        headers=headers,
        timeout=timeout,
    )
    content = _decode_content_payload(payload)

    with _CONTENT_CACHE_LOCK:
        _CONTENT_CACHE[key] = content
    return content


def _repo_cache_dir(base_repo_dir: Optional[str], owner: str, repo: str, commit_sha: str) -> str:
    if base_repo_dir is None:
        base_repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "repo"))
    os.makedirs(base_repo_dir, exist_ok=True)

    safe_name = f"{owner}-{repo}-{commit_sha[:12]}-api"
    return os.path.abspath(os.path.join(base_repo_dir, safe_name))


def prepare_query_snapshot(
    repo_url: str,
    query: str,
    existing_repo_path: Optional[str] = None,
    existing_selected_paths: Optional[Sequence[str]] = None,
    base_repo_dir: Optional[str] = None,
    github_token: Optional[str] = None,
    max_selected_files: int = GITHUB_MAX_SELECTED_FILES,
    max_total_bytes: int = GITHUB_MAX_TOTAL_BYTES,
    max_file_bytes: int = GITHUB_MAX_FILE_BYTES,
    timeout_seconds: int = GITHUB_API_TIMEOUT_SECONDS,
) -> Dict[str, object]:
    from Backend.RepoManager.clone import clone_repository
    owner, repo = parse_github_repo(repo_url)
    token = _load_github_token(github_token)
    headers = _build_headers(token)

    repo_payload = _repo_meta(owner, repo, headers=headers, timeout=timeout_seconds)
    branch = str(repo_payload.get("default_branch", "main"))

    commit_sha, entries = _recursive_tree(
        owner=owner,
        repo=repo,
        branch=branch,
        headers=headers,
        timeout=timeout_seconds,
    )

    cache_key = (owner, repo, commit_sha)
    with _TREE_CACHE_LOCK:
        cached_entries = _TREE_CACHE.get(cache_key)
        if cached_entries is None:
            _TREE_CACHE[cache_key] = list(entries)
            cached_entries = _TREE_CACHE[cache_key]

    ranked = _rank_entries(cached_entries, query)
    overview_mode = _is_overview_query(query)

    ranked_candidates = ranked
    if overview_mode:
        ranked_candidates = _diversify_entries(
            ranked,
            max_items=max_selected_files,
            max_per_bucket=3,
        )

    selected_paths: List[str] = []
    total_bytes = 0

    for entry in ranked_candidates:
        if entry.size > max_file_bytes:
            continue
        if total_bytes + entry.size > max_total_bytes:
            continue
        selected_paths.append(entry.path)
        total_bytes += max(entry.size, 0)
        if len(selected_paths) >= max_selected_files:
            break

    existing = set(existing_selected_paths or [])
    selected_union = sorted(existing.union(selected_paths))

    snapshot_path = existing_repo_path or _repo_cache_dir(
        base_repo_dir=base_repo_dir,
        owner=owner,
        repo=repo,
        commit_sha=commit_sha,
    )
    os.makedirs(snapshot_path, exist_ok=True)

    newly_downloaded = 0
    downloaded_bytes = 0
    for path in selected_union:
        local_path = os.path.join(snapshot_path, path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if path in existing and os.path.isfile(local_path):
            continue

        content = _fetch_file_content(
            owner=owner,
            repo=repo,
            commit_sha=commit_sha,
            path=path,
            headers=headers,
            timeout=timeout_seconds,
        )
        if not content.strip():
            continue

        with open(local_path, "w", encoding="utf-8", errors="ignore") as handle:
            handle.write(content)

        newly_downloaded += 1
        downloaded_bytes += len(content.encode("utf-8", errors="ignore"))

    selection_hash = hashlib.sha256("\n".join(selected_union).encode("utf-8")).hexdigest()

    # Also clone the full repository for complete indexing
    full_repo_path = clone_repository(
        repo_url=repo_url,
        base_repo_dir=base_repo_dir,
    )

    return {
        "repo_path": snapshot_path,
        "full_repo_path": full_repo_path,
        "owner": owner,
        "repo": repo,
        "branch": branch,
        "commit_sha": commit_sha,
        "tree_file_count": len(cached_entries),
        "selected_file_count": len(selected_union),
        "newly_downloaded": newly_downloaded,
        "downloaded_bytes": downloaded_bytes,
        "selected_paths": selected_union,
        "selection_hash": selection_hash,
        "mode": "github_api",
    }
