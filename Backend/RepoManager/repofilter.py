import os
from typing import Iterable, Optional

from Backend.RepoManager.clone import clone_repository

IGNORE_DIRS = {
    ".git", "__pycache__", "dist", "build",
    ".next", "venv", ".venv", "env", "coverage", ".cache",
    "node_modules", "vendor",
    "tests", "test", "spec", "fixtures", "__tests__"
}


IGNORE_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Dockerfile",
    ".dockerignore",
    ".gitignore"
}

IGNORE_EXTENSIONS = {
    ".txt", ".md", ".log", ".lock", ".json",
    ".yml", ".yaml", ".toml", ".ini", ".css",
    ".svg"
}

ALWAYS_INCLUDE_BASENAMES = {
    "readme.md",
    "package.json",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
    "pubspec.yaml",
}

MAX_FILE_SIZE_BYTES = 2_000_000


def _path_parts(path_value: str) -> Iterable[str]:
    return [part for part in path_value.replace("\\", "/").split("/") if part]


def is_ignored_repo_path(path_value: str) -> bool:
    normalized = path_value.strip().replace("\\", "/")
    if not normalized:
        return True

    file_name = os.path.basename(normalized)
    file_name_lower = file_name.lower()
    _, ext = os.path.splitext(file_name)
    ext = ext.lower()

    for part in _path_parts(normalized):
        lowered = part.lower()
        if lowered.startswith("."):
            return True
        if lowered in IGNORE_DIRS:
            return True

    # Preserve key project metadata files that strongly improve repository-overview quality.
    if file_name_lower in ALWAYS_INCLUDE_BASENAMES:
        return False

    if file_name in IGNORE_FILES:
        return True
    if ext in IGNORE_EXTENSIONS:
        return True

    return False


def is_valid_repo_path(path_value: str, size_bytes: Optional[int] = None) -> bool:
    if is_ignored_repo_path(path_value):
        return False
    if size_bytes is not None and size_bytes > MAX_FILE_SIZE_BYTES:
        return False
    return True



def is_valid_file(file_path):
    if not is_valid_repo_path(file_path):
        return False

    try:
        if os.path.getsize(file_path) > MAX_FILE_SIZE_BYTES:
            return False
    except Exception:
        return False

    return True


def get_filtered_files(repo_path):
    valid_files = []

    for root, dirs, files in os.walk(repo_path):

        dirs[:] = [
            d for d in dirs
            if d not in IGNORE_DIRS and not d.startswith(".")
        ]

        for file in files:
            full_path = os.path.join(root, file)
            

            if is_valid_file(full_path):
                valid_files.append(full_path)

    return valid_files


def get_default_filtered_files(repo_path: Optional[str] = None, repo_url: Optional[str] = None):
    if repo_path:
        return get_filtered_files(repo_path)

    if repo_url:
        resolved_repo_path = clone_repository(repo_url)
        return get_filtered_files(resolved_repo_path)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "repo"))
    if os.path.isdir(repo_root):
        candidates = [
            os.path.join(repo_root, d)
            for d in os.listdir(repo_root)
            if os.path.isdir(os.path.join(repo_root, d))
        ]
        if candidates:
            latest_repo = max(candidates, key=os.path.getmtime)
            return get_filtered_files(latest_repo)

    return []


if __name__ == "__main__":
    files = get_default_filtered_files()
    print(f"Total valid files: {len(files)}")
    print(files[:])