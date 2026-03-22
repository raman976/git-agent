import os
import re

from git import Repo


GITHUB_REPO_URL_PATTERN = re.compile(
    r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?",
    re.IGNORECASE,
)

GITHUB_REPO_PARTS_PATTERN = re.compile(
    r"https?://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)(?:\.git)?(?:/|$)",
    re.IGNORECASE,
)


def _normalize_repo_url(raw_value: str) -> str:
    """Extract a valid GitHub repository URL from user input text."""
    if not raw_value or not raw_value.strip():
        raise ValueError("Repository URL is required")

    candidates = GITHUB_REPO_URL_PATTERN.findall(raw_value.strip())
    if not candidates:
        raise ValueError("Invalid GitHub repository URL")

    # Use the last valid URL when users paste concatenated links.
    return candidates[-1]


def clone_repository(repo_url: str, base_repo_dir: str | None = None) -> str:
    """Clone a repository if missing and return local absolute path."""
    repo_url = _normalize_repo_url(repo_url)

    parts_match = GITHUB_REPO_PARTS_PATTERN.search(repo_url)
    if not parts_match:
        raise ValueError("Invalid GitHub repository URL")

    owner = parts_match.group("owner")
    repo = parts_match.group("repo")
    repo_folder = f"{owner}-{repo}"

    if base_repo_dir is None:
        base_repo_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "repo")
        )

    os.makedirs(base_repo_dir, exist_ok=True)
    repo_path = os.path.abspath(os.path.join(base_repo_dir, repo_folder))

    if os.path.isdir(repo_path) and os.listdir(repo_path):
        Repo(repo_path)
    else:
        Repo.clone_from(repo_url, repo_path, depth=1)

    return repo_path
