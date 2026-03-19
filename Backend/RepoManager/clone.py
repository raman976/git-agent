import os

from git import Repo

test_repo_url = "https://github.com/raman976/AwesomeRag.git"
path = test_repo_url.split("/")[-1][:-4]
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "repo", path))

if os.path.isdir(repo_path) and os.listdir(repo_path):
    repo = Repo(repo_path)
else:
    repo = Repo.clone_from(test_repo_url, repo_path, depth=1)
