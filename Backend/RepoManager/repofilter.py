import clone
import os

IGNORE_DIRS = {
    ".git", "__pycache__", "dist", "build",
    ".next", "venv", "env", "coverage", ".cache"
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



def is_valid_file(file_path):
    file_name = os.path.basename(file_path)
    _, ext = os.path.splitext(file_name)

    if file_name in IGNORE_FILES:
        return False
    if ext in IGNORE_EXTENSIONS:
        return False

    try:
        if os.path.getsize(file_path) > 2_000_000:
            return False
    except:
        return False

    return True


def get_filtered_files(repo_path):
    valid_files = []

    for root, dirs, files in os.walk(repo_path):

        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            full_path = os.path.join(root, file)
            

            if is_valid_file(full_path):
                valid_files.append(full_path)

    return valid_files


def get_default_filtered_files():
    return get_filtered_files(clone.repo_path)


if __name__ == "__main__":
    files = get_default_filtered_files()
    print(f"Total valid files: {len(files)}")
    print(files[:])