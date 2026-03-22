import threading
import time
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from Backend.RepoManager.clone import clone_repository
from Backend.RepoManager.github_source import is_github_url, prepare_query_snapshot
from Backend.LLMHandler.runtime import FunctionContextRuntime


@dataclass
class SessionState:
    session_id: str
    repo_url: Optional[str] = None
    repo_path: Optional[str] = None
    repo_mode: str = "clone"
    repo_commit_sha: Optional[str] = None
    selected_paths: List[str] = field(default_factory=list)
    runtime: Optional[FunctionContextRuntime] = None
    history: List[Dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)


class SessionManager:
    def __init__(
        self,
        ttl_seconds: int = 1800,
        max_history_turns: int = 6,
        cleanup_interval_seconds: int = 60,
        orphan_repo_ttl_seconds: Optional[int] = None,
    ):
        self.ttl_seconds = ttl_seconds
        self.max_history_turns = max_history_turns
        self.cleanup_interval_seconds = max(5, cleanup_interval_seconds)
        self.orphan_repo_ttl_seconds = (
            orphan_repo_ttl_seconds
            if orphan_repo_ttl_seconds is not None
            else self.ttl_seconds
        )
        self.base_repo_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "repo")
        )
        self.sessions_root_dir = os.path.join(self.base_repo_dir, "sessions")
        os.makedirs(self.sessions_root_dir, exist_ok=True)
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="session-cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()

    def _safe_session_segment(self, session_id: str) -> str:
        return "".join(ch for ch in session_id if ch.isalnum() or ch in {"-", "_"}) or "session"

    def _session_dir(self, session_id: str) -> str:
        return os.path.abspath(os.path.join(self.sessions_root_dir, self._safe_session_segment(session_id)))

    def _session_repo_base_dir(self, session_id: str) -> str:
        session_dir = self._session_dir(session_id)
        repo_base = os.path.join(session_dir, "repo")
        os.makedirs(repo_base, exist_ok=True)
        return repo_base

    def _touch_session_dir(self, session_id: str) -> None:
        session_dir = self._session_dir(session_id)
        os.makedirs(session_dir, exist_ok=True)
        now = time.time()
        try:
            os.utime(session_dir, (now, now))
        except OSError:
            pass

    def _delete_session_dir(self, session_id: str) -> bool:
        session_dir = self._session_dir(session_id)
        if not os.path.isdir(session_dir):
            return False
        shutil.rmtree(session_dir, ignore_errors=True)
        return not os.path.exists(session_dir)

    def _cleanup_expired_locked(self) -> None:
        now = time.time()
        expired = [
            session_id
            for session_id, state in self._sessions.items()
            if (now - state.last_access) > self.ttl_seconds
        ]
        for session_id in expired:
            self._expire_session_locked(session_id, delete_repo=True)

    def _cleanup_loop(self) -> None:
        while True:
            time.sleep(self.cleanup_interval_seconds)
            with self._lock:
                self._cleanup_expired_locked()
                self._cleanup_orphan_session_dirs_locked()
                self._cleanup_orphan_legacy_repo_dirs_locked()

    def _cleanup_orphan_session_dirs_locked(self) -> None:
        if self.orphan_repo_ttl_seconds <= 0:
            return
        if not os.path.isdir(self.sessions_root_dir):
            return

        now = time.time()
        active_session_dirs = {
            self._session_dir(state.session_id)
            for state in self._sessions.values()
        }

        for name in os.listdir(self.sessions_root_dir):
            session_dir = os.path.abspath(os.path.join(self.sessions_root_dir, name))
            if not os.path.isdir(session_dir):
                continue
            if session_dir in active_session_dirs:
                continue

            try:
                age_seconds = now - os.path.getmtime(session_dir)
            except OSError:
                continue

            if age_seconds > self.orphan_repo_ttl_seconds:
                shutil.rmtree(session_dir, ignore_errors=True)

    def _cleanup_orphan_legacy_repo_dirs_locked(self) -> None:
        """Clean legacy non-session repo directories left from older layout."""
        if self.orphan_repo_ttl_seconds <= 0:
            return
        if not os.path.isdir(self.base_repo_dir):
            return

        now = time.time()
        active_repo_paths = {
            os.path.abspath(state.repo_path)
            for state in self._sessions.values()
            if state.repo_path
        }

        for name in os.listdir(self.base_repo_dir):
            repo_path = os.path.abspath(os.path.join(self.base_repo_dir, name))
            if name == "sessions":
                continue
            if not os.path.isdir(repo_path):
                continue
            if repo_path in active_repo_paths:
                continue

            try:
                age_seconds = now - os.path.getmtime(repo_path)
            except OSError:
                continue

            if age_seconds > self.orphan_repo_ttl_seconds:
                self._delete_repo_dir(repo_path)

    def _is_repo_path_in_use_locked(self, repo_path: str, excluding_session_id: Optional[str] = None) -> bool:
        for sid, state in self._sessions.items():
            if excluding_session_id and sid == excluding_session_id:
                continue
            if state.repo_path and os.path.abspath(state.repo_path) == repo_path:
                return True
        return False

    def _delete_repo_dir(self, repo_path: str) -> bool:
        """Delete a repository directory only if it is within Backend/repo."""
        if not repo_path:
            return False

        abs_repo_path = os.path.abspath(repo_path)
        base_repo_dir = self.base_repo_dir

        try:
            common = os.path.commonpath([abs_repo_path, base_repo_dir])
        except ValueError:
            return False

        if common != base_repo_dir:
            return False
        if not os.path.isdir(abs_repo_path):
            return False

        shutil.rmtree(abs_repo_path, ignore_errors=True)
        return not os.path.exists(abs_repo_path)

    def _expire_session_locked(self, session_id: str, delete_repo: bool = False) -> Dict[str, object]:
        state = self._sessions.pop(session_id, None)
        session_found = state is not None
        repo_deleted = False
        repo_path = (
            os.path.abspath(state.repo_path)
            if state is not None and state.repo_path
            else None
        )
        session_dir_deleted = False

        if delete_repo:
            if (
                repo_path
                and not self._is_repo_path_in_use_locked(
                    repo_path,
                    excluding_session_id=session_id,
                )
            ):
                repo_deleted = self._delete_repo_dir(repo_path)
            session_dir_deleted = self._delete_session_dir(session_id)
            repo_deleted = repo_deleted or session_dir_deleted

        return {
            "session_found": session_found,
            "repo_deleted": repo_deleted,
            "repo_path": repo_path,
            "session_dir_deleted": session_dir_deleted,
        }

    def expire_session(self, session_id: str, delete_repo: bool = False) -> Dict[str, object]:
        with self._lock:
            return self._expire_session_locked(session_id, delete_repo=delete_repo)

    def get_session(self, session_id: str) -> SessionState:
        with self._lock:
            self._cleanup_expired_locked()
            state = self._sessions.get(session_id)
            if not state:
                state = SessionState(session_id=session_id)
                self._sessions[session_id] = state
            state.last_access = time.time()
            self._touch_session_dir(session_id)
            return state

    def prepare_repo(self, state: SessionState, repo_url: str, query: str = "") -> None:
        normalized_url = repo_url.strip()
        if not normalized_url:
            return

        previous_url = state.repo_url or ""
        same_repo = previous_url == normalized_url

        if is_github_url(normalized_url):
            session_repo_base_dir = self._session_repo_base_dir(state.session_id)
            with self._lock:
                previous_repo_path = state.repo_path
                existing_repo_path = state.repo_path if same_repo else None
                existing_selected = list(state.selected_paths) if same_repo else []
                previous_selection = set(existing_selected)
                previous_commit = state.repo_commit_sha

            snapshot = prepare_query_snapshot(
                repo_url=normalized_url,
                query=query,
                existing_repo_path=existing_repo_path,
                existing_selected_paths=existing_selected,
                base_repo_dir=session_repo_base_dir,
            )

            new_selection = set(snapshot.get("selected_paths", []))
            snapshot_changed = (
                not same_repo
                or snapshot.get("commit_sha") != previous_commit
                or previous_selection != new_selection
            )

            with self._lock:
                state.repo_url = normalized_url
                # Use full clone for indexing; snapshot_path is metadata-only
                state.repo_path = str(snapshot.get("full_repo_path", snapshot.get("repo_path", "")))
                new_repo_path = state.repo_path
                state.repo_mode = "github_api"
                state.repo_commit_sha = str(snapshot.get("commit_sha", ""))
                state.selected_paths = list(snapshot.get("selected_paths", []))
                if snapshot_changed:
                    state.runtime = None
                    if not same_repo:
                        state.history = []
                state.last_access = time.time()
                self._touch_session_dir(state.session_id)

            if (
                not same_repo
                and previous_repo_path
                and os.path.abspath(previous_repo_path) != os.path.abspath(new_repo_path)
            ):
                self._delete_repo_dir(previous_repo_path)
            return

        with self._lock:
            # Re-clone only when repository changes for this session.
            if state.repo_url == normalized_url and state.repo_path:
                state.last_access = time.time()
                self._touch_session_dir(state.session_id)
                return

            previous_repo_path = state.repo_path

        repo_path = clone_repository(
            normalized_url,
            base_repo_dir=self._session_repo_base_dir(state.session_id),
        )

        with self._lock:
            state.repo_url = normalized_url
            state.repo_path = repo_path
            state.repo_mode = "clone"
            state.repo_commit_sha = None
            state.selected_paths = []
            state.runtime = None
            state.history = []
            state.last_access = time.time()
            self._touch_session_dir(state.session_id)

        if (
            previous_repo_path
            and os.path.abspath(previous_repo_path) != os.path.abspath(repo_path)
        ):
            self._delete_repo_dir(previous_repo_path)

    def get_runtime(
        self,
        state: SessionState,
        embedding_model_name: str,
        batch_size: int,
    ) -> FunctionContextRuntime:
        with self._lock:
            repo_path = state.repo_path
            runtime = state.runtime

        if not repo_path:
            raise ValueError("Session repository is not initialized")

        if runtime is None:
            runtime = FunctionContextRuntime.create(
                repo_path=repo_path,
                embedding_model_name=embedding_model_name,
                batch_size=batch_size,
            )
            with self._lock:
                state.runtime = runtime

        with self._lock:
            state.last_access = time.time()
            self._touch_session_dir(state.session_id)
            return state.runtime if state.runtime is not None else runtime

    def get_history(self, state: SessionState) -> List[Dict[str, str]]:
        with self._lock:
            state.last_access = time.time()
            self._touch_session_dir(state.session_id)
            return list(state.history)

    def append_turn(self, state: SessionState, user_query: str, assistant_answer: str) -> None:
        with self._lock:
            state.history.append(
                {
                    "user": user_query,
                    "assistant": assistant_answer,
                }
            )
            if len(state.history) > self.max_history_turns:
                state.history = state.history[-self.max_history_turns :]
            state.last_access = time.time()
            self._touch_session_dir(state.session_id)
