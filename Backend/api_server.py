"""
FastAPI server for Gh Agent with streaming support.
Exposes endpoints for repo analysis and real-time agent execution.
"""

import json
import asyncio
import os
import threading
from typing import AsyncGenerator, Optional
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

from Backend.RepoManager.clone import clone_repository
from Backend.LLMHandler.agent_graph import classify_query_scope, run_multi_reasoning_agent
from Backend.LLMHandler.config import DEFAULT_EMBEDDING_MODEL
from Backend.LLMHandler.session_manager import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Gh Agent API", version="1.0.0")
session_manager = SessionManager(
    ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "600")),
    max_history_turns=6,
    cleanup_interval_seconds=int(os.getenv("SESSION_CLEANUP_INTERVAL_SECONDS", "30")),
    orphan_repo_ttl_seconds=int(os.getenv("ORPHAN_REPO_TTL_SECONDS", "600")),
    max_repo_storage_bytes=int(os.getenv("MAX_REPO_STORAGE_BYTES", str(2 * 1024 * 1024 * 1024))),
)


def _parse_allowed_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "")
    configured = [origin.strip() for origin in raw.split(",") if origin.strip()]
    defaults = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    # Keep order stable and remove duplicates.
    return list(dict.fromkeys(defaults + configured))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_allowed_origins(),
    allow_origin_regex=os.getenv("CORS_ALLOW_ORIGIN_REGEX", r"https://.*\.vercel\.app"),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Request model for agent queries."""
    repo_url: str
    query: str
    session_id: Optional[str] = None
    budget_mode: bool = True
    max_reasoning_steps: int = 2


class SessionClearRequest(BaseModel):
    """Request model for clearing session state and cached repos."""
    session_id: Optional[str] = None
    delete_repo: bool = True


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", message="Gh Agent API is running")


async def stream_agent_execution(
    repo_url: str,
    query: str,
    budget_mode: bool,
    max_reasoning_steps: int,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    Stream agent execution with real-time updates.
    Yields JSON-formatted events for frontend consumption.
    """
    def sync_execute():
        """Synchronous wrapper to capture streaming context."""
        try:
            query_scope = classify_query_scope(query)

            # Event: Starting
            yield json.dumps({
                "type": "status",
                "stage": "initializing",
                "message": "Preparing repository and agent...",
                "query_scope": query_scope,
                "session_id": session_id,
            }) + "\n"

            session = session_manager.get_session(session_id)

            repo_path = None
            runtime = None
            history = session_manager.get_history(session)
            runtime_reused = False

            if query_scope != "general":
                # Prepare repository only for repo/hybrid questions.
                try:
                    previous_repo_path = session.repo_path
                    session_manager.prepare_repo(session, repo_url, query=query)
                    repo_path = session.repo_path
                    if not repo_path:
                        raise ValueError("Repository path could not be prepared")
                    yield json.dumps({
                        "type": "status",
                        "stage": "repo_ready",
                        "message": f"Repository ready: {repo_path.split('/')[-1]}",
                        "repo_reused": previous_repo_path == repo_path,
                        "repo_mode": session.repo_mode,
                        "repo_commit_sha": session.repo_commit_sha,
                        "session_id": session_id,
                    }) + "\n"
                except Exception as e:
                    yield json.dumps({
                        "type": "error",
                        "error": f"Repository error: {str(e)}",
                        "stage": "repo_init",
                        "session_id": session_id,
                    }) + "\n"
                    return

                runtime_reused = session.runtime is not None
                runtime = session_manager.get_runtime(
                    session,
                    embedding_model_name=DEFAULT_EMBEDDING_MODEL,
                    batch_size=32,
                )

            # Event: Planning phase starting
            yield json.dumps({
                "type": "status",
                "stage": "planning",
                "message": "Generating optimal search strategy...",
                "runtime_reused": runtime_reused,
                "session_id": session_id,
            }) + "\n"

            # Run agent in a worker thread so we can keep the stream alive with heartbeats.
            result_container: dict[str, object] = {}
            error_container: dict[str, Exception] = {}
            done = threading.Event()

            def _run_agent() -> None:
                try:
                    result_container["result"] = run_multi_reasoning_agent(
                        query,
                        repo_path=repo_path,
                        runtime=runtime,
                        conversation_context=history,
                        query_scope=query_scope,
                        budget_mode=budget_mode,
                        max_reasoning_steps=max_reasoning_steps,
                    )
                except Exception as agent_error:
                    error_container["error"] = agent_error
                finally:
                    done.set()

            threading.Thread(target=_run_agent, name="agent-runner", daemon=True).start()

            while not done.wait(timeout=5):
                yield json.dumps({
                    "type": "status",
                    "stage": "reasoning",
                    "message": "Still processing...",
                    "session_id": session_id,
                }) + "\n"

            if "error" in error_container:
                raise error_container["error"]

            result = result_container.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("Agent returned an invalid response")

            session_manager.append_turn(
                session,
                user_query=query,
                assistant_answer=result.get("final_answer", ""),
            )

            # Event: Planning complete
            if result.get("plan"):
                yield json.dumps({
                    "type": "plan",
                    "plan": result["plan"],
                    "planning_model": result.get("planning_model_used", "unknown"),
                    "session_id": session_id,
                }) + "\n"

            # Event: Tool execution phase
            tool_calls = result.get("tool_calls_executed", [])
            yield json.dumps({
                "type": "status",
                "stage": "tools",
                "message": f"Executing {len(tool_calls)} tool calls...",
                "session_id": session_id,
            }) + "\n"

            # Event: Each tool execution
            for idx, tool_call in enumerate(tool_calls, 1):
                yield json.dumps({
                    "type": "tool_call",
                    "tool_name": tool_call.get("name", "unknown"),
                    "status": "executing",
                    "step": f"{idx}/{len(tool_calls)}",
                    "timestamp": tool_call.get("timestamp", ""),
                    "session_id": session_id,
                }) + "\n"

                # Brief pause to show it's running
                import time
                time.sleep(0.1)

                # Tool result
                result_preview = tool_call.get("result_preview")
                if result_preview:
                    yield json.dumps({
                        "type": "tool_result",
                        "tool_name": tool_call.get("name"),
                        "step": f"{idx}/{len(tool_calls)}",
                        "result_preview": str(result_preview),
                        "session_id": session_id,
                    }) + "\n"

            # Event: Final answer generation
            yield json.dumps({
                "type": "status",
                "stage": "reasoning",
                "message": "Synthesizing final answer...",
                "session_id": session_id,
            }) + "\n"

            # Event: Final answer (streamed in chunks for real-time rendering)
            final_answer = result.get("final_answer", "No answer generated")
            chunk_size = 80  # Optimal for streaming UI updates
            
            for i in range(0, len(final_answer), chunk_size):
                chunk = final_answer[i : i + chunk_size]
                yield json.dumps({
                    "type": "answer_chunk",
                    "chunk": chunk,
                    "is_last": i + chunk_size >= len(final_answer),
                    "session_id": session_id,
                }) + "\n"
                # Small delay for realistic streaming effect
                import time
                time.sleep(0.03)

            # Event: Complete
            yield json.dumps({
                "type": "complete",
                "model_used": result.get("model_used", "unknown"),
                "total_tool_calls": len(tool_calls),
                "session_id": session_id,
                "execution_stats": {
                    "budget_mode": budget_mode,
                    "max_reasoning_steps": max_reasoning_steps,
                    "planning_model": result.get("planning_model_used"),
                    "runtime_reused": runtime_reused,
                    "query_scope": query_scope,
                    "repo_mode": session.repo_mode,
                    "repo_commit_sha": session.repo_commit_sha,
                }
            }) + "\n"

        except Exception as e:
            logger.error(f"Error in stream_agent_execution: {str(e)}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "error": str(e),
                "stage": "execution",
                "session_id": session_id,
            }) + "\n"

    # Convert sync generator to async
    for item in sync_execute():
        yield item
        await asyncio.sleep(0)  # Allow other tasks to run


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Stream agent execution for a query.
    Returns server-sent events with real-time updates.
    """
    session_id = request.session_id or str(uuid4())
    return StreamingResponse(
        stream_agent_execution(
            request.repo_url,
            request.query,
            request.budget_mode,
            request.max_reasoning_steps,
            session_id,
        ),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/query")
async def query_sync(request: QueryRequest):
    """
    Synchronous query execution (non-streaming).
    Returns complete result.
    """
    try:
        query_scope = classify_query_scope(request.query)
        session_id = request.session_id or str(uuid4())
        session = session_manager.get_session(session_id)

        repo_path = None
        runtime = None
        if query_scope != "general":
            session_manager.prepare_repo(session, request.repo_url, query=request.query)
            repo_path = session.repo_path
            if not repo_path:
                raise ValueError("Repository path could not be prepared")

            runtime = session_manager.get_runtime(
                session,
                embedding_model_name=DEFAULT_EMBEDDING_MODEL,
                batch_size=32,
            )

        history = session_manager.get_history(session)

        result = run_multi_reasoning_agent(
            request.query,
            repo_path=repo_path,
            runtime=runtime,
            conversation_context=history,
            query_scope=query_scope,
            budget_mode=request.budget_mode,
            max_reasoning_steps=request.max_reasoning_steps,
        )
        session_manager.append_turn(
            session,
            user_query=request.query,
            assistant_answer=result.get("final_answer", ""),
        )
        result["repo_path"] = repo_path
        result["session_id"] = session_id
        result["query_scope"] = query_scope
        result["repo_mode"] = session.repo_mode
        result["repo_commit_sha"] = session.repo_commit_sha
        return result
    except Exception as e:
        logger.error(f"Error in query_sync: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/repo/validate")
async def validate_repo(repo_url: str):
    """Validate if a repository URL is valid."""
    try:
        repo_path = clone_repository(repo_url)
        return {
            "valid": True,
            "repo_url": repo_url,
            "local_path": repo_path,
            "message": "Repository is valid and ready for analysis"
        }
    except Exception as e:
        logger.error(f"Error validating repo: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid repository: {str(e)}"
        )


@app.post("/session/clear")
async def clear_session(request: SessionClearRequest):
    """Clear/expire a session."""
    try:
        session_id = request.session_id
        if session_id:
            cleanup_result = session_manager.expire_session(
                session_id,
                delete_repo=request.delete_repo,
            )
            logger.info(f"Session cleared: {session_id}")
            return {
                "status": "success",
                "message": f"Session {session_id} cleared",
                "session_id": session_id,
                "session_found": cleanup_result.get("session_found", False),
                "repo_path": cleanup_result.get("repo_path"),
                "repo_deleted": cleanup_result.get("repo_deleted", False),
            }
        else:
            logger.warning("clear_session called without session_id")
            return {
                "status": "success",
                "message": "No session to clear",
                "session_id": None,
                "session_found": False,
                "repo_deleted": False,
            }
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
