"use client";

import { useCallback, useEffect } from "react";
import { useQueryStore } from "./store";
import { getOrCreateSessionId } from "./session-id";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://gh-agent.nstsdc.org";

export function useStreamingAgent() {
  const {
    setIsExecuting,
    setRepoPreparing,
    setRepoPrepared,
    addEvent,
    setPlan,
    addToolCall,
    addAnswerChunk,
    setComplete,
    setError,
    setSessionId,
    reset,
    repoPrepared,
    preparedRepoUrl,
  } = useQueryStore();

  const processRepository = useCallback(async (repoUrl: string) => {
    const normalizedRepoUrl = repoUrl.trim();
    if (!normalizedRepoUrl) {
      setError("Repository URL is required.");
      return;
    }

    reset();
    setRepoPrepared(false, null);
    setRepoPreparing(true);
    addEvent({
      type: "status",
      data: { stage: "preparing", message: "Starting repository preparation..." },
    });

    let sessionId = getOrCreateSessionId();

    try {
      const prepareResponse = await fetch(`${API_URL}/repo/prepare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repo_url: normalizedRepoUrl,
          query: "Repository preprocessing for upcoming questions",
          session_id: sessionId,
        }),
      });

      if (!prepareResponse.ok) {
        throw new Error(`Failed to start preparation: ${prepareResponse.statusText}`);
      }

      const { task_id, session_id: returnedSessionId } = await prepareResponse.json();
      sessionId = returnedSessionId;
      setSessionId(sessionId);

      const startedAt = Date.now();
      const timeoutMs = 300000;
      let preparing = true;

      while (preparing) {
        if (Date.now() - startedAt > timeoutMs) {
          throw new Error("Repository preparation timed out.");
        }

        const statusResponse = await fetch(`${API_URL}/repo/prepare/status/${task_id}`);
        if (!statusResponse.ok) {
          if (statusResponse.status === 404 && Date.now() - startedAt < 10000) {
            await new Promise((resolve) => setTimeout(resolve, 2000));
            continue;
          }
          throw new Error(`Failed to get preparation status: ${statusResponse.statusText}`);
        }

        const statusResult = await statusResponse.json();
        switch (statusResult.status) {
          case "completed":
            preparing = false;
            setRepoPrepared(true, normalizedRepoUrl);
            addEvent({
              type: "status",
              data: { stage: "preparing", message: "Repository processing completed." },
            });
            break;
          case "error":
            throw new Error(`Preparation failed: ${statusResult.message || "Unknown error"}`);
          default:
            addEvent({
              type: "status",
              data: { stage: "preparing", message: "Preparing repository..." },
            });
            await new Promise((resolve) => setTimeout(resolve, 2000));
            break;
        }
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error occurred";
      setRepoPrepared(false, null);
      setError(errorMessage);
      addEvent({
        type: "error",
        data: { error: errorMessage, stage: "preparing" },
      });
    } finally {
      setRepoPreparing(false);
    }
  }, [addEvent, reset, setError, setRepoPrepared, setRepoPreparing, setSessionId]);

  const executeQuery = useCallback(
    async (repoUrl: string, query: string) => {
      const normalizedRepoUrl = repoUrl.trim();
      if (!normalizedRepoUrl || !query.trim()) {
        setError("Repository URL and question are required.");
        return;
      }

      if (!repoPrepared || preparedRepoUrl !== normalizedRepoUrl) {
        setError("Process the repository first before asking a question.");
        addEvent({
          type: "error",
          data: {
            error: "Repository is not processed for this URL yet.",
            stage: "validation",
          },
        });
        return;
      }

      reset();
      setIsExecuting(true);
      addEvent({
        type: "status",
        data: { stage: "starting", message: "Starting analysis..." },
      });

      const sessionId = getOrCreateSessionId();
      setSessionId(sessionId);

      try {
        addEvent({
          type: "status",
          data: { stage: "reasoning", message: "Generating response..." },
        });

        const streamResponse = await fetch(`${API_URL}/query/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            repo_url: normalizedRepoUrl,
            query,
            session_id: sessionId,
            budget_mode: true,
            max_reasoning_steps: 2,
          }),
        });

        if (!streamResponse.ok) {
          throw new Error(`API error: ${streamResponse.statusText}`);
        }
        
        if (!streamResponse.body) {
          throw new Error("The response body is empty.");
        }

        const reader = streamResponse.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");

          for (let i = 0; i < lines.length - 1; i++) {
            const line = lines[i];
            if (line.trim() === "") continue;
            try {
              const event = JSON.parse(line);
              addEvent(event); // Log every event

              switch (event.type) {
                case "plan":
                  setPlan(event.plan, event.planning_model || "unknown");
                  break;
                case "tool_call":
                  addToolCall({
                    step: event.step,
                    tool_name: event.tool_name,
                    status: "executing",
                  });
                  break;
                case "tool_result":
                  addToolCall({
                    step: event.step,
                    tool_name: event.tool_name,
                    status: "completed",
                    result_preview: event.result_preview,
                  });
                  break;
                case "answer_chunk":
                  addAnswerChunk(event.chunk);
                  break;

                case "complete":
                  setComplete(event.model_used || "unknown", event.total_tool_calls || 0);
                  break;
                case "error":
                  throw new Error(event.error);
              }
            } catch (e) {
              console.error("Failed to parse stream line:", line, e);
            }
          }
          buffer = lines[lines.length - 1];
        }

      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : "Unknown error occurred";
        setError(errorMessage);
        addEvent({
          type: "error",
          data: { error: errorMessage, stage: "execution" },
        });
      } finally {
        setIsExecuting(false);
      }
    },
    [
      setIsExecuting,
      addEvent,
      setPlan,
      addToolCall,
      addAnswerChunk,
      setComplete,
      setError,
      setSessionId,
      reset,
      repoPrepared,
      preparedRepoUrl,
    ]
  );

  return { processRepository, executeQuery };
}

export function useSessionCleanup() {
  // Intentionally disabled: cleanup is handled only by explicit New Session
  // and backend inactivity TTL to avoid browser lifecycle edge cases.
  useEffect(() => {}, []);
}
