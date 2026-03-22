"use client";

import { useCallback, useEffect } from "react";
import { useQueryStore } from "./store";
import { getOrCreateSessionId } from "./session-id";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://gh-agent.nstsdc.org";

export function useStreamingAgent() {
  const {
    setIsExecuting,
    addEvent,
    setPlan,
    addToolCall,
    addAnswerChunk,
    setComplete,
    setError,
    reset,
  } = useQueryStore();

  const executeQuery = useCallback(
    async (repoUrl: string, query: string) => {
      reset();
      setIsExecuting(true);
      addEvent({ type: "status", data: { stage: "starting", message: "Initializing..." } });

      try {
        const sessionId = getOrCreateSessionId();
        const response = await fetch(`${API_URL}/query/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            repo_url: repoUrl,
            query: query,
            session_id: sessionId,
            budget_mode: true,
            max_reasoning_steps: 2,
          }),
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.statusText}`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          buffer += decoder.decode(value, { stream: !done });

          if (done) break;

          // Process complete lines
          const lines = buffer.split("\n");
          buffer = lines.pop() || ""; // Keep the incomplete line in the buffer

          for (const line of lines) {
            if (!line.trim()) continue;

            try {
              const event = JSON.parse(line);

              // Add event to timeline
              addEvent({ type: event.type, data: event });

              // Process specific event types
              switch (event.type) {
                case "plan":
                  setPlan(event.plan, event.planning_model);
                  break;

                case "tool_call":
                  addToolCall({
                    step: event.step,
                    tool_name: event.tool_name,
                    status: event.status,
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
                  setComplete(event.model_used, event.total_tool_calls);
                  setIsExecuting(false);
                  break;

                case "error":
                  throw new Error(event.error);
              }
            } catch (e) {
              if (!(e instanceof SyntaxError)) {
                throw e;
              }
              // Ignore JSON parse errors for now
            }
          }
        }

        // Process any remaining data
        if (buffer.trim()) {
          try {
            const event = JSON.parse(buffer);
            addEvent({ type: event.type, data: event });

            if (event.type === "complete") {
              setComplete(event.model_used, event.total_tool_calls);
            }
          } catch (e) {
            // Ignore
          }
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
      reset,
    ]
  );

  return { executeQuery };
}

export function useSessionCleanup() {
  // Intentionally disabled: cleanup is handled only by explicit New Session
  // and backend inactivity TTL to avoid browser lifecycle edge cases.
  useEffect(() => {}, []);
}
