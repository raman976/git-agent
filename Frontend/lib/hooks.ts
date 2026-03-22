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
        addEvent({
          type: "status",
          data: { stage: "reasoning", message: "Generating complete response..." },
        });

        const response = await fetch(`${API_URL}/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            repo_url: repoUrl,
            query,
            session_id: sessionId,
            budget_mode: true,
            max_reasoning_steps: 2,
          }),
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.statusText}`);
        }
        const payload = await response.json();

        if (payload?.plan) {
          setPlan(payload.plan, String(payload?.planning_model_used || "unknown"));
          addEvent({
            type: "plan",
            data: {
              plan: payload.plan,
              planning_model: String(payload?.planning_model_used || "unknown"),
            },
          });
        }

        const toolCalls = Array.isArray(payload?.tool_calls_executed)
          ? payload.tool_calls_executed
          : [];
        for (const [index, toolCall] of toolCalls.entries()) {
          const toolName = String(toolCall?.name || "unknown");
          const step = `${index + 1}/${toolCalls.length}`;
          addToolCall({ step, tool_name: toolName, status: "completed" });
          addEvent({
            type: "tool_result",
            data: {
              step,
              tool_name: toolName,
              result_preview: toolCall?.result_preview,
            },
          });
        }

        const finalAnswer = String(payload?.final_answer || "");
        if (finalAnswer) {
          addAnswerChunk(finalAnswer);
        }

        const modelUsed = String(payload?.model_used || "unknown");
        const totalToolCalls = toolCalls.length;
        setComplete(modelUsed, totalToolCalls);
        addEvent({
          type: "complete",
          data: {
            model_used: modelUsed,
            total_tool_calls: totalToolCalls,
            mode: "sync",
          },
        });
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
