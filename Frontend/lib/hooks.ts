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
        // Step 1: Start the cloning process
        addEvent({
          type: "status",
          data: { stage: "cloning", message: "Starting repository clone..." },
        });

        const cloneResponse = await fetch(`${API_URL}/repo/clone`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ repo_url: repoUrl }),
        });

        if (!cloneResponse.ok) {
          throw new Error(`Failed to start cloning: ${cloneResponse.statusText}`);
        }

        const { task_id } = await cloneResponse.json();

        // Step 2: Poll for cloning status
        let cloning = true;
        let finalStatus = null;
        const startTime = Date.now();
        const timeout = 300000; // 5 minutes

        while (cloning) {
          if (Date.now() - startTime > timeout) {
            throw new Error("Repository cloning timed out.");
          }

          const statusResponse = await fetch(`${API_URL}/repo/clone/status/${task_id}`);
          if (!statusResponse.ok) {
            // Allow 404 for a short period as the job might not be in the dict yet
            if (statusResponse.status === 404 && Date.now() - startTime < 10000) {
              await new Promise((resolve) => setTimeout(resolve, 2000));
              continue;
            }
            throw new Error(`Failed to get cloning status: ${statusResponse.statusText}`);
          }

          const statusResult = await statusResponse.json();

          switch (statusResult.status) {
            case "completed":
              cloning = false;
              finalStatus = statusResult;
              addEvent({
                type: "status",
                data: { stage: "cloning", message: "Repository cloned successfully." },
              });
              break;
            case "error":
              throw new Error(`Cloning failed: ${statusResult.message}`);
            case "cloning":
              addEvent({
                type: "status",
                data: { stage: "cloning", message: "Cloning in progress..." },
              });
              await new Promise((resolve) => setTimeout(resolve, 2000));
              break;
            default:
              await new Promise((resolve) => setTimeout(resolve, 2000));
          }
        }

        // Step 3: Proceed with the original query logic
        const sessionId = getOrCreateSessionId();
        addEvent({
          type: "status",
          data: { stage: "reasoning", message: "Generating complete response..." },
        });

        const queryResponse = await fetch(`${API_URL}/query`, {
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

        if (!queryResponse.ok) {
          throw new Error(`API error: ${queryResponse.statusText}`);
        }
        const payload = await queryResponse.json();

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
