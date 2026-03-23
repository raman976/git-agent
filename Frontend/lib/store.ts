import { create } from "zustand";

export interface ToolCall {
  step?: string;
  tool_name: string;
  status: "executing" | "completed";
  result_preview?: string;
}

export interface ExecutionEvent {
  id: string;
  type:
    | "status"
    | "plan"
    | "tools_info"
    | "tool_call"
    | "tool_result"
    | "answer_chunk"
    | "complete"
    | "error";
  data: Record<string, any>;
  timestamp: number;
}

export interface QueryState {
  repoUrl: string;
  query: string;
  isPreparingRepo: boolean;
  repoPrepared: boolean;
  preparedRepoUrl: string | null;
  isExecuting: boolean;
  events: ExecutionEvent[];
  plan: string | null;
  planningModel: string | null;
  toolCalls: ToolCall[];
  answerChunks: string[];
  finalModel: string | null;
  totalToolCalls: number;
  error: string | null;
  sessionId: string | null;

  setRepoUrl: (url: string) => void;
  setQuery: (query: string) => void;
  setRepoPreparing: (preparing: boolean) => void;
  setRepoPrepared: (prepared: boolean, repoUrl?: string | null) => void;
  setIsExecuting: (executing: boolean) => void;
  addEvent: (event: Omit<ExecutionEvent, "id" | "timestamp">) => void;
  setPlan: (plan: string | string[], model: string) => void;
  addToolCall: (call: ToolCall) => void;
  addAnswerChunk: (chunk: string) => void;
  setComplete: (model: string, toolCount: number) => void;
  setError: (error: string) => void;
  setSessionId: (sessionId: string | null) => void;
  reset: () => void;
  clearSession: () => void;
}

export const useQueryStore = create<QueryState>((set) => ({
  repoUrl: "",
  query: "",
  isPreparingRepo: false,
  repoPrepared: false,
  preparedRepoUrl: null,
  isExecuting: false,
  events: [],
  plan: null,
  planningModel: null,
  toolCalls: [],
  answerChunks: [],
  finalModel: null,
  totalToolCalls: 0,
  error: null,
  sessionId: null,

  setRepoUrl: (url: string) =>
    set((state) => {
      const normalizedIncoming = url.trim();
      const normalizedPrepared = (state.preparedRepoUrl || "").trim();
      const samePreparedRepo =
        normalizedPrepared.length > 0 && normalizedIncoming === normalizedPrepared;
      return {
        repoUrl: url,
        repoPrepared: samePreparedRepo ? state.repoPrepared : false,
        preparedRepoUrl: samePreparedRepo ? state.preparedRepoUrl : null,
      };
    }),
  setQuery: (query: string) => set({ query }),
  setRepoPreparing: (preparing: boolean) => set({ isPreparingRepo: preparing }),
  setRepoPrepared: (prepared: boolean, repoUrl?: string | null) =>
    set({
      repoPrepared: prepared,
      preparedRepoUrl: prepared ? repoUrl?.trim() || null : null,
    }),
  setIsExecuting: (executing: boolean) => set({ isExecuting: executing }),

  addEvent: (event: Omit<ExecutionEvent, "id" | "timestamp">) =>
    set((state) => {
      const rawEvent = event as Record<string, unknown>;
      const rawData = rawEvent.data;
      const normalizedData =
        rawData && typeof rawData === "object"
          ? (rawData as Record<string, unknown>)
          : Object.fromEntries(
              Object.entries(rawEvent).filter(
                ([key]) => key !== "type" && key !== "id" && key !== "timestamp"
              )
            );

      return {
        events: [
          ...state.events,
          {
            type: (rawEvent.type as ExecutionEvent["type"]) || "status",
            data: normalizedData,
            id: Math.random().toString(36).slice(2),
            timestamp: Date.now(),
          },
        ],
      };
    }),

  setPlan: (plan: string | string[], model: string) => {
    const normalizedPlan = Array.isArray(plan) ? plan.join(" -> ") : plan;
    set({ plan: normalizedPlan, planningModel: model });
  },

  addToolCall: (call: ToolCall) =>
    set((state) => ({
      toolCalls: (() => {
        const existingIndex = state.toolCalls.findIndex(
          (item) => item.step === call.step && item.tool_name === call.tool_name
        );

        if (existingIndex === -1) {
          return [...state.toolCalls, call];
        }

        const next = [...state.toolCalls];
        next[existingIndex] = {
          ...next[existingIndex],
          ...call,
          status: call.status,
        };
        return next;
      })(),
    })),

  addAnswerChunk: (chunk: string) =>
    set((state) => ({
      answerChunks: [...state.answerChunks, chunk],
    })),

  setComplete: (model: string, toolCount: number) =>
    set({ finalModel: model, totalToolCalls: toolCount }),

  setError: (error: string) => set({ error, isExecuting: false }),

  setSessionId: (sessionId: string | null) => set({ sessionId }),

  reset: () =>
    set({
      isPreparingRepo: false,
      isExecuting: false,
      events: [],
      plan: null,
      planningModel: null,
      toolCalls: [],
      answerChunks: [],
      finalModel: null,
      totalToolCalls: 0,
      error: null,
    }),

  clearSession: () =>
    set({
      repoUrl: "",
      query: "",
      isPreparingRepo: false,
      repoPrepared: false,
      preparedRepoUrl: null,
      isExecuting: false,
      events: [],
      plan: null,
      planningModel: null,
      toolCalls: [],
      answerChunks: [],
      finalModel: null,
      totalToolCalls: 0,
      error: null,
      sessionId: null,
    }),
}));
