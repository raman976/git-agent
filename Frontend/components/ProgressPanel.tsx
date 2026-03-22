"use client";

import { useQueryStore } from "../lib/store";
import {
  CheckCircle,
  Clock,
  Brain,
  Wrench,
} from "lucide-react";

export function ProgressPanel() {
  const {
    plan,
    planningModel,
    toolCalls,
    totalToolCalls,
    isExecuting,
    events,
  } = useQueryStore();

  if (!plan && toolCalls.length === 0 && !isExecuting) {
    return null;
  }

  return (
    <div className="space-y-4 fade-in">
      {/* Planning Phase */}
      {plan && (
        <div className="bg-white rounded-lg border-2 border-primary-100 p-4 shadow-sm">
          <div className="flex items-start gap-3">
            <Brain className="w-5 h-5 text-primary-600 mt-1 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="font-semibold text-slate-900 mb-2">Agent Planning</h3>
              <p className="text-sm text-slate-700 bg-primary-50 p-3 rounded text-sm">
                {plan}
              </p>
              {planningModel && (
                <p className="text-xs text-slate-500 mt-2">
                  Planning Model: <span className="font-mono">{planningModel}</span>
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tool Execution Timeline */}
      {toolCalls.length > 0 && (
        <div className="bg-white rounded-lg border-2 border-slate-200 p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <Wrench className="w-5 h-5 text-slate-600" />
            <h3 className="font-semibold text-slate-900">
              Tool Execution ({toolCalls.length}/{totalToolCalls || "?"})
            </h3>
          </div>

          <div className="space-y-3">
            {toolCalls.map((tool, idx) => (
              <div
                key={idx}
                className="flex items-start gap-3 p-3 bg-slate-50 rounded-lg"
              >
                <div className="flex-shrink-0 mt-1">
                  {tool.status === "completed" ? (
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  ) : (
                    <div className="w-5 h-5 border-2 border-primary-600 border-t-transparent rounded-full animate-spin" />
                  )}
                </div>
                <div className="flex-1">
                  <p className="font-mono text-sm font-semibold text-slate-900">
                    {tool.tool_name}
                  </p>
                  {tool.result_preview && (
                    <p className="text-xs text-slate-600 mt-1 line-clamp-2">
                      {tool.result_preview}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Real-time Events */}
      {events.length > 0 && (
        <div className="bg-white rounded-lg border-2 border-slate-200 p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <Clock className="w-5 h-5 text-slate-600" />
            <h3 className="font-semibold text-slate-900">Execution Events</h3>
          </div>

          <div className="space-y-2 max-h-48 overflow-y-auto">
            {events.slice(-5).map((event) => (
              <div
                key={event.id}
                className="text-xs p-2 rounded bg-slate-50 border-l-2 border-slate-300"
              >
                <div className="flex items-start justify-between">
                  <span className="font-mono text-primary-600">
                    {event.type}
                  </span>
                  <span className="text-slate-500">
                    {new Date(event.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                {event.data.message && (
                  <p className="text-slate-700 mt-1">{event.data.message}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Current Status */}
      {isExecuting && (
        <div className="flex items-center gap-2 p-3 bg-blue-50 border-2 border-primary-200 rounded-lg animate-pulse">
          <div className="w-2 h-2 bg-primary-600 rounded-full pulse-dot" />
          <p className="text-sm font-medium text-slate-700 flex-1">
            Agent is working on your query...
          </p>
        </div>
      )}
    </div>
  );
}
