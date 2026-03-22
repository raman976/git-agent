"use client";

import { Search, Github, Send, RotateCcw } from "lucide-react";
import { useQueryStore } from "../lib/store";
import { getOrCreateSessionId, rotateSessionId } from "../lib/session-id";
import clsx from "clsx";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function QueryForm({
  onSubmit,
}: {
  onSubmit: (repoUrl: string, query: string) => void;
}) {
  const { repoUrl, query, setRepoUrl, setQuery, isExecuting, clearSession } = useQueryStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!repoUrl.trim() || !query.trim()) {
      return;
    }
    onSubmit(repoUrl, query);
  };

  const handleNewSession = async () => {
    const previousSessionId = getOrCreateSessionId();
    try {
      await fetch(`${API_URL}/session/clear`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: previousSessionId,
          delete_repo: true,
        }),
      });
    } catch (error) {
      console.error("Failed to clear session:", error);
    }

    rotateSessionId();
    clearSession();
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* New Session Button */}
      <button
        type="button"
        onClick={handleNewSession}
        disabled={isExecuting}
        className={clsx(
          "w-full py-2 px-4 rounded-lg font-semibold flex items-center justify-center gap-2",
          "transition-all duration-200 text-sm",
          isExecuting
            ? "bg-slate-300 text-slate-600 cursor-not-allowed"
            : "bg-slate-200 text-slate-700 hover:bg-slate-300 active:scale-95"
        )}
      >
        <RotateCcw className="w-4 h-4" />
        New Session
      </button>

      {/* Repository URL Input */}
      <div className="space-y-2">
        <label className="flex items-center gap-2 text-slate-700 font-semibold">
          <Github className="w-5 h-5 text-primary-600" />
          Repository URL
        </label>
        <input
          type="url"
          value={repoUrl}
          onChange={(e) => setRepoUrl(e.target.value)}
          placeholder="https://github.com/owner/repository"
          disabled={isExecuting}
          className={clsx(
            "w-full px-4 py-3 rounded-lg border-2 font-mono text-sm",
            "bg-white focus:outline-none transition-colors",
            isExecuting
              ? "border-slate-300 bg-slate-50 cursor-not-allowed"
              : "border-slate-300 focus:border-primary-600 focus:ring-2 focus:ring-primary-100"
          )}
        />
      </div>

      {/* Query Input */}
      <div className="space-y-2">
        <label className="flex items-center gap-2 text-slate-700 font-semibold">
          <Search className="w-5 h-5 text-primary-600" />
          Question
        </label>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="What does this repository do? How does authentication work? What are the main components?"
          disabled={isExecuting}
          rows={3}
          className={clsx(
            "w-full px-4 py-3 rounded-lg border-2 resize-none",
            "bg-white focus:outline-none transition-colors",
            "font-sans text-base",
            isExecuting
              ? "border-slate-300 bg-slate-50 cursor-not-allowed"
              : "border-slate-300 focus:border-primary-600 focus:ring-2 focus:ring-primary-100"
          )}
        />
        <p className="text-xs text-slate-500">
          {query.length > 0 && `${query.length} characters`}
        </p>
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={isExecuting || !repoUrl.trim() || !query.trim()}
        className={clsx(
          "w-full py-3 px-4 rounded-lg font-semibold flex items-center justify-center gap-2",
          "transition-all duration-200 text-white",
          isExecuting || !repoUrl.trim() || !query.trim()
            ? "bg-slate-300 cursor-not-allowed"
            : "bg-primary-600 hover:bg-primary-700 active:scale-95 shadow-md hover:shadow-lg"
        )}
      >
        {isExecuting ? (
          <>
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            Processing...
          </>
        ) : (
          <>
            <Send className="w-4 h-4" />
            Analyze Repository
          </>
        )}
      </button>
    </form>
  );
}
