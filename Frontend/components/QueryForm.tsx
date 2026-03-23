"use client";

import { Search, Github, Send, RotateCcw } from "lucide-react";
import { useQueryStore } from "../lib/store";
import { getOrCreateSessionId, rotateSessionId } from "../lib/session-id";
import clsx from "clsx";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://gh-agent.nstsdc.org";

export function QueryForm({
  onProcessRepository,
  onSubmit,
}: {
  onProcessRepository: (repoUrl: string) => void;
  onSubmit: (repoUrl: string, query: string) => void;
}) {
  const {
    repoUrl,
    query,
    setRepoUrl,
    setQuery,
    isExecuting,
    isPreparingRepo,
    repoPrepared,
    preparedRepoUrl,
    clearSession,
  } = useQueryStore();

  const normalizedRepoUrl = repoUrl.trim();
  const isCurrentRepoPrepared =
    repoPrepared && preparedRepoUrl === normalizedRepoUrl && normalizedRepoUrl.length > 0;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!normalizedRepoUrl || !query.trim() || !isCurrentRepoPrepared) {
      return;
    }
    onSubmit(normalizedRepoUrl, query);
  };

  const handleProcessRepository = () => {
    if (!normalizedRepoUrl) {
      return;
    }
    onProcessRepository(normalizedRepoUrl);
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
        disabled={isExecuting || isPreparingRepo}
        className={clsx(
          "w-full py-2 px-4 rounded-lg font-semibold flex items-center justify-center gap-2",
          "transition-all duration-200 text-sm",
          isExecuting || isPreparingRepo
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
          disabled={isExecuting || isPreparingRepo}
          className={clsx(
            "w-full px-4 py-3 rounded-lg border-2 font-mono text-sm",
            "bg-white focus:outline-none transition-colors",
            isExecuting || isPreparingRepo
              ? "border-slate-300 bg-slate-50 cursor-not-allowed"
              : "border-slate-300 focus:border-primary-600 focus:ring-2 focus:ring-primary-100"
          )}
        />
      </div>

      {/* Process Repository Button */}
      <button
        type="button"
        onClick={handleProcessRepository}
        disabled={isExecuting || isPreparingRepo || !normalizedRepoUrl}
        className={clsx(
          "w-full py-3 px-4 rounded-lg font-semibold flex items-center justify-center gap-2",
          "transition-all duration-200 text-white",
          isExecuting || isPreparingRepo || !normalizedRepoUrl
            ? "bg-slate-300 cursor-not-allowed"
            : "bg-emerald-600 hover:bg-emerald-700 active:scale-95 shadow-md hover:shadow-lg"
        )}
      >
        {isPreparingRepo ? (
          <>
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            Processing Repository...
          </>
        ) : isCurrentRepoPrepared ? (
          <>Repository Processed</>
        ) : (
          <>Process Repository</>
        )}
      </button>

      <p className="text-xs text-slate-500">
        {isCurrentRepoPrepared
          ? "Repository is ready. You can now ask questions."
          : "Process the repository first to enable question submission."}
      </p>

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
          disabled={isExecuting || isPreparingRepo || !isCurrentRepoPrepared}
          rows={3}
          className={clsx(
            "w-full px-4 py-3 rounded-lg border-2 resize-none",
            "bg-white focus:outline-none transition-colors",
            "font-sans text-base",
            isExecuting || isPreparingRepo || !isCurrentRepoPrepared
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
        disabled={isExecuting || isPreparingRepo || !isCurrentRepoPrepared || !query.trim()}
        className={clsx(
          "w-full py-3 px-4 rounded-lg font-semibold flex items-center justify-center gap-2",
          "transition-all duration-200 text-white",
          isExecuting || isPreparingRepo || !isCurrentRepoPrepared || !query.trim()
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
            Ask Question
          </>
        )}
      </button>
    </form>
  );
}
