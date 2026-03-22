"use client";

import { useQueryStore } from "../lib/store";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { CheckCircle, AlertCircle, Copy } from "lucide-react";
import { useState } from "react";
import clsx from "clsx";

export function AnswerDisplay() {
  const {
    answerChunks,
    isExecuting,
    error,
    totalToolCalls,
    finalModel,
  } = useQueryStore();
  const [copied, setCopied] = useState(false);

  const fullAnswer = answerChunks.join("");

  if (!fullAnswer && !error && !isExecuting) {
    return null;
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(fullAnswer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-4 fade-in">
      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border-2 border-red-200 rounded-lg p-4 flex gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-red-900">Error</p>
            <p className="text-sm text-red-700 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Answer container */}
      {fullAnswer && (
        <div className="bg-white rounded-lg border-2 border-slate-200 shadow-md overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-primary-50 to-primary-100 border-b-2 border-slate-200 px-6 py-4 flex items-start justify-between">
            <div className="flex items-start gap-3">
              <CheckCircle className="w-6 h-6 text-green-600 flex-shrink-0 mt-0.5" />
              <div>
                <h2 className="font-bold text-lg text-slate-900">Analysis Complete</h2>
                <p className="text-sm text-slate-600 mt-1">
                  Model: <span className="font-mono text-primary-600">{finalModel}</span>
                  {" • "}
                  Tools: <span className="font-mono text-primary-600">{totalToolCalls}</span>
                </p>
              </div>
            </div>
            <button
              onClick={handleCopy}
              className={clsx(
                "px-3 py-2 rounded-lg transition-all flex items-center gap-2",
                copied
                  ? "bg-green-100 text-green-700"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              )}
            >
              <Copy className="w-4 h-4" />
              {copied ? "Copied!" : "Copy"}
            </button>
          </div>

          {/* Content */}
          <div className="px-6 py-6 prose prose-sm max-w-none">
            <div className="markdown-content">
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                components={{
                  h1: ({ children }) => (
                    <h1 className="text-2xl font-bold text-slate-900 mt-6 mb-4">
                      {children}
                    </h1>
                  ),
                  h2: ({ children }) => (
                    <h2 className="text-xl font-bold text-slate-800 mt-5 mb-3">
                      {children}
                    </h2>
                  ),
                  h3: ({ children }) => (
                    <h3 className="text-lg font-bold text-slate-800 mt-4 mb-2">
                      {children}
                    </h3>
                  ),
                  p: ({ children }) => (
                    <p className="text-slate-700 leading-relaxed">{children}</p>
                  ),
                  code: ({ children, ...props }: any) =>
                    !props.className ? (
                      <code className="bg-slate-100 px-2 py-1 rounded text-slate-800 font-mono text-sm">
                        {children}
                      </code>
                    ) : (
                      <code className="block bg-slate-900 text-slate-100 p-4 rounded-lg overflow-x-auto font-mono text-sm">
                        {children}
                      </code>
                    ),
                  pre: ({ children }) => (
                    <pre className="bg-slate-900 text-slate-100 p-4 rounded-lg overflow-x-auto font-mono text-sm">
                      {children}
                    </pre>
                  ),
                  a: ({ href, children }) => (
                    <a
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary-600 hover:text-primary-700 underline"
                    >
                      {children}
                    </a>
                  ),
                  ul: ({ children }) => (
                    <ul className="list-disc list-inside space-y-2 text-slate-700">
                      {children}
                    </ul>
                  ),
                  ol: ({ children }) => (
                    <ol className="list-decimal list-inside space-y-2 text-slate-700">
                      {children}
                    </ol>
                  ),
                  li: ({ children }) => <li className="ml-2">{children}</li>,
                  blockquote: ({ children }) => (
                    <blockquote className="border-l-4 border-slate-300 pl-4 italic text-slate-600 my-4">
                      {children}
                    </blockquote>
                  ),
                  table: ({ children }) => (
                    <table className="w-full border-collapse border border-slate-300">
                      {children}
                    </table>
                  ),
                  th: ({ children }) => (
                    <th className="border border-slate-300 bg-slate-100 px-4 py-2 text-left">
                      {children}
                    </th>
                  ),
                  td: ({ children }) => (
                    <td className="border border-slate-300 px-4 py-2">
                      {children}
                    </td>
                  ),
                }}
              >
                {fullAnswer}
              </ReactMarkdown>
            </div>
          </div>
        </div>
      )}

      {/* Loading state */}
      {isExecuting && fullAnswer === "" && (
        <div className="bg-white rounded-lg border-2 border-slate-200 p-8 flex flex-col items-center justify-center gap-3">
          <div className="w-8 h-8 border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin" />
          <p className="text-slate-600 font-medium">Generating response...</p>
          <p className="text-sm text-slate-500">
            The agent is analyzing your repository
          </p>
        </div>
      )}

      {/* Streaming indicator */}
      {isExecuting && fullAnswer !== "" && (
        <div className="flex items-center gap-2 p-3 bg-blue-50 border-l-4 border-primary-600 rounded">
          <div className="w-2 h-2 bg-primary-600 rounded-full pulse-dot" />
          <p className="text-sm text-slate-700">Streaming response...</p>
        </div>
      )}
    </div>
  );
}
