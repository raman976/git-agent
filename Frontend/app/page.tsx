"use client";

import { QueryForm } from "../components/QueryForm";
import { ProgressPanel } from "../components/ProgressPanel";
import { AnswerDisplay } from "../components/AnswerDisplay";
import { useStreamingAgent, useSessionCleanup } from "../lib/hooks";
import { Brain, Zap, BookOpen } from "lucide-react";

export default function Home() {
  const { executeQuery } = useStreamingAgent();
  useSessionCleanup();

  const handleSubmit = (repoUrl: string, query: string) => {
    executeQuery(repoUrl, query);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50">
      {/* Header */}
      <header className="sticky top-0 z-40 bg-white/80 backdrop-blur-md border-b border-slate-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-600 to-primary-700 rounded-lg flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-900">Gh Agent</h1>
                <p className="text-sm text-slate-600">Intelligent Repository Agent</p>
              </div>
            </div>
            <div className="flex items-center gap-4 text-sm">
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-slate-600 hover:text-slate-900 transition-colors"
              >
                GitHub
              </a>
              <a
                href="https://docs.example.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-slate-600 hover:text-slate-900 transition-colors"
              >
                Docs
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        {/* Hero Section */}
        <div className="mb-12 text-center">
          <h2 className="text-4xl font-bold text-slate-900 mb-4">
            Analyze Any Repository
          </h2>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            Get instant insights about repository purpose, architecture, authentication,
            and more. Powered by advanced code analysis and AI reasoning.
          </p>
        </div>

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Query Form */}
          <div className="lg:col-span-1">
            <div className="sticky top-24 space-y-6">
              {/* Query Form Card */}
              <div className="bg-white rounded-xl border-2 border-slate-200 shadow-lg p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Zap className="w-5 h-5 text-primary-600" />
                  <h3 className="font-bold text-slate-900">Query</h3>
                </div>
                <QueryForm onSubmit={handleSubmit} />
              </div>

              {/* Info Cards */}
              <div className="space-y-3">
                <div className="bg-blue-50 rounded-lg border-2 border-blue-200 p-4">
                  <h4 className="font-semibold text-blue-900 mb-2 flex items-center gap-2">
                    <span className="w-2 h-2 bg-blue-600 rounded-full" />
                    About This Tool
                  </h4>
                  <p className="text-sm text-blue-800">
                    Upload any GitHub repository and ask questions about its functionality,
                    structure, or specific implementation details.
                  </p>
                </div>

                <div className="bg-green-50 rounded-lg border-2 border-green-200 p-4">
                  <h4 className="font-semibold text-green-900 mb-2 flex items-center gap-2">
                    <span className="w-2 h-2 bg-green-600 rounded-full" />
                    Example Questions
                  </h4>
                  <ul className="text-sm text-green-800 space-y-1">
                    <li>• What does this repository do?</li>
                    <li>• How does auth work?</li>
                    <li>• What are the main components?</li>
                    <li>• How is data processed?</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Results */}
          <div className="lg:col-span-2 space-y-6">
            {/* Progress Panel */}
            <ProgressPanel />

            {/* Answer Display */}
            <AnswerDisplay />

            {/* Empty State */}
            {false && (
              <div className="text-center py-12 bg-white rounded-xl border-2 border-slate-200">
                <BookOpen className="w-12 h-12 text-slate-400 mx-auto mb-3" />
                <p className="text-slate-600 font-medium">
                  Enter a repository URL and ask a question to get started
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-slate-50 py-8 mt-12">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-sm text-slate-600">
          <p>Built with Next.js, FastAPI, and AI-powered code analysis</p>
          <p className="mt-2">
            © 2026 Gh Agent. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}
