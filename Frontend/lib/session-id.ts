const STORAGE_KEY = "awesomerag_session_id";

function getStorage(): Storage | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

function createSessionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export function getOrCreateSessionId(): string {
  const storage = getStorage();
  if (!storage) {
    return createSessionId();
  }

  const existing = storage.getItem(STORAGE_KEY);
  if (existing && existing.trim()) {
    return existing;
  }

  const sessionId = createSessionId();
  storage.setItem(STORAGE_KEY, sessionId);
  return sessionId;
}

export function getStoredSessionId(): string | null {
  const storage = getStorage();
  if (!storage) {
    return null;
  }
  const existing = storage.getItem(STORAGE_KEY);
  if (!existing || !existing.trim()) {
    return null;
  }
  return existing;
}

export function rotateSessionId(): string {
  const sessionId = createSessionId();
  const storage = getStorage();
  if (storage) {
    storage.setItem(STORAGE_KEY, sessionId);
  }
  return sessionId;
}
