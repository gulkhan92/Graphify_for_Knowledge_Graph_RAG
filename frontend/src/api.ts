import type { AskResponse, SummaryResponse } from "./types";

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.text();
    throw new Error(payload || `Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchSummary(): Promise<SummaryResponse> {
  const response = await fetch("/api/summary");
  return handleResponse<SummaryResponse>(response);
}

export async function triggerIngest(): Promise<SummaryResponse> {
  const response = await fetch("/api/ingest", { method: "POST" });
  return handleResponse<SummaryResponse>(response);
}

export async function askQuestion(question: string): Promise<AskResponse> {
  const response = await fetch(`/api/ask?question=${encodeURIComponent(question)}`);
  return handleResponse<AskResponse>(response);
}
