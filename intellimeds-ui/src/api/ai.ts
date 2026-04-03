import { apiFetch } from "./client";

export type AiConversation = {
  id: string;
  title?: string;
  createdAt?: string;
  updatedAt?: string;
};

export type AiMessage = {
  id: string;
  role: "user" | "assistant" | string;
  content: string;
  createdAt?: string;
};

export type AnalyzeAiRequest = {
  message: string;
  conversationId?: string;
};

export type AnalyzeAiResponse = {
  conversationId?: string;
  reply?: string;
};

export async function listAiConversations(): Promise<unknown> {
  return apiFetch("/api/ai/conversations", {
    method: "GET",
    auth: true,
  });
}

export async function createAiConversation(title?: string): Promise<unknown> {
  return apiFetch("/api/ai/conversations", {
    method: "POST",
    auth: true,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(title ? { title } : {}),
  });
}

export async function getAiMessages(conversationId: string): Promise<unknown> {
  return apiFetch(`/api/ai/conversations/${conversationId}/messages`, {
    method: "GET",
    auth: true,
  });
}

export async function deleteAiConversation(
  conversationId: string
): Promise<unknown> {
  return apiFetch(`/api/ai/conversations/${conversationId}`, {
    method: "DELETE",
    auth: true,
  });
}

export async function analyzeAiMessage(
  req: AnalyzeAiRequest
): Promise<unknown> {
  return apiFetch("/api/ai/analyze", {
    method: "POST",
    auth: true,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}