import React, { useMemo, useRef, useState, useEffect } from "react";
import { ArrowLeft, Bot, Send, ShieldAlert, Package } from "lucide-react";
import { useNavigate } from "react-router-dom";
import type { ChatMessage } from "@/types";
import { analyzeAiMessage, createAiConversation, getAiMessages } from "@/api/ai";

function nowId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  // Fallback for environments where crypto.randomUUID is not available.
  return `id_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function getDefaultGreeting(): ChatMessage {
  return {
    id: nowId(),
    role: "assistant",
    content:
      "Hello! I'm your AI Personal Pharmacist. I can help you with:\n\n• Medication information\n• Usage guidelines\n• Safety warnings\n• Drug interactions\n\nHow can I assist you today?",
    timestamp: Date.now(),
  };
}

function normalizeConversationId(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function normalizeReply(data: unknown): string | undefined {
  if (data && typeof data === "object") {
    const reply = (data as { reply?: unknown }).reply;
    if (typeof reply === "string" && reply.trim()) return reply;
  }
  return undefined;
}

function normalizeMessages(data: unknown): ChatMessage[] {
  if (!Array.isArray(data)) return [];

  return data
    .map((item) => {
      if (!item || typeof item !== "object") return null;
      const msg = item as {
        id?: unknown;
        role?: unknown;
        content?: unknown;
        createdAt?: unknown;
      };
      if (typeof msg.role !== "string" || typeof msg.content !== "string") return null;
      return {
        id: typeof msg.id === "string" ? msg.id : nowId(),
        role: msg.role === "user" ? "user" : "assistant",
        content: msg.content,
        timestamp:
          typeof msg.createdAt === "string" && !Number.isNaN(Date.parse(msg.createdAt))
            ? Date.parse(msg.createdAt)
            : Date.now(),
      } satisfies ChatMessage;
    })
    .filter((msg): msg is ChatMessage => !!msg);
}

export function AiHelpPage() {
  const navigate = useNavigate();

  const [messages, setMessages] = useState<ChatMessage[]>(() => [getDefaultGreeting()]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);

  const listRef = useRef<HTMLDivElement | null>(null);

  const disclaimer = useMemo(
    () =>
      "I am not a doctor. I provide general medication information only. Always consult a pharmacist or doctor for medical advice.",
    []
  );

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, isTyping]);

  useEffect(() => {
    let alive = true;

    const bootConversation = async () => {
      try {
        const created = await createAiConversation();
        const nextConversationId =
          created && typeof created === "object"
            ? normalizeConversationId((created as { id?: unknown }).id)
            : undefined;

        if (!alive || !nextConversationId) return;
        setConversationId(nextConversationId);

        const history = await getAiMessages(nextConversationId);
        if (!alive) return;

        const normalized = normalizeMessages(history);
        if (normalized.length > 0) {
          setMessages(normalized);
        }
      } catch {
        // Keep the local greeting if backend conversation creation fails.
      }
    };

    void bootConversation();
    return () => {
      alive = false;
    };
  }, []);

  const startNewChat = async () => {
    setError(null);
    setMessages([getDefaultGreeting()]);
    setConversationId(undefined);

    try {
      const created = await createAiConversation();
      const nextConversationId =
        created && typeof created === "object"
          ? normalizeConversationId((created as { id?: unknown }).id)
          : undefined;
      setConversationId(nextConversationId);
    } catch {
      setConversationId(undefined);
    }
  };

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || isTyping) return;

    const userMsg: ChatMessage = {
      id: nowId(),
      role: "user",
      content: text,
      timestamp: Date.now(),
    };

    setInput("");
    setError(null);
    setMessages((prev) => [...prev, userMsg]);
    setIsTyping(true);

    try {
      const data = await analyzeAiMessage({
        message: text,
        conversationId,
      });

      const nextConversationId =
        data && typeof data === "object"
          ? normalizeConversationId((data as { conversationId?: unknown }).conversationId)
          : undefined;
      if (nextConversationId) {
        setConversationId(nextConversationId);
      }

      const replyText =
        normalizeReply(data) ??
        "I’m having trouble answering right now. Please consult a pharmacist or doctor for urgent medication questions.";

      setMessages((prev) => [
        ...prev,
        {
          id: nowId(),
          role: "assistant",
          content: replyText,
          timestamp: Date.now(),
        },
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to contact the AI assistant.";
      setError(message);
      setMessages((prev) => [
        ...prev,
        {
          id: nowId(),
          role: "assistant",
          content:
            "I’m having trouble answering right now. Please consult a pharmacist or doctor for urgent medication questions.",
          timestamp: Date.now(),
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="min-h-[100dvh] bg-slate-100 flex justify-center">
      <div className="w-full max-w-[420px] min-h-[100dvh] bg-slate-50 flex flex-col">
        <div className="bg-blue-600 text-white px-6 pt-12 pb-10 rounded-b-[2.5rem] shadow-lg">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate(-1)}
              className="w-12 h-12 rounded-full bg-white/15 backdrop-blur flex items-center justify-center active:scale-95 transition"
              aria-label="Back"
            >
              <ArrowLeft className="w-6 h-6" />
            </button>

            <div className="min-w-0">
              <h1 className="text-2xl font-extrabold leading-tight">AI Personal Pharmacist</h1>
              <p className="text-blue-100 text-sm mt-1">Medication information &amp; safety guidance only</p>
            </div>
          </div>
        </div>

        <div ref={listRef} className="flex-1 overflow-y-auto no-scrollbar px-5 py-6 space-y-5">
          <div className="bg-blue-50 border border-blue-100/70 rounded-[1.75rem] p-5 shadow-sm flex gap-4">
            <div className="w-11 h-11 rounded-2xl bg-blue-100 text-blue-700 flex items-center justify-center shrink-0">
              <ShieldAlert className="w-6 h-6" />
            </div>
            <div>
              <div className="text-blue-900 font-extrabold text-lg">Important Notice</div>
              <p className="text-blue-800/80 text-sm leading-relaxed mt-1">{disclaimer}</p>
            </div>
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-[1.25rem] px-4 py-3 text-sm text-red-700 shadow-sm">
              {error}
            </div>
          )}

          {messages.map((m) => {
            const isUser = m.role === "user";

            if (isUser) {
              return (
                <div key={m.id} className="flex justify-end">
                  <div className="max-w-[80%] bg-blue-600 text-white rounded-[1.5rem] rounded-tr-[0.6rem] px-5 py-4 shadow-sm">
                    <p className="text-sm font-medium whitespace-pre-wrap leading-relaxed">{m.content}</p>
                  </div>
                </div>
              );
            }

            return (
              <div key={m.id} className="flex items-start gap-3">
                <div className="w-12 flex justify-center">
                  <div className="w-11 h-11 rounded-full bg-blue-600 text-white flex items-center justify-center shadow-md">
                    <Bot className="w-5 h-5" />
                  </div>
                </div>

                <div className="flex-1">
                  <div className="bg-white border border-slate-100 rounded-[1.75rem] rounded-tl-[0.6rem] px-5 py-4 shadow-sm">
                    <p className="text-sm font-medium text-slate-800 whitespace-pre-wrap leading-relaxed">
                      {m.content}
                    </p>
                  </div>
                </div>
              </div>
            );
          })}

          {isTyping && (
            <div className="flex items-start gap-3">
              <div className="w-12 flex justify-center">
                <div className="w-11 h-11 rounded-full bg-blue-600 text-white flex items-center justify-center shadow-md">
                  <Bot className="w-5 h-5" />
                </div>
              </div>
              <div className="bg-white border border-slate-100 rounded-[1.75rem] rounded-tl-[0.6rem] px-5 py-4 shadow-sm">
                <div className="flex items-center gap-2">
                  <span className="inline-flex gap-1">
                    <span className="w-2 h-2 rounded-full bg-slate-300 animate-bounce" />
                    <span
                      className="w-2 h-2 rounded-full bg-slate-300 animate-bounce"
                      style={{ animationDelay: "150ms" }}
                    />
                    <span
                      className="w-2 h-2 rounded-full bg-slate-300 animate-bounce"
                      style={{ animationDelay: "300ms" }}
                    />
                  </span>
                  <span className="text-xs text-slate-400 font-bold">Typing…</span>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="bg-white p-5 pb-safe shadow-[0_-4px_10px_rgba(0,0,0,0.02)]">
          <div className="flex items-center gap-3">
            <button
              className="bg-blue-50 p-3 rounded-full text-blue-600 active:scale-90 transition-transform"
              aria-label="New chat"
              type="button"
              onClick={() => {
                void startNewChat();
              }}
            >
              <Package className="w-6 h-6" />
            </button>

            <div className="flex-1">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                placeholder="Ask about your medication..."
                className="w-full bg-slate-50 rounded-full py-4 px-6 focus:outline-none focus:ring-2 focus:ring-blue-500/10 text-slate-800 font-medium text-sm"
              />
            </div>

            <button
              onClick={() => {
                void sendMessage();
              }}
              disabled={!input.trim() || isTyping}
              className="bg-blue-600 w-14 h-14 rounded-full text-white shadow-lg shadow-blue-500/40 active:scale-[0.95] disabled:opacity-50 transition-all flex items-center justify-center"
              aria-label="Send"
              type="button"
            >
              <Send className="w-6 h-6" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
