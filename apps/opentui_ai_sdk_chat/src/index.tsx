import { streamText } from "ai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createCliRenderer } from "@opentui/core";
import { createRoot, useKeyboard } from "@opentui/react";
import { useMemo, useState } from "react";

type ChatRole = "user" | "assistant";

type ChatTurn = {
  role: ChatRole;
  content: string;
};

const provider = createOpenAICompatible({
  name: "picollm",
  baseURL: process.env.PICOLLM_BASE_URL || "http://127.0.0.1:8008/v1",
  apiKey: process.env.PICOLLM_API_KEY || "local-demo-key",
});

const modelId = process.env.PICOLLM_MODEL || "Qwen/Qwen2.5-1.5B-Instruct";

async function generateAssistantReply(history: ChatTurn[], onChunk: (text: string) => void) {
  const result = streamText({
    model: provider(modelId),
    messages: [
      {
        role: "system",
        content:
          "You are a concise and helpful assistant inside a terminal chat application.",
      },
      ...history.map((message) => ({
        role: message.role,
        content: message.content,
      })),
    ],
  });

  for await (const chunk of result.textStream) {
    onChunk(chunk);
  }
}

function App() {
  const [messages, setMessages] = useState<ChatTurn[]>([
    {
      role: "assistant",
      content:
        "Connected to picoLLM. Ask something about transformers, tokenization, or deployment.",
    },
  ]);
  const [draft, setDraft] = useState("");
  const [status, setStatus] = useState<"idle" | "streaming" | "error">("idle");

  useKeyboard((key) => {
    if (key.name === "escape") {
      process.exit(0);
    }
  });

  const visibleMessages = useMemo(() => messages.slice(-10), [messages]);

  const submit = async (value: string) => {
    const text = value.trim();
    if (!text || status === "streaming") {
      return;
    }

    const userTurn: ChatTurn = { role: "user", content: text };
    const nextHistory = [...messages, userTurn];

    setDraft("");
    setStatus("streaming");
    setMessages([...nextHistory, { role: "assistant", content: "" }]);

    let accumulated = "";

    try {
      await generateAssistantReply(nextHistory, (chunk) => {
        accumulated += chunk;
        setMessages((current) => {
          const updated = [...current];
          updated[updated.length - 1] = { role: "assistant", content: accumulated };
          return updated;
        });
      });
      setStatus("idle");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown streaming error";
      setMessages((current) => {
        const updated = [...current];
        updated[updated.length - 1] = {
          role: "assistant",
          content: `Request failed: ${message}`,
        };
        return updated;
      });
      setStatus("error");
    }
  };

  return (
    <box
      style={{
        width: "100%",
        height: "100%",
        flexDirection: "column",
        padding: 1,
        gap: 1,
        backgroundColor: "#111111",
      }}
    >
      <box
        style={{
          border: true,
          borderStyle: "rounded",
          padding: 1,
          flexDirection: "column",
        }}
      >
        <text fg="#7dd3fc">OpenTUI + AI SDK + picoLLM</text>
        <text fg="#9ca3af">
          Backend: {process.env.PICOLLM_BASE_URL || "http://127.0.0.1:8008/v1"}
        </text>
        <text fg="#9ca3af">Model: {modelId}</text>
      </box>

      <box
        style={{
          border: true,
          borderStyle: "rounded",
          padding: 1,
          flexDirection: "column",
          gap: 1,
          flexGrow: 1,
        }}
      >
        {visibleMessages.map((message, index) => (
          <box key={`${message.role}-${index}`} style={{ flexDirection: "column" }}>
            <text fg={message.role === "assistant" ? "#86efac" : "#fca5a5"}>
              {message.role === "assistant" ? "assistant" : "user"}
            </text>
            <text>{message.content || (status === "streaming" && message.role === "assistant" ? "..." : "")}</text>
          </box>
        ))}
      </box>

      <box
        style={{
          border: true,
          borderStyle: "rounded",
          padding: 1,
          flexDirection: "column",
          gap: 1,
        }}
      >
        <text fg="#fcd34d">Prompt</text>
        <input
          value={draft}
          onInput={setDraft}
          onSubmit={submit}
          focused={status !== "streaming"}
          placeholder={status === "streaming" ? "Waiting for model..." : "Ask your picoLLM model something..."}
          width={120}
        />
        <text fg="#9ca3af">
          Enter to send. Esc to exit. This demo shows the terminal UI layer, not a full coding agent.
        </text>
      </box>
    </box>
  );
}

const renderer = await createCliRenderer({
  exitOnCtrlC: true,
});

createRoot(renderer).render(<App />);
