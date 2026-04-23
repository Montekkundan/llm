import { streamText } from "ai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createCliRenderer } from "@opentui/core";
import { createRoot } from "@opentui/react";
import { useState } from "react";

type ChatRole = "user" | "assistant";

type ChatTurn = {
  role: ChatRole;
  content: string;
};

type OpenTuiSubmitEvent = {};

const provider = createOpenAICompatible({
  name: "picollm",
  baseURL: process.env.PICOLLM_BASE_URL || "http://127.0.0.1:8008/v1",
  apiKey: process.env.PICOLLM_API_KEY || "local-demo-key",
});

const modelId = process.env.PICOLLM_MODEL || "picollm-chat";
async function generateAssistantReply(history: ChatTurn[], onChunk: (text: string) => void) {
  const result = streamText({
    model: provider(modelId),
    messages: history.map((message) => ({
      role: message.role,
      content: message.content,
    })),
  });

  for await (const chunk of result.textStream) {
    onChunk(chunk);
  }
}

function isErrorMessage(message: ChatTurn): boolean {
  return message.role === "assistant" && message.content.startsWith("Request failed:");
}

function PromptFrame({
  draft,
  status,
  onInput,
  onSubmit,
}: {
  draft: string;
  status: "idle" | "streaming" | "error";
  onInput: (value: string) => void;
  onSubmit: (valueOrEvent: string | OpenTuiSubmitEvent) => void;
}) {
  return (
    <box
      style={{
        width: "100%",
        border: true,
        borderStyle: "single",
        backgroundColor: "#1a1a1a",
        flexDirection: "row",
        flexShrink: 0,
      }}
    >
      <box
        style={{
          width: 1,
          backgroundColor: "#60a5fa",
          flexShrink: 0,
        }}
      />
      <box
        style={{
          flexDirection: "column",
          flexGrow: 1,
          paddingLeft: 1,
          paddingRight: 1,
          paddingTop: 1,
          paddingBottom: 1,
          gap: 0,
          minHeight: 4,
        }}
      >
        <input
          value={draft}
          onInput={onInput}
          onSubmit={onSubmit}
          focused={status !== "streaming"}
          placeholder={status === "streaming" ? "Waiting for picoLLM..." : "Ask anything..."}
          width="100%"
        />
      </box>
    </box>
  );
}

function HomeView({
  draft,
  status,
  onInput,
  onSubmit,
}: {
  draft: string;
  status: "idle" | "streaming" | "error";
  onInput: (value: string) => void;
  onSubmit: (valueOrEvent: string | OpenTuiSubmitEvent) => void;
}) {
  return (
    <box
      style={{
        width: "100%",
        height: "100%",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        backgroundColor: "#111111",
      }}
    >
      <box
        style={{
          width: "62%",
          minWidth: 44,
          maxWidth: 92,
          flexDirection: "column",
          alignItems: "center",
          gap: 0,
        }}
      >
        <ascii-font text="PICOLLM" font="block" />
        <box style={{ width: "100%", marginTop: 1 }}>
          <PromptFrame draft={draft} status={status} onInput={onInput} onSubmit={onSubmit} />
        </box>
      </box>
    </box>
  );
}

function ChatView({
  messages,
  draft,
  status,
  onInput,
  onSubmit,
}: {
  messages: ChatTurn[];
  draft: string;
  status: "idle" | "streaming" | "error";
  onInput: (value: string) => void;
  onSubmit: (valueOrEvent: string | OpenTuiSubmitEvent) => void;
}) {
  return (
    <box
      style={{
        width: "100%",
        height: "100%",
        flexDirection: "column",
        backgroundColor: "#111111",
        paddingTop: 1,
        paddingBottom: 1,
        paddingLeft: 2,
        paddingRight: 2,
        gap: 0,
      }}
    >
      <scrollbox
        style={{
          flexGrow: 1,
          stickyScroll: true,
          stickyStart: "bottom",
          scrollbarOptions: {
            showArrows: false,
          },
          viewportOptions: {
            paddingRight: 1,
            paddingBottom: 0,
          },
          contentOptions: {
            flexDirection: "column",
            gap: 1,
          },
        }}
      >
        {messages.map((message, index) => {
          if (message.role === "user") {
            return (
              <box
                key={`${message.role}-${index}`}
                style={{
                  width: "100%",
                  flexDirection: "row",
                  backgroundColor: "#181818",
                  marginBottom: 0,
                }}
              >
                <box
                  style={{
                    width: 1,
                    backgroundColor: "#60a5fa",
                    flexShrink: 0,
                  }}
                />
                <box
                  style={{
                    flexGrow: 1,
                    paddingLeft: 1,
                    paddingRight: 1,
                    paddingTop: 1,
                    paddingBottom: 1,
                  }}
                >
                  <text selectable content={message.content} />
                </box>
              </box>
            );
          }

          const errorState = isErrorMessage(message);

          return (
            <box
              key={`${message.role}-${index}`}
              style={{
                width: "100%",
                flexDirection: "column",
                gap: 0,
              }}
            >
              <box
                style={{
                  width: "100%",
                  flexDirection: "row",
                  backgroundColor: "#181818",
                }}
              >
                <box
                  style={{
                    width: 1,
                    backgroundColor: errorState ? "#fb7185" : "#6b7280",
                    flexShrink: 0,
                  }}
                />
                <box
                  style={{
                    flexGrow: 1,
                    paddingLeft: 1,
                    paddingRight: 1,
                    paddingTop: 1,
                    paddingBottom: 1,
                  }}
                >
                  <text
                    selectable
                    fg={errorState ? "#9ca3af" : "#e5e7eb"}
                    content={message.content || (status === "streaming" ? "..." : "")}
                  />
                </box>
              </box>
            </box>
          );
        })}
      </scrollbox>

      <PromptFrame draft={draft} status={status} onInput={onInput} onSubmit={onSubmit} />
    </box>
  );
}

function App() {
  const [messages, setMessages] = useState<ChatTurn[]>([]);
  const [draft, setDraft] = useState("");
  const [status, setStatus] = useState<"idle" | "streaming" | "error">("idle");

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

  function handleInputSubmit(value: string): void;
  function handleInputSubmit(event: OpenTuiSubmitEvent): void;
  function handleInputSubmit(valueOrEvent: string | OpenTuiSubmitEvent) {
    const value = typeof valueOrEvent === "string" ? valueOrEvent : draft;
    void submit(value);
  }

  const hasStarted = messages.length > 0 || status === "streaming";

  return hasStarted ? (
    <ChatView
      messages={messages}
      draft={draft}
      status={status}
      onInput={setDraft}
      onSubmit={handleInputSubmit}
    />
  ) : (
    <HomeView draft={draft} status={status} onInput={setDraft} onSubmit={handleInputSubmit} />
  );
}

const renderer = await createCliRenderer({
  exitOnCtrlC: true,
});

createRoot(renderer).render(<App />);
