'use client';
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation';
import { Message, MessageAction, MessageActions, MessageContent, MessageResponse } from '@/components/ai-elements/message';
import {
  PromptInput,
  PromptInputActionMenu,
  PromptInputBody,
  type PromptInputMessage,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputFooter,
  PromptInputTools,
} from '@/components/ai-elements/prompt-input';
import { CopyIcon, RefreshCcwIcon } from 'lucide-react';
import { Fragment, useMemo, useState } from 'react';
import { Suggestion, Suggestions } from '@/components/ai-elements/suggestion';
import { Settings } from '@/components/settings';

const suggestions = [
  "Explain self-attention to a beginner in four sentences.",
  "Why is the sky blue?",
  "Give me a two-step study plan for learning transformers.",
];

export default function Chat() {
  const [input, setInput] = useState('');
  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: '/api/chat',
        body: () => ({
          apiKey: localStorage.getItem('PICOLLM_API_KEY') || undefined,
          baseUrl: localStorage.getItem('PICOLLM_BASE_URL') || undefined,
          modelId: localStorage.getItem('PICOLLM_MODEL') || undefined,
        }),
      }),
    [],
  );
  const { messages, sendMessage, regenerate, stop, status } = useChat({ transport });

  const handleSubmit = (message: PromptInputMessage) => {
    if (!('text' in message) || typeof message.text !== 'string') {
      return;
    }

    const text = message.text.trim();
    if (!text) {
      return;
    }

    sendMessage({ text });
    setInput('');
  };

  const handleSuggestionClick = (suggestion: string) => {
    sendMessage({ text: suggestion });
    setInput('');
  };

  return (
    <>
      <div className="fixed right-4 top-4 z-50">
        <Settings />
      </div>
      <div className="max-w-4xl mx-auto p-6 relative size-full h-screen">
        <div className="flex flex-col h-full">
          <Conversation>
            <ConversationContent>
              {messages.map((message, messageIndex) => (
                <Fragment key={message.id}>
                  {message.parts.map((part, i) => {
                    switch (part.type) {
                      case 'text':
                        const isLastMessage =
                          messageIndex === messages.length - 1;
                        return (
                          <Fragment key={`${message.id}-${i}`}>
                            <Message from={message.role}>
                              <MessageContent>
                                <MessageResponse>{part.text}</MessageResponse>
                              </MessageContent>
                            </Message>
                            {message.role === 'assistant' && isLastMessage && (
                              <MessageActions>
                                <MessageAction
                                  onClick={() => regenerate()}
                                  label="Retry"
                                >
                                  <RefreshCcwIcon className="size-3" />
                                </MessageAction>
                                <MessageAction
                                  onClick={() =>
                                    navigator.clipboard.writeText(part.text)
                                  }
                                  label="Copy"
                                >
                                  <CopyIcon className="size-3" />
                                </MessageAction>
                              </MessageActions>
                            )}
                          </Fragment>
                        );
                      default:
                        return null;
                    }
                  })}
                </Fragment>
              ))}
            </ConversationContent>
            <ConversationScrollButton />
          </Conversation>

          <div className="grid gap-4 mt-4">
            <Suggestions>
              {suggestions.map((suggestion) => (
                <Suggestion
                  disabled={status === 'streaming' || status === 'submitted'}
                  key={suggestion}
                  onClick={() => handleSuggestionClick(suggestion)}
                  suggestion={suggestion}
                />
              ))}
            </Suggestions>

            <PromptInput onSubmit={handleSubmit} globalDrop multiple>
              <PromptInputBody>
                <PromptInputTextarea
                  onChange={(e) => setInput(e.target.value)}
                  value={input}
                  placeholder="Ask your picoLLM model something..."
                  disabled={status === 'streaming' || status === 'submitted'}
                />
              </PromptInputBody>
              <PromptInputFooter>
                <PromptInputTools>
                  <PromptInputActionMenu>
                  </PromptInputActionMenu>
                </PromptInputTools>
                <PromptInputSubmit stop={stop} disabled={!input && !status} status={status} />
              </PromptInputFooter>

            </PromptInput>
            <p className="text-xs mx-auto font-light">
              Thank you{" "}
              <a
                href="https://ai-sdk.dev"
                target="_blank"
                rel="noopener noreferrer nofollow"
                className="text-black hover:text-green-700 transition-colors"
              >
                AI SDK
              </a>
              .
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
