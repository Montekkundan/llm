import { streamText, UIMessage, convertToModelMessages } from 'ai';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
  const {
    messages,
    apiKey,
    baseUrl,
    modelId,
  }: {
    messages: UIMessage[];
    apiKey?: string;
    baseUrl?: string;
    modelId?: string;
  } = await req.json();

  const provider = createOpenAICompatible({
    name: 'picollm',
    baseURL: baseUrl || process.env.PICOLLM_BASE_URL || 'http://127.0.0.1:8008/v1',
    apiKey: apiKey || process.env.PICOLLM_API_KEY || 'local-demo-key',
  });

  const result = streamText({
    model: provider(modelId || process.env.PICOLLM_MODEL || 'Qwen/Qwen2.5-1.5B-Instruct'),
    messages: convertToModelMessages(messages),
    system:
      'You are a helpful assistant that can answer questions and help with tasks.',
  });

  return result.toUIMessageStreamResponse();
}
