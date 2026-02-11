'use server';

/**
 * @fileOverview An AI agent that determines the most relevant chat type based on the user's initial message.
 *
 * - determineChatType - A function that determines the chat type.
 * - DetermineChatTypeInput - The input type for the determineChatType function.
 * - DetermineChatTypeOutput - The return type for the determineChatType function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const DetermineChatTypeInputSchema = z.object({
  message: z.string().describe('The user message to analyze.'),
});
export type DetermineChatTypeInput = z.infer<typeof DetermineChatTypeInputSchema>;

const DetermineChatTypeOutputSchema = z.object({
  chatType: z
    .enum(['Normal', 'YouTube', 'Web', 'Git', 'PDF'])
    .describe('The determined chat type based on the user message.'),
});
export type DetermineChatTypeOutput = z.infer<typeof DetermineChatTypeOutputSchema>;

export async function determineChatType(input: DetermineChatTypeInput): Promise<DetermineChatTypeOutput> {
  return determineChatTypeFlow(input);
}

const prompt = ai.definePrompt({
  name: 'determineChatTypePrompt',
  input: {schema: DetermineChatTypeInputSchema},
  output: {schema: DetermineChatTypeOutputSchema},
  prompt: `You are an AI assistant that analyzes user messages to determine the most relevant chat type.

  Based on the user's message, determine whether the user is asking a general question (Normal), asking about content on YouTube (YouTube), asking about content on a specific Website (Web), asking about code or repositories on Git (Git), or asking about content within a PDF (PDF).

  Message: {{{message}}}

  Return the chat type that is most relevant to the message provided.
  `,
});

const determineChatTypeFlow = ai.defineFlow(
  {
    name: 'determineChatTypeFlow',
    inputSchema: DetermineChatTypeInputSchema,
    outputSchema: DetermineChatTypeOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
