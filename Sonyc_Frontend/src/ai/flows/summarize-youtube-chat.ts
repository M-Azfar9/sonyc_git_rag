'use server';

/**
 * @fileOverview A flow that summarizes YouTube video content when a new YouTube chat is started.
 *
 * - summarizeYouTubeChat - A function that summarizes the YouTube video content.
 * - SummarizeYouTubeChatInput - The input type for the summarizeYouTubeChat function.
 * - SummarizeYouTubeChatOutput - The return type for the summarizeYouTubeChat function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const SummarizeYouTubeChatInputSchema = z.object({
  videoContent: z.string().describe('The content of the YouTube video to summarize.'),
});
export type SummarizeYouTubeChatInput = z.infer<typeof SummarizeYouTubeChatInputSchema>;

const SummarizeYouTubeChatOutputSchema = z.object({
  summary: z.string().describe('A summary of the YouTube video content.'),
});
export type SummarizeYouTubeChatOutput = z.infer<typeof SummarizeYouTubeChatOutputSchema>;

export async function summarizeYouTubeChat(input: SummarizeYouTubeChatInput): Promise<SummarizeYouTubeChatOutput> {
  return summarizeYouTubeChatFlow(input);
}

const summarizeYouTubeChatPrompt = ai.definePrompt({
  name: 'summarizeYouTubeChatPrompt',
  input: {schema: SummarizeYouTubeChatInputSchema},
  output: {schema: SummarizeYouTubeChatOutputSchema},
  prompt: `Summarize the following YouTube video content:\n\n{{{videoContent}}}`,
});

const summarizeYouTubeChatFlow = ai.defineFlow(
  {
    name: 'summarizeYouTubeChatFlow',
    inputSchema: SummarizeYouTubeChatInputSchema,
    outputSchema: SummarizeYouTubeChatOutputSchema,
  },
  async input => {
    const {output} = await summarizeYouTubeChatPrompt(input);
    return output!;
  }
);
