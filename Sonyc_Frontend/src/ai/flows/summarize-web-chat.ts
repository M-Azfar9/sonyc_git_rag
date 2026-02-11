'use server';

/**
 * @fileOverview A flow for summarizing the content of a website. 
 *
 * - summarizeWebChat - A function that summarizes the content of a website.
 * - SummarizeWebChatInput - The input type for the summarizeWebChat function.
 * - SummarizeWebChatOutput - The return type for the summarizeWebChat function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const SummarizeWebChatInputSchema = z.object({
  websiteContent: z.string().describe('The content of the website to summarize.'),
});
export type SummarizeWebChatInput = z.infer<typeof SummarizeWebChatInputSchema>;

const SummarizeWebChatOutputSchema = z.object({
  summary: z.string().describe('A short summary of the website content.'),
});
export type SummarizeWebChatOutput = z.infer<typeof SummarizeWebChatOutputSchema>;

export async function summarizeWebChat(input: SummarizeWebChatInput): Promise<SummarizeWebChatOutput> {
  return summarizeWebChatFlow(input);
}

const summarizeWebChatPrompt = ai.definePrompt({
  name: 'summarizeWebChatPrompt',
  input: {schema: SummarizeWebChatInputSchema},
  output: {schema: SummarizeWebChatOutputSchema},
  prompt: `You are an AI assistant whose job is to summarize the content of websites.

  Please provide a concise summary of the following website content:

  {{websiteContent}}`,
});

const summarizeWebChatFlow = ai.defineFlow(
  {
    name: 'summarizeWebChatFlow',
    inputSchema: SummarizeWebChatInputSchema,
    outputSchema: SummarizeWebChatOutputSchema,
  },
  async input => {
    const {output} = await summarizeWebChatPrompt(input);
    return output!;
  }
);
