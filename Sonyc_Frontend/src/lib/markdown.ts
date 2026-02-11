import { marked } from 'marked';
import katex from 'katex';

// Function to render math using KaTeX
function renderMath(source: unknown) {
    if (typeof source !== 'string') {
        return '';
    }
    try {
        const withDisplayMath = source.replace(/\$\$(.*?)\$\$/g, (_, math) => {
            return katex.renderToString(math.trim(), { displayMode: true, throwOnError: false });
        });
        const processedText = withDisplayMath.replace(/\$(.*?)\$/g, (_, math) => {
            return katex.renderToString(math.trim(), { displayMode: false, throwOnError: false });
        });
        return processedText;
    } catch (e) {
        console.error('Error rendering math:', e);
        return source; // Return original source on error
    }
}

// Custom renderer for marked
const renderer = new marked.Renderer();
const originalCodeRenderer = renderer.code;
renderer.code = ({ text, lang }: { text: string; lang?: string; escaped?: boolean }) => {
    // Ensure `text` is a string before trying to process it.
    if (typeof text !== 'string') {
        // Fallback to default rendering if text is not a string
        return `<pre><code>${String(text)}</code></pre>`;
    }
    const language = lang || 'plaintext';
    // a simple escapecode function
    function escapecode(code: string): string {
        return code.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
    const highlighted = escapecode(text);
    return `<pre><code class="language-${language}">${highlighted}</code></pre>`;
};

marked.setOptions({
    renderer,
    gfm: true,
    breaks: false,
});


export function parseMarkdown(markdown: unknown): string {
    if (typeof markdown !== 'string') {
        return '';
    }
    try {
        // First, process the entire markdown for math blocks
        const mathProcessed = renderMath(markdown);
        if (typeof mathProcessed !== 'string') {
          return '';
        }
        // Then, parse the result with marked
        const dirtyHtml = marked.parse(mathProcessed);
        if (typeof dirtyHtml !== 'string') {
          return '';
        }
        return dirtyHtml;
    } catch (e: any) {
        console.error("Error parsing markdown:", e.message);
        return `<p class="text-destructive">Error parsing content.</p>`;
    }
}
