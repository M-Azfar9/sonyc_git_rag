import type { Chat, Message } from "./types";

export const dummyHistory: Chat[] = [
  {
    id: "1",
    title: "React custom hooks",
    type: "Normal",
    createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
  },
  {
    id: "2",
    title: 'Summary of "Next.js in 100s"',
    type: "YouTube",
    createdAt: new Date(Date.now() - 24 * 60 * 60 * 1000), // 1 day ago
  },
  {
    id: "3",
    title: "Bugs in shadcn/ui repository",
    type: "Git",
    createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000), // 2 days ago
  },
];

export const dummyMessages: Message[] = [
  {
    id: "1",
    role: "user",
    content: "How do I create a custom hook in React?",
  },
  {
    id: "2",
    role: "assistant",
    content:
      'To create a custom hook in React, you start by creating a function whose name begins with "use". This convention is important for React to automatically check for violations of rules of hooks.',
  },
  {
    id: "3",
    role: "user",
    content: "Can you give me an example?",
  },
  {
    id: "4",
    role: "assistant",
    content:
      "Certainly! A common example is a `useLocalStorage` hook. It would look something like this:\n```javascript\nimport { useState } from 'react';\n\nfunction useLocalStorage(key, initialValue) {\n  const [storedValue, setStoredValue] = useState(() => {\n    try {\n      const item = window.localStorage.getItem(key);\n      return item ? JSON.parse(item) : initialValue;\n    } catch (error) {\n      console.log(error);\n      return initialValue;\n    }\n  });\n\n  const setValue = value => {\n    try {\n      const valueToStore = value instanceof Function ? value(storedValue) : value;\n      setStoredValue(valueToStore);\n      window.localStorage.setItem(key, JSON.stringify(valueToStore));\n    } catch (error) {\n      console.log(error);\n    }\n  };\n\n  return [storedValue, setValue];\n}\n```",
  },
];
