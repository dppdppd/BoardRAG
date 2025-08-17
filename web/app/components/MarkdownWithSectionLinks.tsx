"use client";

import React from "react";
import ReactMarkdown from "react-markdown";

type Props = {
  content: string;
  onSectionClick: (section: string) => void;
};

export default function MarkdownWithSectionLinks({ content, onSectionClick }: Props) {
  return (
    <ReactMarkdown
      {...({ urlTransform: (url: string) => url } as any)}
      components={{
        a({ href, children, ...props }: { href?: string; children?: any }) {
          if (href && typeof href === 'string' && href.startsWith('section:')) {
            const sec = decodeURIComponent(href.slice('section:'.length));
            return (
              <button
                className="btn link"
                onClick={(e) => {
                  e.preventDefault();
                  onSectionClick(sec);
                }}
                style={{
                  padding: 0,
                  background: 'none',
                  border: 'none',
                  color: 'var(--accent)',
                  textDecoration: 'underline',
                  cursor: 'pointer',
                }}
              >
                {"["}{children}{"]"}
              </button>
            );
          }
          return <a href={href as string} {...(props as any)}>{children}</a>;
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
}


