import "./globals.css";
import AuthGate from "./AuthGate";
import { Analytics } from "@vercel/analytics/react";

export const metadata = {
  title: "Board Game Jippity",
  description: "Board Game Jippity",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
        <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap" rel="stylesheet" />
      </head>
      <body>
        <AuthGate>{children}</AuthGate>
        <Analytics />
      </body>
    </html>
  );
}


