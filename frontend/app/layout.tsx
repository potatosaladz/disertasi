import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SHCR / Experiment Control",
  description: "Structured consensus research control surface",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
