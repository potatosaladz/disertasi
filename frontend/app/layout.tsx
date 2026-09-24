import type { Metadata } from "next";
import "@xyflow/react/dist/style.css";
import "./globals.css";
import { I18nProvider } from "./i18n";

export const metadata: Metadata = {
  title: "SHCR / Sequential Deliberation Framework",
  description: "Sequential setup, individual argument, DDR conflict detection, simulation arbitration, and final convergence analytics",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="id">
      <body><I18nProvider>{children}</I18nProvider></body>
    </html>
  );
}
