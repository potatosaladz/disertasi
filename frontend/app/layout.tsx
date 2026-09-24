import type { Metadata } from "next";
import "@xyflow/react/dist/style.css";
import "./globals.css";
import { I18nProvider } from "./i18n";

export const metadata: Metadata = {
  title: "SHCR / Research Analytics",
  description: "Live consensus, divergence, and reproducibility dashboard",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="id">
      <body><I18nProvider>{children}</I18nProvider></body>
    </html>
  );
}
