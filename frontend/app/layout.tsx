import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SHCR / Research Analytics",
  description: "Live consensus, divergence, and reproducibility dashboard",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
