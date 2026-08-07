import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { AppHeader } from "@/components/AppHeader";
import { AppFooter } from "@/components/AppFooter";
import { ServiceWorkerRegistration } from "@/components/ServiceWorkerRegistration";

// Self-hosted at build time, so the static export needs no external font host
// and stays inside the `default-src 'self'` content-security policy.
const sans = Inter({ subsets: ["latin"], display: "swap", variable: "--font-talon-sans" });
const mono = JetBrains_Mono({ subsets: ["latin"], display: "swap", variable: "--font-talon-mono" });

export const metadata: Metadata = {
  title: { default: "TalonCV · \nAI Interview Practice Lab", template: "%s · TalonCV" },
  description:
    "TalonCV is a browser-local multimodal AI interview-practice coach. Speech, vocal delivery, semantic relevance, and visible delivery cues are analyzed on your own device.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`} data-scroll-behavior="smooth">
      <body>
        <a className="skip-link" href="#main-content">Skip to main content</a>
        <AppHeader />
        <ServiceWorkerRegistration />
        <main id="main-content">{children}</main>
        <AppFooter />
      </body>
    </html>
  );
}
