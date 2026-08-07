import type { Metadata } from "next";
import "./globals.css";
import { AppHeader } from "@/components/AppHeader";
import { ServiceWorkerRegistration } from "@/components/ServiceWorkerRegistration";

export const metadata: Metadata = {
  title: { default: "TalonCV", template: "%s · TalonCV" },
  description: "Private, evidence-based multimodal interview-practice coaching.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <AppHeader />
        <ServiceWorkerRegistration />
        <main>{children}</main>
        <footer className="site-footer">
          <p>TalonCV provides interview-practice coaching, not candidate assessment or employment decisions.</p>
        </footer>
      </body>
    </html>
  );
}
