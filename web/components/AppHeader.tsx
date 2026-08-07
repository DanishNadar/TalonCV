"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LocalAIStatus, TalonLogo, TalonWordmark } from "@/components/ui/primitives";

const links = [
  { href: "/interview/new", label: "New Interview" },
  { href: "/dashboard", label: "History" },
  { href: "/models", label: "Models" },
  { href: "/about", label: "About" },
];

export function AppHeader() {
  const pathname = usePathname();
  const isActive = (href: string) => (href === "/dashboard" ? pathname === href : pathname.startsWith(href));

  return (
    <header className="site-header">
      <Link className="brand" href="/" aria-label="TalonCV home">
        <TalonLogo />
        <TalonWordmark />
      </Link>
      <nav className="site-nav" aria-label="Primary">
        {links.map((link) => (
          <Link key={link.href} href={link.href} aria-current={isActive(link.href) ? "page" : undefined}>
            {link.label}
          </Link>
        ))}
      </nav>
      <span className="spacer" />
      <LocalAIStatus variant="compact" />
    </header>
  );
}
