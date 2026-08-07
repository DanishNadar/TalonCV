"use client";

import Link from "next/link";
export function AppHeader() {
  return (
    <header className="site-header">
      <Link className="brand" href="/" aria-label="TalonCV home">
        <span className="brand-mark" aria-hidden="true">T</span>
        <span>TalonCV</span>
      </Link>
      <nav aria-label="Primary navigation">
        <Link href="/dashboard">Interviews</Link>
        <Link href="/interview/new">New practice</Link>
        <Link href="/#privacy">Privacy</Link>
      </nav>
    </header>
  );
}
