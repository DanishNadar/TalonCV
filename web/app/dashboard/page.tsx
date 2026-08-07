import Link from "next/link";
import { LocalInterviewHistory } from "@/components/LocalInterviewHistory";

export default function DashboardPage() {
  return (
    <div className="shell page-stack">
      <header className="page-heading split"><div><div className="eyebrow">Browser-local history</div><h1>Your practice sessions</h1><p>Open, reanalyze, export, or remove the recordings and reports stored in this browser.</p></div><Link className="button primary" href="/interview/new">New interview</Link></header>
      <LocalInterviewHistory />
    </div>
  );
}
