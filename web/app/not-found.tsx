import Link from "next/link";

export default function NotFound() {
  return <section className="shell centered-state"><h1>Interview not found</h1><p>It may have expired or been deleted.</p><Link className="button primary" href="/dashboard">Back to interviews</Link></section>;
}
