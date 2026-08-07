import Link from "next/link";

export function AppFooter() {
  return (
    <footer className="site-footer">
      <div className="shell footer-inner">
        <div className="stack-4">
          <span className="footer-affiliation">Developed at Illinois Institute of Technology</span>
          <p>
            TalonCV provides AI-powered interview practice and multimodal coaching. It is not a hiring tool, personality
            assessment, lie detector, psychological evaluator, or employability classifier.
          </p>
        </div>
        <div className="stack-4">
          <span className="footer-affiliation">Runtime</span>
          <p>
            Browser-local inference · IndexedDB persistence · no account, database, or server-side analysis.{" "}
            <Link href="/about" className="text-button">
              Architecture
            </Link>
          </p>
        </div>
      </div>
    </footer>
  );
}
