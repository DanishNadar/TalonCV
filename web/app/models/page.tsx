import { LocalModelManager } from "@/components/LocalModelManager";

export const metadata = { title: "Models" };

export default function ModelsPage() {
  return (
    <div className="shell page-stack">
      <header className="section-header">
        <div className="stack">
          <span className="eyebrow">Runtime</span>
          <h1 className="section-title">Local model manager</h1>
          <p>
            Inspect what TalonCV has cached on this device, preload models before an interview, or reclaim the space.
          </p>
        </div>
      </header>
      <LocalModelManager />
    </div>
  );
}
