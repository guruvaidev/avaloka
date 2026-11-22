import { Component, type ReactNode } from "react";
import { Button } from "@/components/ui/button";

interface Props { label: string; children: ReactNode; resetKey?: unknown }
interface State { error: Error | null }

/** Contains render errors to one Inference section so the page shell keeps working. */
export class SectionBoundary extends Component<Props, State> {
  state: State = { error: null };
  static getDerivedStateFromError(error: unknown): State {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }
  componentDidCatch(error: unknown) { console.error(`[inference] ${this.props.label} failed`, error); }
  componentDidUpdate(prev: Props) { if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null }); }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <section className="inf-card" role="alert" aria-label={this.props.label}>
        <p className="text-sm font-medium">This section couldn't load</p>
        <Button variant="outline" className="mt-3 min-h-11" onClick={() => this.setState({ error: null })}>Retry</Button>
        <details className="mt-3 text-xs text-muted-foreground">
          <summary className="cursor-pointer">Details</summary>
          <pre className="mt-2 whitespace-pre-wrap break-all">{this.state.error.message}</pre>
        </details>
      </section>
    );
  }
}
