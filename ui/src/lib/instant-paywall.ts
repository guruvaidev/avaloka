// The paywall card's copy (/instant/:runId when the run is locked). This is
// configuration the owner edits to run the willingness-to-pay experiment
// (1.7-INSTANT-ANALYSIS.md §5, PR-18 moves it behind a per-cohort flag). No
// price appears in code or copy: the price is Checkout's, when PR-8 lands.
export interface PaywallCopy {
  title: string;
  body: string;
  /** The button label. The button is disabled until Checkout exists (PR-8). */
  cta: string;
  /** One line under the button saying why it does nothing yet. */
  note: string;
}

export const PAYWALL_COPY: PaywallCopy = {
  title: "Unlock the full report",
  body: "This is the free preview. The full report carries every figure, chart and check, the interactive export and the notebook that reproduces it.",
  cta: "Unlock",
  note: "Payments arrive with the next release.",
};
