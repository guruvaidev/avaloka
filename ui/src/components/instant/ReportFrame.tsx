// The report itself: the stored HTML export inside a sandboxed frame. One
// renderer, not two (2.0-OUTCOMES.md §8, challenge 4): the UI never draws a
// figure or a chart from the spec; the export the backend verified is what the
// user sees, here and in the download.
//
// `sandbox="allow-scripts"` and nothing else (§4.7): the frame has an opaque
// origin, so no cookies, no storage and no reach into this page; the export's
// own meta CSP (connect-src 'none', scripts by hash) keeps it off the network.
// Never add `allow-same-origin`.
//
// Height: the export's init script posts {type: "avaloka-report-height", h}
// to its parent on load and on resize (render_html.py). The frame grows to
// that height, accepting the message only from its own window; until it
// arrives the frame has a fixed minimum height and scrolls inside.
import { useEffect, useRef, useState } from "react";

export const REPORT_HEIGHT_MESSAGE = "avaloka-report-height";
export const REPORT_FRAME_MIN_HEIGHT = 480;

export function ReportFrame({ html, title = "Report" }: { html: string; title?: string }) {
  const ref = useRef<HTMLIFrameElement>(null);
  const [height, setHeight] = useState(REPORT_FRAME_MIN_HEIGHT);

  useEffect(() => {
    const onMessage = (e: MessageEvent) => {
      const frame = ref.current;
      if (!frame || e.source !== frame.contentWindow) return;
      const data = e.data as { type?: unknown; h?: unknown } | null;
      if (!data || typeof data !== "object" || data.type !== REPORT_HEIGHT_MESSAGE) return;
      const h = Number(data.h);
      if (Number.isFinite(h) && h > 0) setHeight(Math.max(REPORT_FRAME_MIN_HEIGHT, Math.ceil(h)));
    };
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, []);

  return (
    <iframe
      ref={ref}
      title={title}
      sandbox="allow-scripts"
      srcDoc={html}
      referrerPolicy="no-referrer"
      data-testid="instant-report-frame"
      className="block w-full rounded-xl border border-secondary bg-white"
      style={{ height }}
    />
  );
}
