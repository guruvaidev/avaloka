import { Link } from "@tanstack/react-router";

/** Preserve plain chat text; only matching job IDs become navigation links. */
export function RayJobText({ content }: { content: string }) {
  return <>{content.split(/(\bdta-[0-9a-f]{16}\b)/g).map((part, index) => index % 2
    ? <Link key={index} to="/ray-jobs" search={{ job: part, status: "", kind: "" }} className="text-brand-secondary underline">{part}</Link>
    : part)}</>;
}