import type { ReactNode } from "react";
import { Button } from "@/components/base/buttons/button";
import { useRayDashboardLauncher } from "@/lib/use-ray-dashboard";

export function RayJobLink({ jobId, children }: { jobId: string; children: ReactNode }) {
  const ray = useRayDashboardLauncher();
  return <>{children}{ray.allowed ? <> <Button color="link-color" size="sm" onClick={() => ray.open(jobId)}>Open in Ray dashboard</Button></> : null}{ray.dialog}</>;
}

export function RayJobText({ content }: { content: string }) {
  return <>{content.split(/(\bdta-[0-9a-f]{16}\b)/g).map((part, index) => index % 2
    ? <RayJobLink key={index} jobId={part}>{part}</RayJobLink>
    : part)}</>;
}