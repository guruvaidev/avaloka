import { createFileRoute, redirect } from "@tanstack/react-router";

export const Route = createFileRoute("/ray-jobs")({
  beforeLoad: () => { throw redirect({ to: "/", replace: true }); },
});