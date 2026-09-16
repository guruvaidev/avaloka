import { createFileRoute, redirect } from "@tanstack/react-router";

export const Route = createFileRoute("/_authenticated/datasets")({
  beforeLoad: () => {
    throw redirect({ to: "/database" });
  },
});
