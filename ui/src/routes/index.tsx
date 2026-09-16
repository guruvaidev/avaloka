import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useServerFn } from "@tanstack/react-start";
import { useState, useEffect } from "react";
import { ArrowLeft, ArrowRight, Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";
import { Logomark } from "@/components/brand/Logomark";



const slide1 = { url: "/assets/slide-1.png" };
const slide2 = { url: "/assets/slide-2.png" };
const slide3 = { url: "/assets/slide-3.png" };
const slide4 = { url: "/assets/slide4.png" };
const slide5 = { url: "/assets/slide5.png" };
const slide6 = { url: "/assets/slide-6.png" };
const slide7 = { url: "/assets/slide-7.png" };
import { supabase } from "@/integrations/supabase/client";
import {
  checkMyAccountActive,
  generateAdminMagicLink,
  generateSupportAdminMagicLink,
  sendLoginMagicLink,
} from "@/lib/admin-login.functions";

import { acceptAppUserInvite } from "@/lib/configurations.functions";
import { getMyActivePlan } from "@/lib/plan.functions";
import { ACTIVE_PLAN_KEY } from "@/lib/use-active-plan";
import { useQueryClient } from "@tanstack/react-query";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Log in · Avaloka AI" },
      {
        name: "description",
        content:
          "Sign in to Avaloka AI — the unified data workspace for SQL warehouses, dashboards, and AI-driven analytics.",
      },
    ],
  }),
  component: LoginPage,
});

const slides = [
  {
    image: slide1.url,
    title: "Unified Data Workspace",
    description:
      "Connect SQL warehouses, cloud datasets, and files in a single secure platform.",
  },
  {
    image: slide2.url,
    title: "AI-Driven Data Intelligence",
    description:
      "Auto-modeling, analyze datasets and generate alerts, reports, and actionable insights.",
  },
  {
    image: slide3.url,
    title: "Predictive Analytics",
    description:
      "Use built-in machine learning models to predict trends and detect anomalies before they impact business.",
  },
  {
    image: slide4.url,
    title: "Collaborative Data Analysis",
    description:
      "Share dashboards, insights, and reports with your team in real time.",
  },
  {
    image: slide5.url,
    title: "Enterprise Data Governance",
    description:
      "Centralized roles, audit trails, and lineage tracking to keep your data secure and compliant.",
  },
  {
    image: slide6.url,
    title: "AI Data Quality Detection",
    description:
      "Detect anomalies, missing values, and duplicate records before they affect decisions.",
  },
  {
    image: slide7.url,
    title: "Flexible Deployment",
    description:
      "Deploy Avaloka on the cloud, on-prem, or in a hybrid setup — your data, your rules.",
  },
];

function LoginPage() {
  const navigate = useNavigate();
  const acceptInviteFn = useServerFn(acceptAppUserInvite);
  const sendLoginMagicLinkFn = useServerFn(sendLoginMagicLink);
  const resolvePlanFn = useServerFn(getMyActivePlan);
  const checkAccountActiveFn = useServerFn(checkMyAccountActive);
  const queryClient = useQueryClient();
  const [email, setEmail] = useState("");
  const [blockedMessage, setBlockedMessage] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [index, setIndex] = useState(0);
  const [theme, setTheme] = useState<"light" | "dark">("light");

  const getPostLoginRoute = (userEmail?: string | null) =>
    userEmail?.toLowerCase() === "support@avaloka.ai"
      ? "/support-queries"
      : "/proanalysis";

  // When we arrive from a magic link the URL carries the auth hash — render a

  // redirect state immediately so the login screen never flashes.
  const [exchanging, setExchanging] = useState(
    () =>
      typeof window !== "undefined" &&
      window.location.hash.includes("access_token="),
  );

  const slide = slides[index];

  useEffect(() => {
    const stored = (typeof window !== "undefined" && localStorage.getItem("theme")) as
      | "light"
      | "dark"
      | null;
    const initial = stored ?? "light";
    setTheme(initial);
    document.documentElement.classList.toggle("dark", initial === "dark");
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.classList.toggle("dark", next === "dark");
    localStorage.setItem("theme", next);
  };

  useEffect(() => {
    const id = setInterval(() => {
      setIndex((i) => (i + 1) % slides.length);
    }, 5000);
    return () => clearInterval(id);
  }, []);


  useEffect(() => {
    // Resolve the caller's active plan and prime the React Query cache so
    // downstream screens read from cache instead of re-querying.
    const primePlan = async () => {
      try {
        const { data } = await supabase.auth.getUser();
        const uid = data.user?.id ?? null;
        const plan = await resolvePlanFn();
        queryClient.setQueryData(ACTIVE_PLAN_KEY(uid), plan);
      } catch (err) {
        console.error("Failed to resolve active plan", err);
      }
    };

    // Safety net: a deactivated user may still hold a valid link/session.
    // Returns true when the session was rejected and cleared.
    const rejectIfDeactivated = async () => {
      try {
        const res = await checkAccountActiveFn();
        if (res?.active === false) {
          await supabase.auth.signOut();
          if (typeof window !== "undefined") {
            history.replaceState(
              null,
              "",
              window.location.pathname + window.location.search,
            );
          }
          setExchanging(false);
          setBlockedMessage(
            res.message ??
              "Your account has been deactivated. Please contact your organization administrator.",
          );
          toast.error(res.message ?? "Your account has been deactivated.");
          return true;
        }
      } catch (err) {
        console.error("Failed to verify account status", err);
      }
      return false;
    };

    // If we landed here from a magic link, exchange the URL hash for a session.
    const hash = typeof window !== "undefined" ? window.location.hash : "";
    if (hash && hash.includes("access_token=")) {
      const params = new URLSearchParams(hash.replace(/^#/, ""));
      const access_token = params.get("access_token");
      const refresh_token = params.get("refresh_token");
      if (access_token && refresh_token) {
        setExchanging(true);
        supabase.auth
          .setSession({ access_token, refresh_token })
          .then(async ({ data, error }) => {
            if (error) {
              setExchanging(false);
              toast.error(error.message);
              return;
            }
            if (await rejectIfDeactivated()) return;
            try {
              await acceptInviteFn();
            } catch (err) {
              console.error("Failed to activate invited user", err);
            } finally {
              await primePlan();
              // Clear the hash so we don't re-process it
              history.replaceState(null, "", window.location.pathname + window.location.search);
              localStorage.removeItem("admin-bypass");
              navigate({
                to: getPostLoginRoute(data.user?.email),
                replace: true,
              });
            }
          });
        return;
      }
    }

    supabase.auth.getSession().then(async ({ data }) => {
      if (data.session?.access_token) {
        setExchanging(true);
        if (await rejectIfDeactivated()) return;
        acceptInviteFn()
          .catch((err) => {
            console.error("Failed to activate invited user", err);
          })
          .finally(async () => {
            await primePlan();
            localStorage.removeItem("admin-bypass");
            navigate({
              to: getPostLoginRoute(data.session?.user?.email),
              replace: true,
            });
          });
      }
    });
    const { data: sub } = supabase.auth.onAuthStateChange(async (_event, session) => {
      if (session?.access_token) {
        setExchanging(true);
        if (await rejectIfDeactivated()) return;
        acceptInviteFn().catch((err) => {
          console.error("Failed to activate invited user", err);
        });
        primePlan().finally(() => {
          localStorage.removeItem("admin-bypass");
          navigate({
            to: getPostLoginRoute(session?.user?.email),
            replace: true,
          });
        });
      }
    });
    return () => sub.subscription.unsubscribe();
  }, [acceptInviteFn, checkAccountActiveFn, navigate, queryClient, resolvePlanFn]);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) return;
    setSubmitting(true);
    setBlockedMessage(null);
    const normalized = email.trim().toLowerCase();
    try {
      await sendLoginMagicLinkFn({
        data: { email: normalized, redirectTo: window.location.origin },
      });
      toast.success("Check your email for the magic link");
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to send login link";
      if (/deactivated|organization administrator/i.test(message))
        setBlockedMessage(message);
      toast.error(message);
    } finally {
      setSubmitting(false);
    }
  };

  if (exchanging) {
    return (
      <main className="grid min-h-screen w-full place-items-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <Logomark />
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-muted border-t-[#1565ef]" />
          <p className="text-sm text-muted-foreground">Signing you in…</p>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen w-full bg-background">

      <div className="grid min-h-screen lg:h-screen lg:grid-cols-2">
        {/* Left — form */}
        <section className="relative flex flex-col px-5 py-6 sm:px-10 sm:py-8 lg:px-16 lg:h-screen lg:overflow-y-auto">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 sm:gap-3">
              <Logomark />
              <span
                className="text-2xl font-medium leading-tight tracking-tight text-foreground sm:text-[30px] sm:leading-[38px]"
                style={{ fontFamily: '"General Sans", ui-sans-serif, system-ui, sans-serif' }}
              >
                Avaloka AI
              </span>
            </div>

            <button
              type="button"
              aria-label="Toggle theme"
              onClick={toggleTheme}
              className="grid h-10 w-10 place-items-center rounded-lg border border-border bg-background text-foreground transition-colors hover:bg-accent"
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>


          <div className="flex flex-1 items-center justify-center py-10">
            <div className="w-full max-w-[420px]">
              <h1 className="text-2xl font-semibold leading-tight tracking-tight text-foreground sm:text-[30px] sm:leading-[38px]">
                Welcome
              </h1>

              {blockedMessage ? (
                <div
                  role="alert"
                  className="mt-6 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm font-medium text-destructive"
                >
                  {blockedMessage}
                </div>
              ) : null}

              <form onSubmit={onSubmit} className="mt-6 space-y-5">

                <div className="space-y-1.5">
                  <label
                    htmlFor="email"
                    className="text-sm font-medium text-[color:var(--text-secondary)]"
                  >
                    Work Email
                  </label>
                  <Input
                    id="email"
                    type="email"
                    autoComplete="email"
                    required
                    placeholder="Enter your email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="h-11 rounded-lg border-input bg-background text-base text-foreground placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-primary/30 dark:bg-input/30"
                  />
                </div>

                <Button
                  type="submit"
                  disabled={submitting}
                  className="h-11 w-full rounded-lg bg-[#1565ef] px-4 py-1.5 text-base font-semibold text-white shadow-none hover:bg-[#1257d6] focus-visible:ring-2 focus-visible:ring-[#1565ef]/40"
                >
                  {submitting ? "Sending…" : "Send Login Link"}
                </Button>

                <Button
                  type="button"
                  disabled={submitting}
                  onClick={async () => {
                    setSubmitting(true);
                    try {
                      const { token_hash, email: adminEmail } =
                        await generateAdminMagicLink();
                      setEmail(adminEmail);
                      const { error } = await supabase.auth.verifyOtp({
                        token_hash,
                        type: "recovery",
                      });
                      if (error) throw error;
                      localStorage.removeItem("admin-bypass");
                      toast.success("Signed in as admin");
                      navigate({ to: "/proanalysis" });
                    } catch (err) {
                      toast.error(
                        err instanceof Error ? err.message : "Admin login failed",
                      );
                    } finally {
                      setSubmitting(false);
                    }
                  }}
                  className="h-11 w-full rounded-lg border border-dashed border-[#1565ef] bg-transparent px-4 py-1.5 text-base font-semibold text-[#1565ef] shadow-none hover:bg-[#1565ef]/5"
                >
                  {submitting ? "Signing in…" : "Go to Admin View"}
                </Button>

                <Button
                  type="button"
                  disabled={submitting}
                  onClick={async () => {
                    setSubmitting(true);
                    try {
                      const { token_hash, email: supportEmail } =
                        await generateSupportAdminMagicLink();
                      setEmail(supportEmail);
                      const { error } = await supabase.auth.verifyOtp({
                        token_hash,
                        type: "recovery",
                      });
                      if (error) throw error;
                      localStorage.removeItem("admin-bypass");
                      toast.success("Signed in as support admin");
                      navigate({ to: "/support-queries" });
                    } catch (err) {
                      toast.error(
                        err instanceof Error
                          ? err.message
                          : "Support admin login failed",
                      );
                    } finally {
                      setSubmitting(false);
                    }
                  }}
                  className="h-11 w-full rounded-lg border border-dashed border-border bg-transparent px-4 py-1.5 text-base font-semibold text-foreground shadow-none hover:bg-muted"
                >
                  {submitting ? "Signing in…" : "Go to Support Admin"}
                </Button>



                <p className="text-center text-xs text-[color:var(--text-tertiary)]">
                  A secure login link will be sent to your inbox.
                </p>
              </form>

              <p className="mt-8 text-center text-xs leading-5 text-[color:var(--text-tertiary)]">
                By logging in, you agree to our{" "}
                <a href="#" className="font-semibold text-foreground hover:underline">
                  Terms of service
                </a>{" "}
                and{" "}
                <a href="#" className="font-semibold text-foreground hover:underline">
                  Privacy policy
                </a>
                .
              </p>
            </div>
          </div>

          <footer className="text-xs text-[color:var(--text-tertiary)]">
            © Avaloka AI 2026
          </footer>
        </section>

        {/* Right — hero carousel */}
        <section className="hidden p-4 lg:block lg:h-screen lg:p-6">
          <div className="relative h-full max-h-screen w-full overflow-hidden rounded-2xl bg-black">
            <img
              key={slide.image}
              src={slide.image}
              alt={slide.title}
              className="h-full max-h-screen w-full object-cover object-center animate-in fade-in duration-700"
            />

            {/* Bottom card */}
            <div className="absolute inset-x-0 bottom-0 flex flex-col gap-6 border-t border-white/30 bg-white/30 p-6 backdrop-blur-[12px] sm:gap-8 sm:p-8">
              <div className="flex flex-col gap-3 text-white">
                <h2 className="text-2xl font-semibold leading-tight sm:text-[30px] sm:leading-[38px]">{slide.title}</h2>
                <p className="text-base font-medium leading-relaxed sm:text-xl sm:leading-[30px]">{slide.description}</p>
              </div>
              <div className="flex justify-end">
                <div className="flex gap-4 sm:gap-8">
                  <button
                    type="button"
                    aria-label="Previous slide"
                    onClick={() => setIndex((i) => (i - 1 + slides.length) % slides.length)}
                    className="grid h-12 w-12 place-items-center rounded-full border border-white/50 text-white transition-colors hover:bg-white/10 sm:h-14 sm:w-14"
                  >
                    <ArrowLeft className="h-5 w-5 sm:h-6 sm:w-6" />
                  </button>
                  <button
                    type="button"
                    aria-label="Next slide"
                    onClick={() => setIndex((i) => (i + 1) % slides.length)}
                    className="grid h-12 w-12 place-items-center rounded-full border border-white/50 text-white transition-colors hover:bg-white/10 sm:h-14 sm:w-14"
                  >
                    <ArrowRight className="h-5 w-5 sm:h-6 sm:w-6" />
                  </button>
                </div>
              </div>
            </div>

          </div>
        </section>
      </div>
    </main>
  );
}

