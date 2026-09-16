import { useEffect, useMemo, useState } from "react";
import { Check, X, ChevronDown, Download, Users, MoreHorizontal, Loader2 } from "lucide-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useServerFn } from "@tanstack/react-start";
import { toast } from "sonner";
import { CircularLoaderWithLabel } from "@/components/ui/circular-loader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { validateBillingDetails, type BillingFieldErrors } from "@/lib/billing-details";
import { AutoPayConsent, FieldError } from "@/components/dashboard/AutoPayConsent";

import { supabase } from "@/integrations/supabase/client";
import { getBillingSummary, upgradeToEnterprise } from "@/lib/enterprise.functions";
import { createEnterpriseCheckoutSession, setupTrialPaymentMethod, cancelSubscriptionAtPeriodEnd, openBillingPortal } from "@/lib/billing.functions";
import { getActiveGatewayPaymentMethod, type GatewayPaymentMethod } from "@/lib/gateway-payment-method.functions";
import { createPayPalSubscription } from "@/lib/paypal.functions";
import { changePayPalPaymentMethod } from "@/lib/paypal-method.functions";
import { finalizeEnterprisePayPalSubscription } from "@/lib/paypal-finalize.functions";
import { createStripeSetupIntent, saveStripePaymentMethod } from "@/lib/stripe-cards.functions";
import { STRIPE_PUBLISHABLE_KEY } from "@/integrations/supabase/config";
import { loadStripe, type Stripe as StripeJs } from "@stripe/stripe-js";
import { Elements, CardElement, useStripe, useElements } from "@stripe/react-stripe-js";
import { useActivePlan } from "@/lib/use-active-plan";
import {
  AmexIcon,
  DinersClubIcon,
  DiscoverIcon,
  JCBIcon,
  MastercardIcon,
  PayPalIcon,
  UnionPayIcon,
  VisaIcon,
} from "@/components/foundations/payment-icons";

type Cycle = "monthly" | "annual";
type PlanId = "basic" | "pro" | "enterprise";
type Step = "billing" | "method" | "success";
type BuyStep = "billing" | "method" | "success";
type PayMethod = "stripe" | "paypal";

// PayPal checkout is available alongside Stripe.
const SHOW_PAYPAL = true;

const BILLING_SUMMARY_KEY = ["billing", "summary"] as const;
const PLANS_KEY = ["billing", "plans", "active"] as const;

type PlanType = "free" | "professional" | "enterprise";
type BillingIntervalDb = "month" | "year" | null;

export type PlanRow = {
  id: string;
  plan_type: PlanType;
  name: string;
  description: string | null;
  billing_interval: BillingIntervalDb;
  price: number;
  stripe_price_id: string | null;
  paypal_plan_id: string | null;
};

// Features are intentionally hardcoded (per requirements: monthly and yearly
// share the same feature list per plan type; only pricing changes).
const FREE_FEATURES = [
  "Basic File Management",
  "Limited Storage",
  "Community support",
  "Basic Analytics",
  "Conversational AI (limited queries)",
];
const PRO_FEATURES = [
  "Advanced file & cloud dataset management",
  "SQL warehouse access",
  "Auto Insights",
  "Conversational AI (extended usage)",
  "Inference model access",
  "Advanced analytics",
];
const ENTERPRISE_FEATURES = [
  "Unlimited users & multiple teams",
  "Enterprise-grade security & compliance",
  "Dedicated account manager",
  "Custom AI model deployment",
  "Custom integrations",
  "Advanced admin controls",
];

function formatPlanPrice(price: number): string {
  if (!Number.isFinite(price) || price <= 0) return "$0";
  const hasCents = Math.round(price * 100) % 100 !== 0;
  return `$${price.toFixed(hasCents ? 2 : 0)}`;
}

function usePlans() {
  return useQuery({
    queryKey: PLANS_KEY,
    queryFn: async (): Promise<PlanRow[]> => {
      const { data, error } = await (supabase as any)
        .from("plans")
        .select("id, plan_type, name, description, billing_interval, price, stripe_price_id, paypal_plan_id")
        .eq("is_active", true);
      if (error) throw error;
      return (data ?? []) as PlanRow[];
    },
  });
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      day: "2-digit",
      month: "short",
      year: "numeric",
    });
  } catch {
    return "—";
  }
}

export function BillingTab() {
  const queryClient = useQueryClient();
  const fetchSummary = useServerFn(getBillingSummary);
  const upgrade = useServerFn(upgradeToEnterprise);
  const createCheckout = useServerFn(createEnterpriseCheckoutSession);
  const setupPaymentMethod = useServerFn(setupTrialPaymentMethod);
  const cancelSubscription = useServerFn(cancelSubscriptionAtPeriodEnd);
  const createPayPal = useServerFn(createPayPalSubscription);
  const finalizePayPal = useServerFn(finalizeEnterprisePayPalSubscription);
  const [cancelConfirmOpen, setCancelConfirmOpen] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  const { data: summary, isLoading } = useQuery({
    queryKey: BILLING_SUMMARY_KEY,
    queryFn: () => fetchSummary(),
  });

  const { data: plans = [] } = usePlans();
  const { data: activePlan } = useActivePlan();
  const currentPlan = activePlan?.plan ?? "free";
  // Billing cycle of the active subscription ("monthly" / "yearly"), so the
  // pricing grid only marks the matching cycle as the current plan.
  const currentInterval = (activePlan?.planRow?.billing_interval ?? null) as string | null;

  const [modal, setModal] = useState<Step | null>(null);
  // "Manage" on an active subscription opens the full plan list so the user
  // can switch plans and re-subscribe.
  const [managePlansOpen, setManagePlansOpen] = useState(false);

  const [cycle, setCycle] = useState<Cycle>("monthly");
  const [selectedPlan, setSelectedPlan] = useState<PlanId>("enterprise");
  const [selectedPriceId, setSelectedPriceId] = useState<string | null>(null);

  // Buy Now (direct purchase) flow — UI only for now
  const [buyStep, setBuyStep] = useState<BuyStep | null>(null);
  const [buyPlan, setBuyPlan] = useState<PlanId>("enterprise");
  const [buyPrice, setBuyPrice] = useState<{ price: string; unit: string }>({ price: "", unit: "" });
  const [buyPriceId, setBuyPriceId] = useState<string | null>(null);
  const [buyPaypalPlanId, setBuyPaypalPlanId] = useState<string | null>(null);
  const [buyBilling, setBuyBilling] = useState<{
    company_name: string;
    billing_address: string;
    vat_number: string;
  } | null>(null);
  const [checkoutSubmitting, setCheckoutSubmitting] = useState(false);
  const subscribed = Boolean(summary?.profile?.subscription_active);
  const savedBilling = extractSavedBilling((summary as any)?.customerAccount);

  const [paypalRecoveryAttempted, setPaypalRecoveryAttempted] = useState(false);

  // Handle return from Stripe/PayPal checkout — finalize (PayPal only) then
  // poll until the webhook activates the subscription.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const billing = params.get("billing");
    const provider = params.get("provider");
    // PayPal appends the approved subscription id to the return URL.
    const paypalSubscriptionId = params.get("subscription_id") ?? undefined;
    if (!billing) return;

    // Clean the URL immediately so refreshes don't re-trigger. Preserve
    // ?tab=billing so the user stays on the correct tab.
    params.delete("billing");
    params.delete("session_id");
    params.delete("provider");
    params.delete("subscription_id");
    params.delete("ba_token");
    params.delete("token");
    const search = params.toString();
    window.history.replaceState({}, "", window.location.pathname + (search ? `?${search}` : ""));

    if (billing === "cancelled") {
      toast.info("Checkout cancelled.");
      return;
    }
    if (billing !== "success") return;

    let cancelled = false;
    const MAX_ATTEMPTS = 15;
    const INTERVAL_MS = 2000;
    const toastId = toast.loading("Finalizing your subscription…");

    (async () => {
      const runFinalize = async () => {
        try {
          const res = await finalizePayPal({
            data: { subscription_id: paypalSubscriptionId },
          });
          console.log("[billing] paypal finalize result", res);
          if (
            provider === "paypal" &&
            res?.ok &&
            String(res.status ?? "").toUpperCase() === "ACTIVE"
          ) {
            await queryClient.invalidateQueries({ queryKey: ["active-plan"] });
          }
          return res;
        } catch (err) {
          console.warn("[billing] paypal finalize failed (will retry/poll)", err);
          return null;
        }
      };

      // For PayPal, actively finalize server-side before polling — the
      // webhook may still be in flight, and this closes the gap.
      if (provider === "paypal") await runFinalize();

      for (let attempt = 0; attempt < MAX_ATTEMPTS && !cancelled; attempt++) {
        try {
          const fresh = await fetchSummary();
          if (cancelled) return;
          queryClient.setQueryData(BILLING_SUMMARY_KEY, fresh);
          if (fresh?.profile?.subscription_active) {
            await queryClient.invalidateQueries({ queryKey: ["active-plan"] });
            toast.success("Subscription activated.", { id: toastId });
            return;
          }
        } catch (err) {
          console.error("[billing] poll error", err);
        }
        // Retry the finalizer a few times — the first call can land before
        // the auth token is attached or before PayPal marks the sub approved.
        if (provider === "paypal" && !cancelled && (attempt === 1 || attempt === 4 || attempt === 9)) {
          await runFinalize();
        }
        await new Promise((r) => setTimeout(r, INTERVAL_MS));
      }
      if (!cancelled) {
        toast.error("Subscription sync is taking longer than expected. Please refresh in a moment.", {
          id: toastId,
        });
      }
    })();


    return () => {
      cancelled = true;
    };
  }, [queryClient, fetchSummary, finalizePayPal]);

  // If a previous PayPal redirect removed the URL params before the backend
  // cache caught up, retry finalization from the stashed organization id on load.
  useEffect(() => {
    if (paypalRecoveryAttempted || isLoading || subscribed) return;
    const org = summary?.organization as any;
    if (org?.payment_provider !== "paypal" || !org?.paypal_subscription_id) return;

    let cancelled = false;
    setPaypalRecoveryAttempted(true);
    (async () => {
      try {
        const res = await finalizePayPal({ data: {} });
        console.log("[billing] paypal recovery finalize result", res);
        if (cancelled || !res?.ok) return;
        const fresh = await fetchSummary();
        if (cancelled) return;
        queryClient.setQueryData(BILLING_SUMMARY_KEY, fresh);
        await queryClient.invalidateQueries({ queryKey: ["active-plan"] });
        if (fresh?.profile?.subscription_active) {
          toast.success("Subscription activated.");
        }
      } catch (err) {
        console.warn("[billing] paypal recovery finalize failed", err);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [
    fetchSummary,
    finalizePayPal,
    isLoading,
    paypalRecoveryAttempted,
    queryClient,
    subscribed,
    summary?.organization,
  ]);


  const upgradeMutation = useMutation({
    mutationFn: (input: { company_name: string; billing_address: string; vat_number: string; auto_pay_authorized: boolean }) =>
      upgrade({ data: input }),

    onSuccess: (data) => {
      queryClient.setQueryData(BILLING_SUMMARY_KEY, data);
      setModal("success");
    },
    onError: (err: unknown) => {
      const message = err instanceof Error ? err.message : "Failed to start Enterprise trial";
      toast.error(message);
    },
  });
  const startCheckout = (p: PlanId, priceId?: string | null) => {
    setSelectedPlan(p);
    setSelectedPriceId(priceId ?? null);
    if (p === "enterprise") {
      setModal("billing");
      return;
    }
    toast.info("Stripe checkout coming soon for this plan.");
  };

  const startBuyNow = (
    p: PlanId,
    price: string,
    unit: string,
    priceId?: string | null,
    paypalPlanId?: string | null,
  ) => {
    setBuyPlan(p);
    setBuyPrice({ price, unit });
    setBuyPriceId(priceId ?? null);
    setBuyPaypalPlanId(paypalPlanId ?? null);
    setBuyBilling(null);
    setBuyStep("billing");
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-24">
        <CircularLoaderWithLabel label="Loading billing..." />
      </div>
    );
  }

  return (
    <>
      {subscribed && !managePlansOpen ? (
        <ActiveSubscription
          summary={summary!}
          onUpgrade={() => setManagePlansOpen(true)}
          onCancel={() => setCancelConfirmOpen(true)}
          onPayNow={async () => {
            try {
              const interval =
                ((summary?.profile as any)?.billing_interval as "month" | "year" | null) ??
                "month";
              const res = await setupPaymentMethod({ data: { interval } });
              if (res?.url) {
                window.location.href = res.url;
              } else {
                toast.error("Could not start payment setup. Please try again.");
              }
            } catch (err) {
              const msg = err instanceof Error ? err.message : "Failed to start payment setup";
              toast.error(msg);
            }
          }}
        />
      ) : (
        <>
        {subscribed && managePlansOpen && (
          <div className="mb-4 flex items-center justify-between">
            <button
              onClick={() => setManagePlansOpen(false)}
              className="text-sm font-semibold text-[#1565EF] hover:underline"
            >
              ← Back to subscription
            </button>
            <span className="text-sm text-muted-foreground">
              Choose a plan below to switch — you'll be asked to subscribe again.
            </span>
          </div>
        )}

        <Pricing
          cycle={cycle}
          setCycle={setCycle}
          plans={plans}
          onSelect={startCheckout}
          onBuyNow={startBuyNow}
          enterpriseLoading={upgradeMutation.isPending}
          currentPlan={currentPlan}
          currentInterval={currentInterval}
        />
        </>
      )}


      {modal === "billing" && (
        <PaymentModal
          plan={selectedPlan}
          submitting={upgradeMutation.isPending}
          savedBilling={savedBilling}
          onClose={() => setModal(null)}
          onSubmit={(values) => upgradeMutation.mutate(values)}
        />
      )}
      {modal === "success" && (
        <SuccessModal serial={summary?.serialNumber ?? summary?.license?.serial_number ?? null} onClose={() => setModal(null)} />
      )}

      {cancelConfirmOpen && (
        <Modal onClose={() => (cancelling ? null : setCancelConfirmOpen(false))} maxWidth="max-w-md">
          <div className="p-6">
            <h3 className="text-lg font-bold">Cancel subscription?</h3>
            <p className="mt-3 text-sm text-muted-foreground">
              Your subscription will be scheduled for cancellation. You will keep full
              access to all features until the end of your current
              {summary?.profile?.subscription_status === "trialing" ? " trial" : " billing period"}
              {summary?.profile?.next_billing_date || summary?.profile?.trial_ends_at ? (
                <>
                  {" "}
                  (<b>{formatDate(summary?.profile?.next_billing_date ?? summary?.profile?.trial_ends_at)}</b>)
                </>
              ) : null}
              . You will not be charged again.
            </p>
            <div className="mt-6 flex justify-end gap-3">
              <Button
                variant="outline"
                onClick={() => setCancelConfirmOpen(false)}
                disabled={cancelling}
              >
                Keep subscription
              </Button>
              <Button
                onClick={async () => {
                  try {
                    setCancelling(true);
                    await cancelSubscription();
                    const fresh = await fetchSummary();
                    queryClient.setQueryData(BILLING_SUMMARY_KEY, fresh);
                    toast.success("Subscription scheduled for cancellation.");
                    setCancelConfirmOpen(false);
                  } catch (err) {
                    const msg = err instanceof Error ? err.message : "Failed to cancel subscription";
                    toast.error(msg);
                  } finally {
                    setCancelling(false);
                  }
                }}
                disabled={cancelling}
                className="bg-red-600 text-white hover:bg-red-700"
              >
                {cancelling ? <Loader2 className="h-4 w-4 animate-spin" /> : "Yes, cancel"}
              </Button>
            </div>
          </div>
        </Modal>
      )}


      {buyStep === "billing" && (
        <BuyNowBillingModal
          plan={buyPlan}
          savedBilling={savedBilling}
          price={buyPrice.price}
          unit={buyPrice.unit}
          onClose={() => setBuyStep(null)}
          onSubmit={(values) => {
            setBuyBilling(values);
            setBuyStep("method");
          }}
        />
      )}
      {buyStep === "method" && (
        <BuyNowMethodModal
          plan={buyPlan}
          price={buyPrice.price}
          unit={buyPrice.unit}
          submitting={checkoutSubmitting}
          onBack={() => setBuyStep("billing")}
          onClose={() => setBuyStep(null)}
          onConfirm={async (m) => {
            if (buyPlan !== "enterprise" && buyPlan !== "pro") {
              toast.info("Checkout isn't available for this plan.");
              return;
            }
            if (!buyBilling) {
              toast.error("Missing billing information.");
              setBuyStep("billing");
              return;
            }
            try {
              setCheckoutSubmitting(true);
              const intervalFor = cycle === "annual" ? "year" : "month";
              if (m === "paypal") {
                if (!buyPaypalPlanId) {
                  toast.error("No PayPal plan configured for this plan yet.");
                  setCheckoutSubmitting(false);
                  return;
                }
                const { approveUrl } = await createPayPal({
                  data: { ...buyBilling, paypal_plan_id: buyPaypalPlanId },
                });
                window.location.assign(approveUrl);
                return;
              }
              const { url } = await createCheckout({
                data: { ...buyBilling, interval: intervalFor, price_id: buyPriceId ?? undefined },
              });
              window.location.assign(url);

            } catch (err) {
              const message = err instanceof Error ? err.message : "Failed to start checkout";
              toast.error(message);
              setCheckoutSubmitting(false);
            }
          }}
        />
      )}
    </>
  );
}

/* ------------------------------ Pricing ------------------------------ */
function Pricing({
  cycle,
  setCycle,
  plans,
  onSelect,
  onBuyNow,
  enterpriseLoading,
  currentPlan,
  currentInterval,
}: {
  cycle: Cycle;
  setCycle: (c: Cycle) => void;
  plans: PlanRow[];
  onSelect: (p: PlanId, priceId?: string | null) => void;
  onBuyNow: (
    p: PlanId,
    price: string,
    unit: string,
    priceId?: string | null,
    paypalPlanId?: string | null,
  ) => void;
  enterpriseLoading?: boolean;
  currentPlan?: "free" | "professional" | "enterprise";
  currentInterval?: string | null;
}) {
  void enterpriseLoading;
  const [users, setUsers] = useState(20);
  const [compareTab, setCompareTab] = useState<"all" | "overview" | "analytics" | "access">("all");

  const intervalDb = cycle === "annual" ? "yearly" : "monthly";
  const matchesInterval = (val: string | null | undefined) => {
    const v = String(val ?? "").toLowerCase();
    return intervalDb === "yearly"
      ? v === "yearly" || v === "year" || v === "annual" || v === "annually"
      : v === "monthly" || v === "month";
  };
  const unitLabel = cycle === "annual" ? "per user/year" : "per user/month";
  // A paid plan is only "current" when BOTH the tier and the billing cycle
  // match the active subscription (monthly enterprise != yearly enterprise).
  const cycleMatchesCurrent = matchesInterval(currentInterval);
  const isCurrent = (tier: "free" | "professional" | "enterprise") =>
    currentPlan === tier && (tier === "free" || cycleMatchesCurrent);

  const freePlan = useMemo(() => plans.find((p) => p.plan_type === "free") ?? null, [plans]);
  const proPlan = useMemo(
    () =>
      plans.find((p) => p.plan_type === "professional" && matchesInterval(p.billing_interval as any)) ??
      plans.find((p) => p.plan_type === "professional") ??
      null,
    [plans, intervalDb],
  );
  const entPlan = useMemo(
    () =>
      plans.find((p) => p.plan_type === "enterprise" && matchesInterval(p.billing_interval as any)) ??
      plans.find((p) => p.plan_type === "enterprise") ??
      null,
    [plans, intervalDb],
  );

  return (
    <div className="flex flex-col pb-10 pt-6">
      <div className="flex flex-col items-center">
        <p className="text-sm font-semibold text-[#1565EF]">Pricing</p>
        <h2 className="mt-2 text-center text-3xl font-bold tracking-tight sm:text-4xl">Simple, transparent pricing</h2>
        <p className="mt-3 max-w-xl text-center text-sm text-muted-foreground">
          We believe Avaloka.AI should be accessible to all, no matter what your profession is.
        </p>

        {/* Cycle switch */}
        <div className="mt-6 inline-flex items-center rounded-full border border-[#1565EF]/30 bg-[#eaf1ff] p-1 dark:border-[#1565EF]/40 dark:bg-[#1565EF]/10">
          <button
            onClick={() => setCycle("monthly")}
            className={cn(
              "rounded-full px-5 py-2 text-sm font-medium transition-colors",
              cycle === "monthly" ? "bg-white text-foreground shadow dark:bg-card" : "text-foreground/70",
            )}
          >
            Monthly billing
          </button>
          <button
            onClick={() => setCycle("annual")}
            className={cn(
              "flex items-center gap-2 rounded-full px-5 py-2 text-sm font-medium transition-colors",
              cycle === "annual" ? "bg-white text-foreground shadow dark:bg-card" : "text-foreground/70",
            )}
          >
            Annual billing
            <span className="inline-flex items-center gap-1 rounded-full bg-[#dbe7ff] px-2 py-0.5 text-[11px] font-semibold text-[#1565EF] dark:bg-[#1565EF]/20 dark:text-[#8ab4ff]">
              💎 Save ₹4000
            </span>
          </button>
        </div>
      </div>

      {/* Cards */}
      <div className="mt-10 grid w-full grid-cols-1 gap-5 md:grid-cols-3">
        <PlanCard
          tone="light"
          name={freePlan?.name ?? "Free"}
          price={freePlan ? formatPlanPrice(Number(freePlan.price)) : "$0"}
          unit={unitLabel}
          users={freePlan?.description ?? "Perfect for individuals exploring AI-powered data analysis"}
          featuresHeading="Features"
          features={FREE_FEATURES}
          onClick={() => onSelect("basic", freePlan?.stripe_price_id ?? null)}
          selected={isCurrent("free")}
          primaryLabel={isCurrent("free") ? "Current plan" : "Get started"}
          disabled={isCurrent("free")}
        />
        <PlanCard
          tone="dark"
          name={proPlan?.name ?? "Professional"}
          price={proPlan ? formatPlanPrice(Number(proPlan.price)) : "—"}
          unit={unitLabel}
          users={
            proPlan?.description ??
            "For growing teams who need advanced ecommerce and reporting capabilities"
          }
          badge={cycle === "annual" ? "Save 20%" : "Popular"}
          featuresHeading="Everything in Free"
          features={PRO_FEATURES}
          onClick={() =>
            onBuyNow(
              "pro",
              proPlan ? formatPlanPrice(Number(proPlan.price)) : "—",
              unitLabel,
              proPlan?.stripe_price_id ?? null,
              proPlan?.paypal_plan_id ?? null,
            )
          }
          primaryLabel={isCurrent("professional") ? "Current plan" : "Get Started"}
          selected={isCurrent("professional")}
          disabled={isCurrent("professional")}
        />
        <PlanCard
          tone="light"
          name={entPlan?.name ?? "Enterprise"}
          price={entPlan ? formatPlanPrice(Number(entPlan.price)) : "—"}
          unit={unitLabel}
          users={entPlan?.description ?? "Built for organizations with advanced AI and security needs"}
          featuresHeading="Everything in Professional"
          features={ENTERPRISE_FEATURES}
          onClick={() =>
            onBuyNow(
              "enterprise",
              entPlan ? formatPlanPrice(Number(entPlan.price)) : "—",
              unitLabel,
              entPlan?.stripe_price_id ?? null,
              entPlan?.paypal_plan_id ?? null,
            )
          }
          primaryLabel={isCurrent("enterprise") ? "Current plan" : "Get Started"}
          selected={isCurrent("enterprise")}
          disabled={isCurrent("enterprise")}
        />
      </div>



      {/* Comparison */}
      <div className="mt-16">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Overall Comparison</h3>
          <div className="inline-flex rounded-lg border border-border bg-card p-1 text-sm">
            {[
              { id: "all", label: "View all" },
              { id: "overview", label: "Overview" },
              { id: "analytics", label: "Analytics" },
              { id: "access", label: "User Access" },
            ].map((t) => (
              <button
                key={t.id}
                onClick={() => setCompareTab(t.id as typeof compareTab)}
                className={cn(
                  "rounded-md px-3 py-1.5 font-medium transition-colors",
                  compareTab === t.id ? "bg-muted text-foreground" : "text-muted-foreground hover:text-foreground",
                )}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>
        <ComparisonTable tab={compareTab} onSelect={onSelect} onBuyNow={onBuyNow} cycle={cycle} plans={plans} />
      </div>
    </div>
  );
}

function Tick() {
  return (
    <span className="inline-flex h-5 w-5 items-center justify-center rounded-full border-2 border-emerald-500 text-emerald-500">
      <Check className="h-3 w-3" strokeWidth={3} />
    </span>
  );
}
function Dash() {
  return <span className="text-muted-foreground">—</span>;
}

function Rows({ rows }: { rows: [string, React.ReactNode, React.ReactNode, React.ReactNode][] }) {
  return (
    <div>
      {rows.map(([label, a, b, c], i) => (
        <div
          key={label}
          className={cn(
            "grid grid-cols-[1.4fr_1fr_1fr_1fr] items-center px-3 py-4 text-sm",
            i % 2 === 0 ? "bg-muted/40" : "bg-card",
          )}
        >
          <div className="flex items-center gap-1.5 text-foreground">
            {label}
            <span className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-muted-foreground/50 text-[10px] text-muted-foreground">
              ?
            </span>
          </div>
          <div className="flex justify-center text-muted-foreground">{a}</div>
          <div className="flex justify-center text-muted-foreground">{b}</div>
          <div className="flex justify-center text-muted-foreground">{c}</div>
        </div>
      ))}
    </div>
  );
}

function ComparisonTable({
  tab,
  onSelect,
  onBuyNow,
  cycle,
  plans,
}: {
  tab: "all" | "overview" | "analytics" | "access";
  onSelect: (p: PlanId) => void;
  onBuyNow: (
    p: PlanId,
    price: string,
    unit: string,
    priceId?: string | null,
    paypalPlanId?: string | null,
  ) => void;
  cycle: Cycle;
  plans: PlanRow[];
}) {
  const overviewRows: [string, React.ReactNode, React.ReactNode, React.ReactNode][] = [
    ["Basic features", <Tick />, <Tick />, <Tick />],
    ["Users", "10", "20", "Unlimited"],
    ["Individual data", "20 GB", "40 GB", "Unlimited"],
    ["Support", <Tick />, <Tick />, <Tick />],
    ["Automated workflows", <Dash />, <Tick />, <Tick />],
    ["200+ integrations", <Dash />, <Tick />, <Tick />],
  ];
  const analyticsRows: [string, React.ReactNode, React.ReactNode, React.ReactNode][] = [
    ["Analytics", "Basic", "Advanced", "Advanced"],
    ["Export reports", <Tick />, <Tick />, <Tick />],
    ["Scheduled reports", <Tick />, <Tick />, <Tick />],
    ["API Access", <Dash />, <Tick />, <Tick />],
    ["Advanced reports", <Dash />, <Tick />, <Tick />],
    ["Saved reports", <Dash />, <Tick />, <Tick />],
    ["Customer properties", <Dash />, <Dash />, <Tick />],
    ["Custom fields", <Dash />, <Dash />, <Tick />],
  ];
  const accessRows: [string, React.ReactNode, React.ReactNode, React.ReactNode][] = [
    ["SSO/SAML authentication", <Tick />, <Tick />, <Tick />],
    ["Advanced permissions", <Dash />, <Tick />, <Tick />],
    ["Audit log", <Dash />, <Dash />, <Tick />],
    ["Data history", <Dash />, <Dash />, <Tick />],
  ];
  const showO = tab === "all" || tab === "overview";
  const showA = tab === "all" || tab === "analytics";
  const showU = tab === "all" || tab === "access";
  return (
    <div className="mt-6">
      <div className="grid grid-cols-[1.4fr_1fr_1fr_1fr] items-center border-b border-border px-3 pb-4 text-base font-semibold">
        <div>Overview</div>
        <div className="text-center">
          <span className="inline-flex items-center gap-2">
            Free
            <span className="rounded-full border border-[#1565EF]/30 bg-[#eaf1ff] px-2 py-0.5 text-[11px] font-semibold text-[#1565EF] dark:bg-[#1565EF]/20 dark:text-[#8ab4ff]">
              Popular
            </span>
          </span>
        </div>
        <div className="text-center">Professional</div>
        <div className="text-center">Enterprise</div>
      </div>
      {showO && <Rows rows={overviewRows} />}
      {showA && (
        <>
          <div className="mt-6 px-3 py-3 text-sm font-semibold text-[#1565EF]">Reporting and analytics</div>
          <Rows rows={analyticsRows} />
        </>
      )}
      {showU && (
        <>
          <div className="mt-6 px-3 py-3 text-sm font-semibold text-[#1565EF]">User access</div>
          <Rows rows={accessRows} />
        </>
      )}
      <div className="mt-8 grid grid-cols-[1.4fr_1fr_1fr_1fr] gap-4">
        <div />
        <Button
          onClick={() => onSelect("basic")}
          className="h-11 bg-[#1565EF] font-semibold text-white hover:bg-[#1257d4]"
        >
          Get started
        </Button>
        <Button
          onClick={() => {
            const interval = cycle === "annual" ? "year" : "month";
            const pro =
              plans.find((p) => p.plan_type === "professional" && p.billing_interval === interval) ??
              plans.find((p) => p.plan_type === "professional");
            onBuyNow(
              "pro",
              pro ? formatPlanPrice(Number(pro.price)) : "—",
              cycle === "annual" ? "per user/year" : "per user/month",
              pro?.stripe_price_id ?? null,
              pro?.paypal_plan_id ?? null,
            );
          }}
          variant="outline"
          className="h-11 font-semibold"
        >
          Get started
        </Button>

        <Button
          onClick={() => {
            const interval = cycle === "annual" ? "year" : "month";
            const entPlan =
              plans.find((p) => p.plan_type === "enterprise" && p.billing_interval === interval) ??
              plans.find((p) => p.plan_type === "enterprise");
            onBuyNow(
              "enterprise",
              entPlan ? formatPlanPrice(Number(entPlan.price)) : "—",
              cycle === "annual" ? "per user/year" : "per user/month",
              entPlan?.stripe_price_id ?? null,
              entPlan?.paypal_plan_id ?? null,
            );
          }}
          className="h-11 bg-[#1565EF] font-semibold text-white hover:bg-[#1257d4]"
        >
          Get Started
        </Button>
      </div>
    </div>
  );
}

function PlanCard({
  tone,
  name,
  price,
  unit,
  users,
  badge,
  featuresHeading,
  features,
  onClick,
  primaryLabel = "Get started",
  secondaryLabel,
  onSecondaryClick,
  selected = false,
  disabled = false,
}: {
  tone: "light" | "dark";
  name: string;
  price: string;
  unit?: string;
  users: string;
  badge?: string;
  featuresHeading?: string;
  features: string[];
  onClick: () => void;
  primaryLabel?: string;
  secondaryLabel?: string;
  onSecondaryClick?: () => void;
  selected?: boolean;
  disabled?: boolean;
}) {
  const dark = tone === "dark";
  return (
    <div
      className={cn(
        "flex flex-col rounded-2xl border p-6 transition-shadow",
        dark ? "border-transparent bg-[#1565EF] text-white" : "border-border bg-card text-foreground",
        selected && "ring-2 ring-[#1565EF] ring-offset-2 shadow-lg",
      )}
    >
      <div className="flex items-center justify-between">
        <span className={cn("text-sm font-semibold", dark ? "text-white" : "text-foreground")}>{name}</span>
        {selected ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 px-3 py-0.5 text-xs font-semibold text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-300">
            <Check className="h-3 w-3" strokeWidth={3} /> Current plan
          </span>
        ) : (
          badge && (
            <span className="rounded-full bg-white px-3 py-0.5 text-xs font-semibold text-[#1565EF] dark:bg-card">{badge}</span>
          )
        )}
      </div>
      <div className="mt-6 flex items-end gap-2">
        <span className="text-4xl font-bold">{price}</span>
        {unit && <span className={cn("mb-1 text-sm", dark ? "text-white/80" : "text-muted-foreground")}>{unit}</span>}
      </div>
      <div className={cn("mt-3 flex items-center gap-2 text-xs", dark ? "text-white/80" : "text-muted-foreground")}>
        {users}
        <Users className="h-3.5 w-3.5" />
      </div>

      <Button
        onClick={onClick}
        disabled={disabled}
        className={cn(
          "mt-6 h-10 w-full font-semibold",
          dark ? "bg-white text-[#1565EF] hover:bg-white/90" : "bg-[#1565EF] text-white hover:bg-[#1257d4]",
          disabled && "cursor-default opacity-70 hover:bg-current",
        )}
      >
        {primaryLabel}
      </Button>
      {secondaryLabel && onSecondaryClick && (
        <Button
          onClick={onSecondaryClick}
          variant="outline"
          className={cn(
            "mt-2 h-10 w-full font-semibold",
            dark
              ? "border-white/60 bg-transparent text-white hover:bg-white/10 hover:text-white"
              : "border-[#1565EF] bg-transparent text-[#1565EF] hover:bg-[#1565EF]/5",
          )}
        >
          {secondaryLabel}
        </Button>
      )}

      <div className={cn("mt-6 border-t pt-5", dark ? "border-white/20" : "border-border")}>
        <p className={cn("text-xs font-bold tracking-wider", dark ? "text-white" : "text-foreground")}>FEATURES</p>
        {featuresHeading && (
          <p className={cn("mt-1 text-sm", dark ? "text-white/90" : "text-foreground/80")}>{featuresHeading}</p>
        )}
        <ul className="mt-4 space-y-3 text-sm">
          {features.map((f) => (
            <li key={f} className="flex items-start gap-2">
              <span
                className={cn(
                  "mt-0.5 flex h-4 w-4 items-center justify-center rounded-full",
                  dark ? "bg-white/20 text-white" : "bg-[#1565EF]/10 text-[#1565EF]",
                )}
              >
                <Check className="h-3 w-3" strokeWidth={3} />
              </span>
              <span className={cn(dark ? "text-white/90" : "text-foreground/80")}>{f}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

/* ------------------- Saved billing info (reuse / new) ------------------- */
export type SavedBilling = {
  company_name: string;
  billing_address: string;
  vat_number: string;
};

function extractSavedBilling(account: Record<string, any> | null | undefined): SavedBilling | null {
  const company = String(account?.company_name ?? "").trim();
  const address = String(account?.billing_address ?? "").trim();
  const vat = String(account?.vat_number ?? "").trim();
  if (!company || !address || !vat) return null;
  return { company_name: company, billing_address: address, vat_number: vat };
}

function BillingSourceChoice({
  saved,
  source,
  onChange,
}: {
  saved: SavedBilling;
  source: "existing" | "new";
  onChange: (s: "existing" | "new") => void;
}) {
  const option = (value: "existing" | "new", title: string, desc: React.ReactNode) => (
    <button
      key={value}
      type="button"
      onClick={() => onChange(value)}
      className={cn(
        "w-full rounded-lg border p-3 text-left transition",
        source === value ? "border-[#1565EF] bg-[#1565EF]/5" : "border-border hover:border-[#1565EF]/50",
      )}
    >
      <div className="flex items-start gap-3">
        <span
          className={cn(
            "mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border-2",
            source === value ? "border-[#1565EF]" : "border-muted-foreground/40",
          )}
        >
          {source === value && <span className="h-2 w-2 rounded-full bg-[#1565EF]" />}
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold">{title}</p>
          <div className="mt-0.5 text-xs text-muted-foreground">{desc}</div>
        </div>
      </div>
    </button>
  );

  return (
    <div className="space-y-2">
      {option(
        "existing",
        "Use existing billing info",
        <>
          <div className="truncate">{saved.company_name}</div>
          <div className="truncate">{saved.billing_address}</div>
          <div className="truncate">VAT: {saved.vat_number}</div>
        </>,
      )}
      {option("new", "Enter new billing info", "Use a different company, address or VAT number.")}
    </div>
  );
}

/* ----------------------------- Payment Modal ----------------------------- */

function PaymentModal({
  plan,
  submitting,
  savedBilling,
  onClose,
  onSubmit,
}: {
  plan: PlanId;
  submitting?: boolean;
  savedBilling?: SavedBilling | null;
  onClose: () => void;
  onSubmit: (values: {
    company_name: string;
    billing_address: string;
    vat_number: string;
    auto_pay_authorized: boolean;
  }) => void;
}) {
  const [source, setSource] = useState<"existing" | "new">(savedBilling ? "existing" : "new");
  const [company, setCompany] = useState("");
  const [address, setAddress] = useState("");
  const [vat, setVat] = useState("");
  const [autoPay, setAutoPay] = useState(false);
  const [errors, setErrors] = useState<BillingFieldErrors>({});

  const useSaved = Boolean(savedBilling) && source === "existing";

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const { values, errors: errs } = validateBillingDetails({
      company_name: useSaved ? savedBilling!.company_name : company,
      billing_address: useSaved ? savedBilling!.billing_address : address,
      vat_number: useSaved ? savedBilling!.vat_number : vat,
      auto_pay_authorized: autoPay,
    });

    setErrors(errs);
    if (Object.keys(errs).length > 0) {
      toast.error(String(Object.values(errs)[0] ?? "Please check the billing details."));
      return;
    }
    onSubmit(values);
  };


  return (
    <Modal onClose={onClose} maxWidth="max-w-[460px]">
      <form onSubmit={handleSubmit}>
        <div className="flex items-center justify-between p-5">
          <h3 className="text-lg font-semibold">Billing Info</h3>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="text-muted-foreground hover:text-foreground"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Plan summary */}
        <div className="bg-[#f6f7fb] px-5 py-3 text-sm font-semibold">
          <div className="flex items-center justify-between">
            <span>Plan</span>
          </div>
        </div>
        <div className="px-5 pt-3">
          <div className="rounded-lg border border-border p-4">
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-base font-semibold">
                    {plan === "pro" ? "Pro Plan" : plan === "enterprise" ? "Enterprise Plan" : "Basic Plan"}
                  </span>
                  <span className="rounded-full border border-[#1565EF]/30 bg-[#eaf1ff] px-2 py-0.5 text-xs font-semibold text-[#1565EF] dark:bg-[#1565EF]/20 dark:text-[#8ab4ff]">
                    Popular
                  </span>
                </div>
                <p className="mt-1 text-sm text-muted-foreground">15-day free trial. Cancel anytime.</p>
              </div>
              <div className="text-right">
                <p className="text-xl font-bold">Free</p>
                <p className="text-xs text-muted-foreground">Trial</p>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-4 px-5 py-4">
          {savedBilling && (
            <BillingSourceChoice saved={savedBilling} source={source} onChange={setSource} />
          )}
          {!useSaved && (
            <>
          <Field label="Company Name" required>
            <Input
              value={company}
              onChange={(e) => setCompany(e.target.value)}
              placeholder="Your company"
              className="h-10"
              aria-invalid={!!errors.company_name}
            />
            <FieldError message={errors.company_name} />
          </Field>
          <Field label="Billing Address" required>
            <Input
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              placeholder="Street, City, State, ZIP"
              className="h-10"
              aria-invalid={!!errors.billing_address}
            />
            <FieldError message={errors.billing_address} />
          </Field>
          <Field label="VAT Number" required>
            <Input
              value={vat}
              onChange={(e) => setVat(e.target.value)}
              placeholder="000012345"
              className="h-10"
              aria-invalid={!!errors.vat_number}
            />
            <FieldError message={errors.vat_number} />
          </Field>
            </>
          )}

          <AutoPayConsent
            checked={autoPay}
            onChange={setAutoPay}
            error={errors.auto_pay_authorized}
            disabled={submitting}
          />
        </div>


        <div className="px-5 pb-5">
          <Button
            type="submit"
            disabled={submitting}
            className="h-11 w-full bg-[#1565EF] text-base font-semibold hover:bg-[#1257d4]"
          >
            {submitting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Processing…
              </>
            ) : (
              "Proceed"
            )}
          </Button>
        </div>
      </form>
    </Modal>
  );
}

function PaymentMethodStep({ method, setMethod }: { method: "upi" | "card"; setMethod: (m: "upi" | "card") => void }) {
  return (
    <div className="space-y-3 px-5 py-4">
      <h4 className="text-base font-semibold">Payment Method</h4>
      <MethodRow
        active={method === "upi"}
        onClick={() => setMethod("upi")}
        title="UPI"
        desc="Pay by any UPI app"
        badge={<span className="text-xs font-semibold text-[#1565EF]">UPI</span>}
      />
      <MethodRow
        active={method === "card"}
        onClick={() => setMethod("card")}
        title="Credit / Debit / ATM Card"
        desc="Add and secure Cards as per RBI Guidelines"
        badge={<CardGlyph />}
      />
    </div>
  );
}

function MethodRow({
  active,
  onClick,
  title,
  desc,
  badge,
}: {
  active: boolean;
  onClick: () => void;
  title: string;
  desc: string;
  badge: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex w-full items-center gap-3 rounded-lg border p-4 text-left transition-colors",
        active ? "border-[#1565EF]" : "border-border hover:border-muted-foreground/40",
      )}
    >
      <span
        className={cn(
          "flex h-4 w-4 shrink-0 items-center justify-center rounded-full border-2",
          active ? "border-[#1565EF]" : "border-muted-foreground/40",
        )}
      >
        {active && <span className="h-1.5 w-1.5 rounded-full bg-[#1565EF]" />}
      </span>
      <span className="flex h-9 w-9 items-center justify-center rounded-md bg-[#eaf1ff]">{badge}</span>
      <span className="flex-1">
        <span className="block text-sm font-semibold">{title}</span>
        <span className="block text-xs text-muted-foreground">{desc}</span>
      </span>
      <ChevronDown className="h-4 w-4 text-muted-foreground" />
    </button>
  );
}

function CardGlyph() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-5" fill="none" stroke="#1565EF" strokeWidth="2">
      <rect x="2" y="5" width="20" height="14" rx="2" />
      <path d="M2 10h20" />
    </svg>
  );
}

function Field({ label, required, children }: { label: string; required?: boolean; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-1 block text-sm font-medium text-foreground">
        {label} {required && <span className="text-[#1565EF]">*</span>}
      </label>
      {children}
    </div>
  );
}

/* --------------------------- Success Modal --------------------------- */
function SuccessModal({ onClose, serial }: { onClose: () => void; serial?: string | null }) {
  return (
    <Modal onClose={onClose} maxWidth="max-w-[460px]">
      <div className="flex justify-end p-4">
        <button onClick={onClose} aria-label="Close" className="text-muted-foreground hover:text-foreground">
          <X className="h-5 w-5" />
        </button>
      </div>
      <div className="flex flex-col items-center px-6 pb-4 text-center">
        <div className="flex h-20 w-20 items-center justify-center rounded-full bg-[#dbe7ff]">
          <Check className="h-10 w-10 text-[#1565EF]" strokeWidth={3} />
        </div>
        <h3 className="mt-4 text-2xl font-bold">Trial Activated</h3>
        <p className="mt-2 text-sm text-muted-foreground">
          Your 15-day trial is now active. You can upgrade to a paid plan anytime.
        </p>
      </div>
      {serial && (
        <div className="border-t border-border px-6 py-4">
          <p className="text-center text-base font-semibold">License</p>
          <div className="mt-4 space-y-3 text-sm">
            <DetailRow label="Serial Number" value={serial} />
            <DetailRow label="Status" value="Active" />
            <DetailRow label="Trial length" value="15 days" />
          </div>
        </div>
      )}
      <div className="px-5 pb-5">
        <Button onClick={onClose} className="h-11 w-full bg-[#1565EF] text-base font-semibold hover:bg-[#1257d4]">
          View Subscription
        </Button>
      </div>
    </Modal>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-semibold text-foreground">{value}</span>
    </div>
  );
}

/* --------------------- Active Subscription View --------------------- */
type PaymentMethodBrand =
  | "visa"
  | "mastercard"
  | "amex"
  | "discover"
  | "unionpay"
  | "diners"
  | "jcb"
  | "unknown";

type PaymentMethodInfo = {
  brand: PaymentMethodBrand;
  last4: string;
  exp_month: number;
  exp_year: number;
  cardholder: string | null;
};

type BillingInvoice = {
  id?: string;
  provider?: string;
  invoice_number?: string | null;
  amount_cents?: number;
  currency?: string;
  status?: string;
  paid_at?: string | null;
  period_end?: string | null;
  hosted_url?: string | null;
};

type PlanChangeEntry = {
  id?: string;
  from_plan?: string | null;
  to_plan?: string | null;
  to_plan_name?: string | null;
  status?: string | null;
  payment_provider?: string | null;
  created_at?: string | null;
};


type BillingSummary = {
  profile: Record<string, any> | null;
  organization?: Record<string, any> | null;
  customerAccount: Record<string, any> | null;
  license: Record<string, any> | null;
  subscription?: Record<string, any> | null;
  paymentMethod: PaymentMethodInfo | null;
  invoices?: BillingInvoice[];
  planChanges?: PlanChangeEntry[];

  serialNumber?: string | null;
  trialDaysRemaining: number;
};


type LifecycleState = "trialing" | "active" | "past_due" | "canceled" | "expired" | "none";

function deriveLifecycle(summary: BillingSummary): LifecycleState {
  const status = String(summary.profile?.subscription_status ?? "").toLowerCase();
  const trialEnds = summary.profile?.trial_ends_at
    ? new Date(summary.profile.trial_ends_at).getTime()
    : 0;
  const active = Boolean(summary.profile?.subscription_active);
  if (status === "canceled" || status === "cancelled") return "canceled";
  if (status === "expired") return "expired";
  if (status === "past_due") return "past_due";
  if (status === "trialing" || (active && trialEnds > Date.now())) return "trialing";
  if (status === "active" || active) return "active";
  return "none";
}

function formatMoneyCents(cents: number, currency: string): string {
  const amount = (cents || 0) / 100;
  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency: currency || "USD",
      maximumFractionDigits: 2,
    }).format(amount);
  } catch {
    return `$${amount.toFixed(2)}`;
  }
}


function StatusPill({ status }: { status: string }) {
  const active = /active|trial/i.test(status);
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-semibold",
        active ? "bg-emerald-50 text-emerald-700" : "bg-muted text-muted-foreground",
      )}
    >
      {active && <Check className="h-3 w-3" />} {status}
    </span>
  );
}

const BRAND_LABEL: Record<PaymentMethodBrand, string> = {
  visa: "Visa",
  mastercard: "Mastercard",
  amex: "American Express",
  discover: "Discover",
  unionpay: "UnionPay",
  diners: "Diners Club",
  jcb: "JCB",
  unknown: "Card",
};

function BrandMark({ brand }: { brand: PaymentMethodBrand }) {
  const common = "h-7 w-11 rounded-md border border-border bg-card p-0.5 shrink-0";
  if (brand === "visa") return <VisaIcon className={common} />;
  if (brand === "mastercard") return <MastercardIcon className={common} />;
  if (brand === "amex") return <AmexIcon className={common} />;
  if (brand === "discover") return <DiscoverIcon className={common} />;
  if (brand === "unionpay") return <UnionPayIcon className={common} />;
  if (brand === "diners") return <DinersClubIcon className={common} />;
  if (brand === "jcb") return <JCBIcon className={common} />;
  return (
    <div className={cn(common, "flex items-center justify-center text-[10px] font-semibold text-muted-foreground")}>
      CARD
    </div>
  );
}

function PayPalMethodBlock({ payerId }: { payerId: string | null }) {
  const last4 = payerId ? payerId.slice(-4) : "----";
  return (
    <div className="overflow-hidden rounded-xl border border-border">
      <div className="flex items-center justify-between gap-3 p-4">
        <div className="min-w-0">
          <p className="text-sm font-semibold">PayPal</p>
          <p className="mt-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Payer •••• {last4}
          </p>
        </div>
        <PayPalIcon className="h-7 w-11 shrink-0" />
      </div>
      <div className="border-t border-border bg-muted/30 px-4 py-3">
        <p className="text-xs text-muted-foreground">Billing method</p>
        <p className="mt-0.5 text-sm font-semibold">PayPal subscription</p>
      </div>
    </div>
  );
}

function PaymentMethodCard({
  method,
  provider,
  payerId,
  paypalSubscriptionId,
  state,
}: {
  method: PaymentMethodInfo | null;
  provider: string | null;
  payerId: string | null;
  paypalSubscriptionId?: string | null;
  state?: LifecycleState;
}) {
  const [changeOpen, setChangeOpen] = useState(false);
  const fetchGatewayMethod = useServerFn(getActiveGatewayPaymentMethod);

  // Read straight from the gateway (Stripe / PayPal) for the ACTIVE
  // subscription. Fetched once on mount — never on tab focus/reconnect.
  const { data: gw, isLoading } = useQuery<GatewayPaymentMethod | null>({
    queryKey: ["gateway-payment-method"],
    queryFn: () => fetchGatewayMethod(),
    staleTime: Infinity,
    gcTime: Infinity,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
    refetchOnReconnect: false,
    retry: false,
  });

  // Fall back to the subscription summary already loaded on this page.
  const display: GatewayPaymentMethod | null = useMemo(() => {
    if (gw) return gw;
    // A PayPal mandate is a saved payment method even while the free trial
    // runs and no payment has been taken yet, so show it as soon as we know
    // the subscription exists — the payer id is optional.
    if (provider === "paypal" && SHOW_PAYPAL && (payerId || paypalSubscriptionId)) {
      return {
        provider: "paypal",
        brand: "paypal",
        last4: payerId ? payerId.slice(-4) : null,
        exp_month: null,
        exp_year: null,
        cardholder: "PayPal",
        label: null,
        subscription_status: null,
      };
    }
    if (method) {
      return {
        provider: "stripe",
        brand: method.brand,
        last4: method.last4,
        exp_month: method.exp_month,
        exp_year: method.exp_year,
        cardholder: method.cardholder,
        label: null,
        subscription_status: null,
      };
    }
    return null;
  }, [gw, provider, payerId, paypalSubscriptionId, method]);

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-base font-semibold">Payment Method</h3>
        <button
          type="button"
          onClick={() => setChangeOpen(true)}
          className="text-sm font-semibold text-[#1565EF] hover:underline"
        >
          Change
        </button>
      </div>

      {isLoading ? (
        <div className="rounded-xl border border-dashed border-border p-4 text-sm text-muted-foreground">
          Loading payment method…
        </div>
      ) : !display ? (
        <div className="rounded-xl border border-dashed border-border p-4">
          <p className="text-sm font-semibold">No payment method on file</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Click <span className="font-semibold">Change</span> to add or update the
            payment method used for your subscription.
          </p>
        </div>
      ) : (
        <PaymentMethodRowItem method={display} state={state} />
      )}

      {changeOpen && <ChangePaymentMethodModal onClose={() => setChangeOpen(false)} />}
    </div>
  );
}

function PaymentMethodRowItem({
  method,
  state,
}: {
  method: GatewayPaymentMethod;
  state?: LifecycleState;
}) {
  const brand = (method.brand ?? "unknown") as PaymentMethodBrand;
  const isPayPal = method.provider === "paypal";
  return (
    <div className="overflow-hidden rounded-xl border border-border">
      <div className="flex items-center justify-between gap-3 p-4">
        <div className="flex min-w-0 items-center gap-3">
          {isPayPal ? <PayPalIcon className="h-7 w-11 shrink-0" /> : <BrandMark brand={brand} />}
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold">
              {isPayPal
                ? method.label ?? `PayPal • Payer •••• ${method.last4 ?? "----"}`
                : `•••• •••• •••• ${method.last4 ?? "----"}`}
            </p>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {isPayPal
                ? "PayPal subscription"
                : method.exp_month && method.exp_year
                  ? `Exp ${String(method.exp_month).padStart(2, "0")}/${method.exp_year}`
                  : state === "trialing"
                    ? "Card will be charged when trial ends"
                    : "Managed via Stripe"}
              {method.cardholder ? ` • ${method.cardholder}` : ""}
            </p>
          </div>
        </div>
        <span className="inline-flex items-center gap-1 rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-xs font-semibold text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-300">
          <Check className="h-3 w-3" /> Active
        </span>
      </div>
    </div>
  );
}


// Cached Stripe.js promise — created once per browser session.
let _stripePromise: Promise<StripeJs | null> | null = null;
function getStripePromise() {
  if (!_stripePromise) _stripePromise = loadStripe(STRIPE_PUBLISHABLE_KEY);
  return _stripePromise;
}

type GatewayId = "stripe" | "paypal";
type GatewayDef = {
  id: GatewayId;
  name: string;
  description: string;
  icon: React.ReactNode;
  enabled: boolean;
  disabledHint?: string;
};

// Dynamic gateway registry — add new providers here.
const PAYMENT_GATEWAYS: GatewayDef[] = [
  {
    id: "stripe",
    name: "Stripe",
    description: "Pay by credit or debit card — Visa, Mastercard, Amex.",
    icon: <VisaIcon className="h-6 w-10" />,
    enabled: true,
  },
  {
    id: "paypal",
    name: "PayPal",
    description: "Link your PayPal account for recurring billing.",
    icon: <PayPalIcon className="h-6 w-10" />,
    enabled: true,
  },
];

function ChangePaymentMethodModal({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [selected, setSelected] = useState<GatewayId | null>(null);
  const [busy, setBusy] = useState(false);
  const openPortal = useServerFn(openBillingPortal);

  const handleSaved = () => {
    qc.invalidateQueries({ queryKey: ["gateway-payment-method"] });
    onClose();
  };

  const chooseStripe = async () => {
    setBusy(true);
    try {
      const { url } = await openPortal({} as any);
      window.location.assign(url);
    } catch (e: any) {
      toast.error(e?.message ?? "Could not open Stripe billing portal");
      setBusy(false);
    }
  };

  const activeGateway = PAYMENT_GATEWAYS.find((g) => g.id === selected) ?? null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 p-4">
      <div className="w-full max-w-md rounded-2xl border border-border bg-background shadow-xl">
        <div className="flex items-center justify-between border-b border-border px-5 py-3">
          <div className="flex items-center gap-2">
            {selected && (
              <button
                type="button"
                onClick={() => setSelected(null)}
                className="text-xs font-medium text-muted-foreground hover:text-foreground"
              >
                ← Back
              </button>
            )}
            <h3 className="text-base font-semibold">
              {activeGateway ? `Change ${activeGateway.name} method` : "Change payment method"}
            </h3>
          </div>
          <button type="button" onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="space-y-4 px-5 py-4">
          {!selected && (
            <>
              <p className="text-xs text-muted-foreground">
                Choose the gateway you want to bill your subscription through.
                Your details stay with the gateway — nothing is stored here.
              </p>

              <ul className="space-y-2">
                {PAYMENT_GATEWAYS.map((g) => (
                  <li key={g.id}>
                    <button
                      type="button"
                      onClick={() => {
                        if (!g.enabled || busy) return;
                        if (g.id === "stripe") void chooseStripe();
                        else setSelected(g.id);
                      }}
                      disabled={!g.enabled || busy}
                      className={cn(
                        "flex w-full items-center gap-3 rounded-xl border p-4 text-left transition-colors",
                        g.enabled
                          ? "border-border hover:border-[#1565EF] hover:bg-[#eaf1ff]/40"
                          : "cursor-not-allowed border-border/60 opacity-60",
                      )}
                    >
                      <span className="flex h-10 w-14 shrink-0 items-center justify-center rounded-md bg-[#eaf1ff]">
                        {g.icon}
                      </span>
                      <span className="flex-1">
                        <span className="block text-sm font-semibold">{g.name}</span>
                        <span className="block text-xs text-muted-foreground">
                          {g.enabled
                            ? g.id === "stripe"
                              ? "Update your card in the secure Stripe billing portal."
                              : g.description
                            : g.disabledHint ?? "Coming soon"}
                        </span>
                      </span>
                      {g.enabled && (
                        <span className="text-xs font-semibold text-[#1565EF]">
                          {busy && g.id === "stripe" ? "Opening…" : "Select →"}
                        </span>
                      )}
                    </button>

                  </li>
                ))}
              </ul>
            </>
          )}

          {selected === "stripe" && (
            <Elements stripe={getStripePromise()}>
              <StripeCardForm onClose={onClose} onSaved={handleSaved} />
            </Elements>
          )}

          {selected === "paypal" && (
            <PayPalMethodForm onClose={onClose} onSaved={handleSaved} />
          )}
        </div>
      </div>
    </div>
  );
}

function PayPalMethodForm({ onClose, onSaved }: { onClose: () => void; onSaved: () => void }) {
  const changePayPal = useServerFn(changePayPalPaymentMethod);
  const [busy, setBusy] = useState(false);
  const [autoPay, setAutoPay] = useState(false);
  const [consentErr, setConsentErr] = useState<string | undefined>();
  const [err, setErr] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    if (!autoPay) {
      setConsentErr("You must authorize automatic recurring payments to continue.");
      return;
    }
    setConsentErr(undefined);
    setBusy(true);
    try {
      const { approveUrl } = await changePayPal({ data: { auto_pay_authorized: true } } as any);
      toast.info("Redirecting to PayPal to approve the new payment method…");
      onSaved();
      window.location.assign(approveUrl);
    } catch (e: any) {
      const msg = e?.message ?? "Failed to start PayPal flow";
      setErr(msg);
      toast.error(msg);
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit} className="space-y-3">
      <p className="text-xs text-muted-foreground">
        We'll redirect you to PayPal to authorize the new payment method for your
        current subscription. Your saved company, address and VAT details are
        reused — no need to enter them again.
      </p>
      <AutoPayConsent
        checked={autoPay}
        onChange={(v) => {
          setAutoPay(v);
          if (v) setConsentErr(undefined);
        }}
        error={consentErr}
        disabled={busy}
      />
      {err && (
        <div className="rounded-md bg-destructive/10 px-3 py-2 text-xs text-destructive">{err}</div>
      )}

      <div className="flex justify-end gap-2 pt-1">
        <Button type="button" variant="secondary" onClick={onClose} disabled={busy}>
          Cancel
        </Button>
        <Button type="submit" disabled={busy}>
          {busy ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Redirecting…
            </>
          ) : (
            "Continue to PayPal"
          )}
        </Button>
      </div>
    </form>
  );
}


function StripeCardForm({ onClose, onSaved }: { onClose: () => void; onSaved: () => void }) {
  const stripe = useStripe();
  const elements = useElements();
  const createIntent = useServerFn(createStripeSetupIntent);
  const saveMethod = useServerFn(saveStripePaymentMethod);

  const [cardholder, setCardholder] = useState("");
  const [cardholderErr, setCardholderErr] = useState<string | undefined>();
  const [autoPay, setAutoPay] = useState(false);
  const [consentErr, setConsentErr] = useState<string | undefined>();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    const name = cardholder.trim();
    const nameErr =
      !name
        ? "Cardholder name is required."
        : name.length < 2 || !/\p{L}/u.test(name)
          ? "Enter the cardholder name exactly as printed on the card."
          : undefined;
    setCardholderErr(nameErr);
    const cErr = autoPay ? undefined : "You must authorize automatic recurring payments to continue.";
    setConsentErr(cErr);
    if (nameErr || cErr) return;
    if (!stripe || !elements) {
      setErr("Stripe is still loading — please wait a moment.");
      return;
    }
    const card = elements.getElement(CardElement);
    if (!card) {
      setErr("Card field not ready.");
      return;
    }
    setBusy(true);
    try {
      const { client_secret, customer_id } = await createIntent();
      if (!client_secret || !customer_id) throw new Error("Failed to initialize Stripe setup.");

      const result = await stripe.confirmCardSetup(client_secret, {
        payment_method: {
          card,
          billing_details: { name },
        },
      });
      if (result.error) throw new Error(result.error.message ?? "Card verification failed");
      const pmId = result.setupIntent?.payment_method;
      if (!pmId || typeof pmId !== "string") throw new Error("No payment method returned");

      await saveMethod({
        data: {
          payment_method_id: pmId,
          customer_id,
          cardholder: name,
          auto_pay_authorized: true,
        },
      });
      toast.success("Card added. Set it as active from the list to use it for future payments.");
      onSaved();
    } catch (e: any) {
      const msg = e?.message ?? "Failed to save card";
      setErr(msg);
      toast.error(msg);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit} className="space-y-3">
      <div>
        <label className="mb-1 block text-xs font-medium text-muted-foreground">
          Name on card
        </label>
        <Input
          value={cardholder}
          onChange={(e) => setCardholder(e.target.value)}
          placeholder="Full name"
          disabled={busy}
          aria-invalid={!!cardholderErr}
        />
        <FieldError message={cardholderErr} />
      </div>


      <div>
        <label className="mb-1 block text-xs font-medium text-muted-foreground">
          Card details
        </label>
        <div className="rounded-lg border border-border bg-background px-3 py-3">
          <CardElement
            options={{
              hidePostalCode: false,
              style: {
                base: {
                  fontSize: "14px",
                  color: "hsl(var(--foreground))",
                  "::placeholder": { color: "hsl(var(--muted-foreground))" },
                },
                invalid: { color: "#ef4444" },
              },
            }}
          />
        </div>
      </div>

      <AutoPayConsent
        checked={autoPay}
        onChange={(v) => {
          setAutoPay(v);
          if (v) setConsentErr(undefined);
        }}
        error={consentErr}
        disabled={busy}
      />


      {err && (
        <div className="rounded-md bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {err}
        </div>
      )}

      <div className="flex justify-end gap-2 pt-1">
        <Button type="button" variant="secondary" onClick={onClose} disabled={busy}>
          Cancel
        </Button>
        <Button type="submit" disabled={busy || !stripe}>
          {busy ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Saving…
            </>
          ) : (
            "Save card"
          )}
        </Button>
      </div>
    </form>
  );
}

function GatewayTile({
  title,
  desc,
  badge,
  onClick,
  disabled,
  loading,
}: {
  title: string;
  desc: string;
  badge: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  loading?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || loading}
      className={cn(
        "flex w-full items-center gap-3 rounded-lg border p-4 text-left transition-colors",
        disabled
          ? "cursor-not-allowed border-border/60 opacity-60"
          : "border-border hover:border-[#1565EF]",
      )}
    >
      <span className="flex h-9 w-9 items-center justify-center rounded-md bg-[#eaf1ff]">{badge}</span>
      <span className="flex-1">
        <span className="block text-sm font-semibold">{title}</span>
        <span className="block text-xs text-muted-foreground">{desc}</span>
      </span>
      {loading ? <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" /> : null}
    </button>
  );
}

function RowKV({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-sm font-semibold">{value}</p>
    </div>
  );
}

function planTierLabel(value?: string | null) {
  const v = String(value ?? "").toLowerCase();
  if (v === "enterprise") return "Enterprise";
  if (v === "professional" || v === "pro") return "Professional";
  if (!v) return "—";
  return v.charAt(0).toUpperCase() + v.slice(1);
}

function PlanChangeRows({ entries }: { entries: PlanChangeEntry[] }) {
  if (entries.length === 0) {
    return (
      <tr className="border-t border-border">
        <td colSpan={4} className="px-4 py-8 text-center text-sm text-muted-foreground">
          No plan changes yet.
        </td>
      </tr>
    );
  }
  return (
    <>
      {entries.map((e, i) => (
        <tr key={e.id ?? i} className="border-t border-border">
          <td className="px-4 py-3 font-medium">
            {planTierLabel(e.from_plan)} → {planTierLabel(e.to_plan)}
          </td>
          <td className="px-4 py-3">{e.to_plan_name ?? planTierLabel(e.to_plan)}</td>
          <td className="px-4 py-3">{formatDate(e.created_at ?? null)}</td>
          <td className="px-4 py-3 capitalize text-muted-foreground">
            {e.payment_provider ?? "—"}
          </td>
        </tr>
      ))}
    </>
  );
}

function BillingHistoryRows({

  invoices,
  isTrialing,
  planLabel,
  trialEnds,
}: {
  invoices: BillingInvoice[];
  isTrialing: boolean;
  planLabel: string;
  trialEnds: string | null;
}) {
  if (invoices.length === 0 && isTrialing) {
    return (
      <tr className="border-t border-border">
        <td className="px-4 py-3 font-medium">{planLabel}</td>
        <td className="px-4 py-3">$0.00</td>
        <td className="px-4 py-3">{formatDate(trialEnds)}</td>
        <td className="px-4 py-3">
          <span className="inline-flex items-center gap-1 rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-xs font-semibold text-amber-700">
            Trial Active
          </span>
        </td>
        <td className="px-4 py-3 text-muted-foreground"></td>
      </tr>
    );
  }
  if (invoices.length === 0) {
    return (
      <tr className="border-t border-border">
        <td colSpan={5} className="px-4 py-8 text-center text-sm text-muted-foreground">
          No invoices yet.
        </td>
      </tr>
    );
  }
  return (
    <>
      {invoices.map((inv, i) => (
        <tr key={inv.id ?? i} className="border-t border-border">
          <td className="px-4 py-3 font-medium">{inv.invoice_number ?? "Enterprise Plan"}</td>
          <td className="px-4 py-3">
            {formatMoneyCents(Number(inv.amount_cents ?? 0), String(inv.currency ?? "USD"))}
          </td>
          <td className="px-4 py-3">{formatDate(inv.paid_at ?? null)}</td>
          <td className="px-4 py-3">
            <span
              className={cn(
                "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-semibold capitalize",
                (inv.status ?? "paid") === "paid"
                  ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                  : "border-border bg-muted text-muted-foreground",
              )}
            >
              {(inv.status ?? "paid") === "paid" && <Check className="h-3 w-3" />}
              {inv.status ?? "paid"}
            </span>
          </td>
          <td className="px-4 py-3 text-muted-foreground">
            {inv.hosted_url ? (
              <a
                href={inv.hosted_url}
                target="_blank"
                rel="noreferrer"
                className="text-[#1565EF] hover:underline"
                aria-label="Download invoice"
              >
                <Download className="h-4 w-4" />
              </a>
            ) : null}
          </td>
        </tr>
      ))}
    </>
  );
}





function LifecycleBadge({ state }: { state: LifecycleState }) {
  const map: Record<LifecycleState, { label: string; className: string }> = {
    trialing: { label: "Trial", className: "bg-amber-50 text-amber-700 border-amber-200" },
    active: { label: "Active", className: "bg-emerald-50 text-emerald-700 border-emerald-200" },
    past_due: { label: "Past due", className: "bg-orange-50 text-orange-700 border-orange-200" },
    canceled: { label: "Cancelled", className: "bg-muted text-muted-foreground border-border" },
    expired: { label: "Expired", className: "bg-red-50 text-red-700 border-red-200" },
    none: { label: "No subscription", className: "bg-muted text-muted-foreground border-border" },
  };
  const meta = map[state];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold",
        meta.className,
      )}
    >
      {(state === "active" || state === "trialing") && <Check className="h-3 w-3" />}
      {meta.label}
    </span>
  );
}

function statusText(state: LifecycleState): string {
  switch (state) {
    case "trialing": return "Trialing";
    case "active": return "Active Subscription";
    case "past_due": return "Payment past due";
    case "canceled": return "Cancelled";
    case "expired": return "Expired";
    default: return "No subscription";
  }
}

function ActiveSubscription({ summary, onUpgrade, onPayNow, onCancel }: { summary: BillingSummary; onUpgrade: () => void; onPayNow?: () => void; onCancel?: () => void }) {
  const { profile, customerAccount, license, trialDaysRemaining, invoices, planChanges } =
    summary;

  const { data: activePlanInfo } = useActivePlan();
  const isEnterprisePlan = (activePlanInfo?.plan ?? "") === "enterprise";
  const subscriptionSerial = summary.serialNumber ?? null;
  const state = deriveLifecycle(summary);
  const plan = profile?.selected_plan ?? "enterprise";
  const provider = (profile?.payment_provider as string | null) ?? null;
  const interval = ((profile as any)?.billing_interval as string | null) ?? null;
  const priceCents = Number(
    (profile as any)?.unit_amount_cents ??
      profile?.monthly_price_cents ??
      0,
  );
  const currency = String(profile?.currency ?? "USD");
  const hasPrice = priceCents > 0;
  const unitLabel = interval === "year" ? "/year" : "/month";
  const monthlyPrice = hasPrice ? formatMoneyCents(priceCents, currency) : "—";
  const billingEmail =
    ((profile as any)?.billing_email as string | null) ??
    customerAccount?.billing_email ??
    (profile as any)?.company_email ??
    "";
  const trialEnds = profile?.trial_ends_at ?? null;
  const nextBilling = profile?.next_billing_date ?? null;
  const serial = subscriptionSerial ?? "—";
  const subscriptionStatus = String(summary.subscription?.status ?? "").toLowerCase();
  const licenseStatusLabel = ["active", "trialing"].includes(subscriptionStatus) ? "Active" : "Expired";

  const isTrialing = state === "trialing";
  const isActive = state === "active";
  const isCanceled = state === "canceled";
  const isExpired = state === "expired";
  const cancelScheduled = Boolean((profile as any)?.cancel_at_period_end) && !isCanceled && !isExpired;
  const canCancel = (isActive || isTrialing) && (Boolean((profile as any)?.stripe_subscription_id) || Boolean((profile as any)?.paypal_subscription_id));


  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
      {/* Left column */}
      <div className="flex flex-col gap-6">
        <div>
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-base font-semibold">Current Plan</h3>
            <button onClick={onUpgrade} className="text-sm font-semibold text-[#1565EF] hover:underline">
              Manage
            </button>
          </div>
          <div className="rounded-xl border border-border">
            <div className="flex items-start justify-between p-4">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-base font-semibold capitalize">{plan} Plan</span>
                  <LifecycleBadge state={state} />
                </div>
                <p className="mt-1 text-sm text-muted-foreground">
                  {isTrialing
                    ? `${trialDaysRemaining} day${trialDaysRemaining === 1 ? "" : "s"} left in your trial`
                    : statusText(state)}
                </p>
                <p className="mt-2 text-sm font-semibold">
                  {monthlyPrice}
                  <span className="ml-1 text-xs font-normal text-muted-foreground">{unitLabel}</span>
                </p>
              </div>
            </div>
            <div className="rounded-b-xl border-t border-border bg-muted/30 px-4 py-3 space-y-1">
              {isTrialing && (
                <>
                  <RowKV label="Trial ends" value={formatDate(trialEnds)} />
                  <p className="text-xs text-muted-foreground">
                    Your payment will be deducted after{" "}
                    <b>{trialDaysRemaining} day{trialDaysRemaining === 1 ? "" : "s"}</b>.
                  </p>
                </>
              )}
              {isActive && <RowKV label="Next billing date" value={formatDate(nextBilling)} />}
              {isCanceled && (
                <RowKV label="Access ends" value={formatDate(nextBilling ?? trialEnds)} />
              )}
              {isExpired && (
                <RowKV label="Expired on" value={formatDate(nextBilling ?? trialEnds)} />
              )}
              {state === "past_due" && (
                <RowKV label="Access" value="Paused — payment failed" />
              )}
            </div>
          </div>



          {isTrialing && (
            <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 p-3">
              {!provider ? (
                <p className="text-xs text-amber-900">
                  Your free trial ends on <b>{formatDate(trialEnds)}</b>. Set up your
                  payment method now. You won't be charged until your trial ends.
                </p>
              ) : (
                <p className="text-xs text-amber-900">
                  Your free trial ends on <b>{formatDate(trialEnds)}</b>. You will
                  automatically be charged <b>{monthlyPrice}{unitLabel}</b> unless you cancel
                  before then.
                </p>
              )}
              {onPayNow && !provider && (
                <Button
                  size="sm"
                  onClick={onPayNow}
                  className="mt-3 bg-[#1565EF] hover:bg-[#1256CC] text-white"
                >
                  Setup Payment Method
                </Button>
              )}
            </div>
          )}
          {isCanceled && (
            <p className="mt-3 rounded-md border border-border bg-muted/40 p-3 text-xs text-muted-foreground">
              Your subscription is cancelled. You retain access until{" "}
              <b>{formatDate(nextBilling ?? trialEnds)}</b>.
            </p>
          )}
          {isExpired && (
            <p className="mt-3 rounded-md border border-red-200 bg-red-50 p-3 text-xs text-red-900">
              Your subscription has expired. Renew to restore Enterprise access.
            </p>
          )}
          {state === "past_due" && (
            <div className="mt-3 rounded-md border border-red-200 bg-red-50 p-3">
              <p className="text-xs text-red-900">
                <b>Payment failed — update your payment method.</b> Your last payment
                could not be processed, so paid features are unavailable until a
                successful payment is made.
              </p>
              {onPayNow && (
                <Button
                  size="sm"
                  onClick={onPayNow}
                  className="mt-3 bg-[#1565EF] hover:bg-[#1256CC] text-white"
                >
                  Update Payment Method
                </Button>
              )}
            </div>
          )}

          {cancelScheduled && (
            <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 p-3">
              <p className="text-xs text-amber-900">
                Your subscription has been scheduled for cancellation. You will continue
                to have access until{" "}
                <b>{formatDate(nextBilling ?? trialEnds)}</b>.
              </p>
              <Button size="sm" disabled className="mt-3 opacity-70">
                Subscription Scheduled for Cancellation
              </Button>
            </div>
          )}

          {canCancel && !cancelScheduled && onCancel && (
            <div className="mt-3">
              <Button
                size="sm"
                variant="outline"
                onClick={onCancel}
                className="border-red-300 text-red-600 hover:bg-red-50 hover:text-red-700"
              >
                Cancel Subscription
              </Button>
            </div>
          )}
        </div>

        {summary.subscription && isEnterprisePlan && (
          <div>
            <h3 className="mb-3 text-base font-semibold">Enterprise License</h3>
            <div className="rounded-xl border border-border p-4">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Serial Number</p>
              <p className="mt-1 font-mono text-sm font-semibold">{serial}</p>
              <div className="mt-3 flex items-center justify-between">
                <span className="text-xs text-muted-foreground">License status</span>
                <StatusPill status={licenseStatusLabel} />
              </div>
              <div className="mt-2 flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Issued</span>
                <span className="text-xs font-semibold">{formatDate(summary.subscription?.created_at)}</span>
              </div>
              <div className="mt-2 flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Expires</span>
                <span className="text-xs font-semibold">{formatDate(summary.subscription?.current_period_end)}</span>
              </div>
            </div>
          </div>
        )}

        <PaymentMethodCard
          method={summary.paymentMethod}
          provider={provider}
          payerId={(summary.profile?.paypal_payer_id as string | null) ?? null}
          paypalSubscriptionId={
            ((summary.profile?.paypal_subscription_id as string | null) ??
              (summary.subscription?.paypal_subscription_id as string | null)) ?? null
          }
          state={state}
        />
      </div>

      {/* Right column */}
      <div className="flex flex-col gap-8 lg:border-l lg:border-border lg:pl-6">
        <div>
          <h3 className="mb-4 text-base font-semibold">Billing Info</h3>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Field label="Company Name" required>
              <Input readOnly value={customerAccount?.company_name ?? profile?.full_name ?? ""} className="h-10" />
            </Field>
            <Field label="Billing Address" required>
              <Input
                readOnly
                value={customerAccount?.billing_address ?? ""}
                placeholder="Not provided"
                className="h-10"
              />
            </Field>
            <Field label="Country">
              <div className="flex h-10 items-center rounded-md border border-input bg-background px-3 text-sm">
                <span className="mr-2 rounded bg-muted px-1.5 py-0.5 text-[10px] font-semibold uppercase text-muted-foreground">
                  {(customerAccount?.country ?? profile?.country ?? "—").toString().slice(0, 2)}
                </span>
                <span className="flex-1 truncate">{customerAccount?.country ?? profile?.country ?? "—"}</span>
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              </div>
            </Field>
            <Field label="VAT Number" required>
              <Input readOnly value={customerAccount?.vat_number ?? ""} placeholder="—" className="h-10" />
            </Field>
            <Field label="Phone number" required>
              <Input readOnly value={profile?.phone ?? ""} placeholder="—" className="h-10" />
            </Field>
            <Field label="Billing Email">
              <Input
                readOnly
                value={billingEmail}
                placeholder="—"
                className="h-10"
              />
            </Field>
          </div>
        </div>

        <div>
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-base font-semibold">Billing history</h3>
          </div>
          <div className="overflow-hidden rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead className="bg-muted/40 text-xs text-muted-foreground">
                <tr>
                  <th className="px-4 py-3 text-left font-medium">Invoice</th>
                  <th className="px-4 py-3 text-left font-medium">Amount</th>
                  <th className="px-4 py-3 text-left font-medium">Date</th>
                  <th className="px-4 py-3 text-left font-medium">Status</th>
                  <th className="w-10 px-4 py-3"></th>
                </tr>
              </thead>
              <tbody>
                <BillingHistoryRows
                  invoices={invoices ?? []}
                  isTrialing={isTrialing}
                  planLabel={`${(activePlanInfo?.plan ?? "professional").charAt(0).toUpperCase()}${(activePlanInfo?.plan ?? "professional").slice(1)} Trial`}
                  trialEnds={trialEnds}
                />
              </tbody>
            </table>
          </div>
        </div>

        <div>
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-base font-semibold">Plan change history</h3>
          </div>
          <div className="overflow-hidden rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead className="bg-muted/40 text-xs text-muted-foreground">
                <tr>
                  <th className="px-4 py-3 text-left font-medium">Change</th>
                  <th className="px-4 py-3 text-left font-medium">Plan</th>
                  <th className="px-4 py-3 text-left font-medium">Date</th>
                  <th className="px-4 py-3 text-left font-medium">Provider</th>
                </tr>
              </thead>
              <tbody>
                <PlanChangeRows entries={planChanges ?? []} />
              </tbody>
            </table>
          </div>
        </div>


      </div>
    </div>
  );
}

/* ----------------------------- Modal shell ----------------------------- */
function Modal({
  onClose,
  children,
  maxWidth = "max-w-md",
}: {
  onClose: () => void;
  children: React.ReactNode;
  maxWidth?: string;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={onClose}>
      <div
        className={cn("w-full overflow-hidden rounded-2xl bg-card text-foreground shadow-xl", maxWidth)}
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>
  );
}

// suppress unused import warnings if React tree-shake is strict
void MoreHorizontal;

/* ----------------------- Buy Now — Billing Info ----------------------- */
type BuyBillingValues = {
  company_name: string;
  billing_address: string;
  vat_number: string;
  auto_pay_authorized: boolean;
};

function planTitle(plan: PlanId): string {
  if (plan === "pro") return "Professional Plan";
  if (plan === "enterprise") return "Enterprise Plan";
  return "Basic Plan";
}

function BuyNowBillingModal({
  plan,
  price,
  unit,
  savedBilling,
  onClose,
  onSubmit,
}: {
  plan: PlanId;
  price: string;
  unit: string;
  savedBilling?: SavedBilling | null;
  onClose: () => void;
  onSubmit: (values: BuyBillingValues) => void;
}) {
  const [source, setSource] = useState<"existing" | "new">(savedBilling ? "existing" : "new");
  const [company, setCompany] = useState("");
  const [address, setAddress] = useState("");
  const [vat, setVat] = useState("");
  const [autoPay, setAutoPay] = useState(false);
  const [errors, setErrors] = useState<BillingFieldErrors>({});
  const title = planTitle(plan);
  const useSaved = Boolean(savedBilling) && source === "existing";

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const { values, errors: errs } = validateBillingDetails({
      company_name: useSaved ? savedBilling!.company_name : company,
      billing_address: useSaved ? savedBilling!.billing_address : address,
      vat_number: useSaved ? savedBilling!.vat_number : vat,
      auto_pay_authorized: autoPay,
    });

    setErrors(errs);
    if (Object.keys(errs).length > 0) {
      toast.error(String(Object.values(errs)[0] ?? "Please check the billing details."));
      return;
    }
    onSubmit(values);
  };


  return (
    <Modal onClose={onClose} maxWidth="max-w-[460px]">
      <form onSubmit={submit}>
        {/* Header */}
        <div className="flex items-center justify-between px-5 pt-5">
          <h3 className="text-lg font-bold">Payment</h3>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="text-muted-foreground hover:text-foreground"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Stepper */}
        <div className="mt-4 flex items-center justify-center gap-16 px-5 pb-4">
          <div className="flex flex-col items-center gap-1">
            <span className="flex h-5 w-5 items-center justify-center rounded-full border-2 border-[#1565EF]">
              <span className="h-2.5 w-2.5 rounded-full bg-[#1565EF]" />
            </span>
            <span className="text-xs font-semibold text-[#1565EF]">Billing Info</span>
          </div>
          <div className="flex flex-col items-center gap-1">
            <span className="h-5 w-5 rounded-full border-2 border-muted-foreground/40" />
            <span className="text-xs text-muted-foreground">Payment Method</span>
          </div>
        </div>

        {/* Plan */}
        <div className="flex items-center justify-between bg-[#f6f7fb] px-5 py-2 text-sm font-semibold">
          <span>Plan</span>
          <button type="button" onClick={onClose} className="text-sm font-semibold text-[#1565EF] underline">
            Change
          </button>
        </div>
        <div className="px-5 pt-3">
          <div className="rounded-lg border border-border p-4">
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-base font-semibold">{title}</span>
                  <span className="rounded-full border border-[#1565EF]/30 bg-[#eaf1ff] px-2 py-0.5 text-xs font-semibold text-[#1565EF] dark:bg-[#1565EF]/20 dark:text-[#8ab4ff]">
                    Popular
                  </span>
                </div>
                <p className="mt-1 text-sm text-muted-foreground">Advanced features and reporting.</p>
              </div>
              <div className="text-right">
                <p className="text-xl font-bold">{price}</p>
                {unit && <p className="text-xs text-muted-foreground">{unit}</p>}
              </div>
            </div>
          </div>
        </div>

        <div className="px-5 py-4">
          <h4 className="mb-3 text-base font-bold">Billing Info</h4>
          <div className="space-y-4">
            {savedBilling && (
              <BillingSourceChoice saved={savedBilling} source={source} onChange={setSource} />
            )}
            {!useSaved && (
              <>
            <Field label="Company Name" required>
              <Input
                value={company}
                onChange={(e) => setCompany(e.target.value)}
                placeholder="Arina"
                className="h-10"
                aria-invalid={!!errors.company_name}
              />
              <FieldError message={errors.company_name} />
            </Field>
            <Field label="Billing Address" required>
              <Input
                value={address}
                onChange={(e) => setAddress(e.target.value)}
                placeholder="Street, City, State, ZIP"
                className="h-10"
                aria-invalid={!!errors.billing_address}
              />
              <FieldError message={errors.billing_address} />
            </Field>
            <Field label="VAT Number" required>
              <Input
                value={vat}
                onChange={(e) => setVat(e.target.value)}
                placeholder="000012345"
                className="h-10"
                aria-invalid={!!errors.vat_number}
              />
              <FieldError message={errors.vat_number} />
            </Field>
              </>
            )}

            <AutoPayConsent
              checked={autoPay}
              onChange={setAutoPay}
              error={errors.auto_pay_authorized}
            />
          </div>

        </div>

        <div className="px-5 pb-5">
          <Button type="submit" className="h-11 w-full bg-[#1565EF] text-base font-semibold hover:bg-[#1257d4]">
            Proceed
          </Button>
        </div>
      </form>
    </Modal>
  );
}

/* -------------------- Buy Now — Payment Method -------------------- */
function BuyNowMethodModal({
  plan,
  price,
  unit,
  submitting,
  onBack,
  onClose,
  onConfirm,
}: {
  plan: PlanId;
  price: string;
  unit: string;
  submitting?: boolean;
  onBack: () => void;
  onClose: () => void;
  onConfirm: (method: PayMethod) => void;
}) {
  const [method, setMethod] = useState<PayMethod>("stripe");
  const title = planTitle(plan);

  return (
    <Modal onClose={onClose} maxWidth="max-w-[460px]">
      <div className="flex items-center justify-between px-5 pt-5">
        <h3 className="text-lg font-bold">Payment</h3>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          className="text-muted-foreground hover:text-foreground"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      {/* Stepper */}
      <div className="mt-4 flex items-center justify-center gap-16 px-5 pb-4">
        <button type="button" onClick={onBack} className="flex flex-col items-center gap-1">
          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-[#1565EF]">
            <Check className="h-3 w-3 text-white" strokeWidth={3} />
          </span>
          <span className="text-xs font-semibold text-[#1565EF]">Billing Info</span>
        </button>
        <div className="flex flex-col items-center gap-1">
          <span className="flex h-5 w-5 items-center justify-center rounded-full border-2 border-[#1565EF]">
            <span className="h-2.5 w-2.5 rounded-full bg-[#1565EF]" />
          </span>
          <span className="text-xs font-semibold text-[#1565EF]">Payment Method</span>
        </div>
      </div>

      {/* Plan summary */}
      <div className="bg-[#f6f7fb] px-5 py-2 text-sm font-semibold">Plan</div>
      <div className="px-5 pt-3">
        <div className="rounded-lg border border-border p-4">
          <div className="flex items-start justify-between">
            <div>
              <span className="text-base font-semibold">{title}</span>
              <p className="mt-1 text-sm text-muted-foreground">Advanced features and reporting.</p>
            </div>
            <div className="text-right">
              <p className="text-xl font-bold">{price}</p>
              {unit && <p className="text-xs text-muted-foreground">{unit}</p>}
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-3 px-5 py-4">
        <h4 className="text-base font-bold">Payment Method</h4>
        <MethodRow
          active={method === "stripe"}
          onClick={() => setMethod("stripe")}
          title="Stripe"
          desc="Credit / Debit Card via Stripe"
          badge={<CardGlyph />}
        />
        {SHOW_PAYPAL && (
          <MethodRow
            active={method === "paypal"}
            onClick={() => setMethod("paypal")}
            title="PayPal"
            desc="Pay securely with your PayPal account"
            badge={<span className="text-xs font-bold text-[#003087]">Pay<span className="text-[#009cde]">Pal</span></span>}
          />
        )}
      </div>

      <div className="flex gap-3 px-5 pb-5">
        <Button type="button" variant="outline" onClick={onBack} disabled={submitting} className="h-11 flex-1 font-semibold">
          Back
        </Button>
        <Button
          type="button"
          onClick={() => onConfirm(method)}
          disabled={submitting}
          className="h-11 flex-1 bg-[#1565EF] text-base font-semibold hover:bg-[#1257d4]"
        >
          {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : "Pay Now"}
        </Button>
      </div>
    </Modal>
  );
}

