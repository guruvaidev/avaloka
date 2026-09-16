import { useEffect, useMemo, useState } from "react";
import { ChevronDown, CheckCircle2, X, HelpCircle, PlusCircle, Clock, Check } from "lucide-react";
import { useServerFn } from "@tanstack/react-start";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { Dialog, DialogPortal, DialogOverlay } from "@/components/ui/dialog";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { BranchesIcon } from "@/components/dashboard/icons/ConfigSectionIcons";
import { cn } from "@/lib/utils";
import { upsertBranch, listAppUsers } from "@/lib/configurations.functions";

type HeadOption = { name: string; email: string; avatar?: string; profile_id?: string | null };

type Step = "form" | "confirm" | "success";

const COUNTRIES = [
  { name: "Australia", flag: "🇦🇺", tz: "Australia/Sydney" },
  { name: "United States", flag: "🇺🇸", tz: "America/New_York" },
  { name: "India", flag: "🇮🇳", tz: "Asia/Kolkata" },
  { name: "United Kingdom", flag: "🇬🇧", tz: "Europe/London" },
  { name: "Germany", flag: "🇩🇪", tz: "Europe/Berlin" },
];

const STATES_BY_COUNTRY: Record<string, string[]> = {
  Australia: ["New South Wales", "Victoria", "Queensland", "Western Australia"],
  "United States": ["California", "New York", "Texas", "Washington"],
  India: ["Maharashtra", "Karnataka", "Telangana", "Tamil Nadu"],
  "United Kingdom": ["England", "Scotland", "Wales", "Northern Ireland"],
  Germany: ["Bavaria", "Berlin", "Hesse", "Saxony"],
};

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function NewBranchModal({ open, onOpenChange }: Props) {
  const upsertBranchFn = useServerFn(upsertBranch);
  const qc = useQueryClient();

  const [step, setStep] = useState<Step>("form");
  const [name, setName] = useState("");
  const [address, setAddress] = useState("");
  const [country, setCountry] = useState<(typeof COUNTRIES)[number]>(COUNTRIES[0]);
  const [stateName, setStateName] = useState("");
  const [city, setCity] = useState("");
  const [postal, setPostal] = useState("");
  const [head, setHead] = useState<HeadOption | null>(null);
  const [headOpen, setHeadOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [countryOpen, setCountryOpen] = useState(false);
  const [stateOpen, setStateOpen] = useState(false);

  const listUsersFn = useServerFn(listAppUsers);
  const { data: appUsers } = useQuery({
    queryKey: ["config", "appUsers", "active"],
    queryFn: () => listUsersFn(),
    enabled: open,
  });

  const userOptions: HeadOption[] = useMemo(() => {
    if (!appUsers || !Array.isArray(appUsers)) return [];
    return appUsers
      .filter((u: any) => (u.status ?? "Active") === "Active")
      .map((u: any) => ({
        name: u.name,
        email: u.email,
        avatar: u.avatar ?? "",
        profile_id: u.profile_id ?? null,
      }));
  }, [appUsers]);

  useEffect(() => {
    if (!open) {
      setTimeout(() => {
        setStep("form");
        setName("");
        setAddress("");
        setCountry(COUNTRIES[0]);
        setStateName("");
        setCity("");
        setPostal("");
        setHead(null);
        setError(null);
        setSubmitting(false);
      }, 200);
    }
  }, [open]);


  useEffect(() => {
    setStateName("");
  }, [country.name]);

  const timezone = useMemo(() => country.tz, [country]);
  const canSubmit = Boolean(name && address && stateName && city && postal);

  const handleConfirm = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const location = [address, city, stateName, postal, country?.name].filter(Boolean).join(", ");
      await upsertBranchFn({
        data: {
          name,
          location,
          head_profile_id: head?.profile_id ?? null,
          employees: 0,
        },
      });
      await qc.invalidateQueries({ queryKey: ["config", "branches"] });
      setStep("success");
      setTimeout(() => onOpenChange(false), 1500);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to add branch");
      setStep("form");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {step === "form" && (
        <DialogPortal>
          <DialogOverlay />
          <DialogPrimitive.Content className="fixed left-1/2 top-1/2 z-50 max-h-[92vh] w-[760px] max-w-[94vw] -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-2xl bg-white shadow-xl duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 dark:bg-[#0f1216] dark:text-foreground">
            <div className="flex items-center justify-between border-b border-border px-5 py-4 dark:border-white/10">
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-border bg-white dark:border-white/10 dark:bg-white/5">
                  <BranchesIcon className="h-4 w-4 text-foreground" />
                </div>
                <h2 className="text-base font-semibold text-foreground">Branch</h2>
              </div>
              <DialogPrimitive.Close className="text-muted-foreground hover:text-foreground">
                <X className="h-5 w-5" />
              </DialogPrimitive.Close>
            </div>

            <div className="max-h-[68vh] space-y-4 overflow-y-auto px-5 py-5">
              {error && (
                <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                  {error}
                </div>
              )}
              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">Branch Name</label>
                <Input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Enter Branch Name"
                  className="h-10"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">Street Address</label>
                <Textarea
                  value={address}
                  onChange={(e) => setAddress(e.target.value)}
                  placeholder="Add Address"
                  className="min-h-[96px] resize-none"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">Country</label>
                  <Popover open={countryOpen} onOpenChange={setCountryOpen}>
                    <PopoverTrigger asChild>
                      <button
                        type="button"
                        className="flex h-10 w-full items-center justify-between rounded-md border border-border bg-white px-3 text-sm text-foreground dark:border-white/10 dark:bg-white/5"
                      >
                        <span className="flex items-center gap-2">
                          <span className="text-base leading-none">{country.flag}</span>
                          {country.name}
                        </span>
                        <ChevronDown className="h-4 w-4 text-muted-foreground" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-1 dark:border-white/10 dark:bg-[#0f1216]" align="start">
                      {COUNTRIES.map((c) => (
                        <button
                          key={c.name}
                          onClick={() => {
                            setCountry(c);
                            setCountryOpen(false);
                          }}
                          className="flex w-full items-center justify-between rounded-md px-2.5 py-2 text-sm hover:bg-muted dark:hover:bg-white/5"
                        >
                          <span className="flex items-center gap-2">
                            <span className="text-base leading-none">{c.flag}</span>
                            {c.name}
                          </span>
                          {country.name === c.name && <Check className="h-4 w-4 text-primary" />}
                        </button>
                      ))}
                    </PopoverContent>
                  </Popover>
                </div>

                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">State</label>
                  <Popover open={stateOpen} onOpenChange={setStateOpen}>
                    <PopoverTrigger asChild>
                      <button
                        type="button"
                        className={cn(
                          "flex h-10 w-full items-center justify-between rounded-md border border-border bg-white px-3 text-sm dark:border-white/10 dark:bg-white/5",
                          stateName ? "text-foreground" : "text-muted-foreground",
                        )}
                      >
                        {stateName || "Select State"}
                        <ChevronDown className="h-4 w-4 text-muted-foreground" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-1 dark:border-white/10 dark:bg-[#0f1216]" align="start">
                      {(STATES_BY_COUNTRY[country.name] ?? []).map((s) => (
                        <button
                          key={s}
                          onClick={() => {
                            setStateName(s);
                            setStateOpen(false);
                          }}
                          className="flex w-full items-center justify-between rounded-md px-2.5 py-2 text-sm hover:bg-muted dark:hover:bg-white/5"
                        >
                          {s}
                          {stateName === s && <Check className="h-4 w-4 text-primary" />}
                        </button>
                      ))}
                    </PopoverContent>
                  </Popover>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">City</label>
                  <Input
                    value={city}
                    onChange={(e) => setCity(e.target.value)}
                    placeholder="Enter City Name"
                    className="h-10"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">Postal/Zip Code</label>
                  <Input
                    value={postal}
                    onChange={(e) => setPostal(e.target.value)}
                    placeholder="Eg:500051"
                    className="h-10"
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">Time Zone</label>
                <div className="relative">
                  <Clock className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={timezone}
                    readOnly
                    placeholder="Auto-generated"
                    className="h-10 cursor-default bg-muted/40 pl-9 text-muted-foreground dark:bg-white/5"
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">
                  Branch Head <span className="font-normal text-muted-foreground">(Optional)</span>
                </label>
                <Popover open={headOpen} onOpenChange={setHeadOpen}>
                  <PopoverTrigger asChild>
                    <button
                      type="button"
                      className={cn(
                        "flex h-10 w-full items-center justify-between rounded-md border border-border bg-white px-3 text-sm dark:border-white/10 dark:bg-white/5",
                        head ? "text-foreground" : "text-muted-foreground",
                      )}
                    >
                      {head ? (
                        <span className="flex items-center gap-2">
                          <Avatar className="h-6 w-6">
                            <AvatarImage src={head.avatar} alt={head.name} />
                            <AvatarFallback>{head.name?.[0]}</AvatarFallback>
                          </Avatar>
                          {head.name}
                        </span>
                      ) : (
                        "Select Branch Head"
                      )}
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    </button>
                  </PopoverTrigger>
                  <PopoverContent
                    className="max-h-64 w-[var(--radix-popover-trigger-width)] overflow-y-auto p-1 dark:border-white/10 dark:bg-[#0f1216]"
                    align="start"
                  >
                    {userOptions.length === 0 ? (
                      <div className="px-2.5 py-2 text-sm text-muted-foreground">No users found</div>
                    ) : (
                      userOptions.map((h) => (
                        <button
                          key={h.email}
                          onClick={() => {
                            setHead(h);
                            setHeadOpen(false);
                          }}
                          className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-sm hover:bg-muted dark:hover:bg-white/5"
                        >
                          <Avatar className="h-6 w-6">
                            <AvatarImage src={h.avatar} alt={h.name} />
                            <AvatarFallback>{h.name?.[0]}</AvatarFallback>
                          </Avatar>
                          <span className="flex flex-col items-start">
                            <span className="text-sm text-foreground">{h.name}</span>
                            <span className="text-xs text-muted-foreground">{h.email}</span>
                          </span>
                        </button>
                      ))
                    )}
                  </PopoverContent>
                </Popover>
              </div>

            </div>

            <div className="flex justify-end border-t border-border px-5 py-4 dark:border-white/10">
              <Button
                onClick={() => setStep("confirm")}
                disabled={!canSubmit}
                className="h-11 gap-2 bg-[#1565EF] px-5 font-semibold text-white hover:bg-[#1257cf] disabled:bg-[#f4f6fb] disabled:text-muted-foreground dark:disabled:bg-white/5"
              >
                <PlusCircle className="h-4 w-4" />
                Add Branch
              </Button>
            </div>
          </DialogPrimitive.Content>
        </DialogPortal>
      )}

      {step === "confirm" && (
        <DialogPortal>
          <DialogOverlay />
          <DialogPrimitive.Content className="fixed left-1/2 top-1/2 z-50 w-[420px] max-w-[92vw] -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-white p-6 shadow-xl duration-200 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95 dark:bg-[#0f1216]">
            <DialogPrimitive.Close className="absolute right-4 top-4 text-muted-foreground hover:text-foreground">
              <X className="h-5 w-5" />
            </DialogPrimitive.Close>
            <div className="flex flex-col items-center text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-50 dark:bg-[#1565EF]/15">
                <HelpCircle className="h-6 w-6 text-[#1565EF]" />
              </div>
              <h3 className="mt-4 text-lg font-semibold text-foreground">
                Are you sure you want to add
              </h3>
              <p className="mt-1 text-sm text-muted-foreground">{name} Branch?</p>
              <div className="mt-6 grid w-full grid-cols-2 gap-3">
                <Button
                  variant="outline"
                  onClick={() => setStep("form")}
                  disabled={submitting}
                  className="h-11 gap-2 font-semibold dark:border-white/10 dark:bg-transparent dark:text-foreground dark:hover:bg-white/5"
                >
                  <X className="h-4 w-4" />
                  Cancel
                </Button>
                <Button
                  onClick={handleConfirm}
                  disabled={submitting}
                  className="h-11 gap-2 bg-[#1565EF] font-semibold text-white hover:bg-[#1257cf]"
                >
                  <CheckCircle2 className="h-4 w-4" />
                  {submitting ? "Saving..." : "Confirm"}
                </Button>
              </div>
            </div>
          </DialogPrimitive.Content>
        </DialogPortal>
      )}

      {step === "success" && (
        <DialogPortal>
          <DialogOverlay />
          <DialogPrimitive.Content className="fixed left-1/2 top-1/2 z-50 w-[400px] max-w-[92vw] -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-white p-8 shadow-xl duration-200 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95 dark:bg-[#0f1216]">
            <div className="flex flex-col items-center text-center">
              <div className="flex h-14 w-14 items-center justify-center rounded-full bg-emerald-100 dark:bg-emerald-500/15">
                <CheckCircle2 className="h-7 w-7 text-emerald-700 dark:text-emerald-400" strokeWidth={2} />
              </div>
              <h3 className="mt-4 text-lg font-semibold text-foreground">Successfully added</h3>
              <p className="mt-1 text-sm text-muted-foreground">{name} Branch</p>
            </div>
          </DialogPrimitive.Content>
        </DialogPortal>
      )}
    </Dialog>
  );
}
