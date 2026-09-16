import { createFileRoute } from "@tanstack/react-router";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { Search, Settings as SettingsGear, Mail, ChevronDown, HelpCircle, Upload, X, Clock, MoreVertical, Check, Loader2, MessageSquare, AtSign, BellRing, Smartphone, MessageCircle } from "lucide-react";
import { toast } from "sonner";
import { useServerFn } from "@tanstack/react-start";
import { Logomark } from "@/components/brand/Logomark";
import { BillingTab } from "@/components/dashboard/BillingTab";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { supabase } from "@/integrations/supabase/client";
import { CircularLoaderWithLabel } from "@/components/ui/circular-loader";
import { updateOrganizationProfile, getOrganizationProfile } from "@/lib/settings.functions";
import { useActivePlan } from "@/lib/use-active-plan";
import { setTheme } from "@/lib/theme";
import { integrationsApi, type IntegrationState } from "@/lib/api/integrations";
import { IntegrationConfigModal } from "@/components/dashboard/IntegrationConfigModal";


export const Route = createFileRoute("/_authenticated/settings")({
  head: () => ({
    meta: [
      { title: "Settings · Avaloka AI" },
      { name: "description", content: "Manage your personal info, profile, appearance, AI, notifications, integrations, and billing." },
    ],
  }),
  component: SettingsPage,
});

type TabId = "my-details" | "profile" | "appearance" | "email" | "notifications" | "integrations" | "billing";

const TABS: { id: TabId; label: string; ownerOnly?: boolean; billing?: boolean }[] = [
  { id: "my-details", label: "My details" },
  { id: "profile", label: "Profile" },
  { id: "appearance", label: "Appearance" },
  { id: "email", label: "Email" },
  { id: "notifications", label: "Notifications" },
  { id: "integrations", label: "Integrations" },
  { id: "billing", label: "Billing", billing: true },
];

/* --------------------------- Profile form state --------------------------- */

type Channels = { push: boolean; email: boolean; sms: boolean };
type EmailSettings = {
  news: boolean;
  tips: boolean;
  research: boolean;
  reminder: "none" | "important" | "all";
};
type NotificationSettings = {
  comments: Channels;
  tags: Channels;
  reminders: Channels;
};
type IntegrationSettings = Record<string, boolean>;

type ProfileForm = {
  first_name: string;
  last_name: string;
  phone: string;
  country: string;
  timezone: string;
  avatar_url: string | null;
  company_logo_url: string | null;
  company_name: string;
  company_slug: string;
  tagline: string;
  reports_opt_in: boolean;
  emails_opt_in: boolean;
  theme: "system" | "light" | "dark";
  ai_instructions: string;
  ai_depth: "conservative" | "balanced" | "aggressive";
  company_email: string;
  recovery_email: string;
  email_settings: EmailSettings;
  notification_settings: NotificationSettings;
  integration_settings: IntegrationSettings;
};

const DEFAULT_EMAIL: EmailSettings = { news: true, tips: true, research: false, reminder: "all" };
const DEFAULT_NOTIFS: NotificationSettings = {
  comments: { push: true, email: true, sms: false },
  tags: { push: true, email: false, sms: false },
  reminders: { push: false, email: false, sms: false },
};

const EMPTY_FORM: ProfileForm = {
  first_name: "",
  last_name: "",
  phone: "",
  country: "",
  timezone: "",
  avatar_url: null,
  company_logo_url: null,
  company_name: "",
  company_slug: "",
  tagline: "",
  reports_opt_in: true,
  emails_opt_in: true,
  theme: "system",
  ai_instructions: "",
  ai_depth: "conservative",
  company_email: "",
  recovery_email: "",
  email_settings: DEFAULT_EMAIL,
  notification_settings: DEFAULT_NOTIFS,
  integration_settings: {},
};

type ProfileCtx = {
  userId: string | null;
  email: string;
  form: ProfileForm;
  setForm: <K extends keyof ProfileForm>(key: K, value: ProfileForm[K]) => void;
  avatarPreviewUrl: string | null;
  uploadAvatar: (file: File) => Promise<void>;
  uploadingAvatar: boolean;
  companyLogoPreviewUrl: string | null;
  uploadCompanyLogo: (file: File) => Promise<void>;
  uploadingCompanyLogo: boolean;
  profileReadOnly: boolean;
};

const ProfileContext = createContext<ProfileCtx | null>(null);
const useProfileCtx = () => {
  const ctx = useContext(ProfileContext);
  if (!ctx) throw new Error("ProfileContext missing");
  return ctx;
};

function SettingsPage() {
  const updateOrganizationProfileFn = useServerFn(updateOrganizationProfile);
  const getOrganizationProfileFn = useServerFn(getOrganizationProfile);
  const [tab, setTab] = useState<TabId>(() => {
    if (typeof window === "undefined") return "my-details";
    const t = new URLSearchParams(window.location.search).get("tab");
    const allowed: TabId[] = ["my-details", "profile", "appearance", "email", "notifications", "integrations", "billing"];
    return (allowed as string[]).includes(t ?? "") ? (t as TabId) : "my-details";
  });
  // The SSR pass has no query string, so re-read it after hydration; otherwise a
  // deep link like /settings?tab=billing renders "my-details".
  useEffect(() => {
    const t = new URLSearchParams(window.location.search).get("tab");
    const allowed: TabId[] = ["my-details", "profile", "appearance", "email", "notifications", "integrations", "billing"];
    if ((allowed as string[]).includes(t ?? "")) setTab(t as TabId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // Keep the URL in sync so a refresh stays on the same tab.
  useEffect(() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get("tab") === tab) return;
    url.searchParams.set("tab", tab);
    window.history.replaceState(window.history.state, "", url.toString());
  }, [tab]);

  const [search, setSearch] = useState("");

  const [userId, setUserId] = useState<string | null>(null);
  const [orgId, setOrgId] = useState<string | null>(null);
  const [isOrgOwner, setIsOrgOwner] = useState(false);
  const [ownerResolved, setOwnerResolved] = useState(false);

  const [email, setEmail] = useState("");
  const [loaded, setLoaded] = useState<ProfileForm>(EMPTY_FORM);
  const [form, setFormState] = useState<ProfileForm>(EMPTY_FORM);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [avatarPreviewUrl, setAvatarPreviewUrl] = useState<string | null>(null);
  const [uploadingAvatar, setUploadingAvatar] = useState(false);
  const [companyLogoPreviewUrl, setCompanyLogoPreviewUrl] = useState<string | null>(null);
  const [uploadingCompanyLogo, setUploadingCompanyLogo] = useState(false);

  const setForm = useCallback(<K extends keyof ProfileForm>(key: K, value: ProfileForm[K]) => {
    setFormState((prev) => ({ ...prev, [key]: value }));
  }, []);

  const refreshAvatarPreview = useCallback(async (value: string | null) => {
    if (!value) { setAvatarPreviewUrl(null); return; }
    if (/^https?:\/\//i.test(value)) { setAvatarPreviewUrl(value); return; }
    const { data, error } = await supabase.storage.from("avatars").createSignedUrl(value, 60 * 60);
    if (!error && data?.signedUrl) setAvatarPreviewUrl(data.signedUrl);
  }, []);

  const refreshCompanyLogoPreview = useCallback(async (value: string | null) => {
    if (!value) { setCompanyLogoPreviewUrl(null); return; }
    if (/^https?:\/\//i.test(value)) { setCompanyLogoPreviewUrl(value); return; }
    const { data, error } = await supabase.storage.from("avatars").createSignedUrl(value, 60 * 60);
    if (!error && data?.signedUrl) setCompanyLogoPreviewUrl(data.signedUrl);
  }, []);

  // Load profile + user_settings on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const { data: userRes } = await supabase.auth.getUser();
      const u = userRes.user;
      if (!u) { setLoading(false); return; }
      if (cancelled) return;
      setUserId(u.id);
      setEmail(u.email ?? "");

      const [profileRes, settingsRes] = await Promise.all([
        supabase.from("profiles").select("*").eq("user_id", u.id).maybeSingle(),
        (supabase.from("user_settings" as any).select("*").eq("user_id", u.id).maybeSingle() as any),
      ]);
      if (cancelled) return;
      if (profileRes.error) {
        toast.error("Failed to load profile", { description: profileRes.error.message });
      }

      const data: any = profileRes.data;
      const s: any = settingsRes?.data ?? null;

      const orgIdVal: string | null = data?.organization_id ?? null;
      setOrgId(orgIdVal);

      let ent: any = null;
      let org: any = null;
      if (orgIdVal) {
        const [entRes, orgData]: any = await Promise.all([
          (supabase.from("enterprise_settings" as any).select("*").eq("organization_id", orgIdVal).maybeSingle() as any),
          getOrganizationProfileFn({ data: { orgId: orgIdVal } }).catch(() => null),
        ]);
        ent = entRes?.data ?? null;
        org = orgData ?? null;
        const profileId = (data as any)?.id ?? null;
        setIsOrgOwner(!!org?.owner_profile_id && !!profileId && org.owner_profile_id === profileId);
      }
      setOwnerResolved(true);


      const emailSettings: EmailSettings = s
        ? {
            news: s.news_and_updates ?? true,
            tips: s.tips_and_tutorials ?? true,
            research: s.user_research ?? false,
            reminder: (s.reminder_preference as EmailSettings["reminder"]) ?? "all",
          }
        : DEFAULT_EMAIL;

      const notifSettings: NotificationSettings = s
        ? {
            comments: { push: !!s.comments_push, email: !!s.comments_email, sms: !!s.comments_sms },
            tags: { push: !!s.tags_push, email: !!s.tags_email, sms: !!s.tags_sms },
            reminders: { push: !!s.reminders_push, email: !!s.reminders_email, sms: !!s.reminders_sms },
          }
        : DEFAULT_NOTIFS;

      if (data) {
        const fullParts = ((data as any).full_name as string | undefined)?.split(" ") ?? [];
        const next: ProfileForm = {
          first_name: fullParts[0] ?? "",
          last_name: fullParts.slice(1).join(" "),
          phone: data.phone ?? "",
          country: data.country ?? "",
          timezone: data.timezone ?? "",
          avatar_url: data.avatar_url ?? null,
          company_logo_url: org?.logo_url ?? ent?.company_logo_url ?? null,
          company_name: org?.name ?? ent?.company_name ?? data.company_name ?? "",
          company_slug: org?.slug ?? ent?.company_slug ?? data.company_slug ?? "",
          tagline: org?.tagline ?? ent?.company_tagline ?? data.tagline ?? "",
          reports_opt_in: org?.brand_reports ?? ent?.branded_reports ?? data.reports_opt_in ?? true,
          emails_opt_in: org?.brand_emails ?? ent?.branded_emails ?? data.emails_opt_in ?? true,
          theme: (s?.theme as ProfileForm["theme"]) ?? (data.theme as ProfileForm["theme"]) ?? "system",
          ai_instructions: ent?.ai_custom_instruction ?? data.ai_instructions ?? "",
          ai_depth: (ent?.ai_insight_depth as ProfileForm["ai_depth"]) ?? (data.ai_depth as ProfileForm["ai_depth"]) ?? "conservative",
          company_email: ent?.company_email ?? (data as any).company_email ?? "",
          recovery_email: s?.recovery_email ?? (data as any).recovery_email ?? "",
          email_settings: emailSettings,
          notification_settings: notifSettings,
          integration_settings: ((data as any).integration_settings ?? {}) as IntegrationSettings,
        };
        setLoaded(next);
        setFormState(next);
        await refreshAvatarPreview(next.avatar_url);
        await refreshCompanyLogoPreview(next.company_logo_url);
      } else {
        const nameParts = (u.user_metadata?.full_name as string | undefined)?.split(" ") ?? [];
        const next: ProfileForm = {
          ...EMPTY_FORM,
          first_name: nameParts[0] ?? "",
          last_name: nameParts.slice(1).join(" "),
          theme: (s?.theme as ProfileForm["theme"]) ?? "system",
          recovery_email: s?.recovery_email ?? "",
          email_settings: emailSettings,
          notification_settings: notifSettings,
        };
        setLoaded(next);
        setFormState(next);
      }
      setLoading(false);
    })();
    return () => { cancelled = true; };
  }, [refreshAvatarPreview, refreshCompanyLogoPreview, getOrganizationProfileFn]);

  const uploadAvatar = useCallback(async (file: File) => {
    if (!userId) { toast.error("Sign in required"); return; }
    setUploadingAvatar(true);
    try {
      const ext = (file.name.split(".").pop() || "png").toLowerCase();
      const path = `${userId}/avatar.${ext}`;
      const { error: upErr } = await supabase.storage
        .from("avatars")
        .upload(path, file, { upsert: true, contentType: file.type });
      if (upErr) throw upErr;
      const { data: pub } = supabase.storage.from("avatars").getPublicUrl(path);
      // Cache-bust so the same path shows the new file immediately
      const publicUrl = `${pub.publicUrl}?v=${Date.now()}`;
      setForm("avatar_url", publicUrl);
      setAvatarPreviewUrl(publicUrl);
      toast.success("Avatar uploaded", { description: "Click Save to persist your changes." });
    } catch (e: any) {
      toast.error("Upload failed", { description: e?.message ?? String(e) });
    } finally {
      setUploadingAvatar(false);
    }
  }, [userId, setForm]);

  const uploadCompanyLogo = useCallback(async (file: File) => {
    if (!orgId) { toast.error("No organization associated with this account."); return; }
    setUploadingCompanyLogo(true);
    try {
      const ext = (file.name.split(".").pop() || "png").toLowerCase();
      const path = `org/${orgId}/logo.${ext}`;
      const { error: upErr } = await supabase.storage
        .from("avatars")
        .upload(path, file, { upsert: true, contentType: file.type });
      if (upErr) throw upErr;
      const { data: pub } = supabase.storage.from("avatars").getPublicUrl(path);
      const publicUrl = `${pub.publicUrl}?v=${Date.now()}`;
      setForm("company_logo_url", publicUrl);
      setCompanyLogoPreviewUrl(publicUrl);
      toast.success("Logo uploaded", { description: "Click Save to persist your changes." });
    } catch (e: any) {
      toast.error("Upload failed", { description: e?.message ?? String(e) });
    } finally {
      setUploadingCompanyLogo(false);
    }
  }, [orgId, setForm]);

  const tabFieldMap = useMemo<Partial<Record<TabId, (keyof ProfileForm)[]>>>(() => ({
    "my-details": ["first_name", "last_name", "phone", "country", "timezone", "avatar_url"],
    profile: ["company_name", "company_slug", "tagline", "company_logo_url", "reports_opt_in", "emails_opt_in"],
    appearance: ["theme"],
    
    email: ["company_email", "recovery_email", "email_settings", "emails_opt_in"],
    notifications: ["notification_settings"],
    integrations: ["integration_settings"],
  }), []);

  const getScopedPayload = useCallback((activeTab: TabId, source: ProfileForm) => {
    const fields = tabFieldMap[activeTab];
    if (!fields) return source;
    const scoped: Partial<ProfileForm> = {};
    for (const key of fields) (scoped as any)[key] = source[key];
    return scoped as ProfileForm;
  }, [tabFieldMap]);

  const USER_SETTINGS_TABS: TabId[] = ["appearance", "email", "notifications"];
  const ENTERPRISE_TABS: TabId[] = [];

  const buildUserSettingsPayload = useCallback((activeTab: TabId, source: ProfileForm) => {
    const p: Record<string, any> = { user_id: userId };
    if (activeTab === "appearance") {
      p.theme = source.theme;
    }
    if (activeTab === "email") {
      p.recovery_email = source.recovery_email || null;
      p.news_and_updates = source.email_settings.news;
      p.tips_and_tutorials = source.email_settings.tips;
      p.user_research = source.email_settings.research;
      p.reminder_preference = source.email_settings.reminder;
    }
    if (activeTab === "notifications") {
      const n = source.notification_settings;
      p.comments_push = n.comments.push;
      p.comments_email = n.comments.email;
      p.comments_sms = n.comments.sms;
      p.tags_push = n.tags.push;
      p.tags_email = n.tags.email;
      p.tags_sms = n.tags.sms;
      p.reminders_push = n.reminders.push;
      p.reminders_email = n.reminders.email;
      p.reminders_sms = n.reminders.sms;
    }
    return p;
  }, [userId]);

  const buildEnterprisePayload = useCallback((activeTab: TabId, source: ProfileForm) => {
    const p: Record<string, any> = { organization_id: orgId };
    if (activeTab === "email") {
      p.company_email = source.company_email || null;
    }
    return p;
  }, [orgId]);

  const buildOrganizationPayload = useCallback((source: ProfileForm) => {
    const p: Record<string, any> = {
      tagline: source.tagline || null,
      brand_reports: source.reports_opt_in,
      brand_emails: source.emails_opt_in,
      slug: source.company_slug ? source.company_slug : null,
      logo_url: source.company_logo_url || null,
    };
    if (source.company_name && source.company_name.trim()) {
      p.name = source.company_name.trim();
    }
    return p;
  }, []);

  const handleSave = useCallback(async () => {
    if (!userId) return;
    setSaving(true);
    try {
      if (tab === "profile") {
        if (!orgId) throw new Error("No organization associated with this account.");
        const orgPayload = buildOrganizationPayload(form);
        const updatedOrg = await updateOrganizationProfileFn({ data: { orgId, payload: orgPayload } });
        setFormState((prev) => ({
          ...prev,
          company_logo_url: updatedOrg.logo_url,
          company_name: updatedOrg.name ?? prev.company_name,
          company_slug: updatedOrg.slug ?? "",
          tagline: updatedOrg.tagline ?? "",
          reports_opt_in: updatedOrg.brand_reports,
          emails_opt_in: updatedOrg.brand_emails,
        }));
      } else if (ENTERPRISE_TABS.includes(tab)) {
        if (!orgId) throw new Error("No organization associated with this account.");
        const payload = buildEnterprisePayload(tab, form);
        const { error } = await (supabase.from("enterprise_settings" as any) as any).upsert(payload, { onConflict: "organization_id" });
        if (error) throw error;
      } else if (USER_SETTINGS_TABS.includes(tab)) {
        const payload = buildUserSettingsPayload(tab, form);
        const { error } = await (supabase.from("user_settings" as any) as any).upsert(payload, { onConflict: "user_id" });
        if (error) throw error;
        // Company email lives in enterprise_settings — mirror on email tab save
        if (tab === "email" && orgId) {
          const entPayload = buildEnterprisePayload(tab, form);
          const { error: eErr } = await (supabase.from("enterprise_settings" as any) as any).upsert(entPayload, { onConflict: "organization_id" });
          if (eErr) throw eErr;
        }
      } else {
        const payload: Record<string, any> = getScopedPayload(tab, form);
        if ("first_name" in payload || "last_name" in payload) {
          const fn = (payload.first_name ?? form.first_name ?? "").trim();
          const ln = (payload.last_name ?? form.last_name ?? "").trim();
          payload.full_name = [fn, ln].filter(Boolean).join(" ");
          delete payload.first_name;
          delete payload.last_name;
        }
        const { error } = await supabase.from("profiles").upsert({ user_id: userId, ...payload }, { onConflict: "user_id" });
        if (error) throw error;
      }
      setLoaded((prev) => ({ ...prev, ...getScopedPayload(tab, form) }));
      toast.success("Settings saved");
    } catch (e: any) {
      toast.error("Save failed", { description: e?.message ?? String(e) });
    } finally {
      setSaving(false);
    }
  }, [userId, orgId, form, tab, updateOrganizationProfileFn, getScopedPayload, buildUserSettingsPayload, buildEnterprisePayload, buildOrganizationPayload]);

  const handleCancel = useCallback(() => {
    const fields = tabFieldMap[tab];
    if (!fields) {
      setFormState(loaded);
      refreshAvatarPreview(loaded.avatar_url);
      return;
    }
    setFormState((prev) => {
      const next = { ...prev };
      for (const key of fields) (next as any)[key] = loaded[key];
      return next;
    });
    if (fields.includes("avatar_url")) refreshAvatarPreview(loaded.avatar_url);
    if (fields.includes("company_logo_url")) refreshCompanyLogoPreview(loaded.company_logo_url);
  }, [loaded, refreshAvatarPreview, refreshCompanyLogoPreview, tab, tabFieldMap]);

  const profileReadOnly = !isOrgOwner;

  const ctx = useMemo<ProfileCtx>(() => ({
    userId, email, form, setForm,
    avatarPreviewUrl, uploadAvatar, uploadingAvatar,
    companyLogoPreviewUrl, uploadCompanyLogo, uploadingCompanyLogo,
    profileReadOnly,
  }), [userId, email, form, setForm, avatarPreviewUrl, uploadAvatar, uploadingAvatar, companyLogoPreviewUrl, uploadCompanyLogo, uploadingCompanyLogo, profileReadOnly]);

  const activePlanQ = useActivePlan();
  const isFreePlan = (activePlanQ.data?.plan ?? "free") === "free";
  // Ownership can also come from the server-resolved plan, which is the source
  // of truth when the client-side organization lookup fails (e.g. during trial).
  const canSeeBilling = isFreePlan || isOrgOwner || activePlanQ.data?.isOrgOwner === true;

  const visibleTabs = useMemo(
    () =>
      TABS.filter((t) => {
        if (t.ownerOnly && !isOrgOwner) return false;
        if (t.billing && !canSeeBilling) return false;
        return true;
      }),
    [isOrgOwner, canSeeBilling],
  );

  // Redirect away if user lands on a hidden tab — only once ownership and the
  // active plan have actually resolved, otherwise the billing tab flickers
  // back to "my-details" right after a payment redirect.
  useEffect(() => {
    const planResolved = activePlanQ.isSuccess || activePlanQ.isError;
    if (!ownerResolved || !planResolved) return;
    if (!visibleTabs.some((t) => t.id === tab)) setTab("my-details");
  }, [visibleTabs, tab, ownerResolved, activePlanQ.isSuccess, activePlanQ.isError]);


  const showFooter = tab !== "billing" && !(tab === "profile" && profileReadOnly);

  return (
    <ProfileContext.Provider value={ctx}>
      <div className="flex min-h-0 min-w-0 flex-1 bg-[#f4f6fb] text-foreground dark:bg-[#0b0d10]">
        <main className="flex flex-1 flex-col overflow-y-auto">
          <div className="m-4 flex flex-1 flex-col rounded-2xl border border-border bg-white shadow-sm dark:bg-[#111418] sm:m-6">
            {/* Header */}
            <header className="flex items-center justify-between border-b border-border px-6 py-4">
              <div className="flex items-center gap-2.5">
                <SettingsGear className="h-5 w-5 text-foreground" strokeWidth={1.75} />
                <h1 className="text-xl font-semibold tracking-tight">Setting</h1>
              </div>
              <div className="relative w-[320px]">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search"
                  className="h-9 pl-9 pr-12 text-sm"
                />
                <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 rounded border border-border bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
                  ⌘K
                </span>
              </div>
            </header>

            {/* Tabs */}
            <div className="border-b border-border px-6">
              <div className="flex items-center gap-6 overflow-x-auto no-scrollbar">
                {visibleTabs.map((t) => {
                  const active = tab === t.id;
                  return (
                    <button
                      key={t.id}
                      onClick={() => setTab(t.id)}
                      className={cn(
                        "relative whitespace-nowrap py-3 text-sm transition-colors",
                        active ? "font-semibold text-[#1565EF] dark:text-white" : "text-muted-foreground hover:text-foreground",
                      )}
                    >
                      {t.label}
                      {active && <span className="absolute inset-x-0 -bottom-px h-0.5 rounded-full bg-[#1565EF] dark:bg-white" />}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Body */}
            <div className="flex-1 overflow-y-auto px-6 py-6">
              {loading ? (
                <div className="flex h-full min-h-[300px] items-center justify-center">
                  <CircularLoaderWithLabel label="Loading..." />
                </div>
              ) : tab === "my-details" ? (
                <MyDetailsTab />
              ) : tab === "profile" ? (
                <ProfileTab />
              ) : tab === "appearance" ? (
                <AppearanceTab />
              ) : tab === "email" ? (
                <EmailTab />
              ) : tab === "notifications" ? (
                <NotificationsTab />
              ) : tab === "integrations" ? (
                <IntegrationsTab />
              ) : tab === "billing" ? (
                <BillingTab />
              ) : (
                <div className="flex h-full min-h-[300px] items-center justify-center text-sm text-muted-foreground">
                  {TABS.find((t) => t.id === tab)?.label} settings coming soon.
                </div>
              )}
            </div>

            {/* Footer */}
            {showFooter && (
              <footer className="flex items-center justify-end gap-3 border-t border-border px-6 py-4">
                <Button variant="outline" className="gap-2" onClick={handleCancel} disabled={saving}>
                  <X className="h-4 w-4" /> Cancel
                </Button>
                <Button className="gap-2 bg-[#1565EF] hover:bg-[#1257d4]" onClick={handleSave} disabled={saving || !userId}>
                  {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <SaveIcon className="h-4 w-4" />} Save
                </Button>
              </footer>
            )}
          </div>
        </main>
      </div>
    </ProfileContext.Provider>
  );
}

/* ---------------------------- My details tab ---------------------------- */
function MyDetailsTab() {
  const { email, form, setForm } = useProfileCtx();
  return (
    <>
      <div className="mb-5 border-b border-border pb-4">
        <h2 className="text-base font-semibold">Personal info</h2>
        <p className="mt-1 text-sm text-muted-foreground">Update your photo and personal details here.</p>
      </div>

      <div className="flex flex-col gap-5">
        <Row label="Name" required>
          <div className="grid grid-cols-2 gap-3">
            <Input value={form.first_name} onChange={(e) => setForm("first_name", e.target.value)} className="h-10" placeholder="First name" />
            <Input value={form.last_name} onChange={(e) => setForm("last_name", e.target.value)} className="h-10" placeholder="Last name" />
          </div>
        </Row>

        <Row label="Email address" required>
          <div className="relative">
            <Mail className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input value={email} readOnly className="h-10 pl-9 bg-muted/40 dark:bg-white/5" />
          </div>
        </Row>

        <Row label="Contact">
          <div className="relative flex items-center">
            <button
              type="button"
              className="absolute left-0 top-0 flex h-10 items-center gap-1 rounded-l-md border-r border-border px-3 text-sm text-foreground"
            >
              IN <ChevronDown className="h-3.5 w-3.5" />
            </button>
            <Input
              value={form.phone}
              onChange={(e) => setForm("phone", e.target.value)}
              className="h-10 pl-[68px]"
              placeholder="+91 ..."
            />
          </div>
        </Row>

        <Row
          label="Your photo"
          required
          hint={<HelpCircle className="h-3.5 w-3.5 text-muted-foreground" />}
          sublabel="This will be displayed on your profile."
        >
          <PhotoDropzone />
        </Row>

        <Row label="Country">
          <SelectField
            icon={<span className="text-base">🌐</span>}
            value={form.country}
            onChange={(v) => setForm("country", v)}
            placeholder="Select country"
            options={COUNTRIES}
          />
        </Row>

        <Row label="Timezone" hint={<HelpCircle className="h-3.5 w-3.5 text-muted-foreground" />}>
          <SelectField
            icon={<Clock className="h-4 w-4 text-muted-foreground" />}
            value={form.timezone}
            onChange={(v) => setForm("timezone", v)}
            placeholder="Select timezone"
            options={TIMEZONES}
          />
        </Row>
      </div>
    </>
  );
}

function Row({
  label,
  required,
  hint,
  sublabel,
  children,
}: {
  label: string;
  required?: boolean;
  hint?: React.ReactNode;
  sublabel?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid grid-cols-[220px_minmax(0,1fr)] items-start gap-6 border-b border-border pb-5 last:border-0">
      <div>
        <div className="flex items-center gap-1 text-sm font-medium text-foreground">
          <span>{label}</span>
          {required && <span className="text-[#1565EF] dark:text-white">*</span>}
          {hint}
        </div>
        {sublabel && <p className="mt-1 text-xs text-muted-foreground">{sublabel}</p>}
      </div>
      <div className="max-w-[520px]">{children}</div>
    </div>
  );
}

function SelectField({ icon, value, onChange, placeholder, options }: { icon: React.ReactNode; value: string; onChange: (v: string) => void; placeholder?: string; options: string[] }) {
  return (
    <div className="relative flex h-10 items-center rounded-md border border-input bg-background px-3 text-sm focus-within:ring-2 focus-within:ring-ring">
      <span className="mr-2 inline-flex items-center">{icon}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="flex-1 cursor-pointer appearance-none bg-transparent pr-6 outline-none"
      >
        <option value="" disabled>{placeholder ?? "Select…"}</option>
        {options.map((opt) => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
      </select>
      <ChevronDown className="pointer-events-none absolute right-3 h-4 w-4 text-muted-foreground" />
    </div>
  );
}

const COUNTRIES = [
  "United States","United Kingdom","Canada","Australia","India","Germany","France","Spain","Italy","Netherlands","Sweden","Norway","Denmark","Finland","Ireland","Portugal","Switzerland","Belgium","Austria","Poland","Czechia","Greece","Turkey","United Arab Emirates","Saudi Arabia","Israel","South Africa","Egypt","Nigeria","Kenya","Brazil","Mexico","Argentina","Chile","Colombia","Japan","China","South Korea","Singapore","Hong Kong","Taiwan","Thailand","Vietnam","Indonesia","Malaysia","Philippines","New Zealand","Pakistan","Bangladesh","Sri Lanka","Nepal","Russia","Ukraine",
];

const TIMEZONES = (typeof Intl !== "undefined" && typeof (Intl as any).supportedValuesOf === "function")
  ? ((Intl as any).supportedValuesOf("timeZone") as string[])
  : [
      "UTC","Asia/Kolkata","Asia/Dubai","Asia/Singapore","Asia/Tokyo","Asia/Shanghai","Asia/Hong_Kong","Europe/London","Europe/Berlin","Europe/Paris","Europe/Madrid","Europe/Amsterdam","America/New_York","America/Chicago","America/Denver","America/Los_Angeles","America/Toronto","America/Sao_Paulo","Australia/Sydney","Australia/Melbourne","Pacific/Auckland",
    ];

function PhotoDropzone() {
  const { avatarPreviewUrl, uploadAvatar, uploadingAvatar } = useProfileCtx();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = (files: FileList | null) => {
    if (files && files[0]) uploadAvatar(files[0]);
  };

  return (
    <button
      type="button"
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
      className={cn(
        "relative flex w-full cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-6 text-center transition-colors",
        dragOver
          ? "border-[#1565EF] bg-[#e6efff] dark:border-[#1565EF] dark:bg-[#0f1620]"
          : "border-[#1565EF] bg-[#f0f5ff] dark:border-[#1565EF] dark:bg-[#0b1017]",
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".svg,.png,.jpg,.jpeg,.gif"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
      {avatarPreviewUrl ? (
        <img src={avatarPreviewUrl} alt="Avatar" className="mb-2 h-14 w-14 rounded-full object-cover" />
      ) : (
        <div className="mb-2 inline-flex h-9 w-9 items-center justify-center rounded-full border border-border bg-white dark:bg-white/5">
          {uploadingAvatar ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4 text-foreground" />}
        </div>
      )}
      <p className="text-sm">
        <span className="font-semibold text-[#1565EF] dark:text-white">{uploadingAvatar ? "Uploading…" : "Select"}</span>{" "}
        <span className="text-foreground">or drag and drop file</span>
      </p>
      <p className="mt-1 text-xs text-muted-foreground">SVG, PNG, JPG or GIF (max. 800x400px)</p>
      <div className="absolute right-4 top-1/2 hidden -translate-y-1/2 sm:block">
        <div className="flex h-12 w-10 items-center justify-center rounded-md bg-[#1565EF] text-[10px] font-bold text-white shadow-md">
          PNG
        </div>
      </div>
    </button>
  );
}

function CompanyLogoThumb() {
  const { companyLogoPreviewUrl } = useProfileCtx();
  if (!companyLogoPreviewUrl) return null;
  return <img src={companyLogoPreviewUrl} alt="Company logo" className="h-full w-full object-contain" />;
}

function CompanyLogoDropzone() {
  const { companyLogoPreviewUrl, uploadCompanyLogo, uploadingCompanyLogo } = useProfileCtx();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = (files: FileList | null) => {
    if (files && files[0]) uploadCompanyLogo(files[0]);
  };

  return (
    <button
      type="button"
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
      className={cn(
        "relative flex w-full cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-6 text-center transition-colors",
        dragOver
          ? "border-[#1565EF] bg-[#e6efff] dark:border-[#1565EF] dark:bg-[#0f1620]"
          : "border-[#1565EF] bg-[#f0f5ff] dark:border-[#1565EF] dark:bg-[#0b1017]",
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".svg,.png,.jpg,.jpeg,.gif"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
      {companyLogoPreviewUrl ? (
        <img src={companyLogoPreviewUrl} alt="Company logo" className="mb-2 h-14 w-14 rounded-md object-contain" />
      ) : (
        <div className="mb-2 inline-flex h-9 w-9 items-center justify-center rounded-full border border-border bg-white dark:bg-white/5">
          {uploadingCompanyLogo ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4 text-foreground" />}
        </div>
      )}
      <p className="text-sm">
        <span className="font-semibold text-[#1565EF] dark:text-white">{uploadingCompanyLogo ? "Uploading…" : "Select"}</span>{" "}
        <span className="text-foreground">or drag and drop file</span>
      </p>
      <p className="mt-1 text-xs text-muted-foreground">SVG, PNG, JPG or GIF (max. 800x400px)</p>
    </button>
  );
}

function SaveIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
      <polyline points="17 21 17 13 7 13 7 21" />
      <polyline points="7 3 7 8 15 8" />
    </svg>
  );
}

/* ---------------------------- Profile tab ---------------------------- */
function ProfileTab() {
  const { form, setForm, profileReadOnly } = useProfileCtx();
  const ro = profileReadOnly;

  return (
    <>
      <div className="mb-5 border-b border-border pb-4">
        <h2 className="text-base font-semibold">Company profile</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {ro
            ? "Only the organization owner can edit these details."
            : "Update your company photo and details here."}
        </p>
      </div>

      <div className="flex flex-col gap-5">
        <Row label="Public profile" required sublabel="This will be displayed on your profile.">
          <div className="flex flex-col gap-2">
            <Input value={form.company_name} onChange={(e) => setForm("company_name", e.target.value)} className="h-10" placeholder="Avaloka.ai" disabled={ro} readOnly={ro} />
            <div className="flex h-10 items-stretch overflow-hidden rounded-md border border-input">
              <span className="flex items-center bg-muted px-3 text-sm text-muted-foreground">avalokaai.com/profile/</span>
              <input
                value={form.company_slug}
                onChange={(e) => setForm("company_slug", e.target.value)}
                className="flex-1 bg-background px-3 text-sm outline-none disabled:bg-muted/40 disabled:cursor-not-allowed"
                disabled={ro}
                readOnly={ro}
              />
              <span className="flex items-center pr-3 text-muted-foreground">
                <HelpCircle className="h-4 w-4" />
              </span>
            </div>
          </div>
        </Row>

        <Row
          label="Tagline"
          required
          hint={<HelpCircle className="h-3.5 w-3.5 text-muted-foreground" />}
          sublabel="A quick snapshot of your company."
        >
          <div>
            <textarea
              value={form.tagline}
              onChange={(e) => setForm("tagline", e.target.value)}
              rows={3}
              className="w-full resize-none rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:bg-muted/40 disabled:cursor-not-allowed"
              placeholder="Add short description about you company"
              disabled={ro}
              readOnly={ro}
            />
            <p className="mt-1 text-xs text-muted-foreground">
              {Math.max(0, 40 - form.tagline.length)} characters left
            </p>
          </div>
        </Row>

        <Row
          label="Company logo"
          required
          sublabel="Update your company logo and then choose where you want it to display."
        >
          <div className="flex items-start gap-4">
            <div className="flex h-[88px] w-[140px] items-center justify-center rounded-md border border-border bg-white overflow-hidden dark:bg-white/5">
              {form.company_logo_url ? (
                <CompanyLogoThumb />
              ) : (
                <div className="flex items-center gap-2">
                  <Logomark size={28} />
                  <span className="text-base font-semibold">{form.company_name || "Avaloka.ai"}</span>
                </div>
              )}
            </div>
            {!ro && (
              <div className="flex-1">
                <CompanyLogoDropzone />
              </div>
            )}
          </div>
        </Row>

        <Row label="Branding" sublabel="Add your logo to reports and emails.">
          <div className={cn("flex flex-col gap-3", ro && "pointer-events-none opacity-60")}>
            <BrandingCheck
              checked={form.reports_opt_in}
              onChange={(v) => setForm("reports_opt_in", v)}
              title="Reports"
              desc="Include my logo in summary reports."
            />
            <BrandingCheck
              checked={form.emails_opt_in}
              onChange={(v) => setForm("emails_opt_in", v)}
              title="Emails"
              desc="Include my logo in customer emails."
            />
          </div>
          {!ro && (
            <button type="button" className="mt-3 text-sm font-semibold text-[#1565EF] hover:underline dark:text-white">
              View examples
            </button>
          )}
        </Row>
      </div>
    </>
  );
}

function BrandingCheck({
  checked,
  onChange,
  title,
  desc,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  title: string;
  desc: string;
}) {
  return (
    <label className="flex cursor-pointer items-start gap-3">
      <button
        type="button"
        onClick={() => onChange(!checked)}
        className={cn(
          "mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded border transition-colors",
          checked ? "border-[#1565EF] bg-[#1565EF] text-white" : "border-input bg-background",
        )}
        aria-pressed={checked}
      >
        {checked && <Check className="h-3 w-3" strokeWidth={3} />}
      </button>
      <div>
        <p className="text-sm font-medium text-foreground">{title}</p>
        <p className="text-xs text-muted-foreground">{desc}</p>
      </div>
    </label>
  );
}

/* --------------------------- Appearance tab -------------------------- */
function AppearanceTab() {
  const { form, setForm } = useProfileCtx();
  const mode = form.theme;
  return (
    <>
      <div className="mb-5 flex items-start justify-between border-b border-border pb-4">
        <div>
          <h2 className="text-base font-semibold">Appearance</h2>
          <p className="mt-1 text-sm text-muted-foreground">Change how your dashboard looks and feels.</p>
        </div>
        <button type="button" className="text-muted-foreground hover:text-foreground" aria-label="More">
          <MoreVertical className="h-5 w-5" />
        </button>
      </div>

      <Row label="Display preference" sublabel="Switch between light and dark modes.">
        <div className="flex flex-wrap gap-4">
          <ThemeChoice variant="system" active={mode === "system"} label="System preference" onClick={() => { setForm("theme", "system"); setTheme("system"); }} />
          <ThemeChoice variant="light" active={mode === "light"} label="Light mode" onClick={() => { setForm("theme", "light"); setTheme("light"); }} />
          <ThemeChoice variant="dark" active={mode === "dark"} label="Dark mode" onClick={() => { setForm("theme", "dark"); setTheme("dark"); }} />
        </div>
      </Row>
    </>
  );
}

function ThemeChoice({
  variant,
  active,
  label,
  onClick,
}: {
  variant: "system" | "light" | "dark";
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button type="button" onClick={onClick} className="flex flex-col items-start gap-2 text-left">
      <div
        className={cn(
          "relative h-[110px] w-[170px] overflow-hidden rounded-lg border-2 transition-colors",
          active ? "border-[#1565EF]" : "border-transparent",
        )}
      >
        <ThemePreview variant={variant} />
        {active && (
          <span className="absolute bottom-2 left-2 flex h-4 w-4 items-center justify-center rounded-full bg-[#1565EF] ring-2 ring-white">
            <span className="h-1.5 w-1.5 rounded-full bg-white" />
          </span>
        )}
      </div>
      <span className="text-sm font-semibold text-foreground">{label}</span>
    </button>
  );
}

function ThemePreview({ variant }: { variant: "system" | "light" | "dark" }) {
  const Light = (
    <div className="h-full w-full bg-white">
      <div className="flex h-3 items-center gap-1 border-b border-gray-100 px-2">
        <span className="h-1.5 w-1.5 rounded-full bg-red-400" />
        <span className="h-1.5 w-1.5 rounded-full bg-yellow-400" />
        <span className="h-1.5 w-1.5 rounded-full bg-green-400" />
        <span className="ml-2 text-[6px] text-gray-700">Your dashboard</span>
      </div>
      <div className="flex h-[calc(100%-12px)]">
        <div className="w-1/3 space-y-1 border-r border-gray-100 p-1.5">
          <div className="h-1 w-full rounded bg-gray-200" />
          <div className="h-1 w-3/4 rounded bg-gray-200" />
          <div className="h-1 w-2/3 rounded bg-gray-200" />
        </div>
        <div className="flex-1 p-1.5">
          <svg viewBox="0 0 60 30" className="h-full w-full">
            <polyline points="0,25 10,20 20,22 30,12 40,15 50,6 60,8" fill="none" stroke="#1565EF" strokeWidth="1.2" />
          </svg>
        </div>
      </div>
    </div>
  );
  const Dark = (
    <div className="h-full w-full bg-[#1c1c24]">
      <div className="flex h-3 items-center gap-1 border-b border-white/10 px-2">
        <span className="h-1.5 w-1.5 rounded-full bg-red-400" />
        <span className="h-1.5 w-1.5 rounded-full bg-yellow-400" />
        <span className="h-1.5 w-1.5 rounded-full bg-green-400" />
        <span className="ml-2 text-[6px] text-white/70">Your dashboard</span>
      </div>
      <div className="flex h-[calc(100%-12px)]">
        <div className="w-1/3 space-y-1 border-r border-white/10 p-1.5">
          <div className="h-1 w-full rounded bg-white/15" />
          <div className="h-1 w-3/4 rounded bg-white/15" />
          <div className="h-1 w-2/3 rounded bg-white/15" />
        </div>
        <div className="flex-1 p-1.5">
          <svg viewBox="0 0 60 30" className="h-full w-full">
            <polyline points="0,25 10,20 20,22 30,12 40,15 50,6 60,8" fill="none" stroke="#5193FD" strokeWidth="1.2" />
          </svg>
        </div>
      </div>
    </div>
  );

  if (variant === "light") return Light;
  if (variant === "dark") return Dark;
  return (
    <div className="grid h-full w-full grid-cols-2">
      <div className="overflow-hidden">{Light}</div>
      <div className="overflow-hidden">{Dark}</div>
    </div>
  );
}

/* --------------------------- AI tab -------------------------- */
function AITab() {
  const { form, setForm } = useProfileCtx();
  const fileRef = useRef<HTMLInputElement>(null);

  return (
    <>
      <div className="mb-5 border-b border-border pb-4">
        <h2 className="text-base font-semibold">Agent Instructions</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Control how Avaloka.AI interprets data, generates insights, and responds to questions.
        </p>
      </div>

      <Row
        label="User-level agent instructions"
        sublabel="Define guidance that Avaloka.AI should consistently follow when analyzing data and generating insights across your projects."
      >
        <div className="flex flex-col gap-3">
          <textarea
            value={form.ai_instructions}
            onChange={(e) => setForm("ai_instructions", e.target.value)}
            placeholder="Add Custom instruction"
            rows={4}
            className="w-full resize-none rounded-md border border-input bg-background px-3 py-2 text-sm outline-none placeholder:text-muted-foreground focus-visible:ring-1 focus-visible:ring-ring"
          />
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            <span className="h-px flex-1 bg-border" />
            <span>Or</span>
            <span className="h-px flex-1 bg-border" />
          </div>
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border px-6 py-8 text-center hover:bg-muted/50"
          >
            <input ref={fileRef} type="file" accept=".png,.jpg,.jpeg,.pdf" className="hidden" />
            <div className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-border bg-white">
              <Upload className="h-4 w-4 text-foreground" />
            </div>
            <p className="text-sm font-semibold text-[#1565EF]">Click to upload the Instruction file</p>
            <p className="text-xs text-muted-foreground">PNG, JPG or PDF (max. 100MB)</p>
          </button>
        </div>
      </Row>

      <Row
        label="AI Insights Depth"
        sublabel="Control how cautious or exploratory Avaloka.AI should be when generating insights from your data."
      >
        <div className="flex flex-col gap-4">
          <DepthOption
            checked={form.ai_depth === "conservative"}
            onClick={() => setForm("ai_depth", "conservative")}
            title="Conservative"
            desc="Focus on high-confidence patterns and well-supported conclusions."
          />
          <DepthOption
            checked={form.ai_depth === "balanced"}
            onClick={() => setForm("ai_depth", "balanced")}
            title="Balanced"
            desc="Combine reliable insights with moderate exploratory analysis."
          />
          <DepthOption
            checked={form.ai_depth === "aggressive"}
            onClick={() => setForm("ai_depth", "aggressive")}
            title="Aggressive"
            desc="Surface emerging patterns, weak signals, and potential risks earlier."
          />
        </div>
      </Row>
    </>
  );
}

function DepthOption({
  checked,
  onClick,
  title,
  desc,
}: {
  checked: boolean;
  onClick: () => void;
  title: string;
  desc: string;
}) {
  return (
    <button type="button" onClick={onClick} className="flex w-full items-start gap-3 text-left">
      <span
        className={cn(
          "mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border-2 transition-colors",
          checked ? "border-[#1565EF]" : "border-input",
        )}
      >
        {checked && <span className="h-1.5 w-1.5 rounded-full bg-[#1565EF]" />}
      </span>
      <span>
        <span className="block text-sm font-semibold text-foreground">{title}</span>
        <span className="block text-xs text-muted-foreground">{desc}</span>
      </span>
    </button>
  );
}

/* --------------------------- shared toggle -------------------------- */
function Toggle({ checked, onChange, label, disabled }: { checked: boolean; onChange: (v: boolean) => void; label?: string; disabled?: boolean }) {
  const switchEl = (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => !disabled && onChange(!checked)}
      className={cn(
        "relative inline-flex h-5 w-9 shrink-0 items-center rounded-full p-0.5 transition-colors",
        checked ? "bg-[#1565EF]" : "bg-gray-200",
        disabled && "cursor-not-allowed opacity-50",
      )}
    >
      <span
        className={cn(
          "block h-4 w-4 rounded-full bg-white shadow transition-transform",
          checked ? "translate-x-4" : "translate-x-0",
        )}
      />
    </button>
  );

  if (!label) return switchEl;
  return (
    <label className="flex w-full items-center justify-between gap-3">
      <span className="text-sm text-foreground">{label}</span>
      {switchEl}
    </label>
  );
}

/* ---------------------------- Email tab ---------------------------- */
function EmailTab() {
  const { form, setForm } = useProfileCtx();
  const e = form.email_settings;
  const setE = <K extends keyof EmailSettings>(k: K, v: EmailSettings[K]) =>
    setForm("email_settings", { ...e, [k]: v });

  return (
    <>
      <div className="mb-5 border-b border-border pb-4">
        <h2 className="text-base font-semibold">Email</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Get emails to find out what&apos;s going on when you&apos;re not online. You can turn them off anytime.
        </p>
      </div>

      <div className="flex flex-col gap-5">
        <Row label="Company Email Address" required>
          <div className="flex items-center gap-3">
            <div className="relative flex-1">
              <Mail className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input value={form.company_email} onChange={(ev) => setForm("company_email", ev.target.value)} className="h-10 pl-9" placeholder="you@example.com" />
            </div>
            <Button className="h-10 gap-2 bg-[#1565EF] hover:bg-[#1257d4]">
              <ShieldCheckIcon className="h-4 w-4" /> Verify
            </Button>
          </div>
        </Row>

        <Row label="Recovery Email address">
          <div className="relative">
            <Mail className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input value={form.recovery_email} onChange={(ev) => setForm("recovery_email", ev.target.value)} className="h-10 pl-9" placeholder="you@example.com" />
          </div>
        </Row>

        <Row label="Email Notification Preferences">
          <div className="flex items-center justify-between">
            <span className="text-sm text-foreground">Enable email notifications</span>
            <Toggle checked={form.emails_opt_in} onChange={(v) => setForm("emails_opt_in", v)} />
          </div>
        </Row>
      </div>

      <h3 className="mt-2 mb-5 text-base font-semibold">Email Notification</h3>

      <fieldset
        disabled={!form.emails_opt_in}
        aria-disabled={!form.emails_opt_in}
        className={!form.emails_opt_in ? "pointer-events-none opacity-50" : ""}
      >
        <Row
          label="Notifications from us"
          sublabel="Receive the latest news, updates and industry tutorials from us."
        >
          <div className="flex flex-col gap-3">
            <BrandingCheck checked={e.news} onChange={(v) => setE("news", v)} title="News and updates" desc="News about product and feature updates." />
            <BrandingCheck checked={e.tips} onChange={(v) => setE("tips", v)} title="Tips and tutorials" desc="Tips on getting more out of Avaloka.ai." />
            <BrandingCheck checked={e.research} onChange={(v) => setE("research", v)} title="User research" desc="Get involved in our beta testing program or participate in paid product user research." />
          </div>
        </Row>

        <Row
          label="Reminders"
          sublabel="These are notifications to remind you of updates you might have missed."
        >
          <div className="flex flex-col gap-3">
            <DepthOption checked={e.reminder === "none"} onClick={() => setE("reminder", "none")} title="Do not notify me" desc="" />
            <DepthOption checked={e.reminder === "important"} onClick={() => setE("reminder", "important")} title="Important reminders only" desc="Only notify me if the reminder is tagged as important." />
            <DepthOption checked={e.reminder === "all"} onClick={() => setE("reminder", "all")} title="All reminders" desc="" />
          </div>
        </Row>
      </fieldset>

    </>
  );
}

function ShieldCheckIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <polyline points="9 12 11 14 15 10" />
    </svg>
  );
}

/* ----------------------- Notifications tab ----------------------- */
const NOTIFICATION_GROUPS: {
  key: keyof NotificationSettings;
  title: string;
  desc: string;
  icon: React.ComponentType<{ className?: string }>;
}[] = [
  {
    key: "comments",
    title: "Comments",
    desc: "Notifications for comments on your chats and uploaded files.",
    icon: MessageSquare,
  },
  {
    key: "tags",
    title: "Mentions & tags",
    desc: "When someone tags you in a comment, chat or a file.",
    icon: AtSign,
  },
  {
    key: "reminders",
    title: "Reminders",
    desc: "Reminders about updates you might have missed.",
    icon: BellRing,
  },
];

const CHANNELS: { key: keyof Channels; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { key: "push", label: "Push", icon: BellRing },
  { key: "email", label: "Email", icon: Mail },
];

function NotificationsTab() {
  const { form, setForm } = useProfileCtx();
  const n = form.notification_settings;
  const setGroup = (k: keyof NotificationSettings, v: Channels) =>
    setForm("notification_settings", { ...n, [k]: v });

  const activeCount = NOTIFICATION_GROUPS.reduce(
    (acc, g) => acc + CHANNELS.filter((c) => n[g.key]?.[c.key]).length,
    0,
  );
  const totalCount = NOTIFICATION_GROUPS.length * CHANNELS.length;
  const allOn = activeCount === totalCount;

  const setAll = (on: boolean) => {
    const next = { ...n };
    for (const g of NOTIFICATION_GROUPS) next[g.key] = { ...next[g.key], push: on, email: on };
    setForm("notification_settings", next);
  };

  return (
    <>
      <div className="mb-6 flex flex-wrap items-start justify-between gap-4 border-b border-border pb-4">
        <div>
          <h2 className="text-base font-semibold">Notifications</h2>
          <p className="mt-1 text-sm text-muted-foreground">Choose when and how you&apos;d like to be notified.</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="rounded-full bg-muted px-2.5 py-1 text-xs font-medium text-muted-foreground">
            {activeCount} of {totalCount} enabled
          </span>
          <Button variant="outline" size="sm" className="h-8" onClick={() => setAll(!allOn)}>
            {allOn ? "Turn all off" : "Turn all on"}
          </Button>
        </div>
      </div>

      <div className="flex flex-col gap-4">
        {NOTIFICATION_GROUPS.map((g) => {
          const value = n[g.key];
          const Icon = g.icon;
          return (
            <div
              key={g.key}
              className="rounded-xl border border-border bg-card p-4 sm:p-5"
            >
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-start gap-3">
                  <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground">
                    <Icon className="size-4" />
                  </span>
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-foreground">{g.title}</div>
                    <p className="mt-0.5 text-xs text-muted-foreground">{g.desc}</p>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-4 sm:shrink-0">
                  {CHANNELS.map((c) => {
                    const CIcon = c.icon;
                    const checked = !!value?.[c.key];
                    return (
                      <label
                        key={c.key}
                        className="inline-flex cursor-pointer items-center gap-2 text-xs font-medium text-foreground"
                      >
                        <Checkbox
                          checked={checked}
                          onCheckedChange={(v) => setGroup(g.key, { ...value, [c.key]: !!v })}
                          aria-label={`${g.title} ${c.label}`}
                          className="size-4 border-border data-[state=checked]:border-brand-600 data-[state=checked]:bg-brand-600 data-[state=checked]:text-white"
                        />
                        <CIcon className="size-3.5 text-muted-foreground" />
                        {c.label}
                      </label>
                    );
                  })}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <p className="mt-5 flex items-center gap-2 text-xs text-muted-foreground">
        <Smartphone className="size-3.5" />
        Push notifications appear in the app bell; email notifications go to your account address.
      </p>
    </>
  );
}

/* ------------------------- Integrations tab ----------------------- */
type Integration = {
  name: string;
  desc: string;
  defaultEnabled: boolean;
  color: string;
  letter: string;
  categories: string[];
};

const INTEGRATION_CATALOG: Integration[] = [
  { name: "GitHub", desc: "Link pull requests and automate workflows.", defaultEnabled: true, color: "#181717", letter: "G", categories: ["Developer tools", "Productivity"] },
  { name: "Outlook", desc: "Link and sync user group data from Outlook.", defaultEnabled: true, color: "#0078D4", letter: "O", categories: ["Communication", "Productivity"] },
  { name: "Slack", desc: "Send notifications to channels and create projects.", defaultEnabled: true, color: "#4A154B", letter: "S", categories: ["Communication", "Productivity"] },
  { name: "Atlassian JIRA", desc: "Plan, track, and release great software.", defaultEnabled: false, color: "#2684FF", letter: "J", categories: ["Developer tools", "Productivity"] },
];


function RequestIntegrationModal({ onClose }: { onClose: () => void }) {
  const [appName, setAppName] = useState("");
  const [details, setDetails] = useState("");
  const [sending, setSending] = useState(false);

  const submit = async () => {
    const name = appName.trim();
    if (!name) {
      toast.error("Please enter the app name.");
      return;
    }
    setSending(true);
    try {
      const { data: auth } = await supabase.auth.getUser();
      const user = auth?.user;
      if (!user) {
        toast.error("You need to be signed in to request an integration.");
        return;
      }
      const { error } = await supabase.from("support_tickets").insert({
        user_id: user.id,
        subject: `Integration request: ${name}`,
        category: "general",
        message:
          `A new integration has been requested.\n\nApp: ${name}\n\nDetails:\n${details.trim() || "—"}`,
      });
      if (error) {
        toast.error("Could not send your request. Please try again.");
        return;
      }
      toast.success("Your integration request has been sent to support.");
      onClose();
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 p-4">
      <div className="w-full max-w-[460px] rounded-2xl border border-border bg-background p-6 shadow-xl">
        <h3 className="text-base font-semibold">Request an integration</h3>
        <p className="mt-1 text-sm text-muted-foreground">
          Tell us which app you'd like to connect. Your request is sent to the support team.
        </p>

        <label className="mt-4 block text-sm font-medium">App name</label>
        <input
          value={appName}
          onChange={(e) => setAppName(e.target.value)}
          placeholder="e.g. Notion, Zapier, Salesforce"
          className="mt-1.5 h-10 w-full rounded-lg border border-border bg-background px-3 text-sm outline-none focus:border-[#1565EF]"
        />

        <label className="mt-4 block text-sm font-medium">What do you need it for?</label>
        <textarea
          value={details}
          onChange={(e) => setDetails(e.target.value)}
          rows={4}
          placeholder="Describe how you'd use this integration"
          className="mt-1.5 w-full resize-none rounded-lg border border-border bg-background p-3 text-sm outline-none focus:border-[#1565EF]"
        />

        <div className="mt-6 flex justify-end gap-2">
          <Button variant="outline" className="h-9" onClick={onClose} disabled={sending}>
            Cancel
          </Button>
          <Button
            className="h-9 bg-[#1565EF] text-white hover:bg-[#1257cc] disabled:opacity-60"
            onClick={submit}
            disabled={sending}
          >
            {sending ? "Sending…" : "Send request"}
          </Button>

        </div>
      </div>
    </div>
  );
}

const PROVIDER_KEY: Record<string, string> = {
  GitHub: "github",
  Outlook: "outlook",
  Slack: "slack",
  "Atlassian JIRA": "jira",
};

function IntegrationsTab() {
  const [filter, setFilter] = useState("View all");
  const [requestOpen, setRequestOpen] = useState(false);
  const [modalFor, setModalFor] = useState<Integration | null>(null);
  const [states, setStates] = useState<Record<string, IntegrationState>>({});
  const [loading, setLoading] = useState(true);
  const filters = ["View all", "Developer tools", "Communication", "Productivity"];

  const refresh = useCallback(async () => {
    try {
      const list = await integrationsApi.list();
      const next: Record<string, IntegrationState> = {};
      for (const s of list) next[String(s.provider)] = s;
      setStates(next);
    } catch {
      /* leave cards in their default (not connected) state */
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const visible =
    filter === "View all"
      ? INTEGRATION_CATALOG
      : INTEGRATION_CATALOG.filter((i) => i.categories.includes(filter));

  const onToggleGithub = async (next: boolean) => {
    const key = "github";
    const prev = states[key];
    if (!prev?.has_token) {
      setModalFor(INTEGRATION_CATALOG.find((i) => i.name === "GitHub") ?? null);
      return;
    }
    setStates((s) => ({ ...s, [key]: { ...prev, enabled: next } }));
    try {
      await integrationsApi.patchGithub({ enabled: next });
    } catch (e: any) {
      setStates((s) => ({ ...s, [key]: prev }));
      toast.error(e?.message || "Could not update the integration.");
    }
  };

  const onDisconnectGithub = async () => {
    try {
      await integrationsApi.disconnectGithub();
      toast.success("GitHub disconnected.");
      await refresh();
    } catch (e: any) {
      toast.error(e?.message || "Could not disconnect GitHub.");
    }
  };

  return (
    <>
      {requestOpen && <RequestIntegrationModal onClose={() => setRequestOpen(false)} />}
      {modalFor && (
        <IntegrationConfigModal
          provider={PROVIDER_KEY[modalFor.name] ?? "unknown"}
          name={modalFor.name}
          state={states[PROVIDER_KEY[modalFor.name] ?? ""]}
          onClose={() => setModalFor(null)}
          onSaved={() => void refresh()}
        />
      )}
      <div className="mb-4 flex items-start justify-between">
        <div>
          <h2 className="text-base font-semibold">Integrations and connected apps</h2>
          <p className="mt-1 text-sm text-muted-foreground">Supercharge your workflow and connect the tool you use every day.</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" className="h-9 gap-2" onClick={() => setRequestOpen(true)}>
            <span className="text-lg leading-none">+</span> Request integration
          </Button>
          <button className="text-muted-foreground hover:text-foreground" aria-label="More">
            <MoreVertical className="h-5 w-5" />
          </button>
        </div>
      </div>

      <div className="mb-5 border-b border-border">
        <div className="flex items-center gap-6 overflow-x-auto no-scrollbar">
          {filters.map((f) => {
            const active = filter === f;
            return (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={cn(
                  "relative whitespace-nowrap py-2 text-sm transition-colors",
                  active ? "font-semibold text-[#1565EF]" : "text-muted-foreground hover:text-foreground",
                )}
              >
                {f}
                {active && <span className="absolute inset-x-0 -bottom-px h-0.5 rounded-full bg-[#1565EF]" />}
              </button>
            );
          })}
        </div>
      </div>

      <div className={cn("grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3", loading && "animate-pulse opacity-60")}>
        {visible.map((it) => {
          const key = PROVIDER_KEY[it.name] ?? "";
          const isGithub = key === "github";
          return (
            <IntegrationCard
              key={it.name}
              item={it}
              state={states[key]}
              interactive={isGithub}
              onToggle={isGithub ? onToggleGithub : undefined}
              onManage={() => setModalFor(it)}
              onDisconnect={isGithub ? onDisconnectGithub : undefined}
            />
          );
        })}
      </div>
      {visible.length === 0 && (
        <p className="py-10 text-center text-sm text-muted-foreground">
          No integrations in this category yet.
        </p>
      )}
    </>
  );
}

function IntegrationCard({
  item,
  state,
  interactive,
  onToggle,
  onManage,
  onDisconnect,
}: {
  item: Integration;
  state?: IntegrationState;
  interactive: boolean;
  onToggle?: (next: boolean) => void;
  onManage: () => void;
  onDisconnect?: () => void;
}) {
  const connected = !!state?.has_token;
  const systemDefault = !connected && !!state?.using_system_default;
  const enabled = connected ? !!state?.enabled : false;

  const actionLabel = !interactive
    ? "View integration"
    : connected
      ? "Manage"
      : systemDefault
        ? "Connect your account"
        : "Connect";

  return (
    <div className="flex flex-col rounded-xl border border-border bg-white">
      <div className="flex items-start justify-between gap-3 p-4">
        <div className="flex items-center gap-3">
          <span
            className="flex h-9 w-9 items-center justify-center rounded-md text-sm font-semibold text-white"
            style={{ backgroundColor: item.color }}
          >
            {item.letter}
          </span>
          <span className="text-sm font-semibold text-foreground">{item.name}</span>
        </div>
        <Toggle
          checked={enabled}
          disabled={!interactive || !connected}
          onChange={(v) => onToggle?.(v)}
        />
      </div>
      <p className="px-4 pb-2 text-sm text-muted-foreground">{item.desc}</p>
      {interactive && (
        <div className="px-4 pb-4 text-xs">
          {connected ? (
            <span className="font-mono text-foreground">{state?.config?.repo ?? "—"}</span>
          ) : systemDefault ? (
            <span className="rounded-full bg-muted px-2 py-0.5 text-muted-foreground">Using system default</span>
          ) : (
            <span className="text-muted-foreground">Not connected</span>
          )}
        </div>
      )}
      <div className="mt-auto flex items-center justify-between border-t border-border px-4 py-3">
        {interactive && connected && onDisconnect ? (
          <button type="button" className="text-sm font-semibold text-red-600 hover:underline" onClick={onDisconnect}>
            Disconnect
          </button>
        ) : (
          <span />
        )}
        <button type="button" className="text-sm font-semibold text-[#1565EF] hover:underline" onClick={onManage}>
          {actionLabel}
        </button>
      </div>
    </div>
  );
}

