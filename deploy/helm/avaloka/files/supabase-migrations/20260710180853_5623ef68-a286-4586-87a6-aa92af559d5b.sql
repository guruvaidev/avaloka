
-- user_settings
CREATE TABLE IF NOT EXISTS public.user_settings (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
  theme text,
  recovery_email text,
  news_and_updates boolean DEFAULT true,
  tips_and_tutorials boolean DEFAULT true,
  user_research boolean DEFAULT false,
  reminder_preference text DEFAULT 'all',
  comments_push boolean DEFAULT false,
  comments_email boolean DEFAULT false,
  comments_sms boolean DEFAULT false,
  tags_push boolean DEFAULT false,
  tags_email boolean DEFAULT false,
  tags_sms boolean DEFAULT false,
  reminders_push boolean DEFAULT false,
  reminders_email boolean DEFAULT false,
  reminders_sms boolean DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.user_settings TO authenticated;
GRANT ALL ON public.user_settings TO service_role;

ALTER TABLE public.user_settings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own settings"
  ON public.user_settings FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own settings"
  ON public.user_settings FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own settings"
  ON public.user_settings FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own settings"
  ON public.user_settings FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);

CREATE TRIGGER user_settings_set_updated_at
  BEFORE UPDATE ON public.user_settings
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- enterprise_settings
CREATE TABLE IF NOT EXISTS public.enterprise_settings (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid NOT NULL UNIQUE REFERENCES public.organizations(id) ON DELETE CASCADE,
  company_email text,
  company_name text,
  company_slug text,
  company_tagline text,
  company_logo_url text,
  branded_reports boolean DEFAULT true,
  branded_emails boolean DEFAULT true,
  ai_custom_instruction text,
  ai_insight_depth text DEFAULT 'conservative',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.enterprise_settings TO authenticated;
GRANT ALL ON public.enterprise_settings TO service_role;

ALTER TABLE public.enterprise_settings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view enterprise settings"
  ON public.enterprise_settings FOR SELECT
  TO authenticated
  USING (organization_id = public.current_org_id());

CREATE POLICY "Org members can insert enterprise settings"
  ON public.enterprise_settings FOR INSERT
  TO authenticated
  WITH CHECK (organization_id = public.current_org_id());

CREATE POLICY "Org members can update enterprise settings"
  ON public.enterprise_settings FOR UPDATE
  TO authenticated
  USING (organization_id = public.current_org_id())
  WITH CHECK (organization_id = public.current_org_id());

CREATE TRIGGER enterprise_settings_set_updated_at
  BEFORE UPDATE ON public.enterprise_settings
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
