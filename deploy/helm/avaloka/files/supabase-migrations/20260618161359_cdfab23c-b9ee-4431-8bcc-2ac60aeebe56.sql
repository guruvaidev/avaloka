
ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS company_email text,
  ADD COLUMN IF NOT EXISTS recovery_email text,
  ADD COLUMN IF NOT EXISTS email_settings jsonb NOT NULL DEFAULT '{"prefs_on":true,"news":true,"tips":true,"research":false,"reminder":"all"}'::jsonb,
  ADD COLUMN IF NOT EXISTS notification_settings jsonb NOT NULL DEFAULT '{"comments":{"push":true,"email":true,"sms":false},"tags":{"push":true,"email":false,"sms":false},"reminders":{"push":false,"email":false,"sms":false}}'::jsonb,
  ADD COLUMN IF NOT EXISTS integration_settings jsonb NOT NULL DEFAULT '{}'::jsonb;
