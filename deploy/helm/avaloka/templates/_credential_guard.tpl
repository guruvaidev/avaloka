{{/*
Refuse to render when a deployment that asked for its own credentials is still
carrying Supabase's published demo keys.

Those keys are public: anyone can mint a service_role token against a cluster
that still uses them. The chart defaults to them so a local `helm install`
works with no setup, so this guard is opt-in -- but once
`supabase.requireOwnCredentials` is set, shipping the demo values becomes a
render error rather than a quiet security hole.
*/}}
{{- define "avaloka.checkSupabaseCredentials" -}}
{{- if .Values.supabase.requireOwnCredentials -}}
{{- if eq (.Values.supabase.jwtSecret | default "") "super-secret-jwt-token-with-at-least-32-characters-long" -}}
{{- fail "supabase.requireOwnCredentials is true but supabase.jwtSecret is still Supabase's published demo secret. Generate one with: openssl rand -base64 48 | tr -d '\n', then mint anonKey and serviceKey signed with it (HS256)." -}}
{{- end -}}
{{- if contains "supabase-demo" (.Values.supabase.anonKey | default "") -}}
{{- fail "supabase.requireOwnCredentials is true but supabase.anonKey is still the published demo key (iss=supabase-demo). Mint your own, signed with your jwtSecret." -}}
{{- end -}}
{{- if contains "supabase-demo" (.Values.supabase.serviceKey | default "") -}}
{{- fail "supabase.requireOwnCredentials is true but supabase.serviceKey is still the published demo key (iss=supabase-demo). That token grants full read/write on every table and bypasses row-level security. Mint your own." -}}
{{- end -}}
{{- end -}}
{{- end -}}
