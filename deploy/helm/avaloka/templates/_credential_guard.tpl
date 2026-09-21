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
{{- if and .Values.supabase.requireOwnCredentials .Values.supabase.enabled -}}
{{- if not (.Values.supabase.jwtSecret | default "") -}}
{{- fail "supabase.enabled is true but supabase.jwtSecret is empty. The chart ships no credentials on purpose -- a usable secret in a public repository is a forgeable auth stack. Generate one with: openssl rand -base64 48 | tr -d '\n', mint anonKey and serviceKey HS256-signed with it, and pass all three with --set-string (or export SUPABASE_JWT_SECRET / SUPABASE_ANON_KEY / SUPABASE_SERVICE_KEY and let `make up` forward them). Set supabase.requireOwnCredentials=false only for a throwaway loopback cluster." -}}
{{- end -}}
{{- if not (.Values.supabase.anonKey | default "") -}}
{{- fail "supabase.enabled is true but supabase.anonKey is empty. Mint it HS256-signed with your jwtSecret; PostgREST verifies the signature and answers PGRST301 otherwise." -}}
{{- end -}}
{{- if not (.Values.supabase.serviceKey | default "") -}}
{{- fail "supabase.enabled is true but supabase.serviceKey is empty. Mint it HS256-signed with your jwtSecret. This token grants full read/write and bypasses row-level security, so it must be yours." -}}
{{- end -}}
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
