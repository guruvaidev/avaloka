{{- define "avaloka.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "avaloka.fullname" -}}
{{- printf "%s" (include "avaloka.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "avaloka.labels" -}}
app.kubernetes.io/name: {{ include "avaloka.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{- define "avaloka.selectorLabels" -}}
app.kubernetes.io/name: {{ include "avaloka.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "avaloka.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "avaloka.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/* Name of the Secret holding the GROQ keys (existing or chart-managed). */}}
{{- define "avaloka.secretName" -}}
{{- if .Values.existingSecret -}}
{{- .Values.existingSecret -}}
{{- else -}}
{{- printf "%s-secrets" (include "avaloka.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/* Effective Ray address based on connect-vs-provision toggle. */}}
{{- define "avaloka.rayAddress" -}}
{{- if .Values.ray.connectExisting -}}
{{- .Values.ray.address -}}
{{- else -}}
{{- .Values.ray.inClusterAddress -}}
{{- end -}}
{{- end -}}

{{/* Effective Redis URL: chart-managed service when enabled, else external URL. */}}
{{- define "avaloka.redisUrl" -}}
{{- if .Values.redis.enabled -}}
{{- printf "redis://%s-redis:%v/0" (include "avaloka.fullname" .) .Values.redis.port -}}
{{- else -}}
{{- .Values.redis.externalUrl -}}
{{- end -}}
{{- end -}}

{{/* Chroma host: chart-managed service when enabled, else external host. */}}
{{- define "avaloka.chromaHost" -}}
{{- if .Values.chroma.enabled -}}
{{- printf "%s-chroma" (include "avaloka.fullname" .) -}}
{{- else -}}
{{- .Values.chroma.externalHost -}}
{{- end -}}
{{- end -}}

{{/* Postgres URL: chart-managed service when enabled, else external URL. */}}
{{- define "avaloka.postgresUrl" -}}
{{- if .Values.postgres.enabled -}}
{{- printf "postgresql://%s:%s@%s-postgres:%v/%s" .Values.postgres.user .Values.postgres.password (include "avaloka.fullname" .) .Values.postgres.port .Values.postgres.database -}}
{{- else -}}
{{- .Values.postgres.externalUrl -}}
{{- end -}}
{{- end -}}

{{/* Web UI's browser-facing Supabase URL: explicit override, else the in-cluster
     Supabase gateway (kong NodePort), else empty. */}}
{{- define "avaloka.webuiSupabaseUrl" -}}
{{- if .Values.webui.supabaseUrl -}}
{{- .Values.webui.supabaseUrl -}}
{{- else if .Values.supabase.enabled -}}
{{- .Values.supabase.externalUrl -}}
{{- end -}}
{{- end -}}

{{/* API's server-side Supabase URL. Called pod-to-pod, so it uses the kong
     ClusterIP Service DNS; cloud_connections.py reads it for /buckets/list.

     It also selects the JWKS the API verifies user tokens against. Tokens are
     ES256, so this must name the same project the browser signs in to -- the UI
     pins an external one in config.ts. deploy_stack.py passes it from
     SUPABASE_URL; set it explicitly if you deploy another way, or every request
     verifies as anonymous. */}}
{{- define "avaloka.apiSupabaseUrl" -}}
{{- if .Values.config.supabaseUrl -}}
{{- .Values.config.supabaseUrl -}}
{{- else if .Values.supabase.enabled -}}
{{- printf "http://%s-supabase-kong:%v" (include "avaloka.fullname" .) .Values.supabase.kong.port -}}
{{- end -}}
{{- end -}}

{{/* MCP + onboarding URLs the API calls. The chart deploys the MCP service, so
     default to it -- otherwise settings.py falls back to localhost:8080 inside the
     API pod and every tool call fails. */}}
{{- define "avaloka.mcpServerUrl" -}}
{{- if .Values.config.mcpServerUrl -}}
{{- .Values.config.mcpServerUrl -}}
{{- else if .Values.mcp.enabled -}}
{{- printf "http://%s-mcp:%v" (include "avaloka.fullname" .) .Values.mcp.port -}}
{{- end -}}
{{- end -}}

{{- define "avaloka.onboardingApiUrl" -}}
{{- if .Values.config.onboardingApiUrl -}}
{{- .Values.config.onboardingApiUrl -}}
{{- else if .Values.mcp.enabled -}}
{{- printf "http://%s-mcp:%v" (include "avaloka.fullname" .) .Values.mcp.port -}}
{{- end -}}
{{- end -}}

{{- define "avaloka.webuiSupabaseAnonKey" -}}
{{- if .Values.webui.supabaseAnonKey -}}
{{- .Values.webui.supabaseAnonKey -}}
{{- else if .Values.supabase.enabled -}}
{{- .Values.supabase.anonKey -}}
{{- end -}}
{{- end -}}

{{/* Web UI server-side Supabase URL. The browser-facing localhost NodePort is
     unreachable from inside the SSR pod, so bundled Supabase uses Service DNS. */}}
{{- define "avaloka.webuiSupabaseInclusterUrl" -}}
{{- if .Values.webui.supabaseInclusterUrl -}}
{{- .Values.webui.supabaseInclusterUrl -}}
{{- else if .Values.config.supabaseUrl -}}
{{- .Values.config.supabaseUrl -}}
{{- else if .Values.webui.supabaseUrl -}}
{{- .Values.webui.supabaseUrl -}}
{{- else if .Values.supabase.enabled -}}
{{- printf "http://%s-supabase-kong:%v" (include "avaloka.fullname" .) .Values.supabase.kong.port -}}
{{- end -}}
{{- end -}}

{{- /* MCP endpoints surfaced to the Web UI. */ -}}
{{- define "avaloka.webuiMcpApiBase" -}}
{{- if .Values.webui.mcp.apiBase -}}
{{- .Values.webui.mcp.apiBase -}}
{{- else -}}
{{- "" -}}
{{- end -}}
{{- end -}}

{{- define "avaloka.webuiMcpToolBase" -}}
{{- if .Values.webui.mcp.toolBase -}}
{{- .Values.webui.mcp.toolBase -}}
{{- else -}}
{{- "" -}}
{{- end -}}
{{- end -}}

{{/* ---------------------------------------------------------------------------
     MinIO / S3-compatible object storage helpers.
     --------------------------------------------------------------------------- */}}

{{/* Which in-cluster object store this release runs: "seaweedfs", "minio", or "".

     SeaweedFS is the default; minio.yaml is retained, values-gated, for one
     release so an existing on-prem install can roll back. Enabling both is a
     configuration error rather than a merge — they would claim the same role and
     the app can only have one S3 endpoint — so it is refused loudly here instead
     of silently resolving to whichever branch happens to be checked first. */}}
{{- define "avaloka.objectStoreName" -}}
{{- if and .Values.seaweedfs.enabled .Values.minio.enabled -}}
{{- fail "seaweedfs.enabled and minio.enabled are both true; the chart runs exactly one in-cluster object store (minio is the rollback path for one release)" -}}
{{- else if .Values.seaweedfs.enabled -}}
{{- "seaweedfs" -}}
{{- else if .Values.minio.enabled -}}
{{- "minio" -}}
{{- end -}}
{{- end -}}

{{/* True when this release runs an in-cluster object store at all. */}}
{{- define "avaloka.objectStoreEnabled" -}}
{{- if include "avaloka.objectStoreName" . -}}
{{- "true" -}}
{{- end -}}
{{- end -}}

{{/* True when the in-cluster object store should back the app's object storage.

     An explicit cloud backend always wins: a GKE/EKS/AKS overlay sets
     config.storageBackend to gcs/s3/azure, and running an in-cluster store
     alongside it (for something else) must not silently redirect the app's
     datasets into it. So the in-cluster store is adopted only when the configured
     backend is local/empty — or when it is "s3" with no bucket of its own, which
     is exactly the local-S3 case.

     Renamed from avaloka.minioIsStorageBackend: the guard never had anything to do
     with MinIO specifically, and leaving the old name in place while it returned
     true for a SeaweedFS install would have read as a bug at every call site. */}}
{{- define "avaloka.objectStoreIsStorageBackend" -}}
{{- $backend := .Values.config.storageBackend | default "" | lower -}}
{{- if and (include "avaloka.objectStoreEnabled" .) (or (has $backend (list "" "local")) (and (eq $backend "s3") (not .Values.config.s3Bucket))) -}}
{{- "true" -}}
{{- end -}}
{{- end -}}

{{/* Effective STORAGE_BACKEND: s3 when the in-cluster store is adopted, else
     whatever is configured. */}}
{{- define "avaloka.storageBackend" -}}
{{- if include "avaloka.objectStoreIsStorageBackend" . -}}
{{- "s3" -}}
{{- else -}}
{{- .Values.config.storageBackend -}}
{{- end -}}
{{- end -}}

{{/* In-cluster S3 endpoint. Pod-to-pod, so the ClusterIP Service DNS.

     SeaweedFS serves S3 on 8333 (weed's default) where MinIO served 9000, and the
     Service name carries the component suffix, so both the port and the host move
     when the backend does. Everything downstream reads this helper rather than
     rebuilding the hostname, which is why no application code changes. */}}
{{- define "avaloka.objectStoreEndpoint" -}}
{{- $name := include "avaloka.objectStoreName" . -}}
{{- if eq $name "seaweedfs" -}}
{{- printf "http://%s-seaweedfs:%v" (include "avaloka.fullname" .) .Values.seaweedfs.port -}}
{{- else if eq $name "minio" -}}
{{- printf "http://%s-minio:%v" (include "avaloka.fullname" .) .Values.minio.port -}}
{{- end -}}
{{- end -}}

{{/* host:port form of the above, for consumers that want no scheme (Milvus). */}}
{{- define "avaloka.objectStoreAddress" -}}
{{- $name := include "avaloka.objectStoreName" . -}}
{{- if eq $name "seaweedfs" -}}
{{- printf "%s-seaweedfs:%v" (include "avaloka.fullname" .) .Values.seaweedfs.port -}}
{{- else if eq $name "minio" -}}
{{- printf "%s-minio:%v" (include "avaloka.fullname" .) .Values.minio.port -}}
{{- end -}}
{{- end -}}

{{/* Name of the Secret holding the in-cluster store's credentials. */}}
{{- define "avaloka.objectStoreSecretName" -}}
{{- printf "%s-%s" (include "avaloka.fullname" .) (include "avaloka.objectStoreName" .) -}}
{{- end -}}

{{/* Keys inside that Secret. MinIO's server reads its root credentials from
     MINIO_ROOT_USER/MINIO_ROOT_PASSWORD via envFrom, so its Secret is stuck with
     those key names; the SeaweedFS Secret publishes the standard AWS names. A
     consumer that needs a secretKeyRef asks for the key name rather than
     hardcoding one backend's spelling. */}}
{{- define "avaloka.objectStoreAccessKeyName" -}}
{{- if eq (include "avaloka.objectStoreName" .) "minio" -}}
{{- "MINIO_ROOT_USER" -}}
{{- else -}}
{{- "AWS_ACCESS_KEY_ID" -}}
{{- end -}}
{{- end -}}

{{- define "avaloka.objectStoreSecretKeyName" -}}
{{- if eq (include "avaloka.objectStoreName" .) "minio" -}}
{{- "MINIO_ROOT_PASSWORD" -}}
{{- else -}}
{{- "AWS_SECRET_ACCESS_KEY" -}}
{{- end -}}
{{- end -}}

{{/* Credential values for the active in-cluster store. */}}
{{- define "avaloka.objectStoreAccessKey" -}}
{{- if eq (include "avaloka.objectStoreName" .) "minio" -}}
{{- .Values.minio.auth.accessKey -}}
{{- else -}}
{{- .Values.seaweedfs.auth.accessKey -}}
{{- end -}}
{{- end -}}

{{- define "avaloka.objectStoreSecretKey" -}}
{{- if eq (include "avaloka.objectStoreName" .) "minio" -}}
{{- .Values.minio.auth.secretKey -}}
{{- else -}}
{{- .Values.seaweedfs.auth.secretKey -}}
{{- end -}}
{{- end -}}

{{/* Bucket the active in-cluster store holds for the app. */}}
{{- define "avaloka.objectStoreBucket" -}}
{{- if eq (include "avaloka.objectStoreName" .) "minio" -}}
{{- .Values.minio.bucket -}}
{{- else -}}
{{- .Values.seaweedfs.bucket -}}
{{- end -}}
{{- end -}}

{{/* Effective S3 endpoint override: the in-cluster store when adopted, else an
     explicit setting, else empty (real AWS S3). */}}
{{- define "avaloka.s3EndpointUrl" -}}
{{- if include "avaloka.objectStoreIsStorageBackend" . -}}
{{- include "avaloka.objectStoreEndpoint" . -}}
{{- else -}}
{{- .Values.config.s3EndpointUrl -}}
{{- end -}}
{{- end -}}

{{- define "avaloka.s3Bucket" -}}
{{- if include "avaloka.objectStoreIsStorageBackend" . -}}
{{- include "avaloka.objectStoreBucket" . -}}
{{- else -}}
{{- .Values.config.s3Bucket -}}
{{- end -}}
{{- end -}}

{{/* ---------------------------------------------------------------------------
     MLflow helpers. The chart-managed server is the only component that receives
     the PostgreSQL backend URI; clients use its HTTP Service instead.
     --------------------------------------------------------------------------- */}}

{{- define "avaloka.mlflowTrackingUri" -}}
{{- if .Values.mlflow.trackingUri -}}
{{- .Values.mlflow.trackingUri -}}
{{- else if .Values.mlflow.enabled -}}
{{- printf "http://%s-mlflow:%v" (include "avaloka.fullname" .) .Values.mlflow.port -}}
{{- end -}}
{{- end -}}

{{- define "avaloka.mlflowRegistryUri" -}}
{{- default (include "avaloka.mlflowTrackingUri" .) .Values.mlflow.registryUri -}}
{{- end -}}

{{- define "avaloka.mlflowBackendStoreUri" -}}
{{- if .Values.mlflow.backendStoreUri -}}
{{- .Values.mlflow.backendStoreUri -}}
{{- else if or .Values.postgres.enabled .Values.postgres.externalUrl -}}
{{- include "avaloka.postgresUrl" . -}}
{{- end -}}
{{- end -}}

{{- define "avaloka.mlflowArtifactsDestination" -}}
{{- if .Values.mlflow.artifactsDestination -}}
{{- .Values.mlflow.artifactsDestination -}}
{{- else if and (eq (include "avaloka.storageBackend" . | lower) "gcs") .Values.config.gcsBucket -}}
{{- printf "gs://%s/mlflow-artifacts" .Values.config.gcsBucket -}}
{{- else if include "avaloka.objectStoreEnabled" . -}}
{{- printf "s3://%s/mlflow-artifacts" (include "avaloka.objectStoreBucket" .) -}}
{{- else -}}
{{- "" -}}
{{- end -}}
{{- end -}}

{{- define "avaloka.mlflowAllowedHosts" -}}
{{- if .Values.mlflow.allowedHosts -}}
{{- .Values.mlflow.allowedHosts -}}
{{- else -}}
{{- $name := printf "%s-mlflow" (include "avaloka.fullname" .) -}}
{{- printf "%s,%s:%v,%s.%s.svc,%s.%s.svc.cluster.local,localhost,127.0.0.1" $name $name .Values.mlflow.port $name .Release.Namespace $name .Release.Namespace -}}
{{- end -}}
{{- end -}}

{{/* ---------------------------------------------------------------------------
     Gateway API / cloud Application Load Balancer helpers.
     --------------------------------------------------------------------------- */}}

{{- define "avaloka.gatewayName" -}}
{{- default (printf "%s-gateway" (include "avaloka.fullname" .)) .Values.gateway.name -}}
{{- end -}}

{{/* GatewayClass: explicit override wins, else the provider's managed class.
     gcp   -> global external Application Load Balancer (the only class Certificate
              Manager cert maps work with).
     aws   -> a GatewayClass the operator creates with controllerName
              gateway.k8s.aws/alb (AWS Load Balancer Controller >= v2.14).
     azure -> installed by the Application Gateway for Containers ALB Controller. */}}
{{- define "avaloka.gatewayClassName" -}}
{{- if .Values.gateway.className -}}
{{- .Values.gateway.className -}}
{{- else if eq .Values.gateway.provider "gcp" -}}
{{- "gke-l7-global-external-managed" -}}
{{- else if eq .Values.gateway.provider "aws" -}}
{{- "alb" -}}
{{- else if eq .Values.gateway.provider "azure" -}}
{{- "azure-alb-external" -}}
{{- else -}}
{{- fail "gateway.enabled=true requires gateway.provider (gcp|aws|azure) or an explicit gateway.className" -}}
{{- end -}}
{{- end -}}

{{/* Catch-all backend for `/`. The web UI when it ships (its nginx also proxies any
     API path not matched directly at the load balancer), otherwise the API. */}}
{{- define "avaloka.gatewayDefaultBackend" -}}
{{- if .Values.webui.enabled -}}
{{- printf "%s-webui" (include "avaloka.fullname" .) -}}
{{- else -}}
{{- include "avaloka.fullname" . -}}
{{- end -}}
{{- end -}}

{{- define "avaloka.gatewayDefaultBackendPort" -}}
{{- if .Values.webui.enabled -}}
{{- .Values.webui.service.port -}}
{{- else -}}
{{- .Values.service.port -}}
{{- end -}}
{{- end -}}

{{/* Services the Gateway actually sends traffic to, as JSON so the GCP policy
     template can range over them. Only backends referenced by an HTTPRoute get a
     policy — a HealthCheckPolicy/GCPBackendPolicy pointing at a Service with no
     load-balancer backend behind it just reports an unattached condition forever.
     Services named by gateway.extraRoutes are deliberately excluded: the chart
     does not know their health paths or timeouts. */}}
{{- define "avaloka.gatewayGcpBackends" -}}
{{- $g := .Values.gateway -}}
{{- $fullname := include "avaloka.fullname" . -}}
{{- $backends := list -}}
{{- if or (not .Values.webui.enabled) $g.routeApiDirect -}}
{{- $backends = append $backends (dict
      "service" $fullname
      "containerPort" (.Values.service.targetPort | int)
      "healthPath" "/health"
      "timeoutSec" ($g.gcp.api.timeoutSec | int)
      "sessionAffinity" $g.gcp.api.sessionAffinity) -}}
{{- end -}}
{{- if .Values.webui.enabled -}}
{{- $backends = append $backends (dict
      "service" (printf "%s-webui" $fullname)
      "containerPort" (.Values.webui.containerPort | int)
      "healthPath" $g.gcp.webui.healthPath
      "timeoutSec" ($g.gcp.webui.timeoutSec | int)
      "sessionAffinity" $g.gcp.webui.sessionAffinity) -}}
{{- end -}}
{{- if .Values.supabase.enabled -}}
{{- $backends = append $backends (dict
     "service" (printf "%s-supabase-kong" $fullname)
     "containerPort" (.Values.supabase.kong.port | int)
     "healthPath" ""
     "timeoutSec" ($g.gcp.api.timeoutSec | int)
     "sessionAffinity" $g.gcp.api.sessionAffinity) -}}
{{- end -}}
{{- $backends | toJson -}}
{{- end -}}

{{/* LangGraph API URL: chart-managed service when enabled, else external/static URL. */}}
{{- define "avaloka.langgraphUrl" -}}
{{- if .Values.langgraph.enabled -}}
{{- printf "http://%s-langgraph:%v" (include "avaloka.fullname" .) .Values.langgraph.port -}}
{{- else if .Values.langgraph.externalUrl -}}
{{- .Values.langgraph.externalUrl -}}
{{- else -}}
{{- .Values.config.langgraphApiUrl -}}
{{- end -}}
{{- end -}}

{{/* In-cluster OpenAI-compatible local model service. */}}
{{- define "avaloka.localLLMServiceName" -}}
{{- printf "%s-local-llm" (include "avaloka.fullname" .) -}}
{{- end -}}

{{- define "avaloka.localLLMPort" -}}
{{- if eq .Values.localLLM.engine "vllm" -}}8000{{- else -}}11434{{- end -}}
{{- end -}}

{{- define "avaloka.localLLMBaseUrl" -}}
{{- printf "http://%s:%s/v1" (include "avaloka.localLLMServiceName" .) (include "avaloka.localLLMPort" .) -}}
{{- end -}}

{{/*
Storage class for a component's PersistentVolumeClaim.

Resolution order: the component's own persistence.storageClass, then the
chart-wide storage.className, then nothing -- which lets the cluster's default
StorageClass decide, so a `kind` install keeps working with no configuration.

The empty string is deliberately distinct from unset. `storageClassName: ""`
disables dynamic provisioning entirely and binds only a pre-created PV, which
is not what an unset value should mean, so the key is omitted rather than
emitted empty.

Usage: {{ include "avaloka.storageClass" (dict "ctx" . "component" .Values.redis) }}
*/}}
{{- define "avaloka.storageClass" -}}
{{- $component := .component | default dict -}}
{{- $persistence := $component.persistence | default dict -}}
{{- $class := $persistence.storageClass | default (.ctx.Values.storage).className | default "" -}}
{{- if $class }}
storageClassName: {{ $class | quote }}
{{- end }}
{{- end -}}

{{/*
Access mode for a component's PVC. ReadWriteOnce is correct for every store in
this chart: each is a single writer, so no volume here needs RWX.

What RWO does NOT buy is node-loss survival, and this docstring used to imply it
did ("the volume follows it to whichever node it lands on"). With the default
StorageClass it does not follow anything: a `kind`/local-path PV is a hostPath
carrying nodeAffinity pinned to the node that provisioned it, and a pod forced
onto any other node stays Pending (FailedScheduling) indefinitely -- verified on
a live cluster, not reasoned about. Replicated *storage* (Longhorn, Ceph,
Mayastor) is what makes the volume available elsewhere when a node dies; see the
storage.className commentary in values.yaml, which already states this correctly.
*/}}
{{- define "avaloka.accessMode" -}}
{{- $component := .component | default dict -}}
{{- $persistence := $component.persistence | default dict -}}
{{- $persistence.accessMode | default (.ctx.Values.storage).accessMode | default "ReadWriteOnce" -}}
{{- end -}}
