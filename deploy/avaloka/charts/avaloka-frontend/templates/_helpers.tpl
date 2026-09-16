{{/*
Return the name of the chart
*/}}
{{- define "avaloka-frontend.name" -}}
{{- .Chart.Name -}}
{{- end -}}

{{/*
Return the fully qualified app name (release-name + chart-name).
When .Values.color is set, append it so blue/green releases coexist.
*/}}
{{- define "avaloka-frontend.fullname" -}}
{{- $base := printf "%s-%s" .Release.Name .Chart.Name -}}
{{- if .Values.color -}}
{{- printf "%s-%s" $base .Values.color | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $base | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{/*
Return the chart name and version
*/}}
{{- define "avaloka-frontend.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version -}}
{{- end -}}

{{/*
Common labels
*/}}
{{- define "avaloka-frontend.labels" -}}
app.kubernetes.io/name: {{ include "avaloka-frontend.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
helm.sh/chart: {{ include "avaloka-frontend.chart" . }}
{{- end -}}


{{/*
Return the selector labels for the frontend Deployment/Service.
Includes the color label when set so per-color Services target only their
own pods; the router Service uses the color-only selector below.
*/}}
{{- define "avaloka-frontend.selectorLabels" -}}
app.kubernetes.io/name: {{ include "avaloka-frontend.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- if .Values.color }}
avaloka.io/color: {{ .Values.color | quote }}
{{- end }}
{{- end -}}


{{/*
Create the name of the service account to use
*/}}
{{- define "avaloka-frontend.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{ include "avaloka-frontend.fullname" . }}
{{- else }}
{{ .Values.serviceAccount.name }}
{{- end }}
{{- end }}

