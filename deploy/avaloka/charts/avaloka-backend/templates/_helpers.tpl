{{/*
Expand the name of the chart.
*/}}
{{- define "avaloka-backend.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
When .Values.color is set ("blue"|"green"), the color is appended so two parallel
blue-green releases coexist without name collision.
*/}}
{{- define "avaloka-backend.fullname" -}}
{{- $base := "" }}
{{- if .Values.fullnameOverride }}
{{- $base = .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- $base = .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $base = printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- if .Values.color }}
{{- printf "%s-%s" $base .Values.color | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $base }}
{{- end }}
{{- end }}

{{/*
Stable base name (no color suffix). Used by the router chart to discover
per-color Services regardless of which release rendered them.
*/}}
{{- define "avaloka-backend.basename" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- regexReplaceAll "-(blue|green)$" .Release.Name "" | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" (regexReplaceAll "-(blue|green)$" .Release.Name "") $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "avaloka-backend.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "avaloka-backend.labels" -}}
helm.sh/chart: {{ include "avaloka-backend.chart" . }}
{{ include "avaloka-backend.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels — include color when set so per-color Services pick the
right pods. Note: the avaloka-router chart uses ONLY the color label
(without instance) to select across releases.
*/}}
{{- define "avaloka-backend.selectorLabels" -}}
app.kubernetes.io/name: {{ include "avaloka-backend.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- if .Values.color }}
avaloka.io/color: {{ .Values.color | quote }}
{{- end }}
{{- end }}

{{/*
Color-only selector — used by the router Service which must match pods from
either the blue or green release without caring about instance name.
*/}}
{{- define "avaloka-backend.colorSelectorLabels" -}}
app.kubernetes.io/component: backend
avaloka.io/color: {{ .Values.color | quote }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "avaloka-backend.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "avaloka-backend.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
