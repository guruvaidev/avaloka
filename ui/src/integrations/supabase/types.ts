export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "14.5"
  }
  public: {
    Tables: {
      analyses: {
        Row: {
          created_at: string
          created_by_initials: string | null
          created_by_name: string | null
          created_by_role: string | null
          dataset_id: string | null
          dataset_ids: Json
          deleted_at: string | null
          filename: string | null
          id: string
          name: string
          organization_id: string | null
          owner_id: string
          parent_analysis_id: string | null
          project_id: string | null
          samples: Json | null
          schema: Json | null
          session_id: string | null
          team_count: number
          team_label: string
          thread_id: string | null
          updated_at: string
          viz_config: Json | null
        }
        Insert: {
          created_at?: string
          created_by_initials?: string | null
          created_by_name?: string | null
          created_by_role?: string | null
          dataset_id?: string | null
          dataset_ids?: Json
          deleted_at?: string | null
          filename?: string | null
          id?: string
          name: string
          organization_id?: string | null
          owner_id: string
          parent_analysis_id?: string | null
          project_id?: string | null
          samples?: Json | null
          schema?: Json | null
          session_id?: string | null
          team_count?: number
          team_label?: string
          thread_id?: string | null
          updated_at?: string
          viz_config?: Json | null
        }
        Update: {
          created_at?: string
          created_by_initials?: string | null
          created_by_name?: string | null
          created_by_role?: string | null
          dataset_id?: string | null
          dataset_ids?: Json
          deleted_at?: string | null
          filename?: string | null
          id?: string
          name?: string
          organization_id?: string | null
          owner_id?: string
          parent_analysis_id?: string | null
          project_id?: string | null
          samples?: Json | null
          schema?: Json | null
          session_id?: string | null
          team_count?: number
          team_label?: string
          thread_id?: string | null
          updated_at?: string
          viz_config?: Json | null
        }
        Relationships: [
          {
            foreignKeyName: "analyses_parent_analysis_id_fkey"
            columns: ["parent_analysis_id"]
            isOneToOne: false
            referencedRelation: "analyses"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "analyses_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      analysis_comments: {
        Row: {
          analysis_id: string
          attachments: Json
          author_id: string
          content: string
          created_at: string
          id: string
          parent_id: string | null
        }
        Insert: {
          analysis_id: string
          attachments?: Json
          author_id: string
          content: string
          created_at?: string
          id?: string
          parent_id?: string | null
        }
        Update: {
          analysis_id?: string
          attachments?: Json
          author_id?: string
          content?: string
          created_at?: string
          id?: string
          parent_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "analysis_comments_analysis_id_fkey"
            columns: ["analysis_id"]
            isOneToOne: false
            referencedRelation: "analyses"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "analysis_comments_author_id_fkey"
            columns: ["author_id"]
            isOneToOne: false
            referencedRelation: "org_member_directory"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "analysis_comments_author_id_fkey"
            columns: ["author_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "analysis_comments_parent_id_fkey"
            columns: ["parent_id"]
            isOneToOne: false
            referencedRelation: "analysis_comments"
            referencedColumns: ["id"]
          },
        ]
      }
      analysis_dashboards: {
        Row: {
          analysis_id: string
          created_at: string
          id: string
          owner_id: string
          updated_at: string
        }
        Insert: {
          analysis_id: string
          created_at?: string
          id?: string
          owner_id: string
          updated_at?: string
        }
        Update: {
          analysis_id?: string
          created_at?: string
          id?: string
          owner_id?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "analysis_dashboards_analysis_id_fkey"
            columns: ["analysis_id"]
            isOneToOne: true
            referencedRelation: "analyses"
            referencedColumns: ["id"]
          },
        ]
      }
      analysis_feedback: {
        Row: {
          analysis_id: string | null
          comment: string | null
          created_at: string | null
          feedback_type: string | null
          id: string
          insight_id: string | null
          owner_id: string | null
        }
        Insert: {
          analysis_id?: string | null
          comment?: string | null
          created_at?: string | null
          feedback_type?: string | null
          id?: string
          insight_id?: string | null
          owner_id?: string | null
        }
        Update: {
          analysis_id?: string | null
          comment?: string | null
          created_at?: string | null
          feedback_type?: string | null
          id?: string
          insight_id?: string | null
          owner_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "analysis_feedback_analysis_id_fkey"
            columns: ["analysis_id"]
            isOneToOne: false
            referencedRelation: "analyses"
            referencedColumns: ["id"]
          },
        ]
      }
      analysis_messages: {
        Row: {
          analysis_id: string
          author_id: string | null
          content: string
          created_at: string
          id: string
          output: Json | null
          role: Database["public"]["Enums"]["analysis_message_role"]
        }
        Insert: {
          analysis_id: string
          author_id?: string | null
          content: string
          created_at?: string
          id?: string
          output?: Json | null
          role: Database["public"]["Enums"]["analysis_message_role"]
        }
        Update: {
          analysis_id?: string
          author_id?: string | null
          content?: string
          created_at?: string
          id?: string
          output?: Json | null
          role?: Database["public"]["Enums"]["analysis_message_role"]
        }
        Relationships: [
          {
            foreignKeyName: "analysis_messages_analysis_id_fkey"
            columns: ["analysis_id"]
            isOneToOne: false
            referencedRelation: "analyses"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "analysis_messages_author_id_fkey"
            columns: ["author_id"]
            isOneToOne: false
            referencedRelation: "org_member_directory"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "analysis_messages_author_id_fkey"
            columns: ["author_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      app_users: {
        Row: {
          accepted_at: string | null
          auth_user_id: string | null
          branch: string
          created_at: string
          department: string
          email: string
          id: string
          invited_at: string | null
          invited_by: string | null
          organization_id: string | null
          permissions: Json
          profile_id: string | null
          role: string
          status: string
          team_id: string | null
          updated_at: string
          user_id_code: string
        }
        Insert: {
          accepted_at?: string | null
          auth_user_id?: string | null
          branch: string
          created_at?: string
          department: string
          email: string
          id?: string
          invited_at?: string | null
          invited_by?: string | null
          organization_id?: string | null
          permissions?: Json
          profile_id?: string | null
          role: string
          status?: string
          team_id?: string | null
          updated_at?: string
          user_id_code: string
        }
        Update: {
          accepted_at?: string | null
          auth_user_id?: string | null
          branch?: string
          created_at?: string
          department?: string
          email?: string
          id?: string
          invited_at?: string | null
          invited_by?: string | null
          organization_id?: string | null
          permissions?: Json
          profile_id?: string | null
          role?: string
          status?: string
          team_id?: string | null
          updated_at?: string
          user_id_code?: string
        }
        Relationships: [
          {
            foreignKeyName: "app_users_invited_by_fkey"
            columns: ["invited_by"]
            isOneToOne: false
            referencedRelation: "org_member_directory"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "app_users_invited_by_fkey"
            columns: ["invited_by"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "app_users_profile_id_fkey"
            columns: ["profile_id"]
            isOneToOne: false
            referencedRelation: "org_member_directory"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "app_users_profile_id_fkey"
            columns: ["profile_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "app_users_team_id_fkey"
            columns: ["team_id"]
            isOneToOne: false
            referencedRelation: "organization_teams"
            referencedColumns: ["id"]
          },
        ]
      }
      billing_invoices: {
        Row: {
          amount_cents: number
          created_at: string
          currency: string
          hosted_url: string | null
          id: string
          invoice_number: string | null
          organization_id: string
          paid_at: string | null
          period_end: string | null
          period_start: string | null
          provider: string
          provider_invoice_id: string | null
          status: string
        }
        Insert: {
          amount_cents?: number
          created_at?: string
          currency?: string
          hosted_url?: string | null
          id?: string
          invoice_number?: string | null
          organization_id: string
          paid_at?: string | null
          period_end?: string | null
          period_start?: string | null
          provider: string
          provider_invoice_id?: string | null
          status?: string
        }
        Update: {
          amount_cents?: number
          created_at?: string
          currency?: string
          hosted_url?: string | null
          id?: string
          invoice_number?: string | null
          organization_id?: string
          paid_at?: string | null
          period_end?: string | null
          period_start?: string | null
          provider?: string
          provider_invoice_id?: string | null
          status?: string
        }
        Relationships: [
          {
            foreignKeyName: "billing_invoices_organization_id_fkey"
            columns: ["organization_id"]
            isOneToOne: false
            referencedRelation: "organizations"
            referencedColumns: ["id"]
          },
        ]
      }
      branches: {
        Row: {
          created_at: string
          employees: number
          head_avatar: string | null
          head_email: string | null
          head_name: string | null
          id: string
          location: string
          name: string
          organization_id: string | null
          status: string
          updated_at: string
        }
        Insert: {
          created_at?: string
          employees?: number
          head_avatar?: string | null
          head_email?: string | null
          head_name?: string | null
          id?: string
          location: string
          name: string
          organization_id?: string | null
          status?: string
          updated_at?: string
        }
        Update: {
          created_at?: string
          employees?: number
          head_avatar?: string | null
          head_email?: string | null
          head_name?: string | null
          id?: string
          location?: string
          name?: string
          organization_id?: string | null
          status?: string
          updated_at?: string
        }
        Relationships: []
      }
      config_roles: {
        Row: {
          created_at: string
          department: string
          id: string
          organization_id: string | null
          permissions: string[]
          role: string
          status: string
          updated_at: string
        }
        Insert: {
          created_at?: string
          department: string
          id?: string
          organization_id?: string | null
          permissions?: string[]
          role: string
          status?: string
          updated_at?: string
        }
        Update: {
          created_at?: string
          department?: string
          id?: string
          organization_id?: string | null
          permissions?: string[]
          role?: string
          status?: string
          updated_at?: string
        }
        Relationships: []
      }
      customer_accounts: {
        Row: {
          billing_address: string | null
          billing_email: string | null
          company_name: string | null
          country: string | null
          created_at: string
          id: string
          owner_user_id: string
          plan_type: string | null
          updated_at: string
          vat_number: string | null
        }
        Insert: {
          billing_address?: string | null
          billing_email?: string | null
          company_name?: string | null
          country?: string | null
          created_at?: string
          id?: string
          owner_user_id: string
          plan_type?: string | null
          updated_at?: string
          vat_number?: string | null
        }
        Update: {
          billing_address?: string | null
          billing_email?: string | null
          company_name?: string | null
          country?: string | null
          created_at?: string
          id?: string
          owner_user_id?: string
          plan_type?: string | null
          updated_at?: string
          vat_number?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "customer_accounts_owner_user_id_fkey"
            columns: ["owner_user_id"]
            isOneToOne: true
            referencedRelation: "org_member_directory"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "customer_accounts_owner_user_id_fkey"
            columns: ["owner_user_id"]
            isOneToOne: true
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      dashboard_graphs: {
        Row: {
          analysis_id: string
          chart_type: string
          config: Json
          created_at: string
          graph_key: string
          id: string
          owner_id: string
          placed: boolean
          position: number
          tab_id: string
          title: string
          updated_at: string
        }
        Insert: {
          analysis_id: string
          chart_type?: string
          config?: Json
          created_at?: string
          graph_key: string
          id?: string
          owner_id: string
          placed?: boolean
          position?: number
          tab_id: string
          title: string
          updated_at?: string
        }
        Update: {
          analysis_id?: string
          chart_type?: string
          config?: Json
          created_at?: string
          graph_key?: string
          id?: string
          owner_id?: string
          placed?: boolean
          position?: number
          tab_id?: string
          title?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "dashboard_graphs_analysis_id_fkey"
            columns: ["analysis_id"]
            isOneToOne: false
            referencedRelation: "analyses"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "dashboard_graphs_tab_id_fkey"
            columns: ["tab_id"]
            isOneToOne: false
            referencedRelation: "dashboard_tabs"
            referencedColumns: ["id"]
          },
        ]
      }
      dashboard_tabs: {
        Row: {
          created_at: string
          dashboard_id: string
          id: string
          owner_id: string
          tab_index: number
          title: string
          updated_at: string
        }
        Insert: {
          created_at?: string
          dashboard_id: string
          id?: string
          owner_id: string
          tab_index: number
          title: string
          updated_at?: string
        }
        Update: {
          created_at?: string
          dashboard_id?: string
          id?: string
          owner_id?: string
          tab_index?: number
          title?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "dashboard_tabs_dashboard_id_fkey"
            columns: ["dashboard_id"]
            isOneToOne: false
            referencedRelation: "analysis_dashboards"
            referencedColumns: ["id"]
          },
        ]
      }
      data_source_files: {
        Row: {
          created_at: string
          data_source_id: string
          id: string
          owner_id: string
          path: string
          selected: boolean
          size_bytes: number | null
          uploaded: boolean
        }
        Insert: {
          created_at?: string
          data_source_id: string
          id?: string
          owner_id: string
          path: string
          selected?: boolean
          size_bytes?: number | null
          uploaded?: boolean
        }
        Update: {
          created_at?: string
          data_source_id?: string
          id?: string
          owner_id?: string
          path?: string
          selected?: boolean
          size_bytes?: number | null
          uploaded?: boolean
        }
        Relationships: [
          {
            foreignKeyName: "data_source_files_data_source_id_fkey"
            columns: ["data_source_id"]
            isOneToOne: false
            referencedRelation: "data_sources"
            referencedColumns: ["id"]
          },
        ]
      }
      data_sources: {
        Row: {
          config: Json
          created_at: string
          description: string | null
          domain: string | null
          id: string
          last_error: string | null
          last_tested_at: string | null
          name: string
          owner_id: string
          provider: Database["public"]["Enums"]["data_provider"]
          status: Database["public"]["Enums"]["data_source_status"]
          updated_at: string
          use_cases: string | null
        }
        Insert: {
          config?: Json
          created_at?: string
          description?: string | null
          domain?: string | null
          id?: string
          last_error?: string | null
          last_tested_at?: string | null
          name: string
          owner_id: string
          provider: Database["public"]["Enums"]["data_provider"]
          status?: Database["public"]["Enums"]["data_source_status"]
          updated_at?: string
          use_cases?: string | null
        }
        Update: {
          config?: Json
          created_at?: string
          description?: string | null
          domain?: string | null
          id?: string
          last_error?: string | null
          last_tested_at?: string | null
          name?: string
          owner_id?: string
          provider?: Database["public"]["Enums"]["data_provider"]
          status?: Database["public"]["Enums"]["data_source_status"]
          updated_at?: string
          use_cases?: string | null
        }
        Relationships: []
      }
      departments: {
        Row: {
          code: string
          created_at: string
          department: string
          employees: number
          head_profile_id: string | null
          id: string
          organization_id: string | null
          status: string
          updated_at: string
        }
        Insert: {
          code: string
          created_at?: string
          department: string
          employees?: number
          head_profile_id?: string | null
          id?: string
          organization_id?: string | null
          status?: string
          updated_at?: string
        }
        Update: {
          code?: string
          created_at?: string
          department?: string
          employees?: number
          head_profile_id?: string | null
          id?: string
          organization_id?: string | null
          status?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "departments_head_profile_id_fkey"
            columns: ["head_profile_id"]
            isOneToOne: false
            referencedRelation: "org_member_directory"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "departments_head_profile_id_fkey"
            columns: ["head_profile_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      enterprise_licenses: {
        Row: {
          created_at: string
          created_by: string | null
          customer_account_id: string
          expires_at: string | null
          id: string
          issued_at: string
          organization_id: string | null
          paid_at: string | null
          payment_amount: number
          payment_status: string
          serial_number: string
          status: string
          stripe_subscription_id: string | null
          updated_at: string
        }
        Insert: {
          created_at?: string
          created_by?: string | null
          customer_account_id: string
          expires_at?: string | null
          id?: string
          issued_at?: string
          organization_id?: string | null
          paid_at?: string | null
          payment_amount?: number
          payment_status?: string
          serial_number: string
          status?: string
          stripe_subscription_id?: string | null
          updated_at?: string
        }
        Update: {
          created_at?: string
          created_by?: string | null
          customer_account_id?: string
          expires_at?: string | null
          id?: string
          issued_at?: string
          organization_id?: string | null
          paid_at?: string | null
          payment_amount?: number
          payment_status?: string
          serial_number?: string
          status?: string
          stripe_subscription_id?: string | null
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "enterprise_licenses_created_by_fkey"
            columns: ["created_by"]
            isOneToOne: false
            referencedRelation: "org_member_directory"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "enterprise_licenses_created_by_fkey"
            columns: ["created_by"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "enterprise_licenses_customer_account_id_fkey"
            columns: ["customer_account_id"]
            isOneToOne: true
            referencedRelation: "customer_accounts"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "enterprise_licenses_organization_id_fkey"
            columns: ["organization_id"]
            isOneToOne: false
            referencedRelation: "organizations"
            referencedColumns: ["id"]
          },
        ]
      }
      enterprise_settings: {
        Row: {
          ai_custom_instruction: string | null
          ai_insight_depth: string | null
          branded_emails: boolean | null
          branded_reports: boolean | null
          company_email: string | null
          company_logo_url: string | null
          company_name: string | null
          company_slug: string | null
          company_tagline: string | null
          created_at: string
          id: string
          organization_id: string
          updated_at: string
        }
        Insert: {
          ai_custom_instruction?: string | null
          ai_insight_depth?: string | null
          branded_emails?: boolean | null
          branded_reports?: boolean | null
          company_email?: string | null
          company_logo_url?: string | null
          company_name?: string | null
          company_slug?: string | null
          company_tagline?: string | null
          created_at?: string
          id?: string
          organization_id: string
          updated_at?: string
        }
        Update: {
          ai_custom_instruction?: string | null
          ai_insight_depth?: string | null
          branded_emails?: boolean | null
          branded_reports?: boolean | null
          company_email?: string | null
          company_logo_url?: string | null
          company_name?: string | null
          company_slug?: string | null
          company_tagline?: string | null
          created_at?: string
          id?: string
          organization_id?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "enterprise_settings_organization_id_fkey"
            columns: ["organization_id"]
            isOneToOne: true
            referencedRelation: "organizations"
            referencedColumns: ["id"]
          },
        ]
      }
      models: {
        Row: {
          created_at: string
          id: string
          name: string
          organization_id: string | null
          tags: string[]
          task_type: string
          updated_at: string
          updated_label: string
          version: string
        }
        Insert: {
          created_at?: string
          id?: string
          name: string
          organization_id?: string | null
          tags?: string[]
          task_type: string
          updated_at?: string
          updated_label?: string
          version: string
        }
        Update: {
          created_at?: string
          id?: string
          name?: string
          organization_id?: string | null
          tags?: string[]
          task_type?: string
          updated_at?: string
          updated_label?: string
          version?: string
        }
        Relationships: []
      }
      notifications: {
        Row: {
          actor_id: string | null
          body: string | null
          created_at: string
          id: string
          link: string | null
          read_at: string | null
          recipient_id: string
          resource_id: string | null
          resource_type: string | null
          title: string
          type: string
        }
        Insert: {
          actor_id?: string | null
          body?: string | null
          created_at?: string
          id?: string
          link?: string | null
          read_at?: string | null
          recipient_id: string
          resource_id?: string | null
          resource_type?: string | null
          title: string
          type: string
        }
        Update: {
          actor_id?: string | null
          body?: string | null
          created_at?: string
          id?: string
          link?: string | null
          read_at?: string | null
          recipient_id?: string
          resource_id?: string | null
          resource_type?: string | null
          title?: string
          type?: string
        }
        Relationships: [
          {
            foreignKeyName: "notifications_actor_id_fkey"
            columns: ["actor_id"]
            isOneToOne: false
            referencedRelation: "org_member_directory"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "notifications_actor_id_fkey"
            columns: ["actor_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "notifications_recipient_id_fkey"
            columns: ["recipient_id"]
            isOneToOne: false
            referencedRelation: "org_member_directory"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "notifications_recipient_id_fkey"
            columns: ["recipient_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      organization_teams: {
        Row: {
          branch: string
          created_at: string
          description: string | null
          id: string
          lead_user_id: string | null
          name: string
          organization_id: string | null
          status: string
          updated_at: string
        }
        Insert: {
          branch: string
          created_at?: string
          description?: string | null
          id?: string
          lead_user_id?: string | null
          name: string
          organization_id?: string | null
          status?: string
          updated_at?: string
        }
        Update: {
          branch?: string
          created_at?: string
          description?: string | null
          id?: string
          lead_user_id?: string | null
          name?: string
          organization_id?: string | null
          status?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "organization_teams_lead_user_id_fkey"
            columns: ["lead_user_id"]
            isOneToOne: false
            referencedRelation: "app_users"
            referencedColumns: ["id"]
          },
        ]
      }
      organizations: {
        Row: {
          billing_email: string | null
          billing_interval: string | null
          billing_name: string | null
          brand_emails: boolean
          brand_reports: boolean
          canceled_at: string | null
          card_brand: string | null
          card_exp_month: number | null
          card_exp_year: number | null
          card_last4: string | null
          created_at: string
          currency: string
          current_period_end: string | null
          current_period_start: string | null
          id: string
          logo_url: string | null
          monthly_price_cents: number
          name: string
          owner_profile_id: string | null
          payment_provider: string | null
          payment_status: string | null
          paypal_payer_id: string | null
          paypal_subscription_id: string | null
          plan: string | null
          slug: string | null
          stripe_customer_id: string | null
          stripe_default_payment_method_id: string | null
          stripe_price_id: string | null
          stripe_subscription_id: string | null
          subscription_active: boolean
          subscription_status: string | null
          tagline: string | null
          trial_ends_at: string | null
          unit_amount_cents: number | null
          updated_at: string
        }
        Insert: {
          billing_email?: string | null
          billing_interval?: string | null
          billing_name?: string | null
          brand_emails?: boolean
          brand_reports?: boolean
          canceled_at?: string | null
          card_brand?: string | null
          card_exp_month?: number | null
          card_exp_year?: number | null
          card_last4?: string | null
          created_at?: string
          currency?: string
          current_period_end?: string | null
          current_period_start?: string | null
          id?: string
          logo_url?: string | null
          monthly_price_cents?: number
          name: string
          owner_profile_id?: string | null
          payment_provider?: string | null
          payment_status?: string | null
          paypal_payer_id?: string | null
          paypal_subscription_id?: string | null
          plan?: string | null
          slug?: string | null
          stripe_customer_id?: string | null
          stripe_default_payment_method_id?: string | null
          stripe_price_id?: string | null
          stripe_subscription_id?: string | null
          subscription_active?: boolean
          subscription_status?: string | null
          tagline?: string | null
          trial_ends_at?: string | null
          unit_amount_cents?: number | null
          updated_at?: string
        }
        Update: {
          billing_email?: string | null
          billing_interval?: string | null
          billing_name?: string | null
          brand_emails?: boolean
          brand_reports?: boolean
          canceled_at?: string | null
          card_brand?: string | null
          card_exp_month?: number | null
          card_exp_year?: number | null
          card_last4?: string | null
          created_at?: string
          currency?: string
          current_period_end?: string | null
          current_period_start?: string | null
          id?: string
          logo_url?: string | null
          monthly_price_cents?: number
          name?: string
          owner_profile_id?: string | null
          payment_provider?: string | null
          payment_status?: string | null
          paypal_payer_id?: string | null
          paypal_subscription_id?: string | null
          plan?: string | null
          slug?: string | null
          stripe_customer_id?: string | null
          stripe_default_payment_method_id?: string | null
          stripe_price_id?: string | null
          stripe_subscription_id?: string | null
          subscription_active?: boolean
          subscription_status?: string | null
          tagline?: string | null
          trial_ends_at?: string | null
          unit_amount_cents?: number | null
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "organizations_owner_profile_id_fkey"
            columns: ["owner_profile_id"]
            isOneToOne: false
            referencedRelation: "org_member_directory"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "organizations_owner_profile_id_fkey"
            columns: ["owner_profile_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      payment_methods: {
        Row: {
          brand: string | null
          cardholder: string | null
          created_at: string
          exp_month: number | null
          exp_year: number | null
          id: string
          is_default: boolean
          last4: string | null
          metadata: Json
          organization_id: string | null
          owner_profile_id: string
          provider: string
          provider_customer_id: string | null
          provider_payment_method_id: string | null
          provider_subscription_id: string | null
          status: string
          updated_at: string
        }
        Insert: {
          brand?: string | null
          cardholder?: string | null
          created_at?: string
          exp_month?: number | null
          exp_year?: number | null
          id?: string
          is_default?: boolean
          last4?: string | null
          metadata?: Json
          organization_id?: string | null
          owner_profile_id: string
          provider: string
          provider_customer_id?: string | null
          provider_payment_method_id?: string | null
          provider_subscription_id?: string | null
          status?: string
          updated_at?: string
        }
        Update: {
          brand?: string | null
          cardholder?: string | null
          created_at?: string
          exp_month?: number | null
          exp_year?: number | null
          id?: string
          is_default?: boolean
          last4?: string | null
          metadata?: Json
          organization_id?: string | null
          owner_profile_id?: string
          provider?: string
          provider_customer_id?: string | null
          provider_payment_method_id?: string | null
          provider_subscription_id?: string | null
          status?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "payment_methods_organization_id_fkey"
            columns: ["organization_id"]
            isOneToOne: false
            referencedRelation: "organizations"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "payment_methods_owner_profile_id_fkey"
            columns: ["owner_profile_id"]
            isOneToOne: false
            referencedRelation: "org_member_directory"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "payment_methods_owner_profile_id_fkey"
            columns: ["owner_profile_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      plan_change_log: {
        Row: {
          created_at: string
          from_plan: string | null
          from_plan_name: string | null
          id: string
          organization_id: string | null
          owner_profile_id: string | null
          payment_provider: string | null
          status: string | null
          subscription_id: string | null
          to_plan: string | null
          to_plan_name: string | null
        }
        Insert: {
          created_at?: string
          from_plan?: string | null
          from_plan_name?: string | null
          id?: string
          organization_id?: string | null
          owner_profile_id?: string | null
          payment_provider?: string | null
          status?: string | null
          subscription_id?: string | null
          to_plan?: string | null
          to_plan_name?: string | null
        }
        Update: {
          created_at?: string
          from_plan?: string | null
          from_plan_name?: string | null
          id?: string
          organization_id?: string | null
          owner_profile_id?: string | null
          payment_provider?: string | null
          status?: string | null
          subscription_id?: string | null
          to_plan?: string | null
          to_plan_name?: string | null
        }
        Relationships: []
      }
      profiles: {
        Row: {
          ai_depth: string
          ai_instructions: string | null
          avatar_url: string | null
          company_email: string | null
          company_name: string | null
          company_slug: string | null
          country: string | null
          created_at: string
          email_settings: Json
          emails_opt_in: boolean
          first_name: string | null
          full_name: string | null
          id: string
          integration_settings: Json
          is_admin: boolean
          last_name: string | null
          next_billing_date: string | null
          notification_settings: Json
          organization_id: string | null
          payment_provider: string | null
          phone: string | null
          recovery_email: string | null
          reports_opt_in: boolean
          selected_plan: string | null
          stripe_customer_id: string | null
          stripe_subscription_id: string | null
          subscription_active: boolean
          subscription_status: string | null
          tagline: string | null
          theme: string
          timezone: string | null
          trial_ends_at: string | null
          updated_at: string
          user_id: string | null
        }
        Insert: {
          ai_depth?: string
          ai_instructions?: string | null
          avatar_url?: string | null
          company_email?: string | null
          company_name?: string | null
          company_slug?: string | null
          country?: string | null
          created_at?: string
          email_settings?: Json
          emails_opt_in?: boolean
          first_name?: string | null
          full_name?: string | null
          id: string
          integration_settings?: Json
          is_admin?: boolean
          last_name?: string | null
          next_billing_date?: string | null
          notification_settings?: Json
          organization_id?: string | null
          payment_provider?: string | null
          phone?: string | null
          recovery_email?: string | null
          reports_opt_in?: boolean
          selected_plan?: string | null
          stripe_customer_id?: string | null
          stripe_subscription_id?: string | null
          subscription_active?: boolean
          subscription_status?: string | null
          tagline?: string | null
          theme?: string
          timezone?: string | null
          trial_ends_at?: string | null
          updated_at?: string
          user_id?: string | null
        }
        Update: {
          ai_depth?: string
          ai_instructions?: string | null
          avatar_url?: string | null
          company_email?: string | null
          company_name?: string | null
          company_slug?: string | null
          country?: string | null
          created_at?: string
          email_settings?: Json
          emails_opt_in?: boolean
          first_name?: string | null
          full_name?: string | null
          id?: string
          integration_settings?: Json
          is_admin?: boolean
          last_name?: string | null
          next_billing_date?: string | null
          notification_settings?: Json
          organization_id?: string | null
          payment_provider?: string | null
          phone?: string | null
          recovery_email?: string | null
          reports_opt_in?: boolean
          selected_plan?: string | null
          stripe_customer_id?: string | null
          stripe_subscription_id?: string | null
          subscription_active?: boolean
          subscription_status?: string | null
          tagline?: string | null
          theme?: string
          timezone?: string | null
          trial_ends_at?: string | null
          updated_at?: string
          user_id?: string | null
        }
        Relationships: []
      }
      projects: {
        Row: {
          archived: boolean
          bookmarked: boolean
          created_at: string
          deleted_at: string | null
          description: string | null
          id: string
          name: string
          organization_id: string | null
          owner_id: string
          pinned: boolean
          updated_at: string
        }
        Insert: {
          archived?: boolean
          bookmarked?: boolean
          created_at?: string
          deleted_at?: string | null
          description?: string | null
          id?: string
          name: string
          organization_id?: string | null
          owner_id: string
          pinned?: boolean
          updated_at?: string
        }
        Update: {
          archived?: boolean
          bookmarked?: boolean
          created_at?: string
          deleted_at?: string | null
          description?: string | null
          id?: string
          name?: string
          organization_id?: string | null
          owner_id?: string
          pinned?: boolean
          updated_at?: string
        }
        Relationships: []
      }
      report_comments: {
        Row: {
          attachments: Json
          author_id: string
          content: string
          created_at: string
          id: string
          parent_id: string | null
          report_id: string
        }
        Insert: {
          attachments?: Json
          author_id: string
          content: string
          created_at?: string
          id?: string
          parent_id?: string | null
          report_id: string
        }
        Update: {
          attachments?: Json
          author_id?: string
          content?: string
          created_at?: string
          id?: string
          parent_id?: string | null
          report_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "report_comments_author_id_fkey"
            columns: ["author_id"]
            isOneToOne: false
            referencedRelation: "org_member_directory"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "report_comments_author_id_fkey"
            columns: ["author_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "report_comments_parent_id_fkey"
            columns: ["parent_id"]
            isOneToOne: false
            referencedRelation: "report_comments"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "report_comments_report_id_fkey"
            columns: ["report_id"]
            isOneToOne: false
            referencedRelation: "reports"
            referencedColumns: ["id"]
          },
        ]
      }
      reports: {
        Row: {
          analysis_id: string
          archived: boolean
          bookmarked: boolean
          created_at: string
          created_by: string
          dataset_label: string | null
          id: string
          key_insights: Json | null
          organization_id: string
          pinned: boolean
          project_id: string | null
          status: string
          tab_id: string | null
          title: string
          updated_at: string
        }
        Insert: {
          analysis_id: string
          archived?: boolean
          bookmarked?: boolean
          created_at?: string
          created_by: string
          dataset_label?: string | null
          id?: string
          key_insights?: Json | null
          organization_id: string
          pinned?: boolean
          project_id?: string | null
          status?: string
          tab_id?: string | null
          title?: string
          updated_at?: string
        }
        Update: {
          analysis_id?: string
          archived?: boolean
          bookmarked?: boolean
          created_at?: string
          created_by?: string
          dataset_label?: string | null
          id?: string
          key_insights?: Json | null
          organization_id?: string
          pinned?: boolean
          project_id?: string | null
          status?: string
          tab_id?: string | null
          title?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "reports_analysis_id_fkey"
            columns: ["analysis_id"]
            isOneToOne: false
            referencedRelation: "analyses"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "reports_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      resource_collaborators: {
        Row: {
          access_level: string
          app_user_id: string
          created_at: string
          department: string | null
          id: string
          invited_by: string | null
          resource_id: string
          resource_type: string
          updated_at: string
        }
        Insert: {
          access_level?: string
          app_user_id: string
          created_at?: string
          department?: string | null
          id?: string
          invited_by?: string | null
          resource_id: string
          resource_type: string
          updated_at?: string
        }
        Update: {
          access_level?: string
          app_user_id?: string
          created_at?: string
          department?: string | null
          id?: string
          invited_by?: string | null
          resource_id?: string
          resource_type?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "resource_collaborators_app_user_id_fkey"
            columns: ["app_user_id"]
            isOneToOne: false
            referencedRelation: "app_users"
            referencedColumns: ["id"]
          },
        ]
      }
      resource_share_links: {
        Row: {
          created_at: string
          id: string
          is_public: boolean
          link_access: string
          resource_id: string
          resource_type: string
          share_token: string
          updated_at: string
        }
        Insert: {
          created_at?: string
          id?: string
          is_public?: boolean
          link_access?: string
          resource_id: string
          resource_type: string
          share_token?: string
          updated_at?: string
        }
        Update: {
          created_at?: string
          id?: string
          is_public?: boolean
          link_access?: string
          resource_id?: string
          resource_type?: string
          share_token?: string
          updated_at?: string
        }
        Relationships: []
      }
      support_ticket_messages: {
        Row: {
          author_id: string
          author_role: string
          body: string
          created_at: string
          id: string
          ticket_id: string
        }
        Insert: {
          author_id: string
          author_role?: string
          body: string
          created_at?: string
          id?: string
          ticket_id: string
        }
        Update: {
          author_id?: string
          author_role?: string
          body?: string
          created_at?: string
          id?: string
          ticket_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "support_ticket_messages_ticket_id_fkey"
            columns: ["ticket_id"]
            isOneToOne: false
            referencedRelation: "support_tickets"
            referencedColumns: ["id"]
          },
        ]
      }
      support_tickets: {
        Row: {
          category: string
          created_at: string
          id: string
          message: string
          status: string
          subject: string
          updated_at: string
          user_id: string
        }
        Insert: {
          category?: string
          created_at?: string
          id?: string
          message: string
          status?: string
          subject: string
          updated_at?: string
          user_id: string
        }
        Update: {
          category?: string
          created_at?: string
          id?: string
          message?: string
          status?: string
          subject?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      user_roles: {
        Row: {
          created_at: string
          id: string
          role: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Insert: {
          created_at?: string
          id?: string
          role: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Update: {
          created_at?: string
          id?: string
          role?: Database["public"]["Enums"]["app_role"]
          user_id?: string
        }
        Relationships: []
      }
      user_settings: {
        Row: {
          comments_email: boolean | null
          comments_push: boolean | null
          comments_sms: boolean | null
          created_at: string
          id: string
          news_and_updates: boolean | null
          recovery_email: string | null
          reminder_preference: string | null
          reminders_email: boolean | null
          reminders_push: boolean | null
          reminders_sms: boolean | null
          tags_email: boolean | null
          tags_push: boolean | null
          tags_sms: boolean | null
          theme: string | null
          tips_and_tutorials: boolean | null
          updated_at: string
          user_id: string
          user_research: boolean | null
        }
        Insert: {
          comments_email?: boolean | null
          comments_push?: boolean | null
          comments_sms?: boolean | null
          created_at?: string
          id?: string
          news_and_updates?: boolean | null
          recovery_email?: string | null
          reminder_preference?: string | null
          reminders_email?: boolean | null
          reminders_push?: boolean | null
          reminders_sms?: boolean | null
          tags_email?: boolean | null
          tags_push?: boolean | null
          tags_sms?: boolean | null
          theme?: string | null
          tips_and_tutorials?: boolean | null
          updated_at?: string
          user_id: string
          user_research?: boolean | null
        }
        Update: {
          comments_email?: boolean | null
          comments_push?: boolean | null
          comments_sms?: boolean | null
          created_at?: string
          id?: string
          news_and_updates?: boolean | null
          recovery_email?: string | null
          reminder_preference?: string | null
          reminders_email?: boolean | null
          reminders_push?: boolean | null
          reminders_sms?: boolean | null
          tags_email?: boolean | null
          tags_push?: boolean | null
          tags_sms?: boolean | null
          theme?: string | null
          tips_and_tutorials?: boolean | null
          updated_at?: string
          user_id?: string
          user_research?: boolean | null
        }
        Relationships: []
      }
    }
    Views: {
      org_member_directory: {
        Row: {
          avatar_url: string | null
          company_name: string | null
          company_slug: string | null
          country: string | null
          first_name: string | null
          full_name: string | null
          id: string | null
          last_name: string | null
          organization_id: string | null
          tagline: string | null
          timezone: string | null
          user_id: string | null
        }
        Insert: {
          avatar_url?: string | null
          company_name?: string | null
          company_slug?: string | null
          country?: string | null
          first_name?: string | null
          full_name?: string | null
          id?: string | null
          last_name?: string | null
          organization_id?: string | null
          tagline?: string | null
          timezone?: string | null
          user_id?: string | null
        }
        Update: {
          avatar_url?: string | null
          company_name?: string | null
          company_slug?: string | null
          country?: string | null
          first_name?: string | null
          full_name?: string | null
          id?: string | null
          last_name?: string | null
          organization_id?: string | null
          tagline?: string | null
          timezone?: string | null
          user_id?: string | null
        }
        Relationships: []
      }
    }
    Functions: {
      current_org_id: { Args: never; Returns: string }
      current_profile_id: { Args: never; Returns: string }
      ensure_org_owner_app_user: {
        Args: { _org_id: string }
        Returns: undefined
      }
      get_my_report_access: { Args: { _report_id: string }; Returns: string }
      get_report_teams: {
        Args: { _report_ids: string[] }
        Returns: {
          avatar_url: string
          email: string
          member_id: string
          name: string
          report_id: string
        }[]
      }
      has_role: {
        Args: {
          _role: Database["public"]["Enums"]["app_role"]
          _user_id: string
        }
        Returns: boolean
      }
      is_org_admin: { Args: never; Returns: boolean }
      is_org_owner_of_profile: {
        Args: { _profile_id: string }
        Returns: boolean
      }
      is_resource_collaborator: {
        Args: { _resource_id: string; _resource_type: string }
        Returns: boolean
      }
      is_support_admin: { Args: never; Returns: boolean }
    }
    Enums: {
      analysis_message_role: "user" | "assistant" | "comment" | "reply"
      app_role: "admin" | "moderator" | "user"
      data_provider:
        | "mysql"
        | "postgres"
        | "snowflake"
        | "mssql"
        | "clickhouse"
        | "mariadb"
      data_source_status: "connected" | "disconnected" | "error"
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {
      analysis_message_role: ["user", "assistant", "comment", "reply"],
      app_role: ["admin", "moderator", "user"],
      data_provider: [
        "mysql",
        "postgres",
        "snowflake",
        "mssql",
        "clickhouse",
        "mariadb",
      ],
      data_source_status: ["connected", "disconnected", "error"],
    },
  },
} as const
