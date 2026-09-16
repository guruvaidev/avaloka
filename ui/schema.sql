-- =====================================================================
-- Full schema for the external Supabase project this deployment uses.
-- Paste this entire file into the SQL Editor of the external project.
-- Run order: extensions -> enums/tables (raw dump) -> functions ->
--            grants -> RLS enable -> policies -> triggers -> storage.
-- Safe to re-run: uses CREATE OR REPLACE / IF NOT EXISTS where possible.
-- The auth.users trigger and storage bucket are idempotent.
-- =====================================================================

-- EXTENSIONS
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements" WITH SCHEMA extensions;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA extensions;
CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA extensions;
CREATE EXTENSION IF NOT EXISTS "supabase_vault" WITH SCHEMA extensions;
CREATE EXTENSION IF NOT EXISTS "pg_trgm" WITH SCHEMA extensions;

-- ENUMS, TABLES, INDEXES, FKs (from pg_dump)
--
-- PostgreSQL database dump
--


-- Dumped from database version 17.6
-- Dumped by pg_dump version 17.9

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'SQL_ASCII';
SET standard_conforming_strings = off;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET escape_string_warning = off;
SET row_security = off;

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA "public";


--
-- Name: app_role; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE "public"."app_role" AS ENUM (
    'admin',
    'moderator',
    'user'
);


--
-- Name: data_provider; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE "public"."data_provider" AS ENUM (
    'mysql',
    'postgres',
    'snowflake',
    'mssql',
    'clickhouse',
    'mariadb'
);


--
-- Name: data_source_status; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE "public"."data_source_status" AS ENUM (
    'connected',
    'disconnected',
    'error'
);


--
-- Name: handle_new_user(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION "public"."handle_new_user"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
BEGIN
  INSERT INTO public.profiles (id)
  VALUES (NEW.id)
  ON CONFLICT (id) DO NOTHING;
  RETURN NEW;
END;
$$;


--
-- Name: has_role("uuid", "public"."app_role"); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION "public"."has_role"("_user_id" "uuid", "_role" "public"."app_role") RETURNS boolean
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
  SELECT EXISTS (SELECT 1 FROM public.user_roles WHERE user_id = _user_id AND role = _role)
$$;


--
-- Name: set_updated_at(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION "public"."set_updated_at"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    SET "search_path" TO 'public'
    AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END; $$;


SET default_tablespace = '';

SET default_table_access_method = "heap";

--
-- Name: analyses; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."analyses" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "project_id" "uuid" NOT NULL,
    "owner_id" "uuid" NOT NULL,
    "name" "text" NOT NULL,
    "team_label" "text" DEFAULT 'Team'::"text" NOT NULL,
    "team_count" integer DEFAULT 0 NOT NULL,
    "created_by_name" "text",
    "created_by_role" "text",
    "created_by_initials" "text",
    "deleted_at" timestamp with time zone,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


--
-- Name: analysis_dashboards; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."analysis_dashboards" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "analysis_id" "uuid" NOT NULL,
    "owner_id" "uuid" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


--
-- Name: app_users; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."app_users" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "name" "text" NOT NULL,
    "email" "text" NOT NULL,
    "avatar" "text",
    "user_id_code" "text" NOT NULL,
    "role" "text" NOT NULL,
    "department" "text" NOT NULL,
    "branch" "text" NOT NULL,
    "status" "text" DEFAULT 'Active'::"text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "permissions" "jsonb" DEFAULT '{}'::"jsonb" NOT NULL,
    CONSTRAINT "app_users_status_check" CHECK (("status" = ANY (ARRAY['Active'::"text", 'Inactive'::"text"])))
);


--
-- Name: branches; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."branches" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "name" "text" NOT NULL,
    "location" "text" NOT NULL,
    "head_name" "text",
    "head_email" "text",
    "head_avatar" "text",
    "employees" integer DEFAULT 0 NOT NULL,
    "status" "text" DEFAULT 'Active'::"text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    CONSTRAINT "branches_status_check" CHECK (("status" = ANY (ARRAY['Active'::"text", 'Inactive'::"text"])))
);


--
-- Name: config_roles; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."config_roles" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "role" "text" NOT NULL,
    "department" "text" NOT NULL,
    "permissions" "text"[] DEFAULT '{}'::"text"[] NOT NULL,
    "status" "text" DEFAULT 'Active'::"text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    CONSTRAINT "config_roles_status_check" CHECK (("status" = ANY (ARRAY['Active'::"text", 'Inactive'::"text"])))
);


--
-- Name: dashboard_graphs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."dashboard_graphs" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "tab_id" "uuid" NOT NULL,
    "analysis_id" "uuid" NOT NULL,
    "owner_id" "uuid" NOT NULL,
    "graph_key" "text" NOT NULL,
    "title" "text" NOT NULL,
    "chart_type" "text" DEFAULT 'bar'::"text" NOT NULL,
    "config" "jsonb" DEFAULT '{}'::"jsonb" NOT NULL,
    "position" integer DEFAULT 0 NOT NULL,
    "placed" boolean DEFAULT false NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


--
-- Name: dashboard_tabs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."dashboard_tabs" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "dashboard_id" "uuid" NOT NULL,
    "owner_id" "uuid" NOT NULL,
    "tab_index" integer NOT NULL,
    "title" "text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


--
-- Name: data_source_files; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."data_source_files" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "data_source_id" "uuid" NOT NULL,
    "owner_id" "uuid" NOT NULL,
    "path" "text" NOT NULL,
    "size_bytes" bigint,
    "selected" boolean DEFAULT false NOT NULL,
    "uploaded" boolean DEFAULT false NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


--
-- Name: data_sources; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."data_sources" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "owner_id" "uuid" NOT NULL,
    "provider" "public"."data_provider" NOT NULL,
    "name" "text" NOT NULL,
    "description" "text",
    "domain" "text",
    "use_cases" "text",
    "status" "public"."data_source_status" DEFAULT 'disconnected'::"public"."data_source_status" NOT NULL,
    "config" "jsonb" DEFAULT '{}'::"jsonb" NOT NULL,
    "last_tested_at" timestamp with time zone,
    "last_error" "text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


--
-- Name: departments; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."departments" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "department" "text" NOT NULL,
    "code" "text" NOT NULL,
    "head_name" "text",
    "head_email" "text",
    "head_avatar" "text",
    "employees" integer DEFAULT 0 NOT NULL,
    "status" "text" DEFAULT 'Active'::"text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    CONSTRAINT "departments_status_check" CHECK (("status" = ANY (ARRAY['Active'::"text", 'Inactive'::"text"])))
);


--
-- Name: models; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."models" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "name" "text" NOT NULL,
    "task_type" "text" NOT NULL,
    "tags" "text"[] DEFAULT '{}'::"text"[] NOT NULL,
    "version" "text" NOT NULL,
    "updated_label" "text" DEFAULT ''::"text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


--
-- Name: profiles; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."profiles" (
    "id" "uuid" NOT NULL,
    "first_name" "text",
    "last_name" "text",
    "phone" "text",
    "country" "text",
    "timezone" "text",
    "avatar_url" "text",
    "company_name" "text",
    "company_slug" "text",
    "tagline" "text",
    "reports_opt_in" boolean DEFAULT true NOT NULL,
    "emails_opt_in" boolean DEFAULT true NOT NULL,
    "theme" "text" DEFAULT 'system'::"text" NOT NULL,
    "ai_instructions" "text",
    "ai_depth" "text" DEFAULT 'conservative'::"text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "company_email" "text",
    "recovery_email" "text",
    "email_settings" "jsonb" DEFAULT '{"news": true, "tips": true, "prefs_on": true, "reminder": "all", "research": false}'::"jsonb" NOT NULL,
    "notification_settings" "jsonb" DEFAULT '{"tags": {"sms": false, "push": true, "email": false}, "comments": {"sms": false, "push": true, "email": true}, "reminders": {"sms": false, "push": false, "email": false}}'::"jsonb" NOT NULL,
    "integration_settings" "jsonb" DEFAULT '{}'::"jsonb" NOT NULL,
    CONSTRAINT "profiles_ai_depth_check" CHECK (("ai_depth" = ANY (ARRAY['conservative'::"text", 'balanced'::"text", 'aggressive'::"text"]))),
    CONSTRAINT "profiles_theme_check" CHECK (("theme" = ANY (ARRAY['system'::"text", 'light'::"text", 'dark'::"text"])))
);


--
-- Name: projects; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."projects" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "owner_id" "uuid" NOT NULL,
    "name" "text" NOT NULL,
    "bookmarked" boolean DEFAULT false NOT NULL,
    "archived" boolean DEFAULT false NOT NULL,
    "pinned" boolean DEFAULT false NOT NULL,
    "deleted_at" timestamp with time zone,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


--
-- Name: teams; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."teams" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "name" "text" NOT NULL,
    "lead_name" "text",
    "lead_email" "text",
    "lead_avatar" "text",
    "user_count" integer DEFAULT 0 NOT NULL,
    "user_avatars" "text"[] DEFAULT '{}'::"text"[] NOT NULL,
    "branch" "text" NOT NULL,
    "status" "text" DEFAULT 'Active'::"text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "description" "text",
    "member_emails" "text"[] DEFAULT '{}'::"text"[] NOT NULL,
    CONSTRAINT "teams_status_check" CHECK (("status" = ANY (ARRAY['Active'::"text", 'Inactive'::"text"])))
);


--
-- Name: user_roles; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."user_roles" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "role" "public"."app_role" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


--
-- Name: analyses analyses_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."analyses"
    ADD CONSTRAINT "analyses_pkey" PRIMARY KEY ("id");


--
-- Name: analysis_dashboards analysis_dashboards_analysis_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."analysis_dashboards"
    ADD CONSTRAINT "analysis_dashboards_analysis_id_key" UNIQUE ("analysis_id");


--
-- Name: analysis_dashboards analysis_dashboards_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."analysis_dashboards"
    ADD CONSTRAINT "analysis_dashboards_pkey" PRIMARY KEY ("id");


--
-- Name: app_users app_users_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."app_users"
    ADD CONSTRAINT "app_users_pkey" PRIMARY KEY ("id");


--
-- Name: branches branches_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."branches"
    ADD CONSTRAINT "branches_pkey" PRIMARY KEY ("id");


--
-- Name: config_roles config_roles_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."config_roles"
    ADD CONSTRAINT "config_roles_pkey" PRIMARY KEY ("id");


--
-- Name: dashboard_graphs dashboard_graphs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."dashboard_graphs"
    ADD CONSTRAINT "dashboard_graphs_pkey" PRIMARY KEY ("id");


--
-- Name: dashboard_tabs dashboard_tabs_dashboard_id_tab_index_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."dashboard_tabs"
    ADD CONSTRAINT "dashboard_tabs_dashboard_id_tab_index_key" UNIQUE ("dashboard_id", "tab_index");


--
-- Name: dashboard_tabs dashboard_tabs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."dashboard_tabs"
    ADD CONSTRAINT "dashboard_tabs_pkey" PRIMARY KEY ("id");


--
-- Name: data_source_files data_source_files_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."data_source_files"
    ADD CONSTRAINT "data_source_files_pkey" PRIMARY KEY ("id");


--
-- Name: data_sources data_sources_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."data_sources"
    ADD CONSTRAINT "data_sources_pkey" PRIMARY KEY ("id");


--
-- Name: departments departments_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."departments"
    ADD CONSTRAINT "departments_pkey" PRIMARY KEY ("id");


--
-- Name: models models_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."models"
    ADD CONSTRAINT "models_pkey" PRIMARY KEY ("id");


--
-- Name: profiles profiles_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."profiles"
    ADD CONSTRAINT "profiles_pkey" PRIMARY KEY ("id");


--
-- Name: projects projects_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."projects"
    ADD CONSTRAINT "projects_pkey" PRIMARY KEY ("id");


--
-- Name: teams teams_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."teams"
    ADD CONSTRAINT "teams_pkey" PRIMARY KEY ("id");


--
-- Name: user_roles user_roles_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."user_roles"
    ADD CONSTRAINT "user_roles_pkey" PRIMARY KEY ("id");


--
-- Name: user_roles user_roles_user_id_role_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."user_roles"
    ADD CONSTRAINT "user_roles_user_id_role_key" UNIQUE ("user_id", "role");


--
-- Name: idx_analyses_owner; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX "idx_analyses_owner" ON "public"."analyses" USING "btree" ("owner_id");


--
-- Name: idx_analyses_project; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX "idx_analyses_project" ON "public"."analyses" USING "btree" ("project_id") WHERE ("deleted_at" IS NULL);


--
-- Name: idx_dashboard_tabs_dash; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX "idx_dashboard_tabs_dash" ON "public"."dashboard_tabs" USING "btree" ("dashboard_id");


--
-- Name: idx_data_source_files_ds; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX "idx_data_source_files_ds" ON "public"."data_source_files" USING "btree" ("data_source_id");


--
-- Name: idx_graphs_analysis; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX "idx_graphs_analysis" ON "public"."dashboard_graphs" USING "btree" ("analysis_id");


--
-- Name: idx_graphs_tab; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX "idx_graphs_tab" ON "public"."dashboard_graphs" USING "btree" ("tab_id");


--
-- Name: idx_projects_name_trgm; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX "idx_projects_name_trgm" ON "public"."projects" USING "gin" ("name" "extensions"."gin_trgm_ops");


--
-- Name: idx_projects_owner; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX "idx_projects_owner" ON "public"."projects" USING "btree" ("owner_id") WHERE ("deleted_at" IS NULL);


--
-- Name: profiles profiles_set_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "profiles_set_updated_at" BEFORE UPDATE ON "public"."profiles" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();


--
-- Name: app_users set_app_users_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "set_app_users_updated_at" BEFORE UPDATE ON "public"."app_users" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();


--
-- Name: branches set_branches_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "set_branches_updated_at" BEFORE UPDATE ON "public"."branches" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();


--
-- Name: config_roles set_config_roles_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "set_config_roles_updated_at" BEFORE UPDATE ON "public"."config_roles" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();


--
-- Name: departments set_departments_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "set_departments_updated_at" BEFORE UPDATE ON "public"."departments" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();


--
-- Name: models set_models_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "set_models_updated_at" BEFORE UPDATE ON "public"."models" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();


--
-- Name: teams set_teams_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "set_teams_updated_at" BEFORE UPDATE ON "public"."teams" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();


--
-- Name: data_sources tr_data_sources_updated; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "tr_data_sources_updated" BEFORE UPDATE ON "public"."data_sources" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();


--
-- Name: analyses trg_analyses_updated; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "trg_analyses_updated" BEFORE UPDATE ON "public"."analyses" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();


--
-- Name: analysis_dashboards trg_dashboards_updated; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "trg_dashboards_updated" BEFORE UPDATE ON "public"."analysis_dashboards" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();


--
-- Name: dashboard_graphs trg_graphs_updated; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "trg_graphs_updated" BEFORE UPDATE ON "public"."dashboard_graphs" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();


--
-- Name: projects trg_projects_updated; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "trg_projects_updated" BEFORE UPDATE ON "public"."projects" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();


--
-- Name: dashboard_tabs trg_tabs_updated; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "trg_tabs_updated" BEFORE UPDATE ON "public"."dashboard_tabs" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();


--
-- Name: analyses analyses_owner_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."analyses"
    ADD CONSTRAINT "analyses_owner_id_fkey" FOREIGN KEY ("owner_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;


--
-- Name: analyses analyses_project_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."analyses"
    ADD CONSTRAINT "analyses_project_id_fkey" FOREIGN KEY ("project_id") REFERENCES "public"."projects"("id") ON DELETE CASCADE;


--
-- Name: analysis_dashboards analysis_dashboards_analysis_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."analysis_dashboards"
    ADD CONSTRAINT "analysis_dashboards_analysis_id_fkey" FOREIGN KEY ("analysis_id") REFERENCES "public"."analyses"("id") ON DELETE CASCADE;


--
-- Name: analysis_dashboards analysis_dashboards_owner_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."analysis_dashboards"
    ADD CONSTRAINT "analysis_dashboards_owner_id_fkey" FOREIGN KEY ("owner_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;


--
-- Name: dashboard_graphs dashboard_graphs_analysis_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."dashboard_graphs"
    ADD CONSTRAINT "dashboard_graphs_analysis_id_fkey" FOREIGN KEY ("analysis_id") REFERENCES "public"."analyses"("id") ON DELETE CASCADE;


--
-- Name: dashboard_graphs dashboard_graphs_owner_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."dashboard_graphs"
    ADD CONSTRAINT "dashboard_graphs_owner_id_fkey" FOREIGN KEY ("owner_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;


--
-- Name: dashboard_graphs dashboard_graphs_tab_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."dashboard_graphs"
    ADD CONSTRAINT "dashboard_graphs_tab_id_fkey" FOREIGN KEY ("tab_id") REFERENCES "public"."dashboard_tabs"("id") ON DELETE CASCADE;


--
-- Name: dashboard_tabs dashboard_tabs_dashboard_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."dashboard_tabs"
    ADD CONSTRAINT "dashboard_tabs_dashboard_id_fkey" FOREIGN KEY ("dashboard_id") REFERENCES "public"."analysis_dashboards"("id") ON DELETE CASCADE;


--
-- Name: dashboard_tabs dashboard_tabs_owner_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."dashboard_tabs"
    ADD CONSTRAINT "dashboard_tabs_owner_id_fkey" FOREIGN KEY ("owner_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;


--
-- Name: data_source_files data_source_files_data_source_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."data_source_files"
    ADD CONSTRAINT "data_source_files_data_source_id_fkey" FOREIGN KEY ("data_source_id") REFERENCES "public"."data_sources"("id") ON DELETE CASCADE;


--
-- Name: data_source_files data_source_files_owner_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."data_source_files"
    ADD CONSTRAINT "data_source_files_owner_id_fkey" FOREIGN KEY ("owner_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;


--
-- Name: data_sources data_sources_owner_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."data_sources"
    ADD CONSTRAINT "data_sources_owner_id_fkey" FOREIGN KEY ("owner_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;


--
-- Name: profiles profiles_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."profiles"
    ADD CONSTRAINT "profiles_id_fkey" FOREIGN KEY ("id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;


--
-- Name: projects projects_owner_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."projects"
    ADD CONSTRAINT "projects_owner_id_fkey" FOREIGN KEY ("owner_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;


--
-- Name: user_roles user_roles_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."user_roles"
    ADD CONSTRAINT "user_roles_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;


--
-- Name: profiles Users delete own profile; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users delete own profile" ON "public"."profiles" FOR DELETE TO "authenticated" USING (("auth"."uid"() = "id"));


--
-- Name: profiles Users insert own profile; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users insert own profile" ON "public"."profiles" FOR INSERT TO "authenticated" WITH CHECK (("auth"."uid"() = "id"));


--
-- Name: profiles Users select own profile; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users select own profile" ON "public"."profiles" FOR SELECT TO "authenticated" USING (("auth"."uid"() = "id"));


--
-- Name: profiles Users update own profile; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users update own profile" ON "public"."profiles" FOR UPDATE TO "authenticated" USING (("auth"."uid"() = "id")) WITH CHECK (("auth"."uid"() = "id"));


--
-- Name: app_users admin all app_users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "admin all app_users" ON "public"."app_users" TO "authenticated" USING ("public"."has_role"("auth"."uid"(), 'admin'::"public"."app_role")) WITH CHECK ("public"."has_role"("auth"."uid"(), 'admin'::"public"."app_role"));


--
-- Name: branches admin all branches; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "admin all branches" ON "public"."branches" TO "authenticated" USING ("public"."has_role"("auth"."uid"(), 'admin'::"public"."app_role")) WITH CHECK ("public"."has_role"("auth"."uid"(), 'admin'::"public"."app_role"));


--
-- Name: config_roles admin all config_roles; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "admin all config_roles" ON "public"."config_roles" TO "authenticated" USING ("public"."has_role"("auth"."uid"(), 'admin'::"public"."app_role")) WITH CHECK ("public"."has_role"("auth"."uid"(), 'admin'::"public"."app_role"));


--
-- Name: departments admin all departments; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "admin all departments" ON "public"."departments" TO "authenticated" USING ("public"."has_role"("auth"."uid"(), 'admin'::"public"."app_role")) WITH CHECK ("public"."has_role"("auth"."uid"(), 'admin'::"public"."app_role"));


--
-- Name: models admin all models; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "admin all models" ON "public"."models" TO "authenticated" USING ("public"."has_role"("auth"."uid"(), 'admin'::"public"."app_role")) WITH CHECK ("public"."has_role"("auth"."uid"(), 'admin'::"public"."app_role"));


--
-- Name: teams admin all teams; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "admin all teams" ON "public"."teams" TO "authenticated" USING ("public"."has_role"("auth"."uid"(), 'admin'::"public"."app_role")) WITH CHECK ("public"."has_role"("auth"."uid"(), 'admin'::"public"."app_role"));


--
-- Name: user_roles admins manage roles; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "admins manage roles" ON "public"."user_roles" TO "authenticated" USING ("public"."has_role"("auth"."uid"(), 'admin'::"public"."app_role")) WITH CHECK ("public"."has_role"("auth"."uid"(), 'admin'::"public"."app_role"));


--
-- Name: analyses; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE "public"."analyses" ENABLE ROW LEVEL SECURITY;

--
-- Name: analysis_dashboards; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE "public"."analysis_dashboards" ENABLE ROW LEVEL SECURITY;

--
-- Name: app_users; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE "public"."app_users" ENABLE ROW LEVEL SECURITY;

--
-- Name: branches; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE "public"."branches" ENABLE ROW LEVEL SECURITY;

--
-- Name: config_roles; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE "public"."config_roles" ENABLE ROW LEVEL SECURITY;

--
-- Name: dashboard_graphs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE "public"."dashboard_graphs" ENABLE ROW LEVEL SECURITY;

--
-- Name: dashboard_tabs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE "public"."dashboard_tabs" ENABLE ROW LEVEL SECURITY;

--
-- Name: data_source_files; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE "public"."data_source_files" ENABLE ROW LEVEL SECURITY;

--
-- Name: data_sources; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE "public"."data_sources" ENABLE ROW LEVEL SECURITY;

--
-- Name: departments; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE "public"."departments" ENABLE ROW LEVEL SECURITY;

--
-- Name: models; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE "public"."models" ENABLE ROW LEVEL SECURITY;

--
-- Name: analyses owners delete own analyses; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "owners delete own analyses" ON "public"."analyses" FOR DELETE TO "authenticated" USING (("auth"."uid"() = "owner_id"));


--
-- Name: projects owners delete own projects; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "owners delete own projects" ON "public"."projects" FOR DELETE TO "authenticated" USING (("auth"."uid"() = "owner_id"));


--
-- Name: analyses owners insert own analyses; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "owners insert own analyses" ON "public"."analyses" FOR INSERT TO "authenticated" WITH CHECK (("auth"."uid"() = "owner_id"));


--
-- Name: projects owners insert own projects; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "owners insert own projects" ON "public"."projects" FOR INSERT TO "authenticated" WITH CHECK (("auth"."uid"() = "owner_id"));


--
-- Name: analysis_dashboards owners manage own dashboards; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "owners manage own dashboards" ON "public"."analysis_dashboards" TO "authenticated" USING (("auth"."uid"() = "owner_id")) WITH CHECK (("auth"."uid"() = "owner_id"));


--
-- Name: data_source_files owners manage own data_source_files; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "owners manage own data_source_files" ON "public"."data_source_files" TO "authenticated" USING (("owner_id" = "auth"."uid"())) WITH CHECK (("owner_id" = "auth"."uid"()));


--
-- Name: data_sources owners manage own data_sources; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "owners manage own data_sources" ON "public"."data_sources" TO "authenticated" USING (("owner_id" = "auth"."uid"())) WITH CHECK (("owner_id" = "auth"."uid"()));


--
-- Name: dashboard_graphs owners manage own graphs; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "owners manage own graphs" ON "public"."dashboard_graphs" TO "authenticated" USING (("auth"."uid"() = "owner_id")) WITH CHECK (("auth"."uid"() = "owner_id"));


--
-- Name: dashboard_tabs owners manage own tabs; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "owners manage own tabs" ON "public"."dashboard_tabs" TO "authenticated" USING (("auth"."uid"() = "owner_id")) WITH CHECK (("auth"."uid"() = "owner_id"));


--
-- Name: analyses owners read own analyses; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "owners read own analyses" ON "public"."analyses" FOR SELECT TO "authenticated" USING (("auth"."uid"() = "owner_id"));


--
-- Name: projects owners read own projects; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "owners read own projects" ON "public"."projects" FOR SELECT TO "authenticated" USING (("auth"."uid"() = "owner_id"));


--
-- Name: analyses owners update own analyses; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "owners update own analyses" ON "public"."analyses" FOR UPDATE TO "authenticated" USING (("auth"."uid"() = "owner_id")) WITH CHECK (("auth"."uid"() = "owner_id"));


--
-- Name: projects owners update own projects; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "owners update own projects" ON "public"."projects" FOR UPDATE TO "authenticated" USING (("auth"."uid"() = "owner_id")) WITH CHECK (("auth"."uid"() = "owner_id"));


--
-- Name: profiles; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE "public"."profiles" ENABLE ROW LEVEL SECURITY;

--
-- Name: projects; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE "public"."projects" ENABLE ROW LEVEL SECURITY;

--
-- Name: teams; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE "public"."teams" ENABLE ROW LEVEL SECURITY;

--
-- Name: user_roles; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE "public"."user_roles" ENABLE ROW LEVEL SECURITY;

--
-- Name: user_roles users read own roles; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "users read own roles" ON "public"."user_roles" FOR SELECT TO "authenticated" USING (("auth"."uid"() = "user_id"));


--
-- PostgreSQL database dump complete
--



-- FUNCTIONS
CREATE OR REPLACE FUNCTION public.handle_new_user()
 RETURNS trigger
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public'
AS $function$
BEGIN
  INSERT INTO public.profiles (id)
  VALUES (NEW.id)
  ON CONFLICT (id) DO NOTHING;
  RETURN NEW;
END;
$function$
;
CREATE OR REPLACE FUNCTION public.set_updated_at()
 RETURNS trigger
 LANGUAGE plpgsql
 SET search_path TO 'public'
AS $function$
BEGIN NEW.updated_at = now(); RETURN NEW; END; $function$
;
CREATE OR REPLACE FUNCTION public.has_role(_user_id uuid, _role app_role)
 RETURNS boolean
 LANGUAGE sql
 STABLE SECURITY DEFINER
 SET search_path TO 'public'
AS $function$
  SELECT EXISTS (SELECT 1 FROM public.user_roles WHERE user_id = _user_id AND role = _role)
$function$
;

-- GRANTS (required for Supabase Data API)
GRANT SELECT, INSERT, UPDATE, DELETE ON public."analyses" TO authenticated;
GRANT ALL ON public."analyses" TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public."analysis_dashboards" TO authenticated;
GRANT ALL ON public."analysis_dashboards" TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public."app_users" TO authenticated;
GRANT ALL ON public."app_users" TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public."branches" TO authenticated;
GRANT ALL ON public."branches" TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public."config_roles" TO authenticated;
GRANT ALL ON public."config_roles" TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public."dashboard_graphs" TO authenticated;
GRANT ALL ON public."dashboard_graphs" TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public."dashboard_tabs" TO authenticated;
GRANT ALL ON public."dashboard_tabs" TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public."data_source_files" TO authenticated;
GRANT ALL ON public."data_source_files" TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public."data_sources" TO authenticated;
GRANT ALL ON public."data_sources" TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public."departments" TO authenticated;
GRANT ALL ON public."departments" TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public."models" TO authenticated;
GRANT ALL ON public."models" TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public."profiles" TO authenticated;
GRANT ALL ON public."profiles" TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public."projects" TO authenticated;
GRANT ALL ON public."projects" TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public."teams" TO authenticated;
GRANT ALL ON public."teams" TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public."user_roles" TO authenticated;
GRANT ALL ON public."user_roles" TO service_role;

-- ENABLE ROW LEVEL SECURITY
ALTER TABLE public."analyses" ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."analysis_dashboards" ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."app_users" ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."branches" ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."config_roles" ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."dashboard_graphs" ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."dashboard_tabs" ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."data_source_files" ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."data_sources" ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."departments" ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."models" ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."profiles" ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."projects" ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."teams" ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."user_roles" ENABLE ROW LEVEL SECURITY;

-- POLICIES
CREATE POLICY "owners delete own analyses" ON public."analyses" AS PERMISSIVE FOR DELETE TO authenticated USING ((auth.uid() = owner_id));
CREATE POLICY "owners insert own analyses" ON public."analyses" AS PERMISSIVE FOR INSERT TO authenticated WITH CHECK ((auth.uid() = owner_id));
CREATE POLICY "owners read own analyses" ON public."analyses" AS PERMISSIVE FOR SELECT TO authenticated USING ((auth.uid() = owner_id));
CREATE POLICY "owners update own analyses" ON public."analyses" AS PERMISSIVE FOR UPDATE TO authenticated USING ((auth.uid() = owner_id)) WITH CHECK ((auth.uid() = owner_id));
CREATE POLICY "owners manage own dashboards" ON public."analysis_dashboards" AS PERMISSIVE FOR ALL TO authenticated USING ((auth.uid() = owner_id)) WITH CHECK ((auth.uid() = owner_id));
CREATE POLICY "admin all app_users" ON public."app_users" AS PERMISSIVE FOR ALL TO authenticated USING (has_role(auth.uid(), 'admin'::app_role)) WITH CHECK (has_role(auth.uid(), 'admin'::app_role));
CREATE POLICY "admin all branches" ON public."branches" AS PERMISSIVE FOR ALL TO authenticated USING (has_role(auth.uid(), 'admin'::app_role)) WITH CHECK (has_role(auth.uid(), 'admin'::app_role));
CREATE POLICY "admin all config_roles" ON public."config_roles" AS PERMISSIVE FOR ALL TO authenticated USING (has_role(auth.uid(), 'admin'::app_role)) WITH CHECK (has_role(auth.uid(), 'admin'::app_role));
CREATE POLICY "owners manage own graphs" ON public."dashboard_graphs" AS PERMISSIVE FOR ALL TO authenticated USING ((auth.uid() = owner_id)) WITH CHECK ((auth.uid() = owner_id));
CREATE POLICY "owners manage own tabs" ON public."dashboard_tabs" AS PERMISSIVE FOR ALL TO authenticated USING ((auth.uid() = owner_id)) WITH CHECK ((auth.uid() = owner_id));
CREATE POLICY "owners manage own data_source_files" ON public."data_source_files" AS PERMISSIVE FOR ALL TO authenticated USING ((owner_id = auth.uid())) WITH CHECK ((owner_id = auth.uid()));
CREATE POLICY "owners manage own data_sources" ON public."data_sources" AS PERMISSIVE FOR ALL TO authenticated USING ((owner_id = auth.uid())) WITH CHECK ((owner_id = auth.uid()));
CREATE POLICY "admin all departments" ON public."departments" AS PERMISSIVE FOR ALL TO authenticated USING (has_role(auth.uid(), 'admin'::app_role)) WITH CHECK (has_role(auth.uid(), 'admin'::app_role));
CREATE POLICY "admin all models" ON public."models" AS PERMISSIVE FOR ALL TO authenticated USING (has_role(auth.uid(), 'admin'::app_role)) WITH CHECK (has_role(auth.uid(), 'admin'::app_role));
CREATE POLICY "Users delete own profile" ON public."profiles" AS PERMISSIVE FOR DELETE TO authenticated USING ((auth.uid() = id));
CREATE POLICY "Users insert own profile" ON public."profiles" AS PERMISSIVE FOR INSERT TO authenticated WITH CHECK ((auth.uid() = id));
CREATE POLICY "Users select own profile" ON public."profiles" AS PERMISSIVE FOR SELECT TO authenticated USING ((auth.uid() = id));
CREATE POLICY "Users update own profile" ON public."profiles" AS PERMISSIVE FOR UPDATE TO authenticated USING ((auth.uid() = id)) WITH CHECK ((auth.uid() = id));
CREATE POLICY "owners delete own projects" ON public."projects" AS PERMISSIVE FOR DELETE TO authenticated USING ((auth.uid() = owner_id));
CREATE POLICY "owners insert own projects" ON public."projects" AS PERMISSIVE FOR INSERT TO authenticated WITH CHECK ((auth.uid() = owner_id));
CREATE POLICY "owners read own projects" ON public."projects" AS PERMISSIVE FOR SELECT TO authenticated USING ((auth.uid() = owner_id));
CREATE POLICY "owners update own projects" ON public."projects" AS PERMISSIVE FOR UPDATE TO authenticated USING ((auth.uid() = owner_id)) WITH CHECK ((auth.uid() = owner_id));
CREATE POLICY "admin all teams" ON public."teams" AS PERMISSIVE FOR ALL TO authenticated USING (has_role(auth.uid(), 'admin'::app_role)) WITH CHECK (has_role(auth.uid(), 'admin'::app_role));
CREATE POLICY "admins manage roles" ON public."user_roles" AS PERMISSIVE FOR ALL TO authenticated USING (has_role(auth.uid(), 'admin'::app_role)) WITH CHECK (has_role(auth.uid(), 'admin'::app_role));
CREATE POLICY "users read own roles" ON public."user_roles" AS PERMISSIVE FOR SELECT TO authenticated USING ((auth.uid() = user_id));

-- TRIGGERS
CREATE TRIGGER trg_projects_updated BEFORE UPDATE ON public.projects FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_analyses_updated BEFORE UPDATE ON public.analyses FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_dashboards_updated BEFORE UPDATE ON public.analysis_dashboards FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_tabs_updated BEFORE UPDATE ON public.dashboard_tabs FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_graphs_updated BEFORE UPDATE ON public.dashboard_graphs FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER tr_data_sources_updated BEFORE UPDATE ON public.data_sources FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER set_departments_updated_at BEFORE UPDATE ON public.departments FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER set_config_roles_updated_at BEFORE UPDATE ON public.config_roles FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER set_branches_updated_at BEFORE UPDATE ON public.branches FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER set_app_users_updated_at BEFORE UPDATE ON public.app_users FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER set_teams_updated_at BEFORE UPDATE ON public.teams FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER set_models_updated_at BEFORE UPDATE ON public.models FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER profiles_set_updated_at BEFORE UPDATE ON public.profiles FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- Auto-create profile on new auth user
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Storage: avatars bucket
INSERT INTO storage.buckets (id, name, public)
VALUES ('avatars', 'avatars', false)
ON CONFLICT (id) DO NOTHING;

CREATE POLICY "Avatars are publicly readable" ON storage.objects
  FOR SELECT USING (bucket_id = 'avatars');
CREATE POLICY "Users upload own avatar" ON storage.objects
  FOR INSERT WITH CHECK (bucket_id = 'avatars' AND (storage.foldername(name))[1] = (auth.uid())::text);
CREATE POLICY "Users update own avatar" ON storage.objects
  FOR UPDATE USING (bucket_id = 'avatars' AND (storage.foldername(name))[1] = (auth.uid())::text);
CREATE POLICY "Users delete own avatar" ON storage.objects
  FOR DELETE USING (bucket_id = 'avatars' AND (storage.foldername(name))[1] = (auth.uid())::text);
