SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;


CREATE SCHEMA IF NOT EXISTS "public";


ALTER SCHEMA "public" OWNER TO "pg_database_owner";


COMMENT ON SCHEMA "public" IS 'standard public schema';



CREATE TYPE "public"."billing_interval" AS ENUM (
    'monthly',
    'yearly'
);


ALTER TYPE "public"."billing_interval" OWNER TO "postgres";


CREATE TYPE "public"."internal_role" AS ENUM (
    'support_admin',
    'account_manager'
);


ALTER TYPE "public"."internal_role" OWNER TO "postgres";


CREATE TYPE "public"."license_status" AS ENUM (
    'pending',
    'active',
    'expired'
);


ALTER TYPE "public"."license_status" OWNER TO "postgres";


CREATE TYPE "public"."payment_provider" AS ENUM (
    'stripe',
    'paypal'
);


ALTER TYPE "public"."payment_provider" OWNER TO "postgres";


CREATE TYPE "public"."plan_type" AS ENUM (
    'free',
    'professional',
    'enterprise'
);


ALTER TYPE "public"."plan_type" OWNER TO "postgres";


CREATE TYPE "public"."team_role" AS ENUM (
    'admin',
    'member'
);


ALTER TYPE "public"."team_role" OWNER TO "postgres";


CREATE TYPE "public"."user_interest" AS ENUM (
    'ab_experimentation',
    'ml_engineering',
    'data_visualization',
    'predictive_analytics',
    'etl_pipelines',
    'business_intelligence',
    'deep_learning',
    'nlp',
    'computer_vision',
    'time_series_analysis'
);


ALTER TYPE "public"."user_interest" OWNER TO "postgres";


CREATE TYPE "public"."user_profile_type" AS ENUM (
    'data_scientist',
    'data_analyst',
    'data_engineer'
);


ALTER TYPE "public"."user_profile_type" OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."app_users_fill_org"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
BEGIN
  IF NEW.organization_id IS NULL THEN
    SELECT p.organization_id INTO NEW.organization_id
    FROM public.profiles p
    WHERE p.id = COALESCE(NEW.auth_user_id, NEW.profile_id, NEW.invited_by)
      AND p.organization_id IS NOT NULL
    LIMIT 1;
  END IF;

  IF NEW.organization_id IS NULL AND NEW.invited_by IS NOT NULL THEN
    SELECT p.organization_id INTO NEW.organization_id
    FROM public.profiles p
    WHERE p.id = NEW.invited_by
    LIMIT 1;
  END IF;

  RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."app_users_fill_org"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."auto_assign_billing_admin"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
BEGIN
  -- Only for enterprise accounts with an owner
  IF NEW.plan_type = 'enterprise' AND NEW.owner_user_id IS NOT NULL THEN
    -- Insert billing admin record for the owner
    INSERT INTO public.billing_admins (
      customer_account_id,
      user_id,
      assigned_by
    )
    VALUES (
      NEW.id,
      NEW.owner_user_id,
      NEW.owner_user_id -- Self-assigned during account creation
    )
    ON CONFLICT (customer_account_id, user_id) DO NOTHING; -- Prevent duplicates
  END IF;
  
  RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."auto_assign_billing_admin"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."auto_join_enterprise_admin"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
DECLARE
  v_owner_user_id UUID;
BEGIN
  -- Get the owner_user_id from the customer_account
  SELECT owner_user_id INTO v_owner_user_id
  FROM public.customer_accounts
  WHERE id = NEW.customer_account_id
    AND plan_type = 'enterprise'
    AND owner_user_id IS NOT NULL;
  
  -- If this is an enterprise account with an owner, auto-join them as admin
  IF v_owner_user_id IS NOT NULL THEN
    INSERT INTO public.team_memberships (
      team_id,
      user_id,
      role,
      invited_by
    )
    VALUES (
      NEW.id,
      v_owner_user_id,
      'admin',
      v_owner_user_id
    )
    ON CONFLICT (team_id, user_id) DO NOTHING; -- Prevent duplicates
  END IF;
  
  RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."auto_join_enterprise_admin"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."can_create_team"("p_user_id" "uuid", "p_account_id" "uuid") RETURNS boolean
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.customer_accounts ca
    WHERE ca.id = p_account_id
      AND (
        ca.owner_user_id = p_user_id
        OR ca.plan_type = 'professional'
      )
  );
$$;


ALTER FUNCTION "public"."can_create_team"("p_user_id" "uuid", "p_account_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."can_manage_resource_share"("p_type" "text", "p_id" "uuid") RETURNS boolean
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
SELECT CASE
    WHEN p_type = 'project' THEN EXISTS (
        SELECT 1
        FROM public.projects p
        WHERE
            p.id = p_id
            AND p.owner_id = (
                SELECT id
                FROM public.profiles
                WHERE user_id = auth.uid()
            )
            AND p.deleted_at IS NULL
    )

    WHEN p_type = 'analysis' THEN EXISTS (
        SELECT 1
        FROM public.analyses a
        WHERE
            a.id = p_id
            AND a.owner_id = (
                SELECT id
                FROM public.profiles
                WHERE user_id = auth.uid()
            )
    )

    ELSE false
END;
$$;


ALTER FUNCTION "public"."can_manage_resource_share"("p_type" "text", "p_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."check_expiring_licenses"() RETURNS "void"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
DECLARE
  v_license RECORD;
  v_billing_admin RECORD;
  v_days_until_expiry integer;
BEGIN
  -- Find licenses expiring in the next 30 days
  FOR v_license IN
    SELECT 
      el.*,
      ca.company_name
    FROM public.enterprise_licenses el
    JOIN public.customer_accounts ca ON ca.id = el.customer_account_id
    WHERE el.status = 'active'
      AND el.expires_at IS NOT NULL
      AND el.expires_at > NOW()
      AND el.expires_at <= NOW() + INTERVAL '30 days'
  LOOP
    v_days_until_expiry := EXTRACT(DAY FROM (v_license.expires_at - NOW()));
    
    -- Only notify at specific intervals: 30, 14, 7, 3, 1 days before expiration
    IF v_days_until_expiry IN (30, 14, 7, 3, 1) THEN
      -- Notify all billing admins for this account
      FOR v_billing_admin IN
        SELECT user_id
        FROM public.billing_admins
        WHERE customer_account_id = v_license.customer_account_id
      LOOP
        -- Check if notification already exists for this day
        IF NOT EXISTS (
          SELECT 1
          FROM public.notifications
          WHERE user_id = v_billing_admin.user_id
            AND type = 'license_expiration_warning'
            AND data->>'license_id' = v_license.id::text
            AND created_at > NOW() - INTERVAL '1 day'
        ) THEN
          -- Create notification
          INSERT INTO public.notifications (
            user_id,
            type,
            title,
            message,
            data,
            read
          )
          VALUES (
            v_billing_admin.user_id,
            'license_expiration_warning',
            'License Expiring Soon',
            format(
              'License %s for %s expires in %s day%s',
              v_license.serial_number,
              COALESCE(v_license.company_name, 'your account'),
              v_days_until_expiry,
              CASE WHEN v_days_until_expiry = 1 THEN '' ELSE 's' END
            ),
            jsonb_build_object(
              'license_id', v_license.id,
              'serial_number', v_license.serial_number,
              'expires_at', v_license.expires_at,
              'days_remaining', v_days_until_expiry
            ),
            false
          );
        END IF;
      END LOOP;
    END IF;
  END LOOP;
END;
$$;


ALTER FUNCTION "public"."check_expiring_licenses"() OWNER TO "postgres";


COMMENT ON FUNCTION "public"."check_expiring_licenses"() IS 'Checks for licenses expiring in the next 30 days and notifies billing admins at key intervals (30, 14, 7, 3, 1 days before expiration)';



CREATE OR REPLACE FUNCTION "public"."check_license_expiry"() RETURNS "void"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
BEGIN
  UPDATE public.enterprise_licenses
  SET status = 'expired'
  WHERE status = 'active'
    AND expires_at <= now();
END;
$$;


ALTER FUNCTION "public"."check_license_expiry"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."check_onboarding_plan"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
BEGIN
  IF NEW.onboarding_completed = true AND NEW.selected_plan IS NULL THEN
    RAISE EXCEPTION 'Cannot complete onboarding without selecting a plan';
  END IF;
  RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."check_onboarding_plan"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."current_org_id"() RETURNS "uuid"
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
  SELECT organization_id FROM public.profiles WHERE user_id = auth.uid()
$$;


ALTER FUNCTION "public"."current_org_id"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."current_user_email"() RETURNS "text"
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO ''
    AS $$
  SELECT email FROM auth.users WHERE id = auth.uid();
$$;


ALTER FUNCTION "public"."current_user_email"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."encrypt_cloud_credentials"("p_connection_id" "uuid", "p_access_key" "text", "p_secret_key" "text") RETURNS json
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
DECLARE
  v_result JSON;
BEGIN
  -- This function will be called from an edge function that performs the actual encryption
  -- It serves as a secure endpoint to update encrypted credentials
  
  -- Validate that the connection belongs to the current user
  IF NOT EXISTS (
    SELECT 1 FROM public.cloud_datasets 
    WHERE id = p_connection_id 
    AND user_id = auth.uid()
  ) THEN
    RAISE EXCEPTION 'Unauthorized access to connection';
  END IF;
  
  -- Update the encrypted credentials
  UPDATE public.cloud_datasets
  SET 
    access_key_ciphertext = p_access_key,
    secret_key_ciphertext = p_secret_key,
    access_key = NULL,  -- Clear plain text
    secret_key = NULL,   -- Clear plain text
    updated_at = NOW()
  WHERE id = p_connection_id
  AND user_id = auth.uid();
  
  v_result := json_build_object(
    'success', true,
    'message', 'Credentials encrypted successfully'
  );
  
  RETURN v_result;
END;
$$;


ALTER FUNCTION "public"."encrypt_cloud_credentials"("p_connection_id" "uuid", "p_access_key" "text", "p_secret_key" "text") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."find_profile_id_by_email"("p_email" "text") RETURNS "uuid"
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO ''
    AS $$
  SELECT p.id
  FROM public.profiles p
  JOIN auth.users u ON u.id = p.user_id
  WHERE lower(u.email) = lower(p_email)
  LIMIT 1;
$$;


ALTER FUNCTION "public"."find_profile_id_by_email"("p_email" "text") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."find_user_id_by_email"("p_email" "text") RETURNS "uuid"
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO ''
    AS $$
  SELECT id
  FROM auth.users
  WHERE lower(email) = lower(p_email)
  LIMIT 1;
$$;


ALTER FUNCTION "public"."find_user_id_by_email"("p_email" "text") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."generate_license_serial"() RETURNS "text"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
DECLARE
  v_serial TEXT;
  v_exists BOOLEAN;
BEGIN
  LOOP
    -- Generate format: ENT-XXXX-XXXX-XXXX
    v_serial := 'ENT-' || 
                upper(substring(md5(random()::text) from 1 for 4)) || '-' ||
                upper(substring(md5(random()::text) from 1 for 4)) || '-' ||
                upper(substring(md5(random()::text) from 1 for 4));
    
    -- Check if serial already exists
    SELECT EXISTS(
      SELECT 1 FROM public.enterprise_licenses WHERE serial_number = v_serial
    ) INTO v_exists;
    
    -- Exit loop if unique
    EXIT WHEN NOT v_exists;
  END LOOP;
  
  RETURN v_serial;
END;
$$;


ALTER FUNCTION "public"."generate_license_serial"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_my_report_access"("_report_id" "uuid") RETURNS "text"
    LANGUAGE "plpgsql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
DECLARE
  _auth uuid := auth.uid();
  _auth_email text;
  _profile uuid;
  _created_by uuid;
  _analysis_id uuid;
  _project_id uuid;
  _level text;
BEGIN
  IF _auth IS NULL THEN
    RETURN 'none';
  END IF;

  SELECT created_by, analysis_id, project_id
    INTO _created_by, _analysis_id, _project_id
  FROM public.reports WHERE id = _report_id;

  IF NOT FOUND THEN
    RETURN 'none';
  END IF;

  SELECT email INTO _auth_email FROM auth.users WHERE id = _auth LIMIT 1;

  SELECT id INTO _profile FROM public.profiles WHERE user_id = _auth LIMIT 1;
  IF _profile IS NULL THEN
    SELECT id INTO _profile FROM public.profiles WHERE id = _auth LIMIT 1;
  END IF;

  IF _created_by = _auth OR (_profile IS NOT NULL AND _created_by = _profile) THEN
    RETURN 'edit';
  END IF;

  IF public.has_role(_auth, 'admin'::public.app_role) THEN
    RETURN 'edit';
  END IF;

  -- Backfill missing auth_user_id link opportunistically
  IF _auth_email IS NOT NULL THEN
    UPDATE public.app_users
      SET auth_user_id = _auth, updated_at = now()
    WHERE auth_user_id IS NULL
      AND lower(email) = lower(_auth_email);
  END IF;

  SELECT rc.access_level INTO _level
  FROM public.resource_collaborators rc
  JOIN public.app_users au ON au.id = rc.app_user_id
  WHERE (
      au.auth_user_id = _auth
      OR (_auth_email IS NOT NULL AND lower(au.email) = lower(_auth_email))
      OR (_profile IS NOT NULL AND au.profile_id = _profile)
    )
    AND (
      (rc.resource_type = 'report' AND rc.resource_id = _report_id)
      OR (rc.resource_type = 'analysis' AND rc.resource_id = _analysis_id)
      OR (_project_id IS NOT NULL AND rc.resource_type = 'project' AND rc.resource_id = _project_id)
    )
  ORDER BY CASE lower(rc.access_level)
    WHEN 'edit' THEN 1
    WHEN 'comment' THEN 2
    WHEN 'view' THEN 3
    ELSE 4 END
  LIMIT 1;

  RETURN COALESCE(lower(_level), 'none');
END;
$$;


ALTER FUNCTION "public"."get_my_report_access"("_report_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_team_member_public_info"("p_team_id" "uuid") RETURNS TABLE("membership_id" "uuid", "team_id" "uuid", "profile_id" "uuid", "auth_user_id" "uuid", "full_name" "text", "avatar_url" "text", "role" "public"."team_role", "joined_at" timestamp with time zone, "email" "text")
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
  SELECT 
    tm.id AS membership_id,
    tm.team_id,
    p.id AS profile_id,
    p.user_id AS auth_user_id,
    p.full_name,
    p.avatar_url,
    tm.role,
    tm.joined_at,
    public.get_user_email(p.user_id) AS email
  FROM public.team_memberships tm
  JOIN public.profiles p ON p.id = tm.user_id
  WHERE tm.team_id = p_team_id
    AND (
      public.is_team_member((SELECT id FROM public.profiles WHERE user_id = auth.uid()), p_team_id)
      OR public.is_team_admin((SELECT id FROM public.profiles WHERE user_id = auth.uid()), p_team_id)
      OR public.user_owns_account((SELECT id FROM public.profiles WHERE user_id = auth.uid()), (SELECT customer_account_id FROM public.teams WHERE id = p_team_id))
    );
$$;


ALTER FUNCTION "public"."get_team_member_public_info"("p_team_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_user_customer_account"("p_user_id" "uuid") RETURNS "uuid"
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
  SELECT DISTINCT ca.id
  FROM public.customer_accounts ca
  INNER JOIN public.teams t ON t.customer_account_id = ca.id
  INNER JOIN public.team_memberships tm ON tm.team_id = t.id
  WHERE tm.user_id = p_user_id
  LIMIT 1;
$$;


ALTER FUNCTION "public"."get_user_customer_account"("p_user_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_user_email"("user_id" "uuid") RETURNS "text"
    LANGUAGE "plpgsql" STABLE SECURITY DEFINER
    SET "search_path" TO ''
    AS $$
DECLARE
  user_email text;
BEGIN
  SELECT email INTO user_email
  FROM auth.users
  WHERE id = user_id;
  
  RETURN user_email;
END;
$$;


ALTER FUNCTION "public"."get_user_email"("user_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."handle_new_auth_user"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
BEGIN
  INSERT INTO public.profiles (user_id, full_name, onboarding_completed, selected_plan)
  VALUES (NEW.id, COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.raw_user_meta_data->>'name'), false, NULL)
  ON CONFLICT (user_id) DO NOTHING;
  RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."handle_new_auth_user"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."handle_new_user"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public', 'auth'
    AS $$
DECLARE
  v_org uuid;
  v_full_name text;
BEGIN
  v_org := NULLIF(NEW.raw_user_meta_data->>'organization_id','')::uuid;
  v_full_name := NEW.raw_user_meta_data->>'full_name';

  INSERT INTO public.profiles (user_id, organization_id, full_name)
  VALUES (NEW.id, COALESCE(v_org, gen_random_uuid()), v_full_name)
  ON CONFLICT (user_id) DO UPDATE
    SET organization_id = COALESCE(public.profiles.organization_id, EXCLUDED.organization_id),
        full_name       = COALESCE(public.profiles.full_name, EXCLUDED.full_name);

  -- Mark the matching app_users row as Active and link the auth user
  UPDATE public.app_users
  SET auth_user_id = NEW.id,
      accepted_at  = now(),
      status       = 'Active'
  WHERE lower(email) = lower(NEW.email);

  RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."handle_new_user"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."handle_team_invitation_status_change"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
declare
  v_team_name text;
  v_invited_auth_user uuid;
  v_inviter_auth_user uuid;
  v_invited_full_name text;
  v_exists boolean;
begin
  if old.status = 'pending' and new.status in ('accepted', 'declined') then
    select name into v_team_name from public.teams where id = new.team_id;
    select user_id, full_name into v_invited_auth_user, v_invited_full_name from public.profiles where id = new.invited_profile_id;
    select user_id into v_inviter_auth_user from public.profiles where id = new.invited_by;

    if new.status = 'accepted' then
      -- add membership if not already present
      select exists(
        select 1 from public.team_memberships
        where team_id = new.team_id and user_id = new.invited_profile_id
      ) into v_exists;

      if not v_exists then
        insert into public.team_memberships(team_id, user_id, role, invited_by)
        values (new.team_id, new.invited_profile_id, 'member', new.invited_by);
      end if;

      -- notify inviter
      insert into public.notifications (user_id, type, title, message, data, read)
      values (
        v_inviter_auth_user,
        'team_invitation_accepted',
        'Invitation accepted',
        coalesce(v_invited_full_name, 'A user') || ' accepted your invitation to join "' || coalesce(v_team_name, 'Unnamed Team') || '"',
        jsonb_build_object('invitation_id', new.id, 'team_id', new.team_id, 'team_name', v_team_name),
        false
      );
    else
      -- declined: notify inviter
      insert into public.notifications (user_id, type, title, message, data, read)
      values (
        v_inviter_auth_user,
        'team_invitation_declined',
        'Invitation declined',
        coalesce(v_invited_full_name, 'A user') || ' declined your invitation to join "' || coalesce(v_team_name, 'Unnamed Team') || '"',
        jsonb_build_object('invitation_id', new.id, 'team_id', new.team_id, 'team_name', v_team_name),
        false
      );
    end if;
  end if;
  return new;
end;
$$;


ALTER FUNCTION "public"."handle_team_invitation_status_change"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."handle_team_member_added"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
DECLARE
  v_team_name text;
  v_inviter_name text;
  v_target_user_id uuid;
BEGIN
  -- Get team name
  SELECT name INTO v_team_name
  FROM public.teams
  WHERE id = NEW.team_id;
  
  -- Get the auth.users.id from the profile.id
  SELECT user_id INTO v_target_user_id
  FROM public.profiles
  WHERE id = NEW.user_id;
  
  -- Get inviter details
  IF NEW.invited_by IS NOT NULL THEN
    SELECT full_name INTO v_inviter_name
    FROM public.profiles
    WHERE id = NEW.invited_by;
  END IF;
  
  -- Only create notification if the user was invited by someone else (not themselves)
  -- This prevents self-notification when creating a team
  IF NEW.invited_by IS NOT NULL AND NEW.invited_by != NEW.user_id THEN
    INSERT INTO public.notifications (
      user_id,
      type,
      title,
      message,
      data,
      read
    )
    VALUES (
      v_target_user_id,
      'team_invitation_accepted',
      'Added to Team',
      COALESCE(v_inviter_name, 'A team admin') || ' has added you to team "' || COALESCE(v_team_name, 'Unnamed Team') || '". You can now collaborate on shared files and analysis.',
      jsonb_build_object(
        'team_id', NEW.team_id,
        'team_name', v_team_name,
        'invited_by', NEW.invited_by,
        'inviter_name', v_inviter_name,
        'membership_id', NEW.id
      ),
      false
    );
  END IF;
  
  -- DO NOT upgrade user's plan - they keep their original plan
  -- Team membership only grants access to shared files, not professional features
  
  RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."handle_team_member_added"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."handle_team_member_removed"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
BEGIN
  -- DO NOT downgrade user's plan when removed from team
  -- Users maintain their original plan regardless of team membership
  -- Team membership only affects access to shared files, not plan features
  
  RETURN OLD;
END;
$$;


ALTER FUNCTION "public"."handle_team_member_removed"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."has_any_internal_role"("p_user_id" "uuid") RETURNS boolean
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.internal_roles
    WHERE user_id = p_user_id
  );
$$;


ALTER FUNCTION "public"."has_any_internal_role"("p_user_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."is_billing_admin"("p_user_id" "uuid", "p_account_id" "uuid") RETURNS boolean
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.billing_admins
    WHERE customer_account_id = p_account_id
      AND user_id = p_user_id
  );
$$;


ALTER FUNCTION "public"."is_billing_admin"("p_user_id" "uuid", "p_account_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."is_enterprise_admin"("p_user_id" "uuid", "p_account_id" "uuid") RETURNS boolean
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.customer_accounts
    WHERE id = p_account_id
      AND owner_user_id = p_user_id
  );
$$;


ALTER FUNCTION "public"."is_enterprise_admin"("p_user_id" "uuid", "p_account_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."is_internal_staff"("p_user_id" "uuid", "p_required_role" "public"."internal_role") RETURNS boolean
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.internal_roles
    WHERE user_id = p_user_id
      AND role = p_required_role
  );
$$;


ALTER FUNCTION "public"."is_internal_staff"("p_user_id" "uuid", "p_required_role" "public"."internal_role") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."is_license_valid"("p_account_id" "uuid") RETURNS boolean
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.enterprise_licenses
    WHERE customer_account_id = p_account_id
      AND status = 'active'
      AND expires_at > now()
  );
$$;


ALTER FUNCTION "public"."is_license_valid"("p_account_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."is_professional_account"("p_account_id" "uuid") RETURNS boolean
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.customer_accounts
    WHERE id = p_account_id
      AND plan_type = 'professional'
  );
$$;


ALTER FUNCTION "public"."is_professional_account"("p_account_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."is_resource_collaborator"("_resource_type" "text", "_resource_id" "uuid") RETURNS boolean
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
  SELECT EXISTS (
    SELECT 1 FROM public.resource_collaborators rc
    JOIN public.app_users au ON au.id = rc.app_user_id
    WHERE rc.resource_type = _resource_type
      AND rc.resource_id = _resource_id
      AND au.auth_user_id = auth.uid()
  )
$$;


ALTER FUNCTION "public"."is_resource_collaborator"("_resource_type" "text", "_resource_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."is_team_admin"("p_user_id" "uuid", "p_team_id" "uuid") RETURNS boolean
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.team_memberships
    WHERE team_id = p_team_id
      AND user_id = p_user_id
      AND role = 'admin'
  );
$$;


ALTER FUNCTION "public"."is_team_admin"("p_user_id" "uuid", "p_team_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."is_team_member"("p_user_id" "uuid", "p_team_id" "uuid") RETURNS boolean
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.team_memberships
    WHERE team_id = p_team_id
      AND user_id = p_user_id
  );
$$;


ALTER FUNCTION "public"."is_team_member"("p_user_id" "uuid", "p_team_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."notify_file_shared"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
DECLARE
  v_owner_email text;
  v_file_name text;
  v_target_user_id uuid;
BEGIN
  -- Get the owner's email
  SELECT get_user_email(NEW.owner_id) INTO v_owner_email;
  
  -- Get the file name
  SELECT file_name INTO v_file_name
  FROM uploaded_files
  WHERE id = NEW.file_id;
  
  -- Determine the target user ID (prefer shared_with_user_id, fallback to email lookup)
  IF NEW.shared_with_user_id IS NOT NULL THEN
    v_target_user_id := NEW.shared_with_user_id;
  ELSIF NEW.shared_with_email IS NOT NULL THEN
    SELECT id INTO v_target_user_id
    FROM auth.users
    WHERE email = NEW.shared_with_email;
  END IF;
  
  -- Only create notification if we have a target user
  IF v_target_user_id IS NOT NULL THEN
    INSERT INTO public.notifications (
      user_id,
      type,
      title,
      message,
      data,
      read
    )
    VALUES (
      v_target_user_id,
      'file_share_invitation',
      'New file shared with you',
      v_owner_email || ' has shared "' || COALESCE(v_file_name, 'a file') || '" with you',
      jsonb_build_object(
        'share_id', NEW.id,
        'file_id', NEW.file_id,
        'owner_email', v_owner_email,
        'file_name', v_file_name
      ),
      false
    );
  END IF;
  
  RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."notify_file_shared"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."notify_share_status_change"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
DECLARE
  v_file_name text;
  v_action_user_email text;
BEGIN
  -- Only notify if status changed from 'pending' to 'accepted' or 'declined'
  IF OLD.status = 'pending' AND NEW.status IN ('accepted', 'declined') THEN
    -- Get the file name
    SELECT file_name INTO v_file_name
    FROM uploaded_files
    WHERE id = NEW.file_id;
    
    -- Get the email of the user who took the action
    v_action_user_email := NEW.shared_with_email;
    
    -- Notify the owner
    INSERT INTO public.notifications (
      user_id,
      type,
      title,
      message,
      data,
      read
    )
    VALUES (
      NEW.owner_id,
      'file_share_' || NEW.status,
      'File share ' || NEW.status,
      v_action_user_email || ' has ' || NEW.status || ' your share of "' || COALESCE(v_file_name, 'a file') || '"',
      jsonb_build_object(
        'share_id', NEW.id,
        'file_id', NEW.file_id,
        'shared_with_email', v_action_user_email,
        'file_name', v_file_name,
        'status', NEW.status
      ),
      false
    );
  END IF;
  
  RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."notify_share_status_change"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."notify_team_invitation"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
declare
  v_team_name text;
  v_target_auth_user uuid;
  v_inviter_name text;
begin
  select name into v_team_name from public.teams where id = new.team_id;
  select user_id into v_target_auth_user from public.profiles where id = new.invited_profile_id;
  select full_name into v_inviter_name from public.profiles where id = new.invited_by;

  insert into public.notifications (
    user_id,
    type,
    title,
    message,
    data,
    read
  ) values (
    v_target_auth_user,
    'team_invitation',
    'Team Invitation',
    coalesce(v_inviter_name, 'A team admin') || ' invited you to join team "' || coalesce(v_team_name, 'Unnamed Team') || '"',
    jsonb_build_object(
      'invitation_id', new.id,
      'team_id', new.team_id,
      'team_name', v_team_name,
      'invited_by', new.invited_by,
      'inviter_name', v_inviter_name
    ),
    false
  );

  return new;
end;
$$;


ALTER FUNCTION "public"."notify_team_invitation"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."prevent_last_admin_demotion"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
DECLARE
  v_admin_count integer;
BEGIN
  -- Only check if changing from admin to non-admin
  IF OLD.role = 'admin' AND NEW.role != 'admin' THEN
    -- Count remaining admins in the team (excluding the one being changed)
    SELECT COUNT(*) INTO v_admin_count
    FROM public.team_memberships
    WHERE team_id = OLD.team_id
      AND role = 'admin'
      AND id != OLD.id;
    
    -- If this is the last admin, prevent demotion
    IF v_admin_count = 0 THEN
      RAISE EXCEPTION 'Cannot demote the last admin. Please assign another admin first.';
    END IF;
  END IF;
  
  RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."prevent_last_admin_demotion"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."prevent_last_admin_removal"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
DECLARE
  v_admin_count integer;
BEGIN
  -- Only check if the user being removed is an admin
  IF OLD.role = 'admin' THEN
    -- Count remaining admins in the team (excluding the one being deleted)
    SELECT COUNT(*) INTO v_admin_count
    FROM public.team_memberships
    WHERE team_id = OLD.team_id
      AND role = 'admin'
      AND id != OLD.id;
    
    -- If this is the last admin, prevent deletion
    IF v_admin_count = 0 THEN
      RAISE EXCEPTION 'Cannot remove the last admin from the team. Please assign another admin first.';
    END IF;
  END IF;
  
  RETURN OLD;
END;
$$;


ALTER FUNCTION "public"."prevent_last_admin_removal"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."set_updated_at"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    SET "search_path" TO 'public'
    AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END; $$;


ALTER FUNCTION "public"."set_updated_at"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."update_report_comments_updated_at"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    SET "search_path" TO 'public'
    AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."update_report_comments_updated_at"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."update_updated_at_column"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."update_updated_at_column"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."user_owns_account"("p_user_id" "uuid", "p_account_id" "uuid") RETURNS boolean
    LANGUAGE "sql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.customer_accounts
    WHERE id = p_account_id
      AND owner_user_id = p_user_id
  );
$$;


ALTER FUNCTION "public"."user_owns_account"("p_user_id" "uuid", "p_account_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."validate_cloud_credentials"("p_connection_id" "uuid") RETURNS boolean
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 
    FROM public.cloud_datasets 
    WHERE id = p_connection_id 
    AND user_id = auth.uid()
    AND (
      (access_key_ciphertext IS NOT NULL AND secret_key_ciphertext IS NOT NULL)
      OR (access_key IS NOT NULL AND secret_key IS NOT NULL)
    )
  );
END;
$$;


ALTER FUNCTION "public"."validate_cloud_credentials"("p_connection_id" "uuid") OWNER TO "postgres";

SET default_tablespace = '';

SET default_table_access_method = "heap";


CREATE TABLE IF NOT EXISTS "public"."analyses" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "owner_id" "uuid" NOT NULL,
    "project_id" "uuid",
    "name" "text" DEFAULT 'New Analysis'::"text" NOT NULL,
    "team_label" "text" DEFAULT 'Team'::"text",
    "team_count" integer DEFAULT 0,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "dataset_id" "text",
    "thread_id" "text",
    "filename" "text",
    "schema" "jsonb",
    "samples" "jsonb",
    "viz_config" "jsonb",
    "session_id" "text",
    "is_bookmarked" boolean DEFAULT false NOT NULL,
    "is_archived" boolean DEFAULT false NOT NULL,
    "is_pinned" boolean DEFAULT false NOT NULL,
    "parent_analysis_id" "uuid",
    "organization_id" "uuid",
    "session_snapshot" jsonb null
);


ALTER TABLE "public"."analyses" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."analysis_comments" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "analysis_id" "uuid" NOT NULL,
    "content" "text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "parent_id" "uuid",
    "attachments" "jsonb" DEFAULT '[]'::"jsonb" NOT NULL,
    "author_id" "uuid"
);


ALTER TABLE "public"."analysis_comments" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."analysis_feedback" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "analysis_id" "uuid",
    "insight_id" "text",
    "feedback_type" "text",
    "comment" "text",
    "owner_id" "uuid",
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."analysis_feedback" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."analysis_messages" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "analysis_id" "uuid" NOT NULL,
    "role" "text" NOT NULL,
    "content" "text" NOT NULL,
    "output" "jsonb",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "author_id" "uuid",
    CONSTRAINT "analysis_messages_role_check" CHECK (("role" = ANY (ARRAY['user'::"text", 'assistant'::"text"])))
);


ALTER TABLE "public"."analysis_messages" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."app_users" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "organization_id" "uuid" NOT NULL,
    "email" "text" NOT NULL,
    "user_id_code" "text",
    "role" "text",
    "department" "text",
    "branch" "text",
    "status" "text" DEFAULT 'Active'::"text" NOT NULL,
    "permissions" "jsonb" DEFAULT '{}'::"jsonb" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "auth_user_id" "uuid",
    "invited_at" timestamp with time zone,
    "accepted_at" timestamp with time zone,
    "profile_id" "uuid",
    "invited_by" "uuid",
    "team_id" "uuid",
    CONSTRAINT "app_users_status_check" CHECK (("status" = ANY (ARRAY['Active'::"text", 'Inactive'::"text", 'Invited'::"text"])))
);


ALTER TABLE "public"."app_users" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."billing_admins" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "customer_account_id" "uuid" NOT NULL,
    "user_id" "uuid" NOT NULL,
    "assigned_by" "uuid" NOT NULL,
    "assigned_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."billing_admins" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."billing_invoices" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "organization_id" "uuid",
    "owner_profile_id" "uuid",
    "provider" "text" NOT NULL,
    "provider_invoice_id" "text",
    "invoice_number" "text",
    "amount_cents" integer DEFAULT 0 NOT NULL,
    "currency" "text" DEFAULT 'USD'::"text" NOT NULL,
    "status" "text" DEFAULT 'paid'::"text" NOT NULL,
    "paid_at" timestamp with time zone,
    "period_start" timestamp with time zone,
    "period_end" timestamp with time zone,
    "hosted_url" "text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    CONSTRAINT "billing_invoices_provider_check" CHECK (("provider" = ANY (ARRAY['stripe'::"text", 'paypal'::"text"])))
);


ALTER TABLE "public"."billing_invoices" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."branches" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "organization_id" "uuid" NOT NULL,
    "name" "text" NOT NULL,
    "location" "text" NOT NULL,
    "employees" integer DEFAULT 0 NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "head_profile_id" "uuid"
);


ALTER TABLE "public"."branches" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."cloud_datasets" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "name" "text" NOT NULL,
    "provider" "text" NOT NULL,
    "region" "text",
    "access_key" "text",
    "secret_key" "text",
    "bucket_name" "text",
    "endpoint_url" "text",
    "connection_status" "text" DEFAULT 'pending'::"text",
    "last_tested_at" timestamp with time zone,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "access_key_ciphertext" "text",
    "access_key_iv" "text",
    "secret_key_ciphertext" "text",
    "secret_key_iv" "text",
    "dataset_id" "text",
    "session_id" "text",
    "thread_id" "text",
    "file_key" "text",
    "schema" "jsonb",
    "sample_data" "jsonb",
    CONSTRAINT "check_credentials_format" CHECK (((("access_key" IS NOT NULL) AND ("secret_key" IS NOT NULL) AND ("access_key_ciphertext" IS NULL) AND ("secret_key_ciphertext" IS NULL)) OR (("access_key_ciphertext" IS NOT NULL) AND ("secret_key_ciphertext" IS NOT NULL) AND ("access_key" IS NULL) AND ("secret_key" IS NULL))))
);


ALTER TABLE "public"."cloud_datasets" OWNER TO "postgres";


COMMENT ON COLUMN "public"."cloud_datasets"."access_key" IS 'DEPRECATED: Use access_key_ciphertext instead';



COMMENT ON COLUMN "public"."cloud_datasets"."secret_key" IS 'DEPRECATED: Use secret_key_ciphertext instead';



COMMENT ON COLUMN "public"."cloud_datasets"."access_key_ciphertext" IS 'Encrypted access key using AES-256-GCM';



COMMENT ON COLUMN "public"."cloud_datasets"."secret_key_ciphertext" IS 'Encrypted secret key using AES-256-GCM';



CREATE TABLE IF NOT EXISTS "public"."config_roles" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "organization_id" "uuid" NOT NULL,
    "role" "text" NOT NULL,
    "department" "text" NOT NULL,
    "permissions" "text"[] DEFAULT '{}'::"text"[] NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."config_roles" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."customer_accounts" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "owner_user_id" "uuid" NOT NULL,
    "plan_type" "text" DEFAULT 'professional'::"text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "company_name" "text",
    "billing_address" "text",
    "vat_number" "text"
);


ALTER TABLE "public"."customer_accounts" OWNER TO "postgres";


COMMENT ON COLUMN "public"."customer_accounts"."billing_address" IS 'Billing address for enterprise customers';



CREATE TABLE IF NOT EXISTS "public"."dataset_profiles" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "dataset_id" "text" NOT NULL,
    "user_id" "text",
    "file_name" "text",
    "row_count" integer,
    "column_count" integer,
    "columns" "jsonb",
    "summary" "jsonb",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "phase" "text",
    "tier" "text",
    "source_path" "text",
    "source_type" "text",
    "estimated_rows" bigint,
    "exact_rows" bigint,
    "file_size_mb" double precision,
    "sample_pct" double precision,
    "phase_a_elapsed_s" double precision,
    "phase_b_elapsed_s" double precision,
    "completeness_pct" double precision,
    "profiling_result" "jsonb",
    "error" "text"
);


ALTER TABLE "public"."dataset_profiles" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."dataset_samples" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "dataset_id" "text" NOT NULL,
    "sample_data" "jsonb",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "phase" "text",
    "rows_count" integer,
    "columns" "text"[],
    "rows_data" "jsonb"
);


ALTER TABLE "public"."dataset_samples" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."departments" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "organization_id" "uuid" NOT NULL,
    "department" "text" NOT NULL,
    "code" "text" NOT NULL,
    "employees" integer DEFAULT 0 NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "head_profile_id" "uuid"
);


ALTER TABLE "public"."departments" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."enterprise_licenses" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "customer_account_id" "uuid" NOT NULL,
    "serial_number" "text" NOT NULL,
    "issued_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "expires_at" timestamp with time zone NOT NULL,
    "status" "text" DEFAULT 'active'::"text" NOT NULL,
    "created_by" "uuid" NOT NULL,
    "notes" "text",
    "payment_amount" numeric,
    "payment_status" "text" DEFAULT 'unpaid'::"text",
    "paid_at" timestamp with time zone,
    CONSTRAINT "enterprise_licenses_payment_status_check" CHECK (("payment_status" = ANY (ARRAY['unpaid'::"text", 'paid'::"text", 'pending'::"text"]))),
    CONSTRAINT "enterprise_licenses_status_check" CHECK (("status" = ANY (ARRAY['active'::"text", 'expired'::"text", 'revoked'::"text"])))
);


ALTER TABLE "public"."enterprise_licenses" OWNER TO "postgres";


COMMENT ON COLUMN "public"."enterprise_licenses"."payment_amount" IS 'Custom amount agreed upon with customer for license renewal';



COMMENT ON COLUMN "public"."enterprise_licenses"."payment_status" IS 'Payment status: unpaid (initial/trial), pending (processing), paid (completed)';



COMMENT ON COLUMN "public"."enterprise_licenses"."paid_at" IS 'Timestamp when payment was completed';



CREATE TABLE IF NOT EXISTS "public"."enterprise_settings" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "organization_id" "uuid" NOT NULL,
    "company_name" "text",
    "company_slug" "text",
    "company_tagline" "text",
    "company_logo_url" "text",
    "website" "text",
    "company_email" "text",
    "company_email_verified" boolean DEFAULT false NOT NULL,
    "branded_reports" boolean DEFAULT true NOT NULL,
    "branded_emails" boolean DEFAULT true NOT NULL,
    "ai_custom_instruction" "text",
    "ai_instruction_file_url" "text",
    "ai_insight_depth" "text" DEFAULT 'balanced'::"text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    CONSTRAINT "enterprise_settings_ai_insight_depth_check" CHECK (("ai_insight_depth" = ANY (ARRAY['conservative'::"text", 'balanced'::"text", 'aggressive'::"text"])))
);


ALTER TABLE "public"."enterprise_settings" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."file_shares" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "file_id" "uuid" NOT NULL,
    "owner_id" "uuid" NOT NULL,
    "shared_with_email" "text" NOT NULL,
    "status" "text" DEFAULT 'pending'::"text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "expires_at" timestamp with time zone,
    "shared_with_user_id" "uuid",
    CONSTRAINT "file_shares_status_check" CHECK (("status" = ANY (ARRAY['pending'::"text", 'accepted'::"text", 'declined'::"text"])))
);


ALTER TABLE "public"."file_shares" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."folders" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "name" "text" NOT NULL,
    "parent_id" "uuid",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."folders" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."internal_roles" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "role" "public"."internal_role" NOT NULL,
    "assigned_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."internal_roles" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."mcp_connections" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "connection_name" "text" NOT NULL,
    "customer_id" "text" NOT NULL,
    "customer_name" "text" NOT NULL,
    "contact_email" "text" NOT NULL,
    "api_key" "text" NOT NULL,
    "mcp_endpoint" "text" NOT NULL,
    "database_type" "text" NOT NULL,
    "host" "text" NOT NULL,
    "port" integer NOT NULL,
    "database_name" "text" NOT NULL,
    "username" "text" NOT NULL,
    "status" "text" DEFAULT 'active'::"text",
    "query_timeout" integer DEFAULT 30,
    "max_rows" integer DEFAULT 1000,
    "connection_healthy" boolean DEFAULT true,
    "last_health_check" timestamp with time zone,
    "error_message" "text",
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"(),
    "api_key_ciphertext" "text",
    "api_key_iv" "text",
    "username_ciphertext" "text",
    "username_iv" "text"
);


ALTER TABLE "public"."mcp_connections" OWNER TO "postgres";


COMMENT ON COLUMN "public"."mcp_connections"."api_key_ciphertext" IS 'AES-256-GCM encrypted API key';



COMMENT ON COLUMN "public"."mcp_connections"."api_key_iv" IS 'Initialization vector for API key encryption';



COMMENT ON COLUMN "public"."mcp_connections"."username_ciphertext" IS 'AES-256-GCM encrypted username';



COMMENT ON COLUMN "public"."mcp_connections"."username_iv" IS 'Initialization vector for username encryption';



CREATE TABLE IF NOT EXISTS "public"."notifications" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "recipient_id" "uuid" NOT NULL,
    "type" "text" NOT NULL,
    "title" "text" NOT NULL,
    "body" "text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "read_at" timestamp with time zone,
    "actor_id" "uuid",
    "resource_type" "text",
    "resource_id" "uuid",
    "link" "text",
    "metadata" "jsonb" DEFAULT '{}'::"jsonb"
);


ALTER TABLE "public"."notifications" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."organization_teams" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "organization_id" "uuid" NOT NULL,
    "name" "text" NOT NULL,
    "description" "text",
    "branch" "text" NOT NULL,
    "status" "text" DEFAULT 'Active'::"text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "lead_user_id" "uuid"
);


ALTER TABLE "public"."organization_teams" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."organizations" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "name" "text" NOT NULL,
    "owner_profile_id" "uuid" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "slug" "text",
    "tagline" "text",
    "logo_url" "text",
    "brand_reports" boolean DEFAULT true,
    "brand_emails" boolean DEFAULT true
);


ALTER TABLE "public"."organizations" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."plans" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "plan_type" "public"."plan_type" NOT NULL,
    "name" "text" NOT NULL,
    "description" "text",
    "billing_interval" "public"."billing_interval",
    "max_users" integer,
    "price" numeric(10,2) DEFAULT 0 NOT NULL,
    "stripe_price_id" "text",
    "is_active" boolean DEFAULT true NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "paypal_plan_id" "text"
);


ALTER TABLE "public"."plans" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."profiles" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "full_name" "text",
    "phone" "text",
    "avatar_url" "text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "user_profile_type" "public"."user_profile_type",
    "user_interests" "public"."user_interest"[] DEFAULT '{}'::"public"."user_interest"[],
    "onboarding_completed" boolean DEFAULT false NOT NULL,
    "selected_plan" "public"."plan_type" DEFAULT 'free'::"public"."plan_type",
    "stripe_customer_id" "text",
    "stripe_subscription_id" "text",
    "subscription_status" "text",
    "payment_provider" "public"."payment_provider",
    "paypal_subscription_id" "text",
    "paypal_payer_id" "text",
    "trial_ends_at" timestamp with time zone,
    "next_billing_date" timestamp with time zone,
    "subscription_active" boolean DEFAULT false,
    "pending_plan" "public"."plan_type",
    "paypal_order_id" "text",
    "country" "text",
    "timezone" "text",
    "organization_id" "uuid",
    "is_admin" boolean DEFAULT false NOT NULL,
    CONSTRAINT "profiles_subscription_status_check" CHECK (("subscription_status" = ANY (ARRAY['active'::"text", 'canceled'::"text", 'past_due'::"text", 'incomplete'::"text", 'trialing'::"text", 'unpaid'::"text"])))
);


ALTER TABLE "public"."profiles" OWNER TO "postgres";


COMMENT ON COLUMN "public"."profiles"."pending_plan" IS 'Tracks the plan user selected but has not yet paid for';



CREATE TABLE IF NOT EXISTS "public"."projects" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "owner_id" "uuid" NOT NULL,
    "name" "text" NOT NULL,
    "bookmarked" boolean DEFAULT false NOT NULL,
    "archived" boolean DEFAULT false NOT NULL,
    "pinned" boolean DEFAULT false NOT NULL,
    "deleted_at" timestamp with time zone,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "description" "text",
    "organization_id" "uuid"
);


ALTER TABLE "public"."projects" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."report_comments" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "report_id" "uuid" NOT NULL,
    "author_id" "uuid" NOT NULL,
    "parent_id" "uuid",
    "body" "text" NOT NULL,
    "attachments" "jsonb" DEFAULT '[]'::"jsonb",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."report_comments" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."reports" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "organization_id" "uuid" NOT NULL,
    "project_id" "uuid",
    "analysis_id" "uuid" NOT NULL,
    "title" "text" DEFAULT 'Untitled report'::"text" NOT NULL,
    "dataset_label" "text",
    "key_insights" "jsonb" DEFAULT '[]'::"jsonb",
    "status" "text" DEFAULT 'generated'::"text" NOT NULL,
    "created_by" "uuid" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "tab_id" "text"
);


ALTER TABLE "public"."reports" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."resource_collaborators" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "resource_type" "text" NOT NULL,
    "resource_id" "uuid" NOT NULL,
    "app_user_id" "uuid" NOT NULL,
    "access_level" "text" DEFAULT 'view'::"text" NOT NULL,
    "department" "text",
    "invited_by" "uuid",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    CONSTRAINT "resource_collaborators_access_level_check" CHECK (("access_level" = ANY (ARRAY['view'::"text", 'edit'::"text", 'comment'::"text"]))),
    CONSTRAINT "resource_collaborators_resource_type_check" CHECK (("resource_type" = ANY (ARRAY['project'::"text", 'analysis'::"text", 'report'::"text"])))
);


ALTER TABLE "public"."resource_collaborators" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."resource_share_links" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "resource_type" "text" NOT NULL,
    "resource_id" "uuid" NOT NULL,
    "share_token" "text" DEFAULT "encode"("extensions"."gen_random_bytes"(12), 'hex'::"text") NOT NULL,
    "link_access" "text" DEFAULT 'view'::"text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "is_public" boolean DEFAULT false NOT NULL,
    CONSTRAINT "resource_share_links_link_access_check" CHECK (("link_access" = ANY (ARRAY['view'::"text", 'edit'::"text", 'comment'::"text", 'restricted'::"text"]))),
    CONSTRAINT "resource_share_links_resource_type_check" CHECK (("resource_type" = ANY (ARRAY['project'::"text", 'analysis'::"text", 'report'::"text"])))
);


ALTER TABLE "public"."resource_share_links" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."scheduled_runs" (
    "run_id" "text" NOT NULL,
    "schedule_id" "text" NOT NULL,
    "session_id" "text",
    "user_id" "text",
    "run_number" integer,
    "status" "text",
    "started_at" timestamp with time zone,
    "finished_at" timestamp with time zone,
    "duration_ms" integer,
    "result" "jsonb",
    "error" "text"
);


ALTER TABLE "public"."scheduled_runs" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."sql_connections" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "name" "text" NOT NULL,
    "type" "text" NOT NULL,
    "host" "text" NOT NULL,
    "port" integer,
    "database_name" "text",
    "username" "text",
    "use_ssl" boolean DEFAULT true,
    "connection_timeout" integer DEFAULT 30,
    "max_connections" integer DEFAULT 10,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "password_ciphertext" "text",
    "password_iv" "text",
    "credential_type" "text" DEFAULT 'encrypted'::"text",
    "last_accessed" timestamp with time zone,
    "access_count" integer DEFAULT 0,
    "password_salt" "text",
    "connection_method" "text" DEFAULT 'direct'::"text",
    "mcp_connection_id" "text",
    "last_session_id" "text",
    "last_session_created" timestamp with time zone,
    CONSTRAINT "check_encrypted_credentials" CHECK (((("password_ciphertext" IS NOT NULL) AND ("password_iv" IS NOT NULL)) OR (("password_ciphertext" IS NULL) AND ("password_iv" IS NULL))))
);


ALTER TABLE "public"."sql_connections" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."sql_query_history" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "connection_id" "uuid" NOT NULL,
    "session_id" "text",
    "query_text" "text" NOT NULL,
    "dataset_id" "text",
    "thread_id" "text",
    "row_count" integer,
    "execution_time_ms" integer,
    "status" "text" DEFAULT 'success'::"text",
    "error_message" "text",
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."sql_query_history" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."sql_query_sessions" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "connection_id" "uuid" NOT NULL,
    "session_id" "text" NOT NULL,
    "thread_id" "text" NOT NULL,
    "dataset_id" "text",
    "active" boolean DEFAULT true,
    "created_at" timestamp with time zone DEFAULT "now"(),
    "last_activity" timestamp with time zone DEFAULT "now"(),
    "expires_at" timestamp with time zone
);


ALTER TABLE "public"."sql_query_sessions" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."subscriptions" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "owner_profile_id" "uuid" NOT NULL,
    "organization_id" "uuid",
    "plan_id" "uuid" NOT NULL,
    "payment_provider" "public"."payment_provider" NOT NULL,
    "stripe_customer_id" "text",
    "stripe_subscription_id" "text",
    "status" "text" NOT NULL,
    "trial_starts_at" timestamp with time zone,
    "trial_ends_at" timestamp with time zone,
    "current_period_start" timestamp with time zone,
    "current_period_end" timestamp with time zone,
    "cancel_at_period_end" boolean DEFAULT false NOT NULL,
    "canceled_at" timestamp with time zone,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "paypal_subscription_id" "text",
    "paypal_payer_id" "text",
    "serial_number" "text",
    CONSTRAINT "subscriptions_status_check" CHECK (("status" = ANY (ARRAY['trialing'::"text", 'active'::"text", 'past_due'::"text", 'canceled'::"text", 'unpaid'::"text", 'incomplete'::"text", 'incomplete_expired'::"text"])))
);


ALTER TABLE "public"."subscriptions" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."support_tickets" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "subject" "text" NOT NULL,
    "category" "text" DEFAULT 'general'::"text" NOT NULL,
    "message" "text" NOT NULL,
    "status" "text" DEFAULT 'open'::"text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "support_ticket_messages" "jsonb" DEFAULT '[]'::"jsonb" NOT NULL
);


ALTER TABLE "public"."support_tickets" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."team_invitations" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "team_id" "uuid" NOT NULL,
    "invited_profile_id" "uuid" NOT NULL,
    "invited_email" "text" NOT NULL,
    "invited_by" "uuid" NOT NULL,
    "status" "text" DEFAULT 'pending'::"text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."team_invitations" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."team_memberships" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "team_id" "uuid" NOT NULL,
    "user_id" "uuid" NOT NULL,
    "role" "public"."team_role" DEFAULT 'member'::"public"."team_role" NOT NULL,
    "invited_by" "uuid",
    "joined_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."team_memberships" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."teams" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "customer_account_id" "uuid" NOT NULL,
    "name" "text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."teams" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."uploaded_files" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "file_name" "text" NOT NULL,
    "file_path" "text" NOT NULL,
    "file_size" bigint,
    "file_type" "text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "folder_id" "uuid",
    "dataset_id" "text",
    "thread_id" "text",
    "session_id" "text"
);


ALTER TABLE "public"."uploaded_files" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."user_settings" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "theme" "text" DEFAULT 'system'::"text" NOT NULL,
    "recovery_email" "text",
    "news_and_updates" boolean DEFAULT true NOT NULL,
    "tips_and_tutorials" boolean DEFAULT true NOT NULL,
    "user_research" boolean DEFAULT false NOT NULL,
    "reminder_preference" "text" DEFAULT 'all'::"text" NOT NULL,
    "comments_push" boolean DEFAULT true NOT NULL,
    "comments_email" boolean DEFAULT true NOT NULL,
    "comments_sms" boolean DEFAULT false NOT NULL,
    "tags_push" boolean DEFAULT true NOT NULL,
    "tags_email" boolean DEFAULT false NOT NULL,
    "tags_sms" boolean DEFAULT false NOT NULL,
    "reminders_push" boolean DEFAULT false NOT NULL,
    "reminders_email" boolean DEFAULT false NOT NULL,
    "reminders_sms" boolean DEFAULT false NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    CONSTRAINT "user_settings_reminder_preference_check" CHECK (("reminder_preference" = ANY (ARRAY['none'::"text", 'important'::"text", 'all'::"text"]))),
    CONSTRAINT "user_settings_theme_check" CHECK (("theme" = ANY (ARRAY['light'::"text", 'dark'::"text", 'system'::"text"])))
);


ALTER TABLE "public"."user_settings" OWNER TO "postgres";


ALTER TABLE ONLY "public"."analyses"
    ADD CONSTRAINT "analyses_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."analysis_feedback"
    ADD CONSTRAINT "analysis_feedback_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."analysis_comments"
    ADD CONSTRAINT "analysis_insight_comments_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."analysis_messages"
    ADD CONSTRAINT "analysis_messages_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."app_users"
    ADD CONSTRAINT "app_users_auth_user_id_key" UNIQUE ("auth_user_id");



ALTER TABLE ONLY "public"."app_users"
    ADD CONSTRAINT "app_users_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."billing_admins"
    ADD CONSTRAINT "billing_admins_customer_account_id_user_id_key" UNIQUE ("customer_account_id", "user_id");



ALTER TABLE ONLY "public"."billing_admins"
    ADD CONSTRAINT "billing_admins_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."billing_invoices"
    ADD CONSTRAINT "billing_invoices_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."branches"
    ADD CONSTRAINT "branches_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."cloud_datasets"
    ADD CONSTRAINT "cloud_datasets_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."config_roles"
    ADD CONSTRAINT "config_roles_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."customer_accounts"
    ADD CONSTRAINT "customer_accounts_owner_user_id_key" UNIQUE ("owner_user_id");



ALTER TABLE ONLY "public"."customer_accounts"
    ADD CONSTRAINT "customer_accounts_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."dataset_profiles"
    ADD CONSTRAINT "dataset_profiles_dataset_id_phase_key" UNIQUE ("dataset_id", "phase");



ALTER TABLE ONLY "public"."dataset_profiles"
    ADD CONSTRAINT "dataset_profiles_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."dataset_samples"
    ADD CONSTRAINT "dataset_samples_dataset_id_phase_key" UNIQUE ("dataset_id", "phase");



ALTER TABLE ONLY "public"."dataset_samples"
    ADD CONSTRAINT "dataset_samples_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."departments"
    ADD CONSTRAINT "departments_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."enterprise_licenses"
    ADD CONSTRAINT "enterprise_licenses_customer_account_id_key" UNIQUE ("customer_account_id");



ALTER TABLE ONLY "public"."enterprise_licenses"
    ADD CONSTRAINT "enterprise_licenses_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."enterprise_licenses"
    ADD CONSTRAINT "enterprise_licenses_serial_number_key" UNIQUE ("serial_number");



ALTER TABLE ONLY "public"."enterprise_settings"
    ADD CONSTRAINT "enterprise_settings_company_slug_key" UNIQUE ("company_slug");



ALTER TABLE ONLY "public"."enterprise_settings"
    ADD CONSTRAINT "enterprise_settings_organization_id_key" UNIQUE ("organization_id");



ALTER TABLE ONLY "public"."enterprise_settings"
    ADD CONSTRAINT "enterprise_settings_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."file_shares"
    ADD CONSTRAINT "file_shares_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."folders"
    ADD CONSTRAINT "folders_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."folders"
    ADD CONSTRAINT "folders_user_id_name_parent_id_key" UNIQUE ("user_id", "name", "parent_id");



ALTER TABLE ONLY "public"."internal_roles"
    ADD CONSTRAINT "internal_roles_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."internal_roles"
    ADD CONSTRAINT "internal_roles_user_id_role_key" UNIQUE ("user_id", "role");



ALTER TABLE ONLY "public"."mcp_connections"
    ADD CONSTRAINT "mcp_connections_customer_id_key" UNIQUE ("customer_id");



ALTER TABLE ONLY "public"."mcp_connections"
    ADD CONSTRAINT "mcp_connections_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."notifications"
    ADD CONSTRAINT "notifications_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."organization_teams"
    ADD CONSTRAINT "organization_teams_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."organizations"
    ADD CONSTRAINT "organizations_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."organizations"
    ADD CONSTRAINT "organizations_slug_key" UNIQUE ("slug");



ALTER TABLE ONLY "public"."plans"
    ADD CONSTRAINT "plans_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."profiles"
    ADD CONSTRAINT "profiles_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."profiles"
    ADD CONSTRAINT "profiles_user_id_key" UNIQUE ("user_id");



ALTER TABLE ONLY "public"."profiles"
    ADD CONSTRAINT "profiles_user_id_unique" UNIQUE ("user_id");



ALTER TABLE ONLY "public"."projects"
    ADD CONSTRAINT "projects_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."report_comments"
    ADD CONSTRAINT "report_comments_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."reports"
    ADD CONSTRAINT "reports_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."resource_collaborators"
    ADD CONSTRAINT "resource_collaborators_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."resource_collaborators"
    ADD CONSTRAINT "resource_collaborators_resource_type_resource_id_app_user_i_key" UNIQUE ("resource_type", "resource_id", "app_user_id");



ALTER TABLE ONLY "public"."resource_share_links"
    ADD CONSTRAINT "resource_share_links_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."resource_share_links"
    ADD CONSTRAINT "resource_share_links_resource_type_resource_id_key" UNIQUE ("resource_type", "resource_id");



ALTER TABLE ONLY "public"."resource_share_links"
    ADD CONSTRAINT "resource_share_links_share_token_key" UNIQUE ("share_token");



ALTER TABLE ONLY "public"."scheduled_runs"
    ADD CONSTRAINT "scheduled_runs_pkey" PRIMARY KEY ("run_id");



ALTER TABLE ONLY "public"."sql_connections"
    ADD CONSTRAINT "sql_connections_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."sql_query_history"
    ADD CONSTRAINT "sql_query_history_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."sql_query_sessions"
    ADD CONSTRAINT "sql_query_sessions_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."subscriptions"
    ADD CONSTRAINT "subscriptions_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."subscriptions"
    ADD CONSTRAINT "subscriptions_stripe_subscription_id_key" UNIQUE ("stripe_subscription_id");



ALTER TABLE ONLY "public"."support_tickets"
    ADD CONSTRAINT "support_tickets_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."team_invitations"
    ADD CONSTRAINT "team_invitations_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."team_memberships"
    ADD CONSTRAINT "team_memberships_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."team_memberships"
    ADD CONSTRAINT "team_memberships_team_id_user_id_key" UNIQUE ("team_id", "user_id");



ALTER TABLE ONLY "public"."teams"
    ADD CONSTRAINT "teams_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."uploaded_files"
    ADD CONSTRAINT "uploaded_files_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."user_settings"
    ADD CONSTRAINT "user_settings_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."user_settings"
    ADD CONSTRAINT "user_settings_user_id_key" UNIQUE ("user_id");



CREATE INDEX "analyses_is_archived_idx" ON "public"."analyses" USING "btree" ("is_archived") WHERE "is_archived";



CREATE INDEX "analyses_is_bookmarked_idx" ON "public"."analyses" USING "btree" ("is_bookmarked") WHERE "is_bookmarked";



CREATE INDEX "analyses_is_pinned_idx" ON "public"."analyses" USING "btree" ("is_pinned") WHERE "is_pinned";



CREATE INDEX "app_users_org_idx" ON "public"."app_users" USING "btree" ("organization_id");



CREATE INDEX "app_users_team_idx" ON "public"."app_users" USING "btree" ("team_id");



CREATE INDEX "billing_invoices_org_idx" ON "public"."billing_invoices" USING "btree" ("organization_id", "paid_at" DESC);



CREATE INDEX "billing_invoices_owner_idx" ON "public"."billing_invoices" USING "btree" ("owner_profile_id", "paid_at" DESC);



CREATE UNIQUE INDEX "billing_invoices_provider_invoice_uidx" ON "public"."billing_invoices" USING "btree" ("provider", "provider_invoice_id") WHERE ("provider_invoice_id" IS NOT NULL);



CREATE INDEX "branches_head_profile_idx" ON "public"."branches" USING "btree" ("head_profile_id");



CREATE INDEX "branches_org_idx" ON "public"."branches" USING "btree" ("organization_id");



CREATE INDEX "config_roles_org_idx" ON "public"."config_roles" USING "btree" ("organization_id");



CREATE INDEX "departments_org_idx" ON "public"."departments" USING "btree" ("organization_id");



CREATE INDEX "idx_analyses_organization_id" ON "public"."analyses" USING "btree" ("organization_id");



CREATE INDEX "idx_analyses_parent_analysis_id" ON "public"."analyses" USING "btree" ("parent_analysis_id") WHERE ("parent_analysis_id" IS NOT NULL);



CREATE INDEX "idx_analysis_comments_analysis" ON "public"."analysis_comments" USING "btree" ("analysis_id", "created_at");



CREATE INDEX "idx_analysis_comments_author" ON "public"."analysis_comments" USING "btree" ("author_id");



CREATE INDEX "idx_analysis_comments_parent" ON "public"."analysis_comments" USING "btree" ("analysis_id", "parent_id", "created_at");



CREATE INDEX "idx_analysis_messages_author_id" ON "public"."analysis_messages" USING "btree" ("author_id");



CREATE INDEX "idx_enterprise_licenses_account" ON "public"."enterprise_licenses" USING "btree" ("customer_account_id");



CREATE INDEX "idx_enterprise_licenses_serial" ON "public"."enterprise_licenses" USING "btree" ("serial_number");



CREATE INDEX "idx_enterprise_licenses_status" ON "public"."enterprise_licenses" USING "btree" ("status");



CREATE INDEX "idx_file_shares_shared_with_user_id" ON "public"."file_shares" USING "btree" ("shared_with_user_id");



CREATE INDEX "idx_organizations_owner_profile" ON "public"."organizations" USING "btree" ("owner_profile_id");



CREATE INDEX "idx_profiles_payment_provider" ON "public"."profiles" USING "btree" ("payment_provider");



CREATE INDEX "idx_profiles_stripe_customer" ON "public"."profiles" USING "btree" ("stripe_customer_id");



CREATE INDEX "idx_profiles_subscription_status" ON "public"."profiles" USING "btree" ("subscription_status");



CREATE INDEX "idx_projects_organization_id" ON "public"."projects" USING "btree" ("organization_id");



CREATE INDEX "idx_query_history_user_connection" ON "public"."sql_query_history" USING "btree" ("user_id", "connection_id", "created_at" DESC);



CREATE INDEX "idx_query_sessions_user_connection" ON "public"."sql_query_sessions" USING "btree" ("user_id", "connection_id", "active", "last_activity" DESC);



CREATE INDEX "idx_report_comments_author_id" ON "public"."report_comments" USING "btree" ("author_id");



CREATE INDEX "idx_report_comments_parent_id" ON "public"."report_comments" USING "btree" ("parent_id");



CREATE INDEX "idx_report_comments_report_id" ON "public"."report_comments" USING "btree" ("report_id");



CREATE INDEX "idx_reports_analysis" ON "public"."reports" USING "btree" ("analysis_id");



CREATE INDEX "idx_reports_created_by" ON "public"."reports" USING "btree" ("created_by");



CREATE INDEX "idx_reports_org" ON "public"."reports" USING "btree" ("organization_id");



CREATE INDEX "idx_reports_organization_id" ON "public"."reports" USING "btree" ("organization_id");



CREATE INDEX "idx_reports_project" ON "public"."reports" USING "btree" ("project_id");



CREATE INDEX "idx_resource_collaborators_resource" ON "public"."resource_collaborators" USING "btree" ("resource_type", "resource_id");



CREATE INDEX "idx_resource_share_links_resource" ON "public"."resource_share_links" USING "btree" ("resource_type", "resource_id");



CREATE INDEX "idx_subscriptions_organization" ON "public"."subscriptions" USING "btree" ("organization_id");



CREATE INDEX "idx_subscriptions_owner_profile" ON "public"."subscriptions" USING "btree" ("owner_profile_id");



CREATE INDEX "idx_subscriptions_plan" ON "public"."subscriptions" USING "btree" ("plan_id");



CREATE INDEX "idx_subscriptions_status" ON "public"."subscriptions" USING "btree" ("status");



CREATE UNIQUE INDEX "idx_subscriptions_stripe_subscription" ON "public"."subscriptions" USING "btree" ("stripe_subscription_id") WHERE ("stripe_subscription_id" IS NOT NULL);



CREATE INDEX "idx_team_invitations_invited_profile" ON "public"."team_invitations" USING "btree" ("invited_profile_id");



CREATE INDEX "idx_team_invitations_team" ON "public"."team_invitations" USING "btree" ("team_id");



CREATE INDEX "notifications_recipient_created_idx" ON "public"."notifications" USING "btree" ("recipient_id", "created_at" DESC);



CREATE INDEX "notifications_resource_idx" ON "public"."notifications" USING "btree" ("resource_type", "resource_id");



CREATE INDEX "notifications_unread_idx" ON "public"."notifications" USING "btree" ("recipient_id") WHERE ("read_at" IS NULL);



CREATE INDEX "organization_teams_org_idx" ON "public"."organization_teams" USING "btree" ("organization_id");



CREATE UNIQUE INDEX "plans_paypal_plan_id_key" ON "public"."plans" USING "btree" ("paypal_plan_id") WHERE ("paypal_plan_id" IS NOT NULL);



CREATE INDEX "profiles_organization_id_idx" ON "public"."profiles" USING "btree" ("organization_id");



CREATE INDEX "scheduled_runs_schedule_id_idx" ON "public"."scheduled_runs" USING "btree" ("schedule_id", "started_at" DESC);



CREATE UNIQUE INDEX "subscriptions_paypal_subscription_id_key" ON "public"."subscriptions" USING "btree" ("paypal_subscription_id") WHERE ("paypal_subscription_id" IS NOT NULL);



CREATE UNIQUE INDEX "subscriptions_serial_number_key" ON "public"."subscriptions" USING "btree" ("serial_number") WHERE ("serial_number" IS NOT NULL);



CREATE UNIQUE INDEX "uq_team_invite_unique_pending" ON "public"."team_invitations" USING "btree" ("team_id", "invited_profile_id") WHERE ("status" = 'pending'::"text");



CREATE UNIQUE INDEX "ux_profiles_user_id" ON "public"."profiles" USING "btree" ("user_id");



CREATE OR REPLACE TRIGGER "app_users_set_updated_at" BEFORE UPDATE ON "public"."app_users" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();



CREATE OR REPLACE TRIGGER "branches_set_updated_at" BEFORE UPDATE ON "public"."branches" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();



CREATE OR REPLACE TRIGGER "config_roles_set_updated_at" BEFORE UPDATE ON "public"."config_roles" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();



CREATE OR REPLACE TRIGGER "departments_set_updated_at" BEFORE UPDATE ON "public"."departments" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();



CREATE OR REPLACE TRIGGER "enforce_onboarding_plan" BEFORE INSERT OR UPDATE ON "public"."profiles" FOR EACH ROW EXECUTE FUNCTION "public"."check_onboarding_plan"();



CREATE OR REPLACE TRIGGER "handle_team_invitation_status_change_after_update" AFTER UPDATE ON "public"."team_invitations" FOR EACH ROW EXECUTE FUNCTION "public"."handle_team_invitation_status_change"();



CREATE OR REPLACE TRIGGER "notify_team_invitation_after_insert" AFTER INSERT ON "public"."team_invitations" FOR EACH ROW EXECUTE FUNCTION "public"."notify_team_invitation"();



CREATE OR REPLACE TRIGGER "on_enterprise_account_created" AFTER INSERT ON "public"."customer_accounts" FOR EACH ROW EXECUTE FUNCTION "public"."auto_assign_billing_admin"();



CREATE OR REPLACE TRIGGER "on_file_shared" AFTER INSERT ON "public"."file_shares" FOR EACH ROW EXECUTE FUNCTION "public"."notify_file_shared"();



CREATE OR REPLACE TRIGGER "on_share_status_changed" AFTER UPDATE ON "public"."file_shares" FOR EACH ROW EXECUTE FUNCTION "public"."notify_share_status_change"();



CREATE OR REPLACE TRIGGER "on_team_created_join_enterprise_admin" AFTER INSERT ON "public"."teams" FOR EACH ROW EXECUTE FUNCTION "public"."auto_join_enterprise_admin"();



CREATE OR REPLACE TRIGGER "on_team_member_added" AFTER INSERT ON "public"."team_memberships" FOR EACH ROW EXECUTE FUNCTION "public"."handle_team_member_added"();



CREATE OR REPLACE TRIGGER "organization_teams_set_updated_at" BEFORE UPDATE ON "public"."organization_teams" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();



CREATE OR REPLACE TRIGGER "support_tickets_set_updated_at" BEFORE UPDATE ON "public"."support_tickets" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();



CREATE OR REPLACE TRIGGER "team_member_removed_downgrade" AFTER DELETE ON "public"."team_memberships" FOR EACH ROW EXECUTE FUNCTION "public"."handle_team_member_removed"();



CREATE OR REPLACE TRIGGER "trg_app_users_fill_org" BEFORE INSERT ON "public"."app_users" FOR EACH ROW EXECUTE FUNCTION "public"."app_users_fill_org"();



CREATE OR REPLACE TRIGGER "trg_report_comments_updated_at" BEFORE UPDATE ON "public"."report_comments" FOR EACH ROW EXECUTE FUNCTION "public"."update_report_comments_updated_at"();



CREATE OR REPLACE TRIGGER "trg_reports_updated_at" BEFORE UPDATE ON "public"."reports" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();



CREATE OR REPLACE TRIGGER "trg_resource_collaborators_updated" BEFORE UPDATE ON "public"."resource_collaborators" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();



CREATE OR REPLACE TRIGGER "trg_resource_share_links_updated" BEFORE UPDATE ON "public"."resource_share_links" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();



CREATE OR REPLACE TRIGGER "trigger_auto_assign_billing_admin" AFTER INSERT ON "public"."customer_accounts" FOR EACH ROW EXECUTE FUNCTION "public"."auto_assign_billing_admin"();



CREATE OR REPLACE TRIGGER "update_cloud_datasets_updated_at" BEFORE UPDATE ON "public"."cloud_datasets" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_customer_accounts_updated_at" BEFORE UPDATE ON "public"."customer_accounts" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_file_shares_updated_at" BEFORE UPDATE ON "public"."file_shares" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_folders_updated_at" BEFORE UPDATE ON "public"."folders" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_mcp_connections_updated_at" BEFORE UPDATE ON "public"."mcp_connections" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_notifications_updated_at" BEFORE UPDATE ON "public"."notifications" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_organizations_updated_at" BEFORE UPDATE ON "public"."organizations" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_profiles_updated_at" BEFORE UPDATE ON "public"."profiles" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_sql_connections_updated_at" BEFORE UPDATE ON "public"."sql_connections" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_subscriptions_updated_at" BEFORE UPDATE ON "public"."subscriptions" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_team_invitations_updated_at" BEFORE UPDATE ON "public"."team_invitations" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_teams_updated_at" BEFORE UPDATE ON "public"."teams" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



ALTER TABLE ONLY "public"."analyses"
    ADD CONSTRAINT "analyses_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."analyses"
    ADD CONSTRAINT "analyses_owner_id_fkey" FOREIGN KEY ("owner_id") REFERENCES "public"."profiles"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."analyses"
    ADD CONSTRAINT "analyses_parent_analysis_id_fkey" FOREIGN KEY ("parent_analysis_id") REFERENCES "public"."analyses"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."analysis_comments"
    ADD CONSTRAINT "analysis_comments_author_id_fkey" FOREIGN KEY ("author_id") REFERENCES "public"."profiles"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."analysis_feedback"
    ADD CONSTRAINT "analysis_feedback_analysis_id_fkey" FOREIGN KEY ("analysis_id") REFERENCES "public"."analyses"("id");



ALTER TABLE ONLY "public"."analysis_comments"
    ADD CONSTRAINT "analysis_insight_comments_analysis_id_fkey" FOREIGN KEY ("analysis_id") REFERENCES "public"."analyses"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."analysis_comments"
    ADD CONSTRAINT "analysis_insight_comments_parent_id_fkey" FOREIGN KEY ("parent_id") REFERENCES "public"."analysis_comments"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."analysis_messages"
    ADD CONSTRAINT "analysis_messages_analysis_id_fkey" FOREIGN KEY ("analysis_id") REFERENCES "public"."analyses"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."analysis_messages"
    ADD CONSTRAINT "analysis_messages_author_id_fkey" FOREIGN KEY ("author_id") REFERENCES "public"."profiles"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."app_users"
    ADD CONSTRAINT "app_users_auth_user_id_fkey" FOREIGN KEY ("auth_user_id") REFERENCES "auth"."users"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."app_users"
    ADD CONSTRAINT "app_users_invited_by_fkey" FOREIGN KEY ("invited_by") REFERENCES "public"."profiles"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."app_users"
    ADD CONSTRAINT "app_users_organization_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."app_users"
    ADD CONSTRAINT "app_users_profile_fkey" FOREIGN KEY ("profile_id") REFERENCES "public"."profiles"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."app_users"
    ADD CONSTRAINT "app_users_team_fkey" FOREIGN KEY ("team_id") REFERENCES "public"."organization_teams"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."billing_admins"
    ADD CONSTRAINT "billing_admins_assigned_by_fkey" FOREIGN KEY ("assigned_by") REFERENCES "public"."profiles"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."billing_admins"
    ADD CONSTRAINT "billing_admins_customer_account_id_fkey" FOREIGN KEY ("customer_account_id") REFERENCES "public"."customer_accounts"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."billing_admins"
    ADD CONSTRAINT "billing_admins_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "public"."profiles"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."billing_invoices"
    ADD CONSTRAINT "billing_invoices_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."billing_invoices"
    ADD CONSTRAINT "billing_invoices_owner_profile_id_fkey" FOREIGN KEY ("owner_profile_id") REFERENCES "public"."profiles"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."branches"
    ADD CONSTRAINT "branches_head_profile_fkey" FOREIGN KEY ("head_profile_id") REFERENCES "public"."profiles"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."branches"
    ADD CONSTRAINT "branches_organization_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."config_roles"
    ADD CONSTRAINT "config_roles_organization_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."customer_accounts"
    ADD CONSTRAINT "customer_accounts_owner_user_id_fkey" FOREIGN KEY ("owner_user_id") REFERENCES "public"."profiles"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."departments"
    ADD CONSTRAINT "departments_head_profile_id_fkey" FOREIGN KEY ("head_profile_id") REFERENCES "public"."profiles"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."departments"
    ADD CONSTRAINT "departments_organization_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."enterprise_licenses"
    ADD CONSTRAINT "enterprise_licenses_created_by_fkey" FOREIGN KEY ("created_by") REFERENCES "public"."profiles"("id");



ALTER TABLE ONLY "public"."enterprise_licenses"
    ADD CONSTRAINT "enterprise_licenses_customer_account_id_fkey" FOREIGN KEY ("customer_account_id") REFERENCES "public"."customer_accounts"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."file_shares"
    ADD CONSTRAINT "file_shares_shared_with_user_id_fkey" FOREIGN KEY ("shared_with_user_id") REFERENCES "auth"."users"("id");



ALTER TABLE ONLY "public"."file_shares"
    ADD CONSTRAINT "fk_file_shares_file" FOREIGN KEY ("file_id") REFERENCES "public"."uploaded_files"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."folders"
    ADD CONSTRAINT "folders_parent_id_fkey" FOREIGN KEY ("parent_id") REFERENCES "public"."folders"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."internal_roles"
    ADD CONSTRAINT "internal_roles_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "public"."profiles"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."mcp_connections"
    ADD CONSTRAINT "mcp_connections_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."notifications"
    ADD CONSTRAINT "notifications_actor_id_fkey" FOREIGN KEY ("actor_id") REFERENCES "public"."profiles"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."notifications"
    ADD CONSTRAINT "notifications_recipient_fk" FOREIGN KEY ("recipient_id") REFERENCES "public"."profiles"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."organization_teams"
    ADD CONSTRAINT "organization_teams_lead_user_fkey" FOREIGN KEY ("lead_user_id") REFERENCES "public"."app_users"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."organization_teams"
    ADD CONSTRAINT "organization_teams_organization_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."organizations"
    ADD CONSTRAINT "organizations_owner_profile_fkey" FOREIGN KEY ("owner_profile_id") REFERENCES "public"."profiles"("id") ON DELETE RESTRICT;



ALTER TABLE ONLY "public"."profiles"
    ADD CONSTRAINT "profiles_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."projects"
    ADD CONSTRAINT "projects_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."projects"
    ADD CONSTRAINT "projects_owner_id_fkey" FOREIGN KEY ("owner_id") REFERENCES "public"."profiles"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."report_comments"
    ADD CONSTRAINT "report_comments_author_id_fkey" FOREIGN KEY ("author_id") REFERENCES "public"."profiles"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."report_comments"
    ADD CONSTRAINT "report_comments_parent_id_fkey" FOREIGN KEY ("parent_id") REFERENCES "public"."report_comments"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."report_comments"
    ADD CONSTRAINT "report_comments_report_id_fkey" FOREIGN KEY ("report_id") REFERENCES "public"."reports"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."reports"
    ADD CONSTRAINT "reports_analysis_id_fkey" FOREIGN KEY ("analysis_id") REFERENCES "public"."analyses"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."reports"
    ADD CONSTRAINT "reports_created_by_fkey" FOREIGN KEY ("created_by") REFERENCES "auth"."users"("id") ON UPDATE CASCADE ON DELETE RESTRICT;



ALTER TABLE ONLY "public"."reports"
    ADD CONSTRAINT "reports_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations"("id") ON UPDATE CASCADE ON DELETE RESTRICT;



ALTER TABLE ONLY "public"."reports"
    ADD CONSTRAINT "reports_project_id_fkey" FOREIGN KEY ("project_id") REFERENCES "public"."projects"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."resource_collaborators"
    ADD CONSTRAINT "resource_collaborators_app_user_id_fkey" FOREIGN KEY ("app_user_id") REFERENCES "public"."app_users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."resource_collaborators"
    ADD CONSTRAINT "resource_collaborators_invited_by_fkey" FOREIGN KEY ("invited_by") REFERENCES "auth"."users"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."sql_query_history"
    ADD CONSTRAINT "sql_query_history_connection_id_fkey" FOREIGN KEY ("connection_id") REFERENCES "public"."sql_connections"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."sql_query_history"
    ADD CONSTRAINT "sql_query_history_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."sql_query_sessions"
    ADD CONSTRAINT "sql_query_sessions_connection_id_fkey" FOREIGN KEY ("connection_id") REFERENCES "public"."sql_connections"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."sql_query_sessions"
    ADD CONSTRAINT "sql_query_sessions_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."subscriptions"
    ADD CONSTRAINT "subscriptions_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."subscriptions"
    ADD CONSTRAINT "subscriptions_owner_profile_id_fkey" FOREIGN KEY ("owner_profile_id") REFERENCES "public"."profiles"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."subscriptions"
    ADD CONSTRAINT "subscriptions_plan_id_fkey" FOREIGN KEY ("plan_id") REFERENCES "public"."plans"("id");



ALTER TABLE ONLY "public"."support_tickets"
    ADD CONSTRAINT "support_tickets_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."team_invitations"
    ADD CONSTRAINT "team_invitations_invited_by_fkey" FOREIGN KEY ("invited_by") REFERENCES "public"."profiles"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."team_invitations"
    ADD CONSTRAINT "team_invitations_invited_profile_id_fkey" FOREIGN KEY ("invited_profile_id") REFERENCES "public"."profiles"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."team_invitations"
    ADD CONSTRAINT "team_invitations_team_id_fkey" FOREIGN KEY ("team_id") REFERENCES "public"."teams"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."team_memberships"
    ADD CONSTRAINT "team_memberships_invited_by_fkey" FOREIGN KEY ("invited_by") REFERENCES "public"."profiles"("id");



ALTER TABLE ONLY "public"."team_memberships"
    ADD CONSTRAINT "team_memberships_team_id_fkey" FOREIGN KEY ("team_id") REFERENCES "public"."teams"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."team_memberships"
    ADD CONSTRAINT "team_memberships_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "public"."profiles"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."teams"
    ADD CONSTRAINT "teams_customer_account_id_fkey" FOREIGN KEY ("customer_account_id") REFERENCES "public"."customer_accounts"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."uploaded_files"
    ADD CONSTRAINT "uploaded_files_folder_id_fkey" FOREIGN KEY ("folder_id") REFERENCES "public"."folders"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."uploaded_files"
    ADD CONSTRAINT "uploaded_files_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."user_settings"
    ADD CONSTRAINT "user_settings_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



CREATE POLICY "Account owners and enterprise admins can create teams" ON "public"."teams" FOR INSERT WITH CHECK (("public"."user_owns_account"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "customer_account_id") OR "public"."is_enterprise_admin"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "customer_account_id")));



CREATE POLICY "Account owners and enterprise admins can delete teams" ON "public"."teams" FOR DELETE USING (("public"."user_owns_account"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "customer_account_id") OR "public"."is_enterprise_admin"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "customer_account_id")));



CREATE POLICY "Account owners and enterprise admins can update teams" ON "public"."teams" FOR UPDATE USING (("public"."user_owns_account"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "customer_account_id") OR "public"."is_enterprise_admin"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "customer_account_id")));



CREATE POLICY "Active plans are readable" ON "public"."plans" FOR SELECT USING (("is_active" = true));



CREATE POLICY "Admins can delete invites" ON "public"."team_invitations" FOR DELETE USING ("public"."is_team_admin"(( SELECT "p"."id"
   FROM "public"."profiles" "p"
  WHERE ("p"."user_id" = "auth"."uid"())), "team_id"));



CREATE POLICY "Anyone with report access can view comments" ON "public"."report_comments" FOR SELECT TO "authenticated" USING ((EXISTS ( SELECT 1
   FROM "public"."reports" "r"
  WHERE ("r"."id" = "report_comments"."report_id"))));



CREATE POLICY "Authenticated users can read profiles" ON "public"."profiles" FOR SELECT TO "authenticated" USING (true);



CREATE POLICY "Billing admins can update licenses" ON "public"."enterprise_licenses" FOR UPDATE USING ("public"."is_billing_admin"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "customer_account_id"));



CREATE POLICY "Billing admins can view licenses" ON "public"."enterprise_licenses" FOR SELECT USING ("public"."is_billing_admin"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "customer_account_id"));



CREATE POLICY "Billing admins can view their own role" ON "public"."billing_admins" FOR SELECT USING (("user_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



CREATE POLICY "Collaborators can view shared projects" ON "public"."projects" FOR SELECT TO "authenticated" USING ("public"."is_resource_collaborator"('project'::"text", "id"));



CREATE POLICY "Creators delete their reports" ON "public"."reports" FOR DELETE TO "authenticated" USING ((("created_by" = "auth"."uid"()) AND ("organization_id" = "public"."current_org_id"())));



CREATE POLICY "Creators update their reports" ON "public"."reports" FOR UPDATE TO "authenticated" USING ((("created_by" = "auth"."uid"()) AND ("organization_id" = "public"."current_org_id"()))) WITH CHECK ((("created_by" = "auth"."uid"()) AND ("organization_id" = "public"."current_org_id"())));



CREATE POLICY "Enterprise admins can add billing admins" ON "public"."billing_admins" FOR INSERT WITH CHECK ("public"."is_enterprise_admin"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "customer_account_id"));



CREATE POLICY "Enterprise admins can create licenses" ON "public"."enterprise_licenses" FOR INSERT WITH CHECK ("public"."is_enterprise_admin"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "customer_account_id"));



CREATE POLICY "Enterprise admins can remove billing admins" ON "public"."billing_admins" FOR DELETE USING ("public"."is_enterprise_admin"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "customer_account_id"));



CREATE POLICY "Enterprise admins can view billing admins" ON "public"."billing_admins" FOR SELECT USING ("public"."is_enterprise_admin"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "customer_account_id"));



CREATE POLICY "Enterprise admins can view their licenses" ON "public"."enterprise_licenses" FOR SELECT USING ("public"."is_enterprise_admin"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "customer_account_id"));



CREATE POLICY "Internal staff can view all internal roles" ON "public"."internal_roles" FOR SELECT USING ("public"."has_any_internal_role"("auth"."uid"()));



CREATE POLICY "Invited user or admins can update invites" ON "public"."team_invitations" FOR UPDATE USING ((("invited_profile_id" = ( SELECT "p"."id"
   FROM "public"."profiles" "p"
  WHERE ("p"."user_id" = "auth"."uid"()))) OR "public"."is_team_admin"(( SELECT "p"."id"
   FROM "public"."profiles" "p"
  WHERE ("p"."user_id" = "auth"."uid"())), "team_id"))) WITH CHECK ((("invited_profile_id" = ( SELECT "p"."id"
   FROM "public"."profiles" "p"
  WHERE ("p"."user_id" = "auth"."uid"()))) OR "public"."is_team_admin"(( SELECT "p"."id"
   FROM "public"."profiles" "p"
  WHERE ("p"."user_id" = "auth"."uid"())), "team_id")));



CREATE POLICY "Invited user, inviter and admins can view invites" ON "public"."team_invitations" FOR SELECT USING ((("invited_profile_id" = ( SELECT "p"."id"
   FROM "public"."profiles" "p"
  WHERE ("p"."user_id" = "auth"."uid"()))) OR ("invited_by" = ( SELECT "p"."id"
   FROM "public"."profiles" "p"
  WHERE ("p"."user_id" = "auth"."uid"()))) OR "public"."is_team_admin"(( SELECT "p"."id"
   FROM "public"."profiles" "p"
  WHERE ("p"."user_id" = "auth"."uid"())), "team_id")));



CREATE POLICY "Only support admins can manage internal roles" ON "public"."internal_roles" USING ("public"."is_internal_staff"("auth"."uid"(), 'support_admin'::"public"."internal_role")) WITH CHECK ("public"."is_internal_staff"("auth"."uid"(), 'support_admin'::"public"."internal_role"));



CREATE POLICY "Org members can delete enterprise settings" ON "public"."enterprise_settings" FOR DELETE TO "authenticated" USING (("organization_id" = "public"."current_org_id"()));



CREATE POLICY "Org members can insert enterprise settings" ON "public"."enterprise_settings" FOR INSERT TO "authenticated" WITH CHECK (("organization_id" = "public"."current_org_id"()));



CREATE POLICY "Org members can update enterprise settings" ON "public"."enterprise_settings" FOR UPDATE TO "authenticated" USING (("organization_id" = "public"."current_org_id"())) WITH CHECK (("organization_id" = "public"."current_org_id"()));



CREATE POLICY "Org members can view enterprise settings" ON "public"."enterprise_settings" FOR SELECT TO "authenticated" USING (("organization_id" = "public"."current_org_id"()));



CREATE POLICY "Org members insert reports" ON "public"."reports" FOR INSERT TO "authenticated" WITH CHECK ((("organization_id" = "public"."current_org_id"()) AND ("created_by" = "auth"."uid"())));



CREATE POLICY "Owner can insert their organization" ON "public"."organizations" FOR INSERT TO "authenticated" WITH CHECK (("owner_profile_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



CREATE POLICY "Owner can update their organization" ON "public"."organizations" FOR UPDATE TO "authenticated" USING (("owner_profile_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())))) WITH CHECK (("owner_profile_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



CREATE POLICY "Owners delete share links for their reports" ON "public"."resource_share_links" FOR DELETE TO "authenticated" USING ((("resource_type" = 'report'::"text") AND (EXISTS ( SELECT 1
   FROM "public"."reports" "r"
  WHERE (("r"."id" = "resource_share_links"."resource_id") AND ("r"."created_by" = "auth"."uid"()))))));



CREATE POLICY "Owners insert share links for their reports" ON "public"."resource_share_links" FOR INSERT TO "authenticated" WITH CHECK ((("resource_type" = 'report'::"text") AND (EXISTS ( SELECT 1
   FROM "public"."reports" "r"
  WHERE (("r"."id" = "resource_share_links"."resource_id") AND ("r"."created_by" = "auth"."uid"()))))));



CREATE POLICY "Owners read share links for their reports" ON "public"."resource_share_links" FOR SELECT TO "authenticated" USING ((("resource_type" = 'report'::"text") AND (EXISTS ( SELECT 1
   FROM "public"."reports" "r"
  WHERE (("r"."id" = "resource_share_links"."resource_id") AND ("r"."created_by" = "auth"."uid"()))))));



CREATE POLICY "Owners update share links for their reports" ON "public"."resource_share_links" FOR UPDATE TO "authenticated" USING ((("resource_type" = 'report'::"text") AND (EXISTS ( SELECT 1
   FROM "public"."reports" "r"
  WHERE (("r"."id" = "resource_share_links"."resource_id") AND ("r"."created_by" = "auth"."uid"())))))) WITH CHECK ((("resource_type" = 'report'::"text") AND (EXISTS ( SELECT 1
   FROM "public"."reports" "r"
  WHERE (("r"."id" = "resource_share_links"."resource_id") AND ("r"."created_by" = "auth"."uid"()))))));



CREATE POLICY "Read creator profiles of accessible reports" ON "public"."profiles" FOR SELECT TO "authenticated" USING ((EXISTS ( SELECT 1
   FROM "public"."reports" "r"
  WHERE ((("r"."created_by" = "profiles"."id") OR ("r"."created_by" = "profiles"."user_id")) AND ("r"."organization_id" = "public"."current_org_id"()) AND (("r"."created_by" = "auth"."uid"()) OR "public"."is_resource_collaborator"('analysis'::"text", "r"."analysis_id") OR (("r"."project_id" IS NOT NULL) AND "public"."is_resource_collaborator"('project'::"text", "r"."project_id")) OR "public"."is_resource_collaborator"('report'::"text", "r"."id"))))));



CREATE POLICY "Report creators delete collaborators" ON "public"."resource_collaborators" FOR DELETE TO "authenticated" USING ((("resource_type" = 'report'::"text") AND ("invited_by" = "auth"."uid"()) AND (EXISTS ( SELECT 1
   FROM ("public"."reports" "r"
     JOIN "public"."app_users" "invited_user" ON (("invited_user"."id" = "resource_collaborators"."app_user_id")))
  WHERE (("r"."id" = "resource_collaborators"."resource_id") AND ("invited_user"."organization_id" = "r"."organization_id") AND (("r"."created_by" = "auth"."uid"()) OR (EXISTS ( SELECT 1
           FROM "public"."profiles" "creator_profile"
          WHERE ((("creator_profile"."id" = "r"."created_by") OR ("creator_profile"."user_id" = "r"."created_by")) AND (("creator_profile"."user_id" = "auth"."uid"()) OR ("creator_profile"."id" = "auth"."uid"()))))) OR (EXISTS ( SELECT 1
           FROM "public"."app_users" "creator_user"
          WHERE ((("creator_user"."id" = "r"."created_by") OR ("creator_user"."auth_user_id" = "r"."created_by") OR ("creator_user"."profile_id" = "r"."created_by")) AND (("creator_user"."auth_user_id" = "auth"."uid"()) OR ("creator_user"."profile_id" = "auth"."uid"())))))))))));



CREATE POLICY "Report creators insert collaborators" ON "public"."resource_collaborators" FOR INSERT TO "authenticated" WITH CHECK ((("resource_type" = 'report'::"text") AND ("invited_by" = "auth"."uid"()) AND (EXISTS ( SELECT 1
   FROM ("public"."reports" "r"
     JOIN "public"."app_users" "invited_user" ON (("invited_user"."id" = "resource_collaborators"."app_user_id")))
  WHERE (("r"."id" = "resource_collaborators"."resource_id") AND ("invited_user"."organization_id" = "r"."organization_id") AND (("r"."created_by" = "auth"."uid"()) OR (EXISTS ( SELECT 1
           FROM "public"."profiles" "creator_profile"
          WHERE ((("creator_profile"."id" = "r"."created_by") OR ("creator_profile"."user_id" = "r"."created_by")) AND (("creator_profile"."user_id" = "auth"."uid"()) OR ("creator_profile"."id" = "auth"."uid"()))))) OR (EXISTS ( SELECT 1
           FROM "public"."app_users" "creator_user"
          WHERE ((("creator_user"."id" = "r"."created_by") OR ("creator_user"."auth_user_id" = "r"."created_by") OR ("creator_user"."profile_id" = "r"."created_by")) AND (("creator_user"."auth_user_id" = "auth"."uid"()) OR ("creator_user"."profile_id" = "auth"."uid"())))))))))));



CREATE POLICY "Report creators read collaborators" ON "public"."resource_collaborators" FOR SELECT TO "authenticated" USING ((("resource_type" = 'report'::"text") AND ("invited_by" = "auth"."uid"()) AND (EXISTS ( SELECT 1
   FROM ("public"."reports" "r"
     JOIN "public"."app_users" "invited_user" ON (("invited_user"."id" = "resource_collaborators"."app_user_id")))
  WHERE (("r"."id" = "resource_collaborators"."resource_id") AND ("invited_user"."organization_id" = "r"."organization_id") AND (("r"."created_by" = "auth"."uid"()) OR (EXISTS ( SELECT 1
           FROM "public"."profiles" "creator_profile"
          WHERE ((("creator_profile"."id" = "r"."created_by") OR ("creator_profile"."user_id" = "r"."created_by")) AND (("creator_profile"."user_id" = "auth"."uid"()) OR ("creator_profile"."id" = "auth"."uid"()))))) OR (EXISTS ( SELECT 1
           FROM "public"."app_users" "creator_user"
          WHERE ((("creator_user"."id" = "r"."created_by") OR ("creator_user"."auth_user_id" = "r"."created_by") OR ("creator_user"."profile_id" = "r"."created_by")) AND (("creator_user"."auth_user_id" = "auth"."uid"()) OR ("creator_user"."profile_id" = "auth"."uid"())))))))))));



CREATE POLICY "Report creators update collaborators" ON "public"."resource_collaborators" FOR UPDATE TO "authenticated" USING ((("resource_type" = 'report'::"text") AND ("invited_by" = "auth"."uid"()) AND (EXISTS ( SELECT 1
   FROM ("public"."reports" "r"
     JOIN "public"."app_users" "invited_user" ON (("invited_user"."id" = "resource_collaborators"."app_user_id")))
  WHERE (("r"."id" = "resource_collaborators"."resource_id") AND ("invited_user"."organization_id" = "r"."organization_id") AND (("r"."created_by" = "auth"."uid"()) OR (EXISTS ( SELECT 1
           FROM "public"."profiles" "creator_profile"
          WHERE ((("creator_profile"."id" = "r"."created_by") OR ("creator_profile"."user_id" = "r"."created_by")) AND (("creator_profile"."user_id" = "auth"."uid"()) OR ("creator_profile"."id" = "auth"."uid"()))))) OR (EXISTS ( SELECT 1
           FROM "public"."app_users" "creator_user"
          WHERE ((("creator_user"."id" = "r"."created_by") OR ("creator_user"."auth_user_id" = "r"."created_by") OR ("creator_user"."profile_id" = "r"."created_by")) AND (("creator_user"."auth_user_id" = "auth"."uid"()) OR ("creator_user"."profile_id" = "auth"."uid"()))))))))))) WITH CHECK ((("resource_type" = 'report'::"text") AND ("invited_by" = "auth"."uid"()) AND (EXISTS ( SELECT 1
   FROM ("public"."reports" "r"
     JOIN "public"."app_users" "invited_user" ON (("invited_user"."id" = "resource_collaborators"."app_user_id")))
  WHERE (("r"."id" = "resource_collaborators"."resource_id") AND ("invited_user"."organization_id" = "r"."organization_id") AND (("r"."created_by" = "auth"."uid"()) OR (EXISTS ( SELECT 1
           FROM "public"."profiles" "creator_profile"
          WHERE ((("creator_profile"."id" = "r"."created_by") OR ("creator_profile"."user_id" = "r"."created_by")) AND (("creator_profile"."user_id" = "auth"."uid"()) OR ("creator_profile"."id" = "auth"."uid"()))))) OR (EXISTS ( SELECT 1
           FROM "public"."app_users" "creator_user"
          WHERE ((("creator_user"."id" = "r"."created_by") OR ("creator_user"."auth_user_id" = "r"."created_by") OR ("creator_user"."profile_id" = "r"."created_by")) AND (("creator_user"."auth_user_id" = "auth"."uid"()) OR ("creator_user"."profile_id" = "auth"."uid"())))))))))));



CREATE POLICY "Team admins and account owners can create invites" ON "public"."team_invitations" FOR INSERT WITH CHECK (("public"."is_team_admin"(( SELECT "p"."id"
   FROM "public"."profiles" "p"
  WHERE ("p"."user_id" = "auth"."uid"())), "team_id") OR "public"."user_owns_account"(( SELECT "p"."id"
   FROM "public"."profiles" "p"
  WHERE ("p"."user_id" = "auth"."uid"())), ( SELECT "t"."customer_account_id"
   FROM "public"."teams" "t"
  WHERE ("t"."id" = "team_invitations"."team_id")))));



CREATE POLICY "Team admins and account owners can invite members" ON "public"."team_memberships" FOR INSERT WITH CHECK (("public"."is_team_admin"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "team_id") OR "public"."user_owns_account"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), ( SELECT "teams"."customer_account_id"
   FROM "public"."teams"
  WHERE ("teams"."id" = "team_memberships"."team_id")))));



CREATE POLICY "Team admins can remove members" ON "public"."team_memberships" FOR DELETE USING ("public"."is_team_admin"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "team_id"));



CREATE POLICY "Team members can view memberships in their team" ON "public"."team_memberships" FOR SELECT USING ("public"."is_team_member"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "team_id"));



CREATE POLICY "Team members can view their team" ON "public"."teams" FOR SELECT USING (("public"."is_team_member"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "id") OR "public"."user_owns_account"(( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())), "customer_account_id")));



CREATE POLICY "Users can create own sessions" ON "public"."sql_query_sessions" FOR INSERT WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can create query history" ON "public"."sql_query_history" FOR INSERT WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can create shares for their own files" ON "public"."file_shares" FOR INSERT WITH CHECK (("auth"."uid"() = "owner_id"));



CREATE POLICY "Users can create their own MCP connections" ON "public"."mcp_connections" FOR INSERT WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can create their own SQL connections" ON "public"."sql_connections" FOR INSERT WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can create their own accounts" ON "public"."customer_accounts" FOR INSERT TO "authenticated" WITH CHECK (("auth"."uid"() IN ( SELECT "profiles"."user_id"
   FROM "public"."profiles"
  WHERE ("profiles"."id" = "customer_accounts"."owner_user_id"))));



CREATE POLICY "Users can create their own cloud datasets" ON "public"."cloud_datasets" FOR INSERT WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can create their own folders" ON "public"."folders" FOR INSERT WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can create their own support tickets" ON "public"."support_tickets" FOR INSERT TO "authenticated" WITH CHECK (("user_id" = "auth"."uid"()));



CREATE POLICY "Users can delete own sessions" ON "public"."sql_query_sessions" FOR DELETE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can delete own settings" ON "public"."user_settings" FOR DELETE TO "authenticated" USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can delete owned dataset profiles" ON "public"."dataset_profiles" FOR DELETE TO "authenticated" USING ((("user_id" = ("auth"."uid"())::"text") OR (EXISTS ( SELECT 1
   FROM "public"."uploaded_files" "uf"
  WHERE (("uf"."dataset_id" = "dataset_profiles"."dataset_id") AND ("uf"."user_id" = "auth"."uid"())))) OR (EXISTS ( SELECT 1
   FROM "public"."cloud_datasets" "cd"
  WHERE (("cd"."dataset_id" = "dataset_profiles"."dataset_id") AND ("cd"."user_id" = "auth"."uid"()))))));



CREATE POLICY "Users can delete owned dataset samples" ON "public"."dataset_samples" FOR DELETE TO "authenticated" USING (((EXISTS ( SELECT 1
   FROM "public"."dataset_profiles" "dp"
  WHERE (("dp"."dataset_id" = "dataset_samples"."dataset_id") AND ("dp"."user_id" = ("auth"."uid"())::"text")))) OR (EXISTS ( SELECT 1
   FROM "public"."uploaded_files" "uf"
  WHERE (("uf"."dataset_id" = "dataset_samples"."dataset_id") AND ("uf"."user_id" = "auth"."uid"())))) OR (EXISTS ( SELECT 1
   FROM "public"."cloud_datasets" "cd"
  WHERE (("cd"."dataset_id" = "dataset_samples"."dataset_id") AND ("cd"."user_id" = "auth"."uid"()))))));



CREATE POLICY "Users can delete shares they own" ON "public"."file_shares" FOR DELETE USING (("auth"."uid"() = "owner_id"));



CREATE POLICY "Users can delete their own MCP connections" ON "public"."mcp_connections" FOR DELETE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can delete their own SQL connections" ON "public"."sql_connections" FOR DELETE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can delete their own cloud datasets" ON "public"."cloud_datasets" FOR DELETE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can delete their own files" ON "public"."uploaded_files" FOR DELETE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can delete their own folders" ON "public"."folders" FOR DELETE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can delete their own report comments" ON "public"."report_comments" FOR DELETE TO "authenticated" USING ((EXISTS ( SELECT 1
   FROM "public"."reports" "r"
  WHERE (("r"."id" = "report_comments"."report_id") AND ("r"."created_by" = ( SELECT "auth"."uid"() AS "uid"))))));



CREATE POLICY "Users can insert own settings" ON "public"."user_settings" FOR INSERT TO "authenticated" WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can insert owned dataset profiles" ON "public"."dataset_profiles" FOR INSERT TO "authenticated" WITH CHECK ((("user_id" = ("auth"."uid"())::"text") OR (EXISTS ( SELECT 1
   FROM "public"."uploaded_files" "uf"
  WHERE (("uf"."dataset_id" = "dataset_profiles"."dataset_id") AND ("uf"."user_id" = "auth"."uid"())))) OR (EXISTS ( SELECT 1
   FROM "public"."cloud_datasets" "cd"
  WHERE (("cd"."dataset_id" = "dataset_profiles"."dataset_id") AND ("cd"."user_id" = "auth"."uid"()))))));



CREATE POLICY "Users can insert owned dataset samples" ON "public"."dataset_samples" FOR INSERT TO "authenticated" WITH CHECK (((EXISTS ( SELECT 1
   FROM "public"."dataset_profiles" "dp"
  WHERE (("dp"."dataset_id" = "dataset_samples"."dataset_id") AND ("dp"."user_id" = ("auth"."uid"())::"text")))) OR (EXISTS ( SELECT 1
   FROM "public"."uploaded_files" "uf"
  WHERE (("uf"."dataset_id" = "dataset_samples"."dataset_id") AND ("uf"."user_id" = "auth"."uid"())))) OR (EXISTS ( SELECT 1
   FROM "public"."cloud_datasets" "cd"
  WHERE (("cd"."dataset_id" = "dataset_samples"."dataset_id") AND ("cd"."user_id" = "auth"."uid"()))))));



CREATE POLICY "Users can insert their own profile" ON "public"."profiles" FOR INSERT WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can insert their own report comments" ON "public"."report_comments" FOR INSERT TO "authenticated" WITH CHECK ((EXISTS ( SELECT 1
   FROM "public"."reports" "r"
  WHERE (("r"."id" = "report_comments"."report_id") AND ("r"."created_by" = ( SELECT "auth"."uid"() AS "uid"))))));



CREATE POLICY "Users can update own sessions" ON "public"."sql_query_sessions" FOR UPDATE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can update own settings" ON "public"."user_settings" FOR UPDATE TO "authenticated" USING (("auth"."uid"() = "user_id")) WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can update owned dataset profiles" ON "public"."dataset_profiles" FOR UPDATE TO "authenticated" USING ((("user_id" = ("auth"."uid"())::"text") OR (EXISTS ( SELECT 1
   FROM "public"."uploaded_files" "uf"
  WHERE (("uf"."dataset_id" = "dataset_profiles"."dataset_id") AND ("uf"."user_id" = "auth"."uid"())))) OR (EXISTS ( SELECT 1
   FROM "public"."cloud_datasets" "cd"
  WHERE (("cd"."dataset_id" = "dataset_profiles"."dataset_id") AND ("cd"."user_id" = "auth"."uid"())))))) WITH CHECK ((("user_id" = ("auth"."uid"())::"text") OR (EXISTS ( SELECT 1
   FROM "public"."uploaded_files" "uf"
  WHERE (("uf"."dataset_id" = "dataset_profiles"."dataset_id") AND ("uf"."user_id" = "auth"."uid"())))) OR (EXISTS ( SELECT 1
   FROM "public"."cloud_datasets" "cd"
  WHERE (("cd"."dataset_id" = "dataset_profiles"."dataset_id") AND ("cd"."user_id" = "auth"."uid"()))))));



CREATE POLICY "Users can update owned dataset samples" ON "public"."dataset_samples" FOR UPDATE TO "authenticated" USING (((EXISTS ( SELECT 1
   FROM "public"."dataset_profiles" "dp"
  WHERE (("dp"."dataset_id" = "dataset_samples"."dataset_id") AND ("dp"."user_id" = ("auth"."uid"())::"text")))) OR (EXISTS ( SELECT 1
   FROM "public"."uploaded_files" "uf"
  WHERE (("uf"."dataset_id" = "dataset_samples"."dataset_id") AND ("uf"."user_id" = "auth"."uid"())))) OR (EXISTS ( SELECT 1
   FROM "public"."cloud_datasets" "cd"
  WHERE (("cd"."dataset_id" = "dataset_samples"."dataset_id") AND ("cd"."user_id" = "auth"."uid"())))))) WITH CHECK (((EXISTS ( SELECT 1
   FROM "public"."dataset_profiles" "dp"
  WHERE (("dp"."dataset_id" = "dataset_samples"."dataset_id") AND ("dp"."user_id" = ("auth"."uid"())::"text")))) OR (EXISTS ( SELECT 1
   FROM "public"."uploaded_files" "uf"
  WHERE (("uf"."dataset_id" = "dataset_samples"."dataset_id") AND ("uf"."user_id" = "auth"."uid"())))) OR (EXISTS ( SELECT 1
   FROM "public"."cloud_datasets" "cd"
  WHERE (("cd"."dataset_id" = "dataset_samples"."dataset_id") AND ("cd"."user_id" = "auth"."uid"()))))));



CREATE POLICY "Users can update shares they own or are invited to" ON "public"."file_shares" FOR UPDATE USING ((("auth"."uid"() = "owner_id") OR ("auth"."uid"() = "shared_with_user_id") OR ("shared_with_email" = (( SELECT "users"."email"
   FROM "auth"."users"
  WHERE ("users"."id" = "auth"."uid"())))::"text")));



CREATE POLICY "Users can update their own MCP connections" ON "public"."mcp_connections" FOR UPDATE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can update their own SQL connections" ON "public"."sql_connections" FOR UPDATE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can update their own accounts" ON "public"."customer_accounts" FOR UPDATE TO "authenticated" USING (("auth"."uid"() IN ( SELECT "profiles"."user_id"
   FROM "public"."profiles"
  WHERE ("profiles"."id" = "customer_accounts"."owner_user_id"))));



CREATE POLICY "Users can update their own cloud datasets" ON "public"."cloud_datasets" FOR UPDATE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can update their own files" ON "public"."uploaded_files" FOR UPDATE USING (("auth"."uid"() = "user_id")) WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can update their own folders" ON "public"."folders" FOR UPDATE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can update their own profile" ON "public"."profiles" FOR UPDATE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can update their own report comments" ON "public"."report_comments" FOR UPDATE TO "authenticated" USING ((EXISTS ( SELECT 1
   FROM "public"."reports" "r"
  WHERE (("r"."id" = "report_comments"."report_id") AND ("r"."created_by" = ( SELECT "auth"."uid"() AS "uid")))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM "public"."reports" "r"
  WHERE (("r"."id" = "report_comments"."report_id") AND ("r"."created_by" = ( SELECT "auth"."uid"() AS "uid"))))));



CREATE POLICY "Users can update their own tickets" ON "public"."support_tickets" FOR UPDATE TO "authenticated" USING (("user_id" = "auth"."uid"())) WITH CHECK (("user_id" = "auth"."uid"()));



CREATE POLICY "Users can upload their own files" ON "public"."uploaded_files" FOR INSERT WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can view own query history" ON "public"."sql_query_history" FOR SELECT USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can view own sessions" ON "public"."sql_query_sessions" FOR SELECT USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can view own settings" ON "public"."user_settings" FOR SELECT TO "authenticated" USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can view owned dataset profiles" ON "public"."dataset_profiles" FOR SELECT TO "authenticated" USING ((("user_id" = ("auth"."uid"())::"text") OR (EXISTS ( SELECT 1
   FROM "public"."uploaded_files" "uf"
  WHERE (("uf"."dataset_id" = "dataset_profiles"."dataset_id") AND ("uf"."user_id" = "auth"."uid"())))) OR (EXISTS ( SELECT 1
   FROM "public"."cloud_datasets" "cd"
  WHERE (("cd"."dataset_id" = "dataset_profiles"."dataset_id") AND ("cd"."user_id" = "auth"."uid"()))))));



CREATE POLICY "Users can view owned dataset samples" ON "public"."dataset_samples" FOR SELECT TO "authenticated" USING (((EXISTS ( SELECT 1
   FROM "public"."dataset_profiles" "dp"
  WHERE (("dp"."dataset_id" = "dataset_samples"."dataset_id") AND ("dp"."user_id" = ("auth"."uid"())::"text")))) OR (EXISTS ( SELECT 1
   FROM "public"."uploaded_files" "uf"
  WHERE (("uf"."dataset_id" = "dataset_samples"."dataset_id") AND ("uf"."user_id" = "auth"."uid"())))) OR (EXISTS ( SELECT 1
   FROM "public"."cloud_datasets" "cd"
  WHERE (("cd"."dataset_id" = "dataset_samples"."dataset_id") AND ("cd"."user_id" = "auth"."uid"()))))));



CREATE POLICY "Users can view shares they own or are shared with" ON "public"."file_shares" FOR SELECT USING ((("auth"."uid"() = "owner_id") OR ("auth"."uid"() = "shared_with_user_id") OR ("shared_with_email" = (( SELECT "users"."email"
   FROM "auth"."users"
  WHERE ("users"."id" = "auth"."uid"())))::"text")));



CREATE POLICY "Users can view their own MCP connections" ON "public"."mcp_connections" FOR SELECT USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can view their own SQL connections" ON "public"."sql_connections" FOR SELECT USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can view their own accounts" ON "public"."customer_accounts" FOR SELECT TO "authenticated" USING (("auth"."uid"() IN ( SELECT "profiles"."user_id"
   FROM "public"."profiles"
  WHERE ("profiles"."id" = "customer_accounts"."owner_user_id"))));



CREATE POLICY "Users can view their own cloud datasets" ON "public"."cloud_datasets" FOR SELECT USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can view their own files" ON "public"."uploaded_files" FOR SELECT USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can view their own folders" ON "public"."folders" FOR SELECT USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can view their own memberships" ON "public"."team_memberships" FOR SELECT USING (("user_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



CREATE POLICY "Users can view their own profile" ON "public"."profiles" FOR SELECT USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can view their own subscriptions" ON "public"."subscriptions" FOR SELECT TO "authenticated" USING (("owner_profile_id" IN ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



CREATE POLICY "Users can view their own support tickets" ON "public"."support_tickets" FOR SELECT TO "authenticated" USING (("user_id" = "auth"."uid"()));



CREATE POLICY "Users view own or shared reports" ON "public"."reports" FOR SELECT TO "authenticated" USING ((("created_by" = "auth"."uid"()) OR "public"."is_resource_collaborator"('report'::"text", "id")));



ALTER TABLE "public"."analyses" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."analysis_comments" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "analysis_comments_delete" ON "public"."analysis_comments" FOR DELETE TO "authenticated" USING (("author_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



CREATE POLICY "analysis_comments_insert" ON "public"."analysis_comments" FOR INSERT TO "authenticated" WITH CHECK ((("author_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))) AND (EXISTS ( SELECT 1
   FROM "public"."analyses" "a"
  WHERE (("a"."id" = "analysis_comments"."analysis_id") AND (("a"."owner_id" = ( SELECT "profiles"."id"
           FROM "public"."profiles"
          WHERE ("profiles"."user_id" = "auth"."uid"()))) OR "public"."is_resource_collaborator"('analysis'::"text", "a"."id") OR (("a"."project_id" IS NOT NULL) AND "public"."is_resource_collaborator"('project'::"text", "a"."project_id"))))))));



CREATE POLICY "analysis_comments_select" ON "public"."analysis_comments" FOR SELECT TO "authenticated" USING ((EXISTS ( SELECT 1
   FROM "public"."analyses" "a"
  WHERE (("a"."id" = "analysis_comments"."analysis_id") AND (("a"."owner_id" = ( SELECT "profiles"."id"
           FROM "public"."profiles"
          WHERE ("profiles"."user_id" = "auth"."uid"()))) OR "public"."is_resource_collaborator"('analysis'::"text", "a"."id") OR (("a"."project_id" IS NOT NULL) AND "public"."is_resource_collaborator"('project'::"text", "a"."project_id")))))));



CREATE POLICY "analysis_comments_update" ON "public"."analysis_comments" FOR UPDATE TO "authenticated" USING (("author_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())))) WITH CHECK (("author_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



ALTER TABLE "public"."analysis_feedback" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."analysis_messages" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."app_users" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "authenticated notifications insert" ON "public"."notifications" FOR INSERT TO "authenticated" WITH CHECK (("actor_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



ALTER TABLE "public"."billing_admins" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."billing_invoices" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "billing_invoices_read" ON "public"."billing_invoices" FOR SELECT TO "authenticated" USING ((("owner_profile_id" IN ( SELECT "p"."id"
   FROM "public"."profiles" "p"
  WHERE (("p"."id" = "auth"."uid"()) OR ("p"."user_id" = "auth"."uid"())))) OR (("organization_id" IS NOT NULL) AND ("organization_id" IN ( SELECT "p"."organization_id"
   FROM "public"."profiles" "p"
  WHERE (("p"."id" = "auth"."uid"()) OR ("p"."user_id" = "auth"."uid"())))))));



ALTER TABLE "public"."branches" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."cloud_datasets" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "collaborators insert shared analysis messages" ON "public"."analysis_messages" FOR INSERT TO "authenticated" WITH CHECK ((EXISTS ( SELECT 1
   FROM "public"."analyses" "a"
  WHERE (("a"."id" = "analysis_messages"."analysis_id") AND ("public"."is_resource_collaborator"('analysis'::"text", "a"."id") OR (("a"."project_id" IS NOT NULL) AND "public"."is_resource_collaborator"('project'::"text", "a"."project_id")))))));



CREATE POLICY "collaborators read shared analyses" ON "public"."analyses" FOR SELECT TO "authenticated" USING (("public"."is_resource_collaborator"('analysis'::"text", "id") OR (("project_id" IS NOT NULL) AND "public"."is_resource_collaborator"('project'::"text", "project_id"))));



CREATE POLICY "collaborators read shared analysis messages" ON "public"."analysis_messages" FOR SELECT USING ((EXISTS ( SELECT 1
   FROM "public"."analyses" "a"
  WHERE (("a"."id" = "analysis_messages"."analysis_id") AND ("public"."is_resource_collaborator"('analysis'::"text", "a"."id") OR (("a"."project_id" IS NOT NULL) AND "public"."is_resource_collaborator"('project'::"text", "a"."project_id")))))));



CREATE POLICY "collaborators read shared projects" ON "public"."projects" FOR SELECT TO "authenticated" USING ((EXISTS ( SELECT 1
   FROM ("public"."resource_collaborators" "rc"
     JOIN "public"."app_users" "au" ON (("au"."id" = "rc"."app_user_id")))
  WHERE (("rc"."resource_type" = 'project'::"text") AND ("rc"."resource_id" = "projects"."id") AND ("au"."auth_user_id" = "auth"."uid"())))));



ALTER TABLE "public"."config_roles" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."customer_accounts" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."dataset_profiles" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."dataset_samples" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."departments" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "departments delete" ON "public"."departments" FOR DELETE TO "authenticated" USING (("organization_id" = "public"."current_org_id"()));



CREATE POLICY "departments insert" ON "public"."departments" FOR INSERT TO "authenticated" WITH CHECK (("organization_id" = "public"."current_org_id"()));



CREATE POLICY "departments select" ON "public"."departments" FOR SELECT TO "authenticated" USING (("organization_id" = "public"."current_org_id"()));



CREATE POLICY "departments update" ON "public"."departments" FOR UPDATE TO "authenticated" USING (("organization_id" = "public"."current_org_id"())) WITH CHECK (("organization_id" = "public"."current_org_id"()));



ALTER TABLE "public"."enterprise_licenses" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."enterprise_settings" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."file_shares" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."folders" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."internal_roles" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."mcp_connections" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."notifications" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "org members manage app_users" ON "public"."app_users" TO "authenticated" USING (("organization_id" = "public"."current_org_id"())) WITH CHECK (("organization_id" = "public"."current_org_id"()));



CREATE POLICY "org members manage branches" ON "public"."branches" TO "authenticated" USING (("organization_id" = "public"."current_org_id"())) WITH CHECK (("organization_id" = "public"."current_org_id"()));



CREATE POLICY "org members manage config_roles" ON "public"."config_roles" TO "authenticated" USING (("organization_id" = "public"."current_org_id"())) WITH CHECK (("organization_id" = "public"."current_org_id"()));



CREATE POLICY "org members manage organization_teams" ON "public"."organization_teams" TO "authenticated" USING (("organization_id" = "public"."current_org_id"())) WITH CHECK (("organization_id" = "public"."current_org_id"()));



ALTER TABLE "public"."organization_teams" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."organizations" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "own analyses delete" ON "public"."analyses" FOR DELETE TO "authenticated" USING (("owner_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



CREATE POLICY "own analyses insert" ON "public"."analyses" FOR INSERT TO "authenticated" WITH CHECK (("owner_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



CREATE POLICY "own analyses select" ON "public"."analyses" FOR SELECT TO "authenticated" USING (("owner_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



CREATE POLICY "own analyses update" ON "public"."analyses" FOR UPDATE TO "authenticated" USING (("owner_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())))) WITH CHECK (("owner_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



CREATE POLICY "own messages insert" ON "public"."analysis_messages" FOR INSERT TO "authenticated" WITH CHECK ((EXISTS ( SELECT 1
   FROM "public"."analyses" "a"
  WHERE (("a"."id" = "analysis_messages"."analysis_id") AND ("a"."owner_id" = ( SELECT "profiles"."id"
           FROM "public"."profiles"
          WHERE ("profiles"."user_id" = "auth"."uid"())))))));



CREATE POLICY "own messages select" ON "public"."analysis_messages" FOR SELECT TO "authenticated" USING ((EXISTS ( SELECT 1
   FROM "public"."analyses" "a"
  WHERE (("a"."id" = "analysis_messages"."analysis_id") AND ("a"."owner_id" = ( SELECT "profiles"."id"
           FROM "public"."profiles"
          WHERE ("profiles"."user_id" = "auth"."uid"())))))));



CREATE POLICY "own notifications delete" ON "public"."notifications" FOR DELETE TO "authenticated" USING (("recipient_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



CREATE POLICY "own notifications select" ON "public"."notifications" FOR SELECT TO "authenticated" USING (("recipient_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



CREATE POLICY "own notifications update" ON "public"."notifications" FOR UPDATE TO "authenticated" USING (("recipient_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())))) WITH CHECK (("recipient_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



CREATE POLICY "own projects insert" ON "public"."projects" FOR INSERT TO "authenticated" WITH CHECK (("owner_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



CREATE POLICY "own projects select" ON "public"."projects" FOR SELECT TO "authenticated" USING (("owner_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



CREATE POLICY "own projects update" ON "public"."projects" FOR UPDATE TO "authenticated" USING (("owner_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"())))) WITH CHECK (("owner_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))));



CREATE POLICY "owners and collaborators delete projects" ON "public"."projects" FOR DELETE TO "authenticated" USING ((("owner_id" = ( SELECT "profiles"."id"
   FROM "public"."profiles"
  WHERE ("profiles"."user_id" = "auth"."uid"()))) OR "public"."is_resource_collaborator"('project'::"text", "id")));



CREATE POLICY "owners delete resource collaborators" ON "public"."resource_collaborators" FOR DELETE TO "authenticated" USING ("public"."can_manage_resource_share"("resource_type", "resource_id"));



CREATE POLICY "owners delete resource share links" ON "public"."resource_share_links" FOR DELETE TO "authenticated" USING ("public"."can_manage_resource_share"("resource_type", "resource_id"));



CREATE POLICY "owners insert resource collaborators" ON "public"."resource_collaborators" FOR INSERT TO "authenticated" WITH CHECK ("public"."can_manage_resource_share"("resource_type", "resource_id"));



CREATE POLICY "owners insert resource share links" ON "public"."resource_share_links" FOR INSERT TO "authenticated" WITH CHECK ("public"."can_manage_resource_share"("resource_type", "resource_id"));



CREATE POLICY "owners read resource collaborators" ON "public"."resource_collaborators" FOR SELECT TO "authenticated" USING ("public"."can_manage_resource_share"("resource_type", "resource_id"));



CREATE POLICY "owners read resource share links" ON "public"."resource_share_links" FOR SELECT TO "authenticated" USING ("public"."can_manage_resource_share"("resource_type", "resource_id"));



CREATE POLICY "owners update resource collaborators" ON "public"."resource_collaborators" FOR UPDATE TO "authenticated" USING ("public"."can_manage_resource_share"("resource_type", "resource_id")) WITH CHECK ("public"."can_manage_resource_share"("resource_type", "resource_id"));



CREATE POLICY "owners update resource share links" ON "public"."resource_share_links" FOR UPDATE TO "authenticated" USING ("public"."can_manage_resource_share"("resource_type", "resource_id")) WITH CHECK ("public"."can_manage_resource_share"("resource_type", "resource_id"));



ALTER TABLE "public"."plans" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."profiles" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."projects" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "read projects via accessible report" ON "public"."projects" FOR SELECT TO "authenticated" USING ((EXISTS ( SELECT 1
   FROM "public"."reports" "r"
  WHERE (("r"."project_id" = "projects"."id") AND (("r"."created_by" = "auth"."uid"()) OR "public"."is_resource_collaborator"('report'::"text", "r"."id"))))));



CREATE POLICY "report collaborators can delete comments" ON "public"."report_comments" FOR DELETE TO "authenticated" USING ((EXISTS ( SELECT 1
   FROM "public"."resource_collaborators" "rc"
  WHERE (("rc"."app_user_id" = "report_comments"."author_id") AND ("rc"."resource_type" = 'report'::"text") AND ("rc"."resource_id" = "report_comments"."report_id")))));



CREATE POLICY "report collaborators can insert comments" ON "public"."report_comments" FOR INSERT TO "authenticated" WITH CHECK ((EXISTS ( SELECT 1
   FROM "public"."resource_collaborators" "rc"
  WHERE (("rc"."app_user_id" = "report_comments"."author_id") AND ("rc"."resource_type" = 'report'::"text") AND ("rc"."resource_id" = "report_comments"."report_id")))));



CREATE POLICY "report collaborators can update comments" ON "public"."report_comments" FOR UPDATE TO "authenticated" USING ((EXISTS ( SELECT 1
   FROM "public"."resource_collaborators" "rc"
  WHERE (("rc"."app_user_id" = "report_comments"."author_id") AND ("rc"."resource_type" = 'report'::"text") AND ("rc"."resource_id" = "report_comments"."report_id"))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM "public"."resource_collaborators" "rc"
  WHERE (("rc"."app_user_id" = "report_comments"."author_id") AND ("rc"."resource_type" = 'report'::"text") AND ("rc"."resource_id" = "report_comments"."report_id")))));



ALTER TABLE "public"."report_comments" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."reports" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."resource_collaborators" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."resource_share_links" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."scheduled_runs" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."sql_connections" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."sql_query_history" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."sql_query_sessions" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."subscriptions" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."support_tickets" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."team_invitations" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."team_memberships" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."teams" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."uploaded_files" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."user_settings" ENABLE ROW LEVEL SECURITY;


GRANT USAGE ON SCHEMA "public" TO "postgres";
GRANT USAGE ON SCHEMA "public" TO "anon";
GRANT USAGE ON SCHEMA "public" TO "authenticated";
GRANT USAGE ON SCHEMA "public" TO "service_role";



GRANT ALL ON FUNCTION "public"."app_users_fill_org"() TO "anon";
GRANT ALL ON FUNCTION "public"."app_users_fill_org"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."app_users_fill_org"() TO "service_role";



GRANT ALL ON FUNCTION "public"."auto_assign_billing_admin"() TO "anon";
GRANT ALL ON FUNCTION "public"."auto_assign_billing_admin"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."auto_assign_billing_admin"() TO "service_role";



GRANT ALL ON FUNCTION "public"."auto_join_enterprise_admin"() TO "anon";
GRANT ALL ON FUNCTION "public"."auto_join_enterprise_admin"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."auto_join_enterprise_admin"() TO "service_role";



GRANT ALL ON FUNCTION "public"."can_create_team"("p_user_id" "uuid", "p_account_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."can_create_team"("p_user_id" "uuid", "p_account_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."can_create_team"("p_user_id" "uuid", "p_account_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."can_manage_resource_share"("p_type" "text", "p_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."can_manage_resource_share"("p_type" "text", "p_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."can_manage_resource_share"("p_type" "text", "p_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."check_expiring_licenses"() TO "anon";
GRANT ALL ON FUNCTION "public"."check_expiring_licenses"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."check_expiring_licenses"() TO "service_role";



GRANT ALL ON FUNCTION "public"."check_license_expiry"() TO "anon";
GRANT ALL ON FUNCTION "public"."check_license_expiry"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."check_license_expiry"() TO "service_role";



GRANT ALL ON FUNCTION "public"."check_onboarding_plan"() TO "anon";
GRANT ALL ON FUNCTION "public"."check_onboarding_plan"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."check_onboarding_plan"() TO "service_role";



GRANT ALL ON FUNCTION "public"."current_org_id"() TO "anon";
GRANT ALL ON FUNCTION "public"."current_org_id"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."current_org_id"() TO "service_role";



GRANT ALL ON FUNCTION "public"."current_user_email"() TO "anon";
GRANT ALL ON FUNCTION "public"."current_user_email"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."current_user_email"() TO "service_role";



GRANT ALL ON FUNCTION "public"."encrypt_cloud_credentials"("p_connection_id" "uuid", "p_access_key" "text", "p_secret_key" "text") TO "anon";
GRANT ALL ON FUNCTION "public"."encrypt_cloud_credentials"("p_connection_id" "uuid", "p_access_key" "text", "p_secret_key" "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."encrypt_cloud_credentials"("p_connection_id" "uuid", "p_access_key" "text", "p_secret_key" "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."find_profile_id_by_email"("p_email" "text") TO "anon";
GRANT ALL ON FUNCTION "public"."find_profile_id_by_email"("p_email" "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."find_profile_id_by_email"("p_email" "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."find_user_id_by_email"("p_email" "text") TO "anon";
GRANT ALL ON FUNCTION "public"."find_user_id_by_email"("p_email" "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."find_user_id_by_email"("p_email" "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."generate_license_serial"() TO "anon";
GRANT ALL ON FUNCTION "public"."generate_license_serial"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."generate_license_serial"() TO "service_role";



REVOKE ALL ON FUNCTION "public"."get_my_report_access"("_report_id" "uuid") FROM PUBLIC;
GRANT ALL ON FUNCTION "public"."get_my_report_access"("_report_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_my_report_access"("_report_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."get_team_member_public_info"("p_team_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."get_team_member_public_info"("p_team_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_team_member_public_info"("p_team_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."get_user_customer_account"("p_user_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."get_user_customer_account"("p_user_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_user_customer_account"("p_user_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."get_user_email"("user_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."get_user_email"("user_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_user_email"("user_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."handle_new_auth_user"() TO "anon";
GRANT ALL ON FUNCTION "public"."handle_new_auth_user"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."handle_new_auth_user"() TO "service_role";



GRANT ALL ON FUNCTION "public"."handle_new_user"() TO "anon";
GRANT ALL ON FUNCTION "public"."handle_new_user"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."handle_new_user"() TO "service_role";



GRANT ALL ON FUNCTION "public"."handle_team_invitation_status_change"() TO "anon";
GRANT ALL ON FUNCTION "public"."handle_team_invitation_status_change"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."handle_team_invitation_status_change"() TO "service_role";



GRANT ALL ON FUNCTION "public"."handle_team_member_added"() TO "anon";
GRANT ALL ON FUNCTION "public"."handle_team_member_added"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."handle_team_member_added"() TO "service_role";



GRANT ALL ON FUNCTION "public"."handle_team_member_removed"() TO "anon";
GRANT ALL ON FUNCTION "public"."handle_team_member_removed"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."handle_team_member_removed"() TO "service_role";



GRANT ALL ON FUNCTION "public"."has_any_internal_role"("p_user_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."has_any_internal_role"("p_user_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."has_any_internal_role"("p_user_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."is_billing_admin"("p_user_id" "uuid", "p_account_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."is_billing_admin"("p_user_id" "uuid", "p_account_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."is_billing_admin"("p_user_id" "uuid", "p_account_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."is_enterprise_admin"("p_user_id" "uuid", "p_account_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."is_enterprise_admin"("p_user_id" "uuid", "p_account_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."is_enterprise_admin"("p_user_id" "uuid", "p_account_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."is_internal_staff"("p_user_id" "uuid", "p_required_role" "public"."internal_role") TO "anon";
GRANT ALL ON FUNCTION "public"."is_internal_staff"("p_user_id" "uuid", "p_required_role" "public"."internal_role") TO "authenticated";
GRANT ALL ON FUNCTION "public"."is_internal_staff"("p_user_id" "uuid", "p_required_role" "public"."internal_role") TO "service_role";



GRANT ALL ON FUNCTION "public"."is_license_valid"("p_account_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."is_license_valid"("p_account_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."is_license_valid"("p_account_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."is_professional_account"("p_account_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."is_professional_account"("p_account_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."is_professional_account"("p_account_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."is_resource_collaborator"("_resource_type" "text", "_resource_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."is_resource_collaborator"("_resource_type" "text", "_resource_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."is_resource_collaborator"("_resource_type" "text", "_resource_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."is_team_admin"("p_user_id" "uuid", "p_team_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."is_team_admin"("p_user_id" "uuid", "p_team_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."is_team_admin"("p_user_id" "uuid", "p_team_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."is_team_member"("p_user_id" "uuid", "p_team_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."is_team_member"("p_user_id" "uuid", "p_team_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."is_team_member"("p_user_id" "uuid", "p_team_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."notify_file_shared"() TO "anon";
GRANT ALL ON FUNCTION "public"."notify_file_shared"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."notify_file_shared"() TO "service_role";



GRANT ALL ON FUNCTION "public"."notify_share_status_change"() TO "anon";
GRANT ALL ON FUNCTION "public"."notify_share_status_change"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."notify_share_status_change"() TO "service_role";



GRANT ALL ON FUNCTION "public"."notify_team_invitation"() TO "anon";
GRANT ALL ON FUNCTION "public"."notify_team_invitation"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."notify_team_invitation"() TO "service_role";



GRANT ALL ON FUNCTION "public"."prevent_last_admin_demotion"() TO "anon";
GRANT ALL ON FUNCTION "public"."prevent_last_admin_demotion"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."prevent_last_admin_demotion"() TO "service_role";



GRANT ALL ON FUNCTION "public"."prevent_last_admin_removal"() TO "anon";
GRANT ALL ON FUNCTION "public"."prevent_last_admin_removal"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."prevent_last_admin_removal"() TO "service_role";



GRANT ALL ON FUNCTION "public"."set_updated_at"() TO "anon";
GRANT ALL ON FUNCTION "public"."set_updated_at"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."set_updated_at"() TO "service_role";



GRANT ALL ON FUNCTION "public"."update_report_comments_updated_at"() TO "anon";
GRANT ALL ON FUNCTION "public"."update_report_comments_updated_at"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."update_report_comments_updated_at"() TO "service_role";



GRANT ALL ON FUNCTION "public"."update_updated_at_column"() TO "anon";
GRANT ALL ON FUNCTION "public"."update_updated_at_column"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."update_updated_at_column"() TO "service_role";



GRANT ALL ON FUNCTION "public"."user_owns_account"("p_user_id" "uuid", "p_account_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."user_owns_account"("p_user_id" "uuid", "p_account_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."user_owns_account"("p_user_id" "uuid", "p_account_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."validate_cloud_credentials"("p_connection_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."validate_cloud_credentials"("p_connection_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."validate_cloud_credentials"("p_connection_id" "uuid") TO "service_role";



GRANT ALL ON TABLE "public"."analyses" TO "anon";
GRANT ALL ON TABLE "public"."analyses" TO "authenticated";
GRANT ALL ON TABLE "public"."analyses" TO "service_role";



GRANT ALL ON TABLE "public"."analysis_comments" TO "anon";
GRANT ALL ON TABLE "public"."analysis_comments" TO "authenticated";
GRANT ALL ON TABLE "public"."analysis_comments" TO "service_role";



GRANT ALL ON TABLE "public"."analysis_feedback" TO "anon";
GRANT ALL ON TABLE "public"."analysis_feedback" TO "authenticated";
GRANT ALL ON TABLE "public"."analysis_feedback" TO "service_role";



GRANT ALL ON TABLE "public"."analysis_messages" TO "anon";
GRANT ALL ON TABLE "public"."analysis_messages" TO "authenticated";
GRANT ALL ON TABLE "public"."analysis_messages" TO "service_role";



GRANT ALL ON TABLE "public"."app_users" TO "anon";
GRANT ALL ON TABLE "public"."app_users" TO "authenticated";
GRANT ALL ON TABLE "public"."app_users" TO "service_role";



GRANT ALL ON TABLE "public"."billing_admins" TO "anon";
GRANT ALL ON TABLE "public"."billing_admins" TO "authenticated";
GRANT ALL ON TABLE "public"."billing_admins" TO "service_role";



GRANT ALL ON TABLE "public"."billing_invoices" TO "anon";
GRANT ALL ON TABLE "public"."billing_invoices" TO "authenticated";
GRANT ALL ON TABLE "public"."billing_invoices" TO "service_role";



GRANT ALL ON TABLE "public"."branches" TO "anon";
GRANT ALL ON TABLE "public"."branches" TO "authenticated";
GRANT ALL ON TABLE "public"."branches" TO "service_role";



GRANT ALL ON TABLE "public"."cloud_datasets" TO "anon";
GRANT ALL ON TABLE "public"."cloud_datasets" TO "authenticated";
GRANT ALL ON TABLE "public"."cloud_datasets" TO "service_role";



GRANT ALL ON TABLE "public"."config_roles" TO "anon";
GRANT ALL ON TABLE "public"."config_roles" TO "authenticated";
GRANT ALL ON TABLE "public"."config_roles" TO "service_role";



GRANT ALL ON TABLE "public"."customer_accounts" TO "anon";
GRANT ALL ON TABLE "public"."customer_accounts" TO "authenticated";
GRANT ALL ON TABLE "public"."customer_accounts" TO "service_role";



GRANT ALL ON TABLE "public"."dataset_profiles" TO "anon";
GRANT ALL ON TABLE "public"."dataset_profiles" TO "authenticated";
GRANT ALL ON TABLE "public"."dataset_profiles" TO "service_role";



GRANT ALL ON TABLE "public"."dataset_samples" TO "anon";
GRANT ALL ON TABLE "public"."dataset_samples" TO "authenticated";
GRANT ALL ON TABLE "public"."dataset_samples" TO "service_role";



GRANT ALL ON TABLE "public"."departments" TO "anon";
GRANT ALL ON TABLE "public"."departments" TO "authenticated";
GRANT ALL ON TABLE "public"."departments" TO "service_role";



GRANT ALL ON TABLE "public"."enterprise_licenses" TO "anon";
GRANT ALL ON TABLE "public"."enterprise_licenses" TO "authenticated";
GRANT ALL ON TABLE "public"."enterprise_licenses" TO "service_role";



GRANT ALL ON TABLE "public"."enterprise_settings" TO "anon";
GRANT ALL ON TABLE "public"."enterprise_settings" TO "authenticated";
GRANT ALL ON TABLE "public"."enterprise_settings" TO "service_role";



GRANT ALL ON TABLE "public"."file_shares" TO "anon";
GRANT ALL ON TABLE "public"."file_shares" TO "authenticated";
GRANT ALL ON TABLE "public"."file_shares" TO "service_role";



GRANT ALL ON TABLE "public"."folders" TO "anon";
GRANT ALL ON TABLE "public"."folders" TO "authenticated";
GRANT ALL ON TABLE "public"."folders" TO "service_role";



GRANT ALL ON TABLE "public"."internal_roles" TO "anon";
GRANT ALL ON TABLE "public"."internal_roles" TO "authenticated";
GRANT ALL ON TABLE "public"."internal_roles" TO "service_role";



GRANT ALL ON TABLE "public"."mcp_connections" TO "anon";
GRANT ALL ON TABLE "public"."mcp_connections" TO "authenticated";
GRANT ALL ON TABLE "public"."mcp_connections" TO "service_role";



GRANT ALL ON TABLE "public"."notifications" TO "anon";
GRANT ALL ON TABLE "public"."notifications" TO "authenticated";
GRANT ALL ON TABLE "public"."notifications" TO "service_role";



GRANT ALL ON TABLE "public"."organization_teams" TO "anon";
GRANT ALL ON TABLE "public"."organization_teams" TO "authenticated";
GRANT ALL ON TABLE "public"."organization_teams" TO "service_role";



GRANT ALL ON TABLE "public"."organizations" TO "anon";
GRANT ALL ON TABLE "public"."organizations" TO "authenticated";
GRANT ALL ON TABLE "public"."organizations" TO "service_role";



GRANT ALL ON TABLE "public"."plans" TO "anon";
GRANT ALL ON TABLE "public"."plans" TO "authenticated";
GRANT ALL ON TABLE "public"."plans" TO "service_role";



GRANT ALL ON TABLE "public"."profiles" TO "anon";
GRANT ALL ON TABLE "public"."profiles" TO "authenticated";
GRANT ALL ON TABLE "public"."profiles" TO "service_role";



GRANT ALL ON TABLE "public"."projects" TO "anon";
GRANT ALL ON TABLE "public"."projects" TO "authenticated";
GRANT ALL ON TABLE "public"."projects" TO "service_role";



GRANT ALL ON TABLE "public"."report_comments" TO "anon";
GRANT ALL ON TABLE "public"."report_comments" TO "authenticated";
GRANT ALL ON TABLE "public"."report_comments" TO "service_role";



GRANT ALL ON TABLE "public"."reports" TO "anon";
GRANT ALL ON TABLE "public"."reports" TO "authenticated";
GRANT ALL ON TABLE "public"."reports" TO "service_role";



GRANT ALL ON TABLE "public"."resource_collaborators" TO "anon";
GRANT ALL ON TABLE "public"."resource_collaborators" TO "authenticated";
GRANT ALL ON TABLE "public"."resource_collaborators" TO "service_role";



GRANT ALL ON TABLE "public"."resource_share_links" TO "anon";
GRANT ALL ON TABLE "public"."resource_share_links" TO "authenticated";
GRANT ALL ON TABLE "public"."resource_share_links" TO "service_role";



GRANT ALL ON TABLE "public"."scheduled_runs" TO "anon";
GRANT ALL ON TABLE "public"."scheduled_runs" TO "authenticated";
GRANT ALL ON TABLE "public"."scheduled_runs" TO "service_role";



GRANT ALL ON TABLE "public"."sql_connections" TO "anon";
GRANT ALL ON TABLE "public"."sql_connections" TO "authenticated";
GRANT ALL ON TABLE "public"."sql_connections" TO "service_role";



GRANT ALL ON TABLE "public"."sql_query_history" TO "anon";
GRANT ALL ON TABLE "public"."sql_query_history" TO "authenticated";
GRANT ALL ON TABLE "public"."sql_query_history" TO "service_role";



GRANT ALL ON TABLE "public"."sql_query_sessions" TO "anon";
GRANT ALL ON TABLE "public"."sql_query_sessions" TO "authenticated";
GRANT ALL ON TABLE "public"."sql_query_sessions" TO "service_role";



GRANT ALL ON TABLE "public"."subscriptions" TO "anon";
GRANT ALL ON TABLE "public"."subscriptions" TO "authenticated";
GRANT ALL ON TABLE "public"."subscriptions" TO "service_role";



GRANT ALL ON TABLE "public"."support_tickets" TO "anon";
GRANT ALL ON TABLE "public"."support_tickets" TO "authenticated";
GRANT ALL ON TABLE "public"."support_tickets" TO "service_role";



GRANT ALL ON TABLE "public"."team_invitations" TO "anon";
GRANT ALL ON TABLE "public"."team_invitations" TO "authenticated";
GRANT ALL ON TABLE "public"."team_invitations" TO "service_role";



GRANT ALL ON TABLE "public"."team_memberships" TO "anon";
GRANT ALL ON TABLE "public"."team_memberships" TO "authenticated";
GRANT ALL ON TABLE "public"."team_memberships" TO "service_role";



GRANT ALL ON TABLE "public"."teams" TO "anon";
GRANT ALL ON TABLE "public"."teams" TO "authenticated";
GRANT ALL ON TABLE "public"."teams" TO "service_role";



GRANT ALL ON TABLE "public"."uploaded_files" TO "anon";
GRANT ALL ON TABLE "public"."uploaded_files" TO "authenticated";
GRANT ALL ON TABLE "public"."uploaded_files" TO "service_role";



GRANT ALL ON TABLE "public"."user_settings" TO "anon";
GRANT ALL ON TABLE "public"."user_settings" TO "authenticated";
GRANT ALL ON TABLE "public"."user_settings" TO "service_role";



ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "service_role";