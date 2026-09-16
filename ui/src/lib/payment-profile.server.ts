export type ResolvedPaymentProfile = {
  ownerProfileId: string;
  organizationId: string | null;
  profile: Record<string, any>;
};

export async function resolvePaymentProfile(
  supabase: any,
  userId: string,
  claims?: Record<string, any> | null,
): Promise<ResolvedPaymentProfile> {
  // Optional columns are attempted first, then progressively dropped so a
  // schema without first_name/last_name/full_name/company_email still works.
  const columnSets = [
    "id, user_id, organization_id, first_name, last_name, full_name, company_email",
    "id, user_id, organization_id, full_name, company_email",
    "id, user_id, organization_id",
    "id, organization_id",
    "id",
  ];

  let existing: Record<string, any> | null = null;
  let lastError: any = null;
  let usableColumns = columnSets[columnSets.length - 1];

  for (const cols of columnSets) {
    const { data, error } = await supabase
      .from("profiles")
      .select(cols)
      .or(`id.eq.${userId},user_id.eq.${userId}`)
      .maybeSingle();

    if (!error) {
      existing = (data as Record<string, any> | null) ?? null;
      usableColumns = cols;
      lastError = null;
      break;
    }

    lastError = error;
    // Retry with fewer columns only when the failure is a missing column.
    const msg = String(error?.message ?? "");
    if (!/does not exist|column/i.test(msg)) break;
  }

  if (!existing && lastError) {
    // `user_id` itself may not exist; final fallback matches on id only.
    const fallback = await supabase
      .from("profiles")
      .select("id")
      .eq("id", userId)
      .maybeSingle();
    if (fallback.error) {
      throw new Error(`Profile lookup failed: ${lastError.message}`);
    }
    existing = (fallback.data as Record<string, any> | null) ?? null;
    usableColumns = "id";
  }

  if (existing?.id) {
    return {
      ownerProfileId: existing.id as string,
      organizationId: (existing.organization_id ?? null) as string | null,
      profile: existing,
    };
  }


  const fullName =
    (claims?.full_name as string | undefined) ??
    (claims?.name as string | undefined) ??
    (claims?.user_metadata?.full_name as string | undefined) ??
    (claims?.user_metadata?.name as string | undefined) ??
    null;
  const companyEmail =
    (claims?.email as string | undefined) ??
    (claims?.user_metadata?.email as string | undefined) ??
    null;

  const payload: Record<string, any> = { id: userId };
  if (usableColumns.includes("user_id")) payload.user_id = userId;
  if (fullName && usableColumns.includes("full_name")) payload.full_name = fullName;
  if (companyEmail && usableColumns.includes("company_email"))
    payload.company_email = companyEmail;

  let { data: created, error: createError } = await supabase
    .from("profiles")
    .insert(payload)
    .select(usableColumns)
    .single();

  if (createError) {
    // Retry with the bare minimum if optional columns are rejected.
    const retry = await supabase
      .from("profiles")
      .insert({ id: userId })
      .select("id")
      .single();
    created = retry.data;
    createError = retry.error;
  }

  if (createError || !created?.id) {
    throw new Error(
      `Failed to create profile for current user: ${createError?.message ?? "unknown error"}`,
    );
  }


  return {
    ownerProfileId: created.id as string,
    organizationId: (created.organization_id ?? null) as string | null,
    profile: created,
  };
}