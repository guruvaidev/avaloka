// Auth listUsers supports pagination, not an email filter.
export async function findTeamInvitee(admin: any, email: string) {
  const perPage = 1000;
  for (let page = 1; ; page += 1) {
    const { data, error } = await admin.auth.admin.listUsers({ page, perPage });
    if (error) throw new Error(error.message);
    const users = data?.users ?? [];
    const user = users.find((user: any) => String(user.email ?? "").toLowerCase() === email);
    if (user) return user;
    if (users.length < perPage) return undefined;
  }
}
