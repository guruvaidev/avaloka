"""Resolve analysis and inherited project grants using the requester's profile identity."""


def standalone_analysis_access(client, analysis_id: str, profile_id: str, requester_plan: str = "professional") -> int:
    """Return 0/1/2/3 for no access/view/comment/edit, including root grants."""
    # Enterprise continues through the existing organization collaborator path.
    if requester_plan == "enterprise":
        return 0
    rows = (
        client.table("analyses").select("id,parent_analysis_id,project_id")
        .eq("id", analysis_id).limit(1).execute().data or []
    )
    if not rows:
        return 0
    root_id = rows[0].get("parent_analysis_id") or analysis_id
    project_id = rows[0].get("project_id")
    if project_id:
        projects = client.table("projects").select("owner_id").eq("id", project_id).limit(1).execute().data or []
        if projects and projects[0].get("owner_id") == profile_id:
            return 3
    shares = (
        client.table("project_shares" if project_id else "analysis_shares").select("recipient_profile_id,team_id,access_level")
        .in_("project_id" if project_id else "analysis_id", [project_id] if project_id else list({analysis_id, root_id})).execute().data or []
    )
    if not shares:
        return 0
    memberships = (
        client.table("team_memberships").select("team_id")
        .eq("user_id", profile_id).execute().data or []
    )
    teams = (
        client.table("teams").select("id")
        .or_(f"owner_profile_id.eq.{profile_id},lead_profile_id.eq.{profile_id}")
        .execute().data or []
    )
    team_ids = {str(row["team_id"]) for row in memberships} | {str(row["id"]) for row in teams}
    level = max((
        {"view": 1, "comment": 2, "edit": 3}.get(share.get("access_level"), 0)
        for share in shares
        if share.get("recipient_profile_id") == profile_id
        or (share.get("team_id") and str(share["team_id"]) in team_ids)
    ), default=0)

    return min(level, 1) if requester_plan != "professional" else level
