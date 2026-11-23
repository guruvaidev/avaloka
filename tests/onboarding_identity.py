"""Who a request to the onboarding API says it is, for fakes of that API. Not a test module.

/customers/{id}/credentials answers only service-class callers (merge of
develop-1.7 into integration/2.0, decision 1): a signed avu1 user assertion
for a customer that user owns, or the service token. A customer's api_key is
403. Fakes of the route use these helpers to answer the way the real one
does, so a test cannot pass by holding a key the route would refuse.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def assertion_user(headers: Optional[Dict[str, Any]]) -> Optional[str]:
    """The user a valid, unexpired avu1 assertion in ``headers`` names; ``None`` otherwise."""
    from app.mcp_server.credentials import verify_user_assertion

    auth = str((headers or {}).get("Authorization") or "")
    if not auth.lower().startswith("bearer "):
        return None
    return verify_user_assertion(auth[len("bearer "):].strip())


def credentials_status(headers: Optional[Dict[str, Any]], owner: Optional[str]) -> int:
    """The status the real /credentials route gives ``headers`` for a customer ``owner`` owns."""
    auth = str((headers or {}).get("Authorization") or "")
    user = assertion_user(headers)
    if user is None:
        return 403 if auth.lower().startswith("bearer ") else 401
    return 200 if owner and user == owner else 404
