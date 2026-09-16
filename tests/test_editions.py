"""Capability resolution across the deployment x edition matrix."""

from __future__ import annotations

import pytest

from app.core.editions import (EDITIONS, GATE_REASON, Capabilities, GateReason,
                               explain, normalize_deployment,
                               normalize_edition, resolve_capabilities)


def caps(deployment, edition, *, commercial=True):
    return resolve_capabilities(deployment, edition, has_commercial_build=commercial)


# --------------------------------------------------------------------------- #
# The correction: same capability, opposite answer, explained by deployment
# --------------------------------------------------------------------------- #

def test_self_hosted_has_a_scheduler_and_free_hosted_does_not():
    """The case the single-axis tier model could not represent."""
    assert caps("self_hosted", "oss").scheduler is True
    assert caps("self_hosted", "oss").batch_jobs is True
    assert caps("avaloka_hosted", "free").scheduler is False
    assert caps("avaloka_hosted", "free").batch_jobs is False


def test_cloud_connectivity_is_the_professional_line():
    free = caps("avaloka_hosted", "free")
    pro = caps("avaloka_hosted", "professional")
    for capability in ("database_connectors", "cloud_connectors", "cloud_provisioning"):
        assert getattr(free, capability) is False
        assert getattr(pro, capability) is True, capability

def test_swarm_intelligence_is_commercial_and_absent_from_open_source():
    """Swarm ships on the 1.7 line but never in the open-source distribution.

    develop-1.6 contains no swarm code at all, so an open-source build cannot
    resolve this capability whatever edition the environment claims.
    """
    assert GATE_REASON["swarm_intelligence"] is GateReason.COMMERCIAL
    oss = resolve_capabilities("self_hosted", "oss", has_commercial_build=False)
    assert oss.swarm_intelligence is False
    assert caps("avaloka_hosted", "free").swarm_intelligence is False
    assert caps("avaloka_hosted", "professional").swarm_intelligence is True
    assert caps("avaloka_hosted", "enterprise").swarm_intelligence is True


def test_an_oss_build_cannot_claim_swarm_by_setting_an_edition():
    """Capability follows installed code, never configuration."""
    forged = resolve_capabilities("avaloka_hosted", "enterprise",
                                  has_commercial_build=False)
    assert forged.swarm_intelligence is False


def test_swarm_intelligence_is_commercial_and_absent_from_open_source():
    """Swarm ships on the 1.7 line but never in the open-source distribution.

    develop-1.6 contains no swarm code at all, so an open-source build cannot
    resolve this capability whatever edition the environment claims.
    """
    assert GATE_REASON["swarm_intelligence"] is GateReason.COMMERCIAL
    oss = resolve_capabilities("self_hosted", "oss", has_commercial_build=False)
    assert oss.swarm_intelligence is False
    assert caps("avaloka_hosted", "free").swarm_intelligence is False
    assert caps("avaloka_hosted", "professional").swarm_intelligence is True
    assert caps("avaloka_hosted", "enterprise").swarm_intelligence is True


def test_an_oss_build_cannot_claim_swarm_by_setting_an_edition():
    """Capability follows installed code, never configuration."""
    forged = resolve_capabilities("avaloka_hosted", "enterprise",
                                  has_commercial_build=False)
    assert forged.swarm_intelligence is False


def test_collaboration_is_the_enterprise_line():
    pro = caps("avaloka_hosted", "professional")
    ent = caps("avaloka_hosted", "enterprise")
    for capability in ("team_collaboration", "share_analysis",
                       "comment_on_analysis", "notifications", "sso_audit"):
        assert getattr(pro, capability) is False, capability
        assert getattr(ent, capability) is True, capability


# --------------------------------------------------------------------------- #
# The two invariants from the module docstring
# --------------------------------------------------------------------------- #

def test_invariant_self_hosted_oss_never_locks_a_commercial_capability():
    """Invariant 1: an OSS build ships no commercial code, so it never nags.

    Every capability an OSS self-hosted build reports as disabled must be
    disabled because the code is absent -- never because someone should pay.
    The user-facing explanation must therefore never solicit an upgrade.
    """
    resolved = resolve_capabilities("self_hosted", "oss", has_commercial_build=False)
    for capability in resolved.disabled():
        assert GATE_REASON[capability] is GateReason.COMMERCIAL, (
            f"{capability} is withheld from a self-hosted OSS build for a "
            f"{GATE_REASON[capability].value} reason, which protects nobody"
        )


def test_invariant_cost_gates_never_apply_to_self_hosted():
    """Invariant 2: the operator pays their own bill, so nothing is capped."""
    resolved = resolve_capabilities("self_hosted", "oss", has_commercial_build=False)
    cost_gated = [name for name, reason in GATE_REASON.items()
                  if reason is GateReason.COST]
    for capability in cost_gated:
        assert getattr(resolved, capability) is True, (
            f"{capability} is a COST gate but was denied to a self-hosted "
            "deployment that pays for its own compute"
        )


def test_open_capabilities_are_granted_in_every_edition():
    open_caps = [n for n, r in GATE_REASON.items() if r is GateReason.OPEN]
    for edition in EDITIONS:
        resolved = caps("avaloka_hosted", edition)
        for capability in open_caps:
            assert getattr(resolved, capability) is True, (edition, capability)


def test_generated_code_export_is_open_even_on_free():
    """It is the developer's own work product, taken at their own risk."""
    assert caps("avaloka_hosted", "free").export_generated_code is True
    assert resolve_capabilities(
        "self_hosted", "oss", has_commercial_build=False
    ).export_generated_code is True


# --------------------------------------------------------------------------- #
# An OSS build cannot grant what it does not ship
# --------------------------------------------------------------------------- #

def test_oss_build_cannot_grant_commercial_capabilities_even_as_enterprise():
    """Flipping the edition in an OSS checkout must grant nothing commercial.

    This is the honest answer to "can't a self-hoster just edit the flags?" --
    they can, and it buys them nothing, because the code is not on disk.
    """
    forged = resolve_capabilities(
        "avaloka_hosted", "enterprise", has_commercial_build=False
    )
    commercial = [n for n, r in GATE_REASON.items() if r is GateReason.COMMERCIAL]
    for capability in commercial:
        assert getattr(forged, capability) is False, capability


def test_commercial_build_does_grant_them():
    real = resolve_capabilities(
        "avaloka_hosted", "enterprise", has_commercial_build=True
    )
    assert real.team_collaboration is True
    assert real.cloud_provisioning is True


# --------------------------------------------------------------------------- #
# Normalization fails closed
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", [None, "", "  ", "nonsense", "ENTERPRISE_"])
def test_unknown_edition_falls_back_to_oss(value):
    assert normalize_edition(value) == "oss"


@pytest.mark.parametrize("value", [None, "", "cloud", "on-prem"])
def test_unknown_deployment_falls_back_to_self_hosted(value):
    assert normalize_deployment(value) == "self_hosted"


@pytest.mark.parametrize("value,expected", [
    ("Enterprise", "enterprise"), ("  professional ", "professional"), ("FREE", "free"),
])
def test_edition_normalization_is_case_and_space_insensitive(value, expected):
    assert normalize_edition(value) == expected


def test_deployment_normalization_accepts_hyphens():
    assert normalize_deployment("avaloka-hosted") == "avaloka_hosted"


# --------------------------------------------------------------------------- #
# Explanations
# --------------------------------------------------------------------------- #

def test_explanation_never_solicits_upgrade_for_a_cost_gate_on_self_hosted():
    resolved = resolve_capabilities("self_hosted", "oss", has_commercial_build=False)
    assert "available" in explain("scheduler", resolved)


def test_explanation_for_a_commercial_capability_names_the_paid_editions():
    resolved = caps("avaloka_hosted", "free")
    message = explain("team_collaboration", resolved)
    assert "Professional" in message and "Enterprise" in message


def test_explain_rejects_unknown_capability():
    with pytest.raises(KeyError):
        explain("can_haz_cheeseburger", Capabilities())


def test_every_capability_field_has_a_declared_gate_reason():
    """A new capability must state why it is gated before it can ship."""
    from dataclasses import fields
    declared = {f.name for f in fields(Capabilities)}
    assert declared == set(GATE_REASON), (
        f"missing gate reasons: {declared - set(GATE_REASON)}; "
        f"stale entries: {set(GATE_REASON) - declared}"
    )
