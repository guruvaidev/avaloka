"""Per-user Ray clusters: naming, states, start/touch, the idle reaper, and
job routing in ray_job_runner. Kubernetes and the Ray dashboard are faked."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import pytest

from app.infra import ray_job_runner
from app.infra import user_ray_clusters as urc

NOW = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)


class ApiError(Exception):
    def __init__(self, status):
        super().__init__(status)
        self.status = status


class FakeApi:
    """Just enough of kubernetes.client.CustomObjectsApi."""

    def __init__(self):
        self.objs = {}
        self.deleted = []

    def get_namespaced_custom_object(self, group, version, namespace, plural, name):
        if name not in self.objs:
            raise ApiError(404)
        return copy.deepcopy(self.objs[name])

    def create_namespaced_custom_object(self, group, version, namespace, plural, body):
        name = body["metadata"]["name"]
        if name in self.objs:
            raise ApiError(409)
        self.objs[name] = copy.deepcopy(body)
        self.objs[name]["metadata"]["uid"] = f"uid-{name}"

    def patch_namespaced_custom_object(self, group, version, namespace, plural, name, body):
        if name not in self.objs:
            raise ApiError(404)
        anns = self.objs[name]["metadata"].setdefault("annotations", {})
        anns.update(body["metadata"]["annotations"])

    def delete_namespaced_custom_object(self, group, version, namespace, plural, name, body):
        if name not in self.objs:
            raise ApiError(404)
        self.deleted.append(name)
        del self.objs[name]

    def list_namespaced_custom_object(self, group, version, namespace, plural, label_selector):
        key, value = label_selector.split("=")
        return {"items": [copy.deepcopy(o) for o in self.objs.values()
                          if o["metadata"].get("labels", {}).get(key) == value]}

    # helpers
    def ready(self, name):
        self.objs[name]["status"] = {"state": "ready"}

    def last_active(self, name, when):
        self.objs[name]["metadata"]["annotations"][urc.ANN_LAST_ACTIVE] = urc._iso(when)


class FakeNetApi:
    """Just enough of kubernetes.client.NetworkingV1Api."""

    def __init__(self):
        self.policies = {}
        self.fail_with = None

    def create_namespaced_network_policy(self, namespace, body):
        if self.fail_with:
            raise ApiError(self.fail_with)
        name = body["metadata"]["name"]
        if name in self.policies:
            raise ApiError(409)
        self.policies[name] = copy.deepcopy(body)


@pytest.fixture
def net(monkeypatch):
    fake = FakeNetApi()
    monkeypatch.setattr(urc, "_NET_API", fake)
    return fake


@pytest.fixture
def api(monkeypatch, net):
    fake = FakeApi()
    monkeypatch.setattr(urc, "_API", fake)
    monkeypatch.delenv("RAY_USER_CLUSTER_NETWORK_POLICY", raising=False)
    monkeypatch.setenv("RAY_PER_USER_CLUSTERS", "1")
    monkeypatch.setenv("RAY_PER_USER_CLUSTER_PLANS", "*")
    monkeypatch.delenv("RAY_USER_CLUSTER_NAMESPACE", raising=False)
    monkeypatch.delenv("RAY_USER_CLUSTER_IDLE_MINUTES", raising=False)
    return fake


# ------------------------------------------------------------------ naming

def test_name_is_stable_dns_safe_and_hides_the_user_id():
    a = urc.cluster_name("2f6b0c1e-aaaa-bbbb-cccc-1234567890ab")
    assert a == urc.cluster_name("2f6b0c1e-aaaa-bbbb-cccc-1234567890ab")
    assert a != urc.cluster_name("someone-else")
    assert a.startswith("ray-u-") and len(a) == 18
    assert "2f6b0c1e" not in a


def test_upstream_is_the_in_cluster_head_service(api):
    name = urc.cluster_name("u1")
    assert urc.upstream_for(name) == f"{name}-head-svc.ray-users.svc.cluster.local:8265"


# ------------------------------------------------------------- who applies

def test_switch_off_means_nobody(monkeypatch):
    monkeypatch.delenv("RAY_PER_USER_CLUSTERS", raising=False)
    assert urc.applies_to("u1") is False


def test_plan_list_decides(monkeypatch):
    import sys, types
    monkeypatch.setenv("RAY_PER_USER_CLUSTERS", "1")
    monkeypatch.setenv("RAY_PER_USER_CLUSTER_PLANS", "professional")
    plans = {"pro": "professional", "free": "free", "ent": "enterprise"}
    fake = types.SimpleNamespace(try_resolve_plan=lambda uid: plans.get(uid))
    monkeypatch.setitem(sys.modules, "app.core.plan_limits", fake)
    import app.core
    monkeypatch.setattr(app.core, "plan_limits", fake, raising=False)
    assert urc.applies_to("pro") is True
    assert urc.applies_to("free") is False
    assert urc.applies_to("ent") is False
    assert urc.applies_to("unknown") is False  # plan unknown -> shared cluster
    assert urc.applies_to(None) is False


# ---------------------------------------------------------------- manifest

def test_manifest_is_clusterip_small_and_labelled(api, monkeypatch):
    monkeypatch.setenv("RAY_USER_CLUSTER_IMAGE", "us-docker.pkg.dev/x/ray:2.9.3-py310")
    body = urc.build_manifest("u1", now=NOW)
    spec = body["spec"]
    assert spec["headGroupSpec"]["serviceType"] == "ClusterIP"
    assert spec["rayVersion"] == "2.9.3"
    workers = spec["workerGroupSpecs"][0]
    assert (workers["minReplicas"], workers["maxReplicas"]) == (0, 2)  # 1 to 3 nodes
    meta = body["metadata"]
    assert meta["namespace"] == "ray-users"
    assert meta["labels"][urc.LABEL_MANAGED] == "true"
    assert meta["annotations"][urc.ANN_USER_ID] == "u1"
    assert meta["annotations"][urc.ANN_LAST_ACTIVE] == "2026-11-09T12:00:00Z"


# ------------------------------------------------------- states, start, touch

def test_states(api):
    name = urc.cluster_name("u1")
    assert urc.describe("u1")["state"] == "stopped"
    assert urc.start("u1")["state"] == "starting"
    api.ready(name)
    assert urc.describe("u1")["state"] == "running"
    api.objs[name]["metadata"]["deletionTimestamp"] = "2026-11-09T12:00:00Z"
    assert urc.describe("u1")["state"] == "stopping"


def test_start_twice_is_fine_and_counts_as_activity(api):
    name = urc.cluster_name("u1")
    urc.start("u1")
    old = datetime(2020, 1, 1, tzinfo=timezone.utc)
    api.last_active(name, old)
    urc.start("u1")  # 409 -> touch
    last = urc._parse(api.objs[name]["metadata"]["annotations"][urc.ANN_LAST_ACTIVE])
    assert last > old


def test_touch_missing_cluster_is_false(api):
    assert urc.touch("nobody") is False


# ----------------------------------------------------------- ensure_running

def test_ensure_running_starts_and_waits(api, monkeypatch):
    name = urc.cluster_name("u1")
    polls = {"n": 0}

    def fake_sleep(_s):
        polls["n"] += 1
        if polls["n"] == 2:
            api.ready(name)

    monkeypatch.setattr(urc, "_sleep", fake_sleep)
    monkeypatch.setattr(urc, "_dashboard_answers", lambda url: True)
    url = urc.ensure_running("u1", timeout_s=1000, poll_s=1)
    assert url == f"http://{name}-head-svc.ray-users.svc.cluster.local:8265"
    assert polls["n"] == 2


def test_ensure_running_gives_up_after_timeout(api, monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(urc, "_monotonic", lambda: clock["t"])
    monkeypatch.setattr(urc, "_sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    assert urc.ensure_running("u1", timeout_s=30, poll_s=10) is None


# -------------------------------------------------------------------- reaper

def _cluster(api, user, idle, ready=True):
    urc.start(user)
    name = urc.cluster_name(user)
    if ready:
        api.ready(name)
    api.last_active(name, NOW - idle)
    return name


def test_reaper_deletes_idle_keeps_recent(api, monkeypatch):
    monkeypatch.setattr(urc, "_active_jobs", lambda url: 0)
    old = _cluster(api, "old", timedelta(minutes=61))
    recent = _cluster(api, "recent", timedelta(minutes=10))
    out = urc.reap_idle(now=NOW)
    assert out["deleted"] == [old]
    assert recent in api.objs and old not in api.objs


def test_reaper_keeps_a_cluster_with_a_running_job_and_marks_it_active(api, monkeypatch):
    monkeypatch.setattr(urc, "_active_jobs", lambda url: 1)
    name = _cluster(api, "busy", timedelta(hours=5))
    out = urc.reap_idle(now=NOW)
    assert out["busy"] == [name] and name in api.objs
    assert api.objs[name]["metadata"]["annotations"][urc.ANN_LAST_ACTIVE] == urc._iso(NOW)


def test_reaper_unreachable_dashboard_waits_until_twice_the_limit(api, monkeypatch):
    monkeypatch.setattr(urc, "_active_jobs", lambda url: None)
    a = _cluster(api, "a", timedelta(minutes=90))
    b = _cluster(api, "b", timedelta(minutes=121))
    out = urc.reap_idle(now=NOW)
    assert out["unreachable"] == [a]
    assert out["deleted"] == [b]


def test_reaper_deletes_a_cluster_that_never_became_ready(api, monkeypatch):
    monkeypatch.setattr(urc, "_active_jobs", lambda url: pytest.fail("not asked"))
    name = _cluster(api, "stuck", timedelta(minutes=61), ready=False)
    assert urc.reap_idle(now=NOW)["deleted"] == [name]


def test_idle_limit_is_a_setting(api, monkeypatch):
    monkeypatch.setenv("RAY_USER_CLUSTER_IDLE_MINUTES", "15")
    monkeypatch.setattr(urc, "_active_jobs", lambda url: 0)
    name = _cluster(api, "u", timedelta(minutes=16))
    assert urc.reap_idle(now=NOW)["deleted"] == [name]


def test_reaper_does_nothing_when_switched_off(api, monkeypatch):
    monkeypatch.delenv("RAY_PER_USER_CLUSTERS")
    assert urc.reap_idle(now=NOW)["enabled"] is False


# ------------------------------------------------------------ job routing

def test_runner_uses_the_users_cluster(monkeypatch):
    monkeypatch.setattr(urc, "applies_to", lambda uid: True)
    monkeypatch.setattr(urc, "ensure_running", lambda uid: "http://mine:8265")
    assert ray_job_runner._per_user_dashboard_url({"user_id": "u1"}) == "http://mine:8265"


@pytest.mark.parametrize("tags, applies, ensure", [
    ({}, True, lambda uid: "http://mine:8265"),                          # no user on the job
    ({"user_id": "u1"}, False, lambda uid: "http://mine:8265"),          # plan / switch off
    ({"user_id": "u1"}, True, lambda uid: None),                         # not ready in time
    ({"user_id": "u1"}, True, lambda uid: (_ for _ in ()).throw(RuntimeError("k8s"))),
])
def test_runner_falls_back_to_the_shared_cluster(monkeypatch, tags, applies, ensure):
    monkeypatch.setattr(urc, "applies_to", lambda uid: applies)
    monkeypatch.setattr(urc, "ensure_running", ensure)
    assert ray_job_runner._per_user_dashboard_url(tags) is None


def test_explicit_dashboard_url_still_wins(monkeypatch):
    seen = {}
    monkeypatch.setattr(ray_job_runner, "_per_user_dashboard_url",
                        lambda tags: pytest.fail("must not be asked"))
    monkeypatch.setattr(ray_job_runner, "_extract_script_from_yaml", lambda y: "print(1)")
    monkeypatch.setattr(ray_job_runner, "_extract_runtime_env_from_yaml", lambda y: {})

    def fake_direct(**kw):
        seen["url"] = kw["dashboard_url"]
        return "ok"

    monkeypatch.setattr(ray_job_runner, "_run_ray_job_direct", fake_direct)
    ray_job_runner.run_rayjob_from_yaml("x", rayjob_name="j1", dashboard_url="http://explicit:8265",
                                        job_tags={"user_id": "u1"})
    assert seen["url"] == "http://explicit:8265"


def test_users_cluster_beats_the_shared_default(monkeypatch):
    seen = {}
    monkeypatch.setenv("RAY_DASHBOARD_URL", "http://shared:8265")
    monkeypatch.setattr(ray_job_runner, "_per_user_dashboard_url", lambda tags: "http://mine:8265")
    monkeypatch.setattr(ray_job_runner, "_extract_script_from_yaml", lambda y: "print(1)")
    monkeypatch.setattr(ray_job_runner, "_extract_runtime_env_from_yaml", lambda y: {})
    monkeypatch.setattr(ray_job_runner, "_run_ray_job_direct",
                        lambda **kw: seen.setdefault("url", kw["dashboard_url"]))
    ray_job_runner.run_rayjob_from_yaml("x", rayjob_name="j1", job_tags={"user_id": "u1"})
    assert seen["url"] == "http://mine:8265"


def test_local_image_pull_policy_and_version(api, monkeypatch):
    monkeypatch.setenv("RAY_USER_CLUSTER_IMAGE", "avaloka-ray:latest")
    monkeypatch.setenv("RAY_USER_CLUSTER_RAY_VERSION", "2.49.2")
    monkeypatch.setenv("RAY_USER_CLUSTER_IMAGE_PULL_POLICY", "IfNotPresent")
    spec = urc.build_manifest("u1", now=NOW)["spec"]
    assert spec["rayVersion"] == "2.49.2"
    head = spec["headGroupSpec"]["template"]["spec"]["containers"][0]
    worker = spec["workerGroupSpecs"][0]["template"]["spec"]["containers"][0]
    assert head["imagePullPolicy"] == worker["imagePullPolicy"] == "IfNotPresent"


@pytest.mark.parametrize("k8s, ray", [("500m", "1"), ("1500m", "2"), ("2", "2"), ("0.25", "1"), ("bad", "1")])
def test_head_num_cpus_is_a_whole_number(api, monkeypatch, k8s, ray):
    monkeypatch.setenv("RAY_USER_CLUSTER_HEAD_CPU", k8s)
    head = urc.build_manifest("u1", now=NOW)["spec"]["headGroupSpec"]
    assert head["rayStartParams"]["num-cpus"] == ray



# ------------------------------------------------------- network isolation

def test_start_gives_the_cluster_its_own_network_policy(api, net):
    urc.start("u1")
    name = urc.cluster_name("u1")
    pol = net.policies[f"{name}-same-cluster"]
    h = urc.user_hash("u1")
    # Owned by the RayCluster, so it is deleted together with it.
    owner = pol["metadata"]["ownerReferences"][0]
    assert (owner["kind"], owner["name"], owner["uid"]) == ("RayCluster", name, f"uid-{name}")
    # Only this cluster's pods, and only from this cluster's pods.
    assert pol["spec"]["podSelector"] == {"matchLabels": {urc.LABEL_USER_HASH: h}}
    assert pol["spec"]["ingress"] == [{"from": [{"podSelector": {"matchLabels": {urc.LABEL_USER_HASH: h}}}]}]


def test_two_users_policies_do_not_admit_each_other(api, net):
    urc.start("alice")
    urc.start("bob")
    a = net.policies[f"{urc.cluster_name('alice')}-same-cluster"]["spec"]
    b = net.policies[f"{urc.cluster_name('bob')}-same-cluster"]["spec"]
    assert a["podSelector"] != b["podSelector"]
    assert a["ingress"][0]["from"][0]["podSelector"] == a["podSelector"]
    assert b["ingress"][0]["from"][0]["podSelector"] == b["podSelector"]


def test_pods_carry_the_label_the_policy_selects(api):
    body = urc.build_manifest("u1", now=NOW)
    h = urc.user_hash("u1")
    head = body["spec"]["headGroupSpec"]["template"]["metadata"]["labels"]
    worker = body["spec"]["workerGroupSpecs"][0]["template"]["metadata"]["labels"]
    assert head[urc.LABEL_USER_HASH] == worker[urc.LABEL_USER_HASH] == h


def test_existing_cluster_gets_the_policy_on_next_start(api, net):
    urc.start("u1")
    net.policies.clear()          # a cluster made before policies existed
    urc.start("u1")               # 409 path
    assert f"{urc.cluster_name('u1')}-same-cluster" in net.policies


def test_policy_failure_is_raised(api, net):
    net.fail_with = 403           # e.g. the Role lacks networkpolicies
    with pytest.raises(ApiError):
        urc.start("u1")


def test_policy_can_be_switched_off(api, net, monkeypatch):
    monkeypatch.setenv("RAY_USER_CLUSTER_NETWORK_POLICY", "0")
    urc.start("u1")
    assert net.policies == {}


# --------------------------------------------- 15-minute shutdown (asked for)

@pytest.mark.parametrize("idle_min, running_jobs, deleted", [
    (14, 0, False),   # used 14 min ago: kept
    (15, 0, True),    # exactly 15 min unused: deleted
    (16, 0, True),    # 16 min unused: deleted
    (40, 1, False),   # a job still running: kept, however long
])
def test_cluster_shuts_down_after_15_minutes_unused(api, monkeypatch, idle_min, running_jobs, deleted):
    monkeypatch.setenv("RAY_USER_CLUSTER_IDLE_MINUTES", "15")
    monkeypatch.setattr(urc, "_active_jobs", lambda url: running_jobs)
    name = _cluster(api, "u15", timedelta(minutes=idle_min))
    out = urc.reap_idle(now=NOW)
    assert (name in out["deleted"]) is deleted
    assert (name in api.objs) is (not deleted)


def test_dashboard_heartbeat_resets_the_15_minute_clock(api, monkeypatch):
    monkeypatch.setenv("RAY_USER_CLUSTER_IDLE_MINUTES", "15")
    monkeypatch.setattr(urc, "_active_jobs", lambda url: 0)
    name = _cluster(api, "u15", timedelta(minutes=30))
    urc._touch_name(name, NOW - timedelta(minutes=1))   # heartbeat a minute ago
    assert urc.reap_idle(now=NOW)["deleted"] == []


def test_ray_analysis_jobs_are_tagged_with_the_user():
    """Without user_id the job silently goes to the shared cluster."""
    from pathlib import Path
    path = Path("app/agents/execution_agent.py")
    if not path.exists():
        pytest.skip("execution_agent.py not in this checkout")
    src = path.read_text()
    call = src[src.index("outcome = run_rayjob_from_yaml("):]
    call = call[: call.index("\n        )")]
    assert "job_tags=" in call
    assert '"user_id": state.get("user_id")' in call
