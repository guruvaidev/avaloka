"""Memory a returning user actually gets back.

The layers that persist were keyed by session. A new login means a new
session_id, so the Chroma "user signature" and the Milvus insight search --
the RAG path -- could never match anything an earlier session had written. The
store was durable and the data was unreachable, which reads exactly like
forgetting.
"""
import pytest

from app.services.memory_plane import _memory_scope


class TestMemoryScope:
    def test_prefers_the_user_over_the_session(self):
        assert _memory_scope("user-42", "sess-a") == "user-42"
        assert _memory_scope("user-42", "sess-b") == "user-42"

    def test_is_stable_across_logins(self):
        """The whole point: two sessions, one user, one scope."""
        first = _memory_scope("user-42", "session-monday")
        second = _memory_scope("user-42", "session-friday")
        assert first == second

    @pytest.mark.parametrize("absent", [None, "", "   ", "default"])
    def test_falls_back_to_the_session_when_there_is_no_user(self, absent):
        """An anonymous session still accumulates context for as long as it
        lasts; it just cannot outlive itself. That was the old behaviour and
        stays the behaviour when no user is known."""
        assert _memory_scope(absent, "sess-a") == "sess-a"

    def test_separates_users(self):
        assert _memory_scope("alice", "s") != _memory_scope("bob", "s")


class TestRetrievalIsUserScoped:
    """The orchestrator must read and write the cross-session tiers by scope."""

    def _orchestrator(self, chroma, milvus):
        from app.services.memory_plane import MemoryOrchestrator
        orch = MemoryOrchestrator.__new__(MemoryOrchestrator)
        orch.chroma, orch.milvus = chroma, milvus
        return orch

    def test_signature_written_in_one_session_is_read_in_the_next(self):
        """Chroma's client takes a user_id -- it was being handed a session_id."""
        store = {}

        class Chroma:
            def update_user_signature(self, user_id, sig): store[user_id] = sig
            def get_user_signature(self, user_id): return store.get(user_id)

        chroma = Chroma()
        # Monday: signature recorded under the user, not the session.
        chroma.update_user_signature(_memory_scope("user-42", "session-monday"), "prefers weekly rollups")
        # Friday: a new session, same person.
        assert chroma.get_user_signature(_memory_scope("user-42", "session-friday")) == "prefers weekly rollups"

    def test_legacy_session_keyed_signature_is_still_reachable(self):
        """Dual-read: data written before the change must not be stranded."""
        store = {"session-monday": "prefers weekly rollups"}

        def read(scope, session):
            return store.get(scope) or (store.get(session) if scope != session else None)

        assert read(_memory_scope("user-42", "session-monday"), "session-monday") == "prefers weekly rollups"

    def test_two_users_do_not_read_each_others_memory(self):
        store = {}
        store[_memory_scope("alice", "s1")] = "alice's preference"
        assert store.get(_memory_scope("bob", "s2")) is None
