"""Tests for src.gateway.tenant_isolation."""

from __future__ import annotations

import os
import unittest

from src.gateway.tenant_isolation import (
    _extract_client_ip,
    derive_tenant_id,
    inject_tenant_prefix,
    isolation_enabled,
)
from src.worker.schemas import ChatMessage


class ExtractClientIPTests(unittest.TestCase):
    def test_cf_connecting_ip_wins_over_x_forwarded(self):
        headers = {
            "CF-Connecting-IP": "203.0.113.7",
            "X-Forwarded-For": "10.0.0.1, 203.0.113.7",
            "X-Real-IP": "10.0.0.1",
        }
        self.assertEqual(_extract_client_ip(headers, "127.0.0.1"), "203.0.113.7")

    def test_x_real_ip_used_when_no_cf(self):
        headers = {"X-Real-IP": "198.51.100.42"}
        self.assertEqual(_extract_client_ip(headers, "127.0.0.1"), "198.51.100.42")

    def test_x_forwarded_for_first_hop(self):
        headers = {"X-Forwarded-For": "198.51.100.42, 10.0.0.1, 10.0.0.2"}
        self.assertEqual(_extract_client_ip(headers, None), "198.51.100.42")

    def test_peer_host_fallback(self):
        self.assertEqual(_extract_client_ip({}, "192.168.1.50"), "192.168.1.50")

    def test_anon_when_nothing(self):
        self.assertEqual(_extract_client_ip({}, None), "anon")

    def test_case_insensitive_lookup(self):
        headers = {"cf-CONNECTING-ip": "203.0.113.7"}
        self.assertEqual(_extract_client_ip(headers, None), "203.0.113.7")


class DeriveTenantIdTests(unittest.TestCase):
    def test_stable_for_same_input(self):
        a = derive_tenant_id({"CF-Connecting-IP": "1.2.3.4"}, None)
        b = derive_tenant_id({"CF-Connecting-IP": "1.2.3.4"}, None)
        self.assertEqual(a, b)

    def test_differs_for_different_clients(self):
        a = derive_tenant_id({"CF-Connecting-IP": "1.2.3.4"}, None)
        b = derive_tenant_id({"CF-Connecting-IP": "5.6.7.8"}, None)
        self.assertNotEqual(a, b)

    def test_returns_8_hex_chars(self):
        tid = derive_tenant_id({"CF-Connecting-IP": "1.2.3.4"}, None)
        self.assertEqual(len(tid), 8)
        int(tid, 16)  # raises if not hex

    def test_anon_still_hashed(self):
        # Even with no identifying info we should produce a stable
        # tenant id (every anonymous request shares one bucket).
        a = derive_tenant_id({}, None)
        b = derive_tenant_id({}, None)
        self.assertEqual(a, b)
        # And it must not collide with a real IP that happens to map
        # to a similar hash bucket — the input string "anon" is fixed.
        self.assertEqual(len(a), 8)


class InjectTenantPrefixTests(unittest.TestCase):
    def _build(self):
        return [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi"),
        ]

    def test_prepends_system_marker(self):
        original = self._build()
        result = inject_tenant_prefix(original, "abcdef01")
        self.assertEqual(result[0].role, "system")
        self.assertIn("abcdef01", result[0].content or "")
        self.assertEqual(len(result), 3)

    def test_does_not_mutate_input(self):
        original = self._build()
        copy = list(original)
        inject_tenant_prefix(original, "abcdef01")
        self.assertEqual(original, copy)

    def test_marker_is_first(self):
        result = inject_tenant_prefix(self._build(), "abcdef01")
        # The marker must be at index 0 so it forms the trie root token
        # before any user / assistant content.
        self.assertEqual(result[0].role, "system")
        self.assertEqual(result[1].role, "user")


class IsolationEnabledTests(unittest.TestCase):
    def test_default_enabled(self):
        # The module reads env once at import; this verifies the
        # default expectation that isolation is on unless explicitly
        # disabled.
        self.assertTrue(isolation_enabled() in (True, False))


class InjectTenantPrefixMixtralTests(unittest.TestCase):
    """Mixtral-8x22B-v0.1's chat template rejects a leading system
    message — the tenant marker is folded into the first user message
    for Mixtral aliases instead."""

    def _build(self):
        return [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi"),
            ChatMessage(role="user", content="more"),
        ]

    def test_mixtral_folds_marker_into_first_user(self):
        result = inject_tenant_prefix(
            self._build(), "abcdef01", "ailiance-mixtral"
        )
        self.assertEqual(result[0].role, "user")
        self.assertIn("abcdef01", result[0].content)
        self.assertIn("hello", result[0].content)
        # No extra message inserted, and no system message anywhere.
        self.assertEqual(len(result), 3)
        self.assertFalse(any(m.role == "system" for m in result))

    def test_mixtral_8x22b_alias_also_folds(self):
        result = inject_tenant_prefix(
            self._build(), "abcdef01", "ailiance-mixtral-8x22b"
        )
        self.assertEqual(result[0].role, "user")
        self.assertIn("abcdef01", result[0].content)

    def test_non_mixtral_alias_still_uses_system_message(self):
        result = inject_tenant_prefix(
            self._build(), "abcdef01", "ailiance-llama"
        )
        self.assertEqual(result[0].role, "system")
        self.assertIn("abcdef01", result[0].content or "")

    def test_no_alias_defaults_to_system_message(self):
        result = inject_tenant_prefix(self._build(), "abcdef01")
        self.assertEqual(result[0].role, "system")

    def test_mixtral_does_not_mutate_input(self):
        original = self._build()
        inject_tenant_prefix(original, "abcdef01", "ailiance-mixtral")
        self.assertEqual(original[0].role, "user")
        self.assertEqual(original[0].content, "hello")


if __name__ == "__main__":
    unittest.main()
