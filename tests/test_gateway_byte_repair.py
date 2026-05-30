"""Byte-level detokenizer-leak repair in the gateway response path.

Some omlx-served tokenizers (Mistral-Small) leak the GPT-2 byte-level
surface forms instead of decoded text: spaces become ``Ġ``, newlines
``Ċ``, and UTF-8 accents are read as latin-1 mojibake (``Ã©`` for ``é``).
The text and token ids are correct — only the final byte-decode is
missing — so the corruption is deterministically reversible by applying
the GPT-2 byte decoder. The gateway repairs it as a last resort so the
3 router domains that resolve to the omlx Mistral-Small model don't ship
garbage to clients.
"""

from src.gateway.server import _repair_byte_level, _normalize_message_dict


class TestRepairByteLevel:
    def test_spaces_repaired(self):
        assert _repair_byte_level("TheĠproductĠof") == "The product of"

    def test_full_french_message(self):
        corrupted = "PourĠciterĠtroisĠvillesĠenĠfranÃ§ais"
        assert _repair_byte_level(corrupted) == "Pour citer trois villes en français"

    def test_newline_marker(self):
        assert _repair_byte_level("lineĠoneĊĊlineĠtwo") == "line one\n\nline two"

    def test_accents_in_corrupted(self):
        # déjà à café — all spaces are Ġ so the whole thing repairs
        assert _repair_byte_level("dÃ©jÃłĠvisitÃ©") == "déjà visité"

    def test_noop_on_clean_text(self):
        assert _repair_byte_level("Pour citer trois villes") == "Pour citer trois villes"

    def test_noop_preserves_legit_accents(self):
        # Has real spaces -> not corrupted -> untouched, accents preserved
        assert _repair_byte_level("déjà visité Paris") == "déjà visité Paris"

    def test_noop_on_empty(self):
        assert _repair_byte_level("") == ""

    def test_noop_when_normal_space_present(self):
        # A stray Ġ but real spaces -> treat as legit (not fully corrupted)
        assert _repair_byte_level("normal text with Ġ stray") == "normal text with Ġ stray"


class TestNormalizeAppliesRepair:
    def test_message_content_repaired(self):
        msg = {"role": "assistant", "content": "17ĠÃĹĠ23Ġ=Ġ391"}
        _normalize_message_dict(msg)
        assert msg["content"] == "17 × 23 = 391"

    def test_clean_message_untouched(self):
        msg = {"role": "assistant", "content": "17 × 23 = 391"}
        _normalize_message_dict(msg)
        assert msg["content"] == "17 × 23 = 391"
