import pytest
from scripts.kicad_sch.compilers.system_prompts import (
    SKIDL_PROMPT, ATOPILE_PROMPT, TSCIRCUIT_PROMPT, CIRCUIT_SYNTH_PROMPT,
    SYSTEM_PROMPTS,
)


@pytest.mark.parametrize("name,prompt", [
    ("skidl", SKIDL_PROMPT),
    ("atopile", ATOPILE_PROMPT),
    ("tscircuit", TSCIRCUIT_PROMPT),
    ("circuit-synth", CIRCUIT_SYNTH_PROMPT),
])
def test_prompt_is_non_empty_string(name, prompt):
    assert isinstance(prompt, str)
    assert 200 <= len(prompt) <= 2000, f"{name} prompt length {len(prompt)} out of band"


@pytest.mark.parametrize("compiler", ["skidl", "atopile", "tscircuit", "circuit-synth"])
def test_prompt_forbids_markdown(compiler):
    p = SYSTEM_PROMPTS[compiler].lower()
    assert "no markdown" in p or "no code fence" in p or "do not wrap" in p


def test_system_prompts_dict_has_all_four():
    assert set(SYSTEM_PROMPTS.keys()) == {"skidl", "atopile", "tscircuit", "circuit-synth"}


def test_skidl_prompt_mentions_generate_schematic():
    assert "generate_schematic" in SKIDL_PROMPT


def test_atopile_prompt_mentions_ato_extension():
    assert ".ato" in ATOPILE_PROMPT


def test_tscircuit_prompt_mentions_tsx():
    assert ".tsx" in TSCIRCUIT_PROMPT or "TSX" in TSCIRCUIT_PROMPT


def test_circuit_synth_prompt_mentions_module():
    assert "circuit_synth" in CIRCUIT_SYNTH_PROMPT
