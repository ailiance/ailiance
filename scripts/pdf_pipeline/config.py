"""Source registry for PDF pipeline.

Each entry defines a document source with its legal basis under the
EU Digital Single Market Directive (DSM) and AI Act compliance metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
PDF_RAW_DIR = DATA_ROOT / "pdf-raw"
PDF_EXTRACTED_DIR = DATA_ROOT / "pdf-extracted"
HF_TRACED_DIR = DATA_ROOT / "hf-traced"

USER_AGENT = "ailiance-training-pipeline/0.2 (research, EU DSM Art.4 TDM)"
RATE_LIMIT_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PdfSource:
    name: str
    base_url: str
    legal_basis: str
    domains: tuple[str, ...]
    robots_txt: str = ""
    license_note: str = ""
    example_urls: tuple[str, ...] = field(default_factory=tuple)


SOURCES: tuple[PdfSource, ...] = (
    PdfSource(
        name="ST Application Notes",
        base_url="https://www.st.com/resource/en/application_note/",
        legal_basis="DSM_ART4_TDM",
        license_note="ST encourages free distribution of app notes for design-in purposes",
        robots_txt="https://www.st.com/robots.txt",
        domains=("stm32", "embedded", "electronics", "emc-dsp-power"),
        example_urls=(
            "https://www.st.com/resource/en/application_note/an4488-getting-started-with-stm32f4xxxx-mcu-hardware-development-stmicroelectronics.pdf",
            "https://www.st.com/resource/en/application_note/an2867-oscillator-design-guide-for-stm8afals-stm32-mcus-and-mpus-stmicroelectronics.pdf",
        ),
    ),
    PdfSource(
        name="Espressif Technical Reference",
        base_url="https://www.espressif.com/sites/default/files/documentation/",
        legal_basis="DSM_ART4_TDM",
        license_note="Espressif docs are freely available, Apache-2.0 SDK",
        robots_txt="https://www.espressif.com/robots.txt",
        domains=("embedded", "iot", "electronics"),
        example_urls=(
            "https://www.espressif.com/sites/default/files/documentation/esp32_technical_reference_manual_en.pdf",
        ),
    ),
    PdfSource(
        name="TI Application Reports",
        base_url="https://www.ti.com/lit/",
        legal_basis="DSM_ART4_TDM",
        license_note="TI explicitly allows use of app notes for design purposes",
        robots_txt="https://www.ti.com/robots.txt",
        domains=("electronics", "power", "emc-dsp-power"),
        example_urls=(),
    ),
    PdfSource(
        name="NXP Application Notes",
        base_url="https://www.nxp.com/docs/en/application-note/",
        legal_basis="DSM_ART4_TDM",
        robots_txt="https://www.nxp.com/robots.txt",
        domains=("embedded", "electronics"),
        example_urls=(),
    ),
    PdfSource(
        name="KiCad Documentation",
        base_url="https://docs.kicad.org/",
        legal_basis="CC-BY",
        license_note="KiCad docs are CC-BY licensed",
        domains=("kicad-dsl", "kicad-pcb"),
        example_urls=(),
    ),
)


def get_source(name: str) -> PdfSource:
    """Look up a source by name (case-insensitive partial match)."""
    lower = name.lower()
    for src in SOURCES:
        if lower in src.name.lower():
            return src
    msg = f"Unknown source: {name!r}. Available: {[s.name for s in SOURCES]}"
    raise KeyError(msg)
