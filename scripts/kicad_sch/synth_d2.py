"""D2: random circuit synth -> skidl/atopile/circuit-synth -> kicad_sch.

Each template ships a renderer per compiler. ERC-clean rate target
60-80% (rejected outputs are unlinked and logged).
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.kicad_sch.audit_log import AuditLogger
from scripts.kicad_sch.manifest import DatasetManifest

TEMPLATES: list[dict] = [
    {"name": "voltage_divider",
     "params": {"r1_k": (1, 100), "r2_k": (1, 100), "vin": (3, 24)}},
    {"name": "rc_lowpass",
     "params": {"r_k": (1, 100), "c_nf": (1, 1000)}},
    {"name": "rlc_series",
     "params": {"r": (1, 1000), "l_uh": (1, 1000), "c_nf": (1, 1000)}},
    {"name": "ne555_astable",
     "params": {"r1_k": (1, 100), "r2_k": (1, 100), "c_nf": (1, 1000)}},
    {"name": "opamp_noninv",
     "params": {"rf_k": (1, 100), "rg_k": (1, 100)}},
    {"name": "common_emitter",
     "params": {"rc_k": (1, 10), "re": (10, 1000), "rb_k": (10, 1000)}},
    {"name": "led_blinker",
     "params": {"r_led": (100, 1000), "vcc": (3, 12)}},
    {"name": "diode_clamp",
     "params": {"r_in_k": (1, 100)}},
    {"name": "ldo_3v3",
     "params": {"vin": (5, 12), "c_in_uf": (1, 10), "c_out_uf": (1, 10)}},
    {"name": "transistor_inv",
     "params": {"rb_k": (1, 100), "rc_k": (1, 10)}},
]

COMPILERS = ("skidl", "atopile", "circuit-synth")


def randomize_values(tpl: dict, seed: int) -> dict:
    """Deterministic per-seed parameter draw."""
    rng = random.Random(seed)
    out: dict[str, object] = {}
    for k, v in tpl["params"].items():
        if isinstance(v, tuple) and all(isinstance(x, int) for x in v):
            out[k] = rng.randint(*v)
        else:
            out[k] = round(rng.uniform(*v), 3)
    return out


def _compile_skidl(tpl: dict, vals: dict, out: Path) -> int:
    """Stub: emit a minimal parseable v10 skeleton.

    Real skidl call is a follow-up patch once the v10-capable skidl
    wheel is wired into the ``~/eu-kiki/.venv-d2/`` environment.
    """
    out.write_text(
        "(kicad_sch (version 20240101) (generator skidl)\n"
        '  (uuid "00000000-0000-0000-0000-000000000001")\n'
        '  (paper "A4") (lib_symbols))\n',
        encoding="utf-8",
    )
    return 0


def _compile_atopile(tpl: dict, vals: dict, out: Path) -> int:
    r = subprocess.run(
        ["ato", "build", "--template", tpl["name"], "--out", str(out)],
        capture_output=True,
        timeout=120,
    )
    return r.returncode


def _compile_circuit_synth(tpl: dict, vals: dict, out: Path) -> int:
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "circuit_synth.build",
            "--template",
            tpl["name"],
            "--out",
            str(out),
        ],
        capture_output=True,
        timeout=120,
    )
    return r.returncode


def _kicad_erc(path: Path) -> int:
    r = subprocess.run(
        ["kicad-cli", "sch", "erc", str(path)],
        capture_output=True,
        timeout=60,
    )
    return r.returncode


def synth_one(
    template: str, compiler: str, seed: int, out_dir: Path
) -> Path | None:
    """Generate one schematic; unlink+return None if compile or ERC fails."""
    tpl = next((t for t in TEMPLATES if t["name"] == template), None)
    if tpl is None:
        return None
    vals = randomize_values(tpl, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{template}-{compiler}-{seed}.kicad_sch"
    fn = {
        "skidl": _compile_skidl,
        "atopile": _compile_atopile,
        "circuit-synth": _compile_circuit_synth,
    }[compiler]
    if fn(tpl, vals, out) != 0:
        out.unlink(missing_ok=True)
        return None
    if _kicad_erc(out) != 0:
        out.unlink(missing_ok=True)
        return None
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--compilers", default="skidl,atopile,circuit-synth")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / "eu-kiki-data/kicad-sch-synth",
    )
    p.add_argument(
        "--audit-dir",
        type=Path,
        default=Path.home() / "eu-kiki/output/audit/kicad-sch-2026-05-11",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path.home() / "eu-kiki/output/audit/kicad-sch-2026-05-11/d2_manifest.csv",
    )
    a = p.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    comps = a.compilers.split(",")
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_path = a.audit_dir / f"d2-{run_stamp}.ndjson"
    log = AuditLogger(audit_path)
    manifest = DatasetManifest(a.manifest, split="D2")
    rng = random.Random(a.seed_start)
    n_ok = 0
    for i in range(a.n_samples):
        tpl = rng.choice(TEMPLATES)
        comp = rng.choice(comps)
        seed = a.seed_start + i
        out = synth_one(tpl["name"], comp, seed, a.out_dir)
        if out is None:
            log.log(
                "d2_synth_fail",
                template=tpl["name"], compiler=comp, seed=seed,
            )
            continue
        manifest.add(
            source_type="synth",
            source_url=f"gen:{tpl['name']}@seed{seed}@{comp}",
            commit_sha="",
            license_spdx="CC0-1.0",
            dedup_hash=f"{tpl['name']}-{comp}-{seed}",
            file_size_bytes=out.stat().st_size,
            kicad_version_before="10.0.2",
            kicad_version_after="10.0.2",
        )
        n_ok += 1
    manifest.write()
    log.log("d2_done", accepted=n_ok, requested=a.n_samples)
    print(f"D2: {n_ok}/{a.n_samples} files written to {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
