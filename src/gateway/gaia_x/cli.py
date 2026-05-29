"""CLI orchestrating the Gaia-X credential pipeline.

Subcommands:
  render    write did.json + signed VCs into the well-known dir
  (notarize / comply / publish are added in later tasks)
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from gateway.gaia_x.config import GaiaXConfig
from gateway.gaia_x.credentials import (
    build_legal_participant,
    build_service_offering,
    build_terms_and_conditions,
)
from gateway.gaia_x.did import build_did_document
from gateway.gaia_x.keys import ensure_key, public_jwk
from gateway.gaia_x.signing import sign_credential


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _write(out_dir: Path, name: str, doc: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(doc, indent=2))


def render_artifacts(cfg: GaiaXConfig, out_dir: Path, key_path: Path,
                     issuance_date: str | None = None) -> None:
    issuance_date = issuance_date or _now()
    ensure_key(key_path)
    jwk = public_jwk(key_path, cfg)
    _write(out_dir, "did.json", build_did_document(cfg, jwk))
    for name, builder in (
        ("participant.json", build_legal_participant),
        ("gx-terms-and-conditions.json", build_terms_and_conditions),
        ("service-offering.json", build_service_offering),
    ):
        signed = sign_credential(builder(cfg, issuance_date), key_path, cfg)
        _write(out_dir, name, signed)


def _cfg_and_paths(args):
    cfg = GaiaXConfig.from_env()
    out_dir = Path(os.environ.get("GAIA_X_WELL_KNOWN_DIR", "var/well-known"))
    key_path = Path(os.environ.get("GAIA_X_KEY_PATH", "var/gaia-x-signing.pem"))
    return cfg, out_dir, key_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gateway.gaia_x.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("render", help="write did.json + signed VCs")
    args = parser.parse_args(argv)
    cfg, out_dir, key_path = _cfg_and_paths(args)
    if args.command == "render":
        render_artifacts(cfg, out_dir, key_path)
        print(f"rendered artifacts into {out_dir}")
        return 0
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
