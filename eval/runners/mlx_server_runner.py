#!/usr/bin/env python3
"""mlx_server_runner — start an MLX-LM OpenAI-compatible server for evaluation.

Spawns `mlx_lm server` with the given model + optional adapter, waits for it
to become responsive, and exposes a context manager API. Used by Lighteval
and EvalPlus runners that consume an OpenAI-compatible endpoint.

Captures the resolved environment (model SHA, adapter SHA, MLX version,
hardware) for publishable benchmarks.

Usage (CLI):
    python -m runners.mlx_server_runner \\
        --model models/Devstral-Small-2-24B-MLX-4bit \\
        --adapter output/adapters/devstral/python \\
        --port 8000

Usage (Python):
    with MLXServer(model_path, adapter_path, port=8000) as srv:
        env = srv.env_snapshot()       # dict, save as env.json
        run_lighteval(srv.base_url)
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """SHA-256 of the first chunk_bytes bytes (fast fingerprint for big weights)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(chunk_bytes))
    return h.hexdigest()


def _shasum_first_safetensors(model_dir: Path) -> Optional[str]:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        return None
    return _sha256_file(files[0])


def _mlx_version() -> str:
    try:
        import mlx
        return getattr(mlx, "__version__", "unknown")
    except Exception:
        return "not-installed"


def _mlx_lm_version() -> str:
    try:
        import mlx_lm
        return getattr(mlx_lm, "__version__", "unknown")
    except Exception:
        return "not-installed"


def _hardware_snapshot() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "node": platform.node(),
        "python": sys.version.split()[0],
        "mlx_version": _mlx_version(),
        "mlx_lm_version": _mlx_lm_version(),
    }


def _git_describe(repo_path: Path) -> dict:
    """Capture git state for the eu-kiki repo at run time."""
    out: dict = {"repo": str(repo_path)}
    if not (repo_path / ".git").exists():
        out["error"] = "not a git repo"
        return out
    try:
        for key, args in [
            ("commit", ["rev-parse", "HEAD"]),
            ("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
            ("describe", ["describe", "--always", "--dirty", "--tags"]),
            ("status_short", ["status", "--porcelain"]),
        ]:
            r = subprocess.run(
                ["git", "-C", str(repo_path)] + args,
                capture_output=True, text=True, timeout=5,
            )
            out[key] = r.stdout.strip() if r.returncode == 0 else None
    except Exception as e:
        out["error"] = str(e)
    return out


def _pip_freeze() -> list[str]:
    """Capture pinned package versions for full reproducibility."""
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            return r.stdout.splitlines()
    except Exception:
        pass
    return []


# ----------------------------------------------------------------------------
# MLXServer
# ----------------------------------------------------------------------------


@dataclasses.dataclass
class MLXServer:
    model_path: Path
    adapter_path: Optional[Path] = None
    port: int = 8000
    host: str = "127.0.0.1"
    log_file: Optional[Path] = None
    extra_args: tuple[str, ...] = ()
    startup_timeout_s: int = 120

    _process: Optional[subprocess.Popen] = dataclasses.field(default=None, init=False)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _build_cmd(self) -> list[str]:
        cmd = [
            sys.executable, "-m", "mlx_lm", "server",
            "--model", str(self.model_path),
            "--port", str(self.port),
            "--host", self.host,
            "--log-level", "WARNING",
        ]
        if self.adapter_path is not None:
            cmd.extend(["--adapter-path", str(self.adapter_path)])
        cmd.extend(self.extra_args)
        return cmd

    def start(self) -> None:
        cmd = self._build_cmd()
        log_fp = open(self.log_file, "w") if self.log_file else subprocess.DEVNULL
        self._process = subprocess.Popen(
            cmd, stdout=log_fp, stderr=subprocess.STDOUT,
        )
        self._wait_ready()

    def _wait_ready(self) -> None:
        url = f"{self.base_url}/models"
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"mlx_lm server exited early (code={self._process.returncode}). "
                    f"Check log file: {self.log_file}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2):
                    return
            except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
                time.sleep(1)
        raise TimeoutError(
            f"mlx_lm server did not become ready on {url} within "
            f"{self.startup_timeout_s}s"
        )

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        self._process = None

    def __enter__(self) -> "MLXServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def env_snapshot(self, eu_kiki_repo: Path | None = None) -> dict:
        """Capture full environment for publishable runs."""
        adapter_sha = None
        if self.adapter_path:
            adapter_file = self.adapter_path / "adapters.safetensors"
            if adapter_file.exists():
                adapter_sha = _sha256_file(adapter_file)
        if eu_kiki_repo is None:
            eu_kiki_repo = Path(__file__).resolve().parents[2]
        return {
            "schema_version": "eu-kiki-eval-env/1.0",
            "model_path": str(self.model_path),
            "model_first_safetensors_sha256": _shasum_first_safetensors(self.model_path),
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "adapter_sha256": adapter_sha,
            "server": {
                "host": self.host,
                "port": self.port,
                "base_url": self.base_url,
            },
            "hardware": _hardware_snapshot(),
            "git": _git_describe(eu_kiki_repo),
            "pip_freeze": _pip_freeze(),
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Path to MLX model directory")
    parser.add_argument("--adapter", type=Path, default=None, help="Optional adapter path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--env-out", type=Path, default=None,
                        help="Write env snapshot JSON to this path then exit (no server stays up)")
    parser.add_argument("--block", action="store_true",
                        help="Stay running until Ctrl-C (otherwise exits after env snapshot)")
    args = parser.parse_args()

    srv = MLXServer(
        model_path=args.model,
        adapter_path=args.adapter,
        port=args.port,
        host=args.host,
        log_file=args.log_file,
    )

    if args.env_out and not args.block:
        # Just snapshot env without starting server (no model load)
        # Useful to verify SHA/hardware before a long run.
        env = {
            "schema_version": "eu-kiki-eval-env/1.0",
            "model_path": str(args.model),
            "model_first_safetensors_sha256": _shasum_first_safetensors(args.model),
            "adapter_path": str(args.adapter) if args.adapter else None,
            "adapter_sha256": _sha256_file(args.adapter / "adapters.safetensors")
                if args.adapter and (args.adapter / "adapters.safetensors").exists()
                else None,
            "hardware": _hardware_snapshot(),
            "git": _git_describe(Path(__file__).resolve().parents[2]),
            "pip_freeze": _pip_freeze(),
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "snapshot_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        args.env_out.parent.mkdir(parents=True, exist_ok=True)
        args.env_out.write_text(json.dumps(env, indent=2))
        print(f"Env snapshot → {args.env_out}")
        return

    with srv:
        env = srv.env_snapshot()
        if args.env_out:
            args.env_out.parent.mkdir(parents=True, exist_ok=True)
            args.env_out.write_text(json.dumps(env, indent=2))
            print(f"Env snapshot → {args.env_out}")
        print(f"Server up at {srv.base_url}")
        print(json.dumps(env, indent=2))
        if args.block:
            print("Press Ctrl-C to stop.")
            with contextlib.suppress(KeyboardInterrupt):
                while True:
                    time.sleep(60)


if __name__ == "__main__":
    _cli()
