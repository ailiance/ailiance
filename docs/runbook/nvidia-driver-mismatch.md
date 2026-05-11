# NVIDIA Driver Kernel/Userspace Mismatch — Runbook

**Last validated:** 2026-05-11 on kxkm-ai (RTX 4090, Ubuntu)
**Severity:** Blocks new CUDA process initialization but does NOT kill already-loaded contexts
**Recovery time:** ~10 min, zero downtime for existing GPU workers, no reboot needed

## Symptoms

- `nvidia-smi`: `Failed to initialize NVML: Driver/library version mismatch`
- `python -c "import torch; torch.cuda.is_available()"` returns `False`
- CUDA error code 804 "forward compatibility was attempted on non supported HW"
- **Already-running GPU workers continue to function** (they hold pre-mismatch CUDA contexts in RAM)
- New PyTorch / Unsloth / mlx_lm processes fail at `torch.cuda.init()` or equivalent

## Root cause

Unattended `apt upgrade` pulled a new `nvidia-driver-XXX` userspace lib (e.g. `libcuda.so.580.142`) but the kernel module is still the previous version (`NVRM 580.126.09`). DKMS rebuild was not re-run (or the upgrade didn't re-trigger DKMS).

This commonly happens when:
- `unattended-upgrades` runs in the background
- Multiple parallel `nvidia-driver-{550,580,...}` packages installed (legacy + new)
- The system was last rebooted with kernel module XXX, then upgrade pulled XXX+epsilon

Verify diagnosis:
```bash
cat /proc/driver/nvidia/version | head -3     # kernel module version
ldconfig -p | grep libcuda.so.1               # userspace lib version
dpkg -l | grep -E "nvidia-driver-[0-9]"       # installed driver packages
```

## Fix — Surgical APT downgrade (preferred, zero downtime)

This pattern downgrades the userspace libs to match the running kernel module, **without** killing the running GPU workers.

### Step 1 — Identify the matching version

```bash
# What kernel says it loaded
cat /proc/driver/nvidia/version
# What userspace has
ls /usr/lib/x86_64-linux-gnu/libcuda.so.*
```

If kernel = `580.126.09` and userspace = `580.142`, target downgrade is `580.126.09`.

### Step 2 — Find the kernel-matching package versions in apt

```bash
apt-cache madison nvidia-driver-580
apt-cache madison libnvidia-gl-580
# etc — for each nvidia-580 package, find the candidate version that matches kernel
```

Typically the matching version is the "older" one in `apt-cache madison` output.

### Step 3 — Downgrade in one transaction

```bash
sudo apt install --allow-downgrades \
  libnvidia-extra-580=580.126.09-0ubuntu1 \
  libnvidia-fbc1-580=580.126.09-0ubuntu1 \
  libnvidia-gl-580=580.126.09-0ubuntu1 \
  nvidia-compute-utils-580=580.126.09-0ubuntu1 \
  nvidia-dkms-580=580.126.09-0ubuntu1 \
  nvidia-driver-580=580.126.09-0ubuntu1 \
  nvidia-firmware-580-580.126.09=580.126.09-0ubuntu1 \
  nvidia-kernel-common-580=580.126.09-0ubuntu1 \
  nvidia-kernel-source-580=580.126.09-0ubuntu1 \
  xserver-xorg-video-nvidia-580=580.126.09-0ubuntu1
```

Adjust the 10-15 package list to your distro/version. DKMS rebuilds during install — no reboot.

### Step 4 — Verify

```bash
nvidia-smi                                    # should show driver + CUDA info
python -c "import torch; print(torch.cuda.is_available())"    # True
```

### Step 5 — Lock the versions to prevent re-occurrence

```bash
sudo apt-mark hold libnvidia-extra-580 libnvidia-fbc1-580 libnvidia-gl-580 \
  nvidia-compute-utils-580 nvidia-dkms-580 nvidia-driver-580 \
  nvidia-firmware-580-580.126.09 nvidia-kernel-common-580 \
  nvidia-kernel-source-580 xserver-xorg-video-nvidia-580
```

Verify:
```bash
apt-mark showhold | grep nvidia
```

Should list 15 packages (10 above + 5 transitively-pinned).

## Alternative fix — Full reboot (only if surgical downgrade fails)

If APT cache doesn't have the matching version, or DKMS rebuild fails, the fallback is to reboot the box. This kills all running GPU workers. Plan as a maintenance window:

1. Stop GPU workers (e.g. `sudo systemctl stop eu-kiki-qwen-server eu-kiki-granite-server`)
2. `sudo reboot`
3. Wait for boot, verify `nvidia-smi`
4. Restart workers

## Architectural insight — Why this matters

kxkm-ai runs 3 GPU services on a single RTX 4090 (24 GB VRAM):
- llama-server Qwen3-Next 80B Q4_K_M MoE (5.75 GB VRAM, partial offload to CPU via `-ot ffn_*=CPU`)
- llama-server Granite-4.1-30B Q4_K_M (19.3 GB VRAM, partial offload via `-ot blk.5[0-9]=CPU`)
- Qwen3-4B QLoRA training (transient, when active)

This VRAM saturation is **intentional** — the ailiance gateway's PR #49 routes the 10 hardware-domain mascarade aliases to a Tower-Ollama backend (port :8004) precisely so kxkm-ai isn't the bottleneck for that traffic. The surgical downgrade pattern preserves this architecture without needing to take any of the 3 services down.

## Cross-references

- ailiance gateway state: `reference_eu_kiki_gateway_2026_05_10.md` (memory)
- PR #49 mascarade routing: `ailiance/ailiance#49`
- kxkm-ai inventory: `project_kxkm23_models_inventory_2026_04_26.md`
