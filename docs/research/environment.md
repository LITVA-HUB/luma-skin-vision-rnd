# Environment — observed 2026-09-10

Source: `python scripts/inspect_environment.py` → [environment.json](environment.json).
Windows 11 build 26200; CPython 3.12.6; PyTorch 2.8.0+cu128; torchvision 0.23.0+cu128.
CUDA runtime 12.8 is available through installed wheels. GPU: NVIDIA GeForce RTX 4060,
8188 MiB reported by nvidia-smi; driver 610.88. Physical RAM: 33,487,904,768 bytes
(approximately 31.19 GiB available to the OS as installed physical memory).

`Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name` reports
AMD Ryzen 9 7900X 12-Core Processor. This is actual host evidence, distinct from the
prior audit's CPU container. No latency figures from that supplied audit are used here.

The system-wide Python is 3.13.2; the project uses an isolated `.venv` with the already
installed Python 3.12.6. Exact package versions/index hashes are in uv.lock. No global
Python packages or Git identity were changed. Local commits use explicit automation
identity `Codex <codex@localhost>` because the machine had no configured Git author.

Run latency and allocated-memory measurements are in docs/benchmarks and the corresponding
experiment artifacts. They use 64px synthetic crops unless otherwise stated. No result
at that size implies accuracy or throughput at 224–256px real capture resolution.
