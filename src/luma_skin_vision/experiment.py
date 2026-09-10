import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def source_identity():
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "src").rglob("*.py")) + [root / "pyproject.toml", root / "uv.lock"]
    h = hashlib.sha256()
    for path in files:
        if path.exists():
            h.update(path.relative_to(root).as_posix().encode())
            h.update(path.read_bytes())
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except subprocess.CalledProcessError:
        commit, dirty = None, True
    return {"git_commit": commit, "git_dirty": dirty, "source_hash": h.hexdigest()}


def create_run(root, config):
    now = datetime.now(timezone.utc)
    run = Path(root) / f"{now:%Y%m%dT%H%M%S}_{config['method']}_{uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", config)
    return run, {"experiment_id": run.name, "timestamp": now.isoformat(), **source_identity()}
