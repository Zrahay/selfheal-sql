"""Run one of our scripts on an HF Job GPU.

Ships the code folder and mounts the private dataset repo at /data. The HF token is read
from the HF_TOKEN env var and passed as an encrypted job secret (never written to disk).

Usage:
  python launch_job.py --name smoke --flavor a10g-small --timeout 1h \
      --pip "transformers accelerate peft" -- student.py --split pool --limit 20 ...
Everything after "--" is the script and its arguments; --data/--out are added for you.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time

from huggingface_hub import HfApi, Volume, fetch_job_logs, inspect_job, run_job, sync_job_volume

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_REPO = "HideIron/selfheal-sql-spider"
RUNS_REPO = "HideIron/selfheal-sql-runs"
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--flavor", default="a10g-small")
    ap.add_argument("--timeout", default="1h")
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--pip", default="transformers accelerate peft huggingface_hub jinja2>=3.1.6")
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
    script, script_args = rest[0], rest[1:]

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(RUNS_REPO, repo_type="dataset", private=True, exist_ok=True)

    code = sync_job_volume(HERE, "/code", token=token)
    data = Volume(type="dataset", source=DATA_REPO, mount_path="/data")

    inner = (
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        "pip install -q -U " + " ".join(shlex.quote(x) for x in a.pip.split()) + " && "
        f"python /code/{script} --data /data --out /tmp/out "
        f"--push-repo {RUNS_REPO} --push-path runs/{a.name} " + " ".join(shlex.quote(x) for x in script_args)
    )
    job = run_job(
        image=a.image, command=["bash", "-lc", inner], flavor=a.flavor, timeout=a.timeout,
        name=a.name, volumes=[code, data], secrets={"HF_TOKEN": token},
    )
    print("job:", job.url, flush=True)
    if a.detach:
        print("job_id:", job.id)
        return
    while True:
        st = inspect_job(job_id=job.id).status
        if st.stage in ("COMPLETED", "ERROR", "CANCELED", "DELETED"):
            break
        time.sleep(10)
    print("stage:", st.stage, "|", st.message)
    lines = [ln for ln in fetch_job_logs(job_id=job.id) if "[A" not in ln and "it/s]" not in ln and "B/s]" not in ln]
    for line in lines[-80:]:
        print(line)
    sys.exit(0 if st.stage == "COMPLETED" else 1)


if __name__ == "__main__":
    main()
