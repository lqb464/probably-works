"""Run at most one independent experiment per GPU; propagate failures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True, help="JSON list of run.py argument lists")
    parser.add_argument("--gpus", type=int, default=2)
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error("gpus must be positive")
    queue = json.loads(Path(args.jobs).read_text())
    if not isinstance(queue, list) or any(not isinstance(j, list) or not all(isinstance(x, str) for x in j) for j in queue):
        parser.error("jobs must be a list of string argument lists")
    active, failed = {}, []
    try:
        while queue or active:
            for gpu in range(args.gpus):
                if gpu not in active and queue:
                    job = queue.pop(0)
                    output = Path(job[job.index("--output") + 1])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    log = output.with_suffix(".log").open("x")
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
                    process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("run.py")), *job, "--device", "cuda:0"], env=env, stdout=log, stderr=subprocess.STDOUT)
                    active[gpu] = (process, log, output)
                    print(f"GPU {gpu}: {output}; log={output.with_suffix('.log')}", flush=True)
            for gpu, (process, log, output) in list(active.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    print(f"GPU {gpu}: exit={code} {output}", flush=True)
                    if code:
                        failed.append(str(output))
                    del active[gpu]
            if active:
                time.sleep(2)
    finally:
        for process, log, _ in active.values():
            process.terminate()
            process.wait()
            log.close()
    if failed:
        raise SystemExit("FAILED (read .log): " + ", ".join(failed))


if __name__ == "__main__":
    main()
