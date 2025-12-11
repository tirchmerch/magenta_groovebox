#!/usr/bin/env python3
import os
import subprocess

BASE = "/home/pmtirch/groovebox/run"

def is_run_dir(path):
    return os.path.isdir(os.path.join(path, "train"))

for run_id in sorted(os.listdir(BASE)):
    run_path = os.path.join(BASE, run_id)
    if not is_run_dir(run_path):
        continue

    print(f"\n=== Evaluating {run_id} ===")

    cmd = [
        "python3", "music_vae_train.py",
        "--mode=eval",
        f"--config=groovae_2bar_groovebox",
        f"--tfds_name=groove/2bar-midionly",
        f"--run_dir={run_path}",
        "--eval_num_batches=50",
        "--run_once=True"
    ]

    subprocess.run(cmd)
