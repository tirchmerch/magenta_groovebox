#!/usr/bin/env python3
import os
import subprocess
from tensorboard.backend.event_processing import event_accumulator

BASE = "/home/pmtirch/groovebox/run"
OUTPUT_FILE = "/home/pmtirch/groovebox/run/p5_results.txt"

def is_run_dir(path):
    return os.path.isdir(os.path.join(path, "train"))

with open(OUTPUT_FILE, "w") as out:
    out.write("run_id|P@5\n")

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
            "--eval_once"
        ]

        subprocess.run(cmd, check=True)

        eval_dir = os.path.join(run_path, "eval")
        event_files = [os.path.join(eval_dir, f) 
            for f in os.listdir(eval_dir) 
            if f.startswith("events.out")]
        if not event_files:
            print(f"No event files found for {run_id}")
            continue

        event_files.sort()
        ea = event_accumulator.EventAccumulator(event_files[-1])

        ea.Reload()

        p5 = None
        tag = "P@5"
        if tag in ea.scalars.Keys():
            events = ea.scalars.Items(tag)
            p5 = events[-1].value

        if p5 is None:
            print(f"No P@5 found for {run_id}")
            continue

        # 3. Write result
        line = f"{run_id}|{p5}\n"
        out.write(line)


