import os
import subprocess
import re

# ---- CONFIG ----
script_to_run = "frontend/streamlit_app.py"  # Replace with your main script
output_dir = "perf_eval"
log_file = os.path.join(output_dir, "import_times.log")

# ---- STEP 1: Run with `-X importtime` and save output ----
print(f"Running: python -X importtime {script_to_run}")
with open(log_file, "w", encoding="utf-8") as f:
    subprocess.run(["python", "-X", "importtime", script_to_run], stdout=f, stderr=subprocess.STDOUT)

print(f"\n[✓] Import timing log saved to: {log_file}")

# ---- STEP 2: Parse and sort the import timings ----
pattern = re.compile(r'import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(.*)')

entries = []

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.match(line)
        if match:
            self_time = int(match.group(1))
            cumulative = int(match.group(2))
            module = match.group(3)
            entries.append((cumulative, self_time, module))

# Sort by cumulative time descending
entries.sort(reverse=True)

# ---- STEP 3: Print top slowest imports ----
print("\nTop slowest imports by cumulative time:\n")
print(f"{'Cumulative (µs)':>16} | {'Self (µs)':>10} | Module")
print("-" * 50)
for cumulative, self_time, module in entries[:25]:  # Show top 25
    print(f"{cumulative:>16} | {self_time:>10} | {module}")
