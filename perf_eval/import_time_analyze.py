import re

# Adjust the path as needed
with open("import_times.log") as f:
    lines = f.readlines()

pattern = re.compile(r'import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(.*)')

entries = []

for line in lines:
    match = pattern.match(line)
    if match:
        self_time = int(match.group(1))
        cumulative = int(match.group(2))
        module = match.group(3)
        entries.append((cumulative, self_time, module))

# Sort by cumulative time (descending)
entries.sort(reverse=True)

for cumulative, self_time, module in entries:
    print(f"{cumulative:>8} µs | {self_time:>8} µs | {module}")