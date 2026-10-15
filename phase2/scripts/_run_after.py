"""Server-side job chain: wait until no process runs <pattern>, then start <script> detached.

    python scripts/_run_after.py <pattern-of-running-script> <script-to-run-next>
Used to run jobs one at a time (16 GB container) without a watcher on the laptop.
"""
import os
import subprocess
import sys
import time

pattern, nxt = sys.argv[1], sys.argv[2]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pattern.startswith("file:"):
    # wait until a result file exists (e.g. file:results/10_cascade_reject/results.json)
    target = os.path.join(root, pattern[5:])
    while not os.path.exists(target):
        time.sleep(60)
else:
    # "[x]yz" matches "xyz" in other processes' command lines but not this script's own argv
    # nor the shell that runs pgrep (which both contain the literal "[x]yz").
    safe = f"[{pattern[0]}]{pattern[1:]}"
    me = str(os.getpid())
    def running():
        pids = subprocess.run(f"pgrep -f '{safe}'", shell=True, capture_output=True, text=True).stdout.split()
        return [p for p in pids if p != me]          # this script's own argv also contains the pattern
    while running():
        time.sleep(60)
os.makedirs(os.path.join(root, "results", "_runs"), exist_ok=True)
name = os.path.splitext(os.path.basename(nxt))[0]
out = open(os.path.join(root, "results", "_runs", f"{name}.out"), "w")
p = subprocess.Popen([sys.executable, "-u", nxt], cwd=root, stdout=out, stderr=subprocess.STDOUT,
                     start_new_session=True)
print(f"started {nxt} pid {p.pid}")
