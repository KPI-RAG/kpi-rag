"""Small helpers: per-step output folder, logging to screen + file, JSON saving."""
import json
import os
import platform
import sys
from datetime import datetime, timezone

from . import config as C


class StepLog:
    """Prints and records every line; writes <results>/<step>/log.txt on close."""

    def __init__(self, step):
        self.dir = os.path.join(C.RESULTS, step)
        os.makedirs(self.dir, exist_ok=True)
        self.lines = []
        self(f"# {step}  |  {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}  |  "
             f"python {sys.version.split()[0]} on {platform.node()}")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def path(self, name):
        return os.path.join(self.dir, name)

    def save_json(self, name, obj):
        with open(self.path(name), "w") as f:
            json.dump(obj, f, indent=2)
        self(f"saved {os.path.relpath(self.path(name), C.PHASE2)}")

    def close(self):
        with open(self.path("log.txt"), "w") as f:
            f.write("\n".join(self.lines) + "\n")


def library_versions():
    import catboost, numpy, pandas, scipy, sklearn
    return {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "pandas": pandas.__version__, "scikit-learn": sklearn.__version__,
            "catboost": catboost.__version__, "scipy": scipy.__version__}
