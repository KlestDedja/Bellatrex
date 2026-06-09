import os

import matplotlib

# Force Matplotlib to use the "Agg" backend, which is a non-interactive backend suitable for testing environments without a display server.
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)
