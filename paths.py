"""Where the raw public datasets live: $ORGANFUSION_DATA if set, else ./datasets next to the scripts."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("ORGANFUSION_DATA", os.path.join(ROOT, "datasets"))
