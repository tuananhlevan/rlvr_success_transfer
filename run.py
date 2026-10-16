"""Start or resume the configured run: `python run.py`."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/'scripts'))
from cloud_cli import run_main

if __name__=='__main__':run_main()
