"""Set up this research run: use `%run setup.py` in Colab or Kaggle."""
from pathlib import Path
import importlib
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/'scripts'))
if __name__=='__main__':
    # %run shares the notebook kernel. Pick up a git pull without losing the
    # mounted Drive or downloading everything again in a restarted VM.
    importlib.invalidate_caches()
    for name in ['session_store','setup_progress','cloud_cli']:
        if name in sys.modules:importlib.reload(sys.modules[name])
    from cloud_cli import setup_main
    setup_main()
