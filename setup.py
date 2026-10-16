"""Set up this research run: use `%run setup.py` in Colab or Kaggle."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/'scripts'))
from cloud_cli import setup_main

if __name__=='__main__':setup_main()
