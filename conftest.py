import sys
from pathlib import Path

# Put the project root on the path so tests can import config / src.*
sys.path.insert(0, str(Path(__file__).resolve().parent))
