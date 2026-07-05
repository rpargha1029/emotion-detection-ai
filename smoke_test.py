# smoke_test.py — place at C:\Project\emotion-detection-ai\smoke_test.py
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
print("ROOT:", ROOT)
print("Python executable:", sys.executable)
print("CWD:", os.getcwd())
print("sys.path[0]:", sys.path[0])

# Check src import
try:
    import src
    print("import src OK")
except Exception as e:
    print("import src FAILED:", e)

# Check dataset folders
print("data/splits/train exists:", (ROOT / "data" / "splits" / "train").exists())
print("data/splits/val exists:", (ROOT / "data" / "splits" / "val").exists())
print("data/splits/test exists:", (ROOT / "data" / "splits" / "test").exists())
