import os
import sys

# Make the project's src/ importable as top-level modules (data_loader, etc.).
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
