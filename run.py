#!/usr/bin/env python
"""
Murmur - Local Speech-to-Text Hotkey App
Run this script to start Murmur.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.main import main

if __name__ == "__main__":
    main()
