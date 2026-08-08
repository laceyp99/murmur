#!/usr/bin/env python
"""
Murmur - Local Speech-to-Text Hotkey App
Run this script to start Murmur.
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import main

if __name__ == "__main__":
    main()
