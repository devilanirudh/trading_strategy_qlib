#!/usr/bin/env python3
"""
Simple startup script for Qlib Trading Dashboard
"""

import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import main

if __name__ == "__main__":
    main()
