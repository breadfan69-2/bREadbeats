#!/usr/bin/env python3
"""
bREadbeats - Audio Beat Detection to restim

A real-time audio beat detector that sends stroke commands to restim
via TCP using T-code format.

Usage:
    python run.py

Requirements:
    pip install -r requirements.txt

Author: bREadbeats
"""

from main import main

if __name__ == "__main__":
    main()
