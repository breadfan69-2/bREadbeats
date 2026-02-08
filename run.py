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

import sys
import time
import threading

t0 = time.perf_counter()

from pathlib import Path

# Print loading message immediately
print("\n" + "="*60)
print("bREadbeats - Audio Beat Detection & TCode Generator")
print("="*60)
print("Please wait, loading BeatTracker modules....")
print("="*60)
print("[Startup] Preparing GUI framework...")
print("="*60 + "\n", flush=True)

# Import ONLY PyQt6 essentials first for splash screen (fast)
t_pyqt = time.perf_counter()
from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
print(f"[Startup] GUI framework loaded (+{(time.perf_counter()-t_pyqt)*1000:.0f} ms). Initializing application...", flush=True)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    print("[Startup] Displaying splash screen...", flush=True)
    
    # Show splash screen IMMEDIATELY before any heavy imports
    if getattr(sys, 'frozen', False):
        resource_dir = Path(sys._MEIPASS)
    else:
        resource_dir = Path(__file__).parent
    
    splash_path = resource_dir / 'splash_screen.png'
    splash = None
    if splash_path.exists():
        pixmap = QPixmap(str(splash_path))
        splash = QSplashScreen(pixmap)
        splash.show()
        app.processEvents()  # Force display update
    
    # Show banner only after splash is visible to avoid an early "stuck" console
    print("\n" + "=" * 60)
    print("bREadbeats - Audio Beat Detection & TCode Generator")
    print("=" * 60)
    print("Please wait, loading BeatTracker modules....")
    print("=" * 60 + "\n", flush=True)

    # Progress dots while heavy imports load (prevents dead console feel)
    progress_stop = threading.Event()

    def _progress_worker():
        while not progress_stop.is_set():
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(0.3)

    def _start_progress():
        progress_stop.clear()
        t = threading.Thread(target=_progress_worker, daemon=True)
        t.start()
        return t

    def _stop_progress():
        progress_stop.set()
        sys.stdout.write('\n')
        sys.stdout.flush()

    print("[Startup] Loading audio engine and processing modules...", flush=True)
    t_main = time.perf_counter()
    _start_progress()
    
    # NOW import heavy modules (numpy, scipy, pyqtgraph, etc.)
    from main import BREadbeatsWindow
    _stop_progress()
    print(f"[Startup] Loaded main module (+{(time.perf_counter()-t_main)*1000:.0f} ms)", flush=True)

    print("[Startup] Creating main window...", flush=True)
    
    # Create main window
    window = BREadbeatsWindow()
    
    print("\nInitialization complete. Starting GUI...\n", flush=True)
    
    # Close splash and show main window
    if splash:
        splash.finish(window)
    
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
