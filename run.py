#!/usr/bin/env python3
"""
bREadbeats - Audio Beat Detection to restim

A real-time audio beat detector that sends stroke commands to restim
via TCP using T-code format.
"""

import argparse
import cProfile
import sys
import time
from pathlib import Path

# Import ONLY PyQt6 essentials first for splash screen (fast)
t_pyqt = time.perf_counter()
from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

print(
    f"[Startup] GUI framework loaded (+{(time.perf_counter() - t_pyqt) * 1000:.0f} ms). "
    "Initializing application...",
    flush=True,
)


def run_app(app_argv: list[str]) -> int:
    app = QApplication(app_argv)
    app.setStyle("Fusion")

    print("[Startup] Displaying splash screen...", flush=True)

    # Show splash screen immediately before any heavy imports
    if getattr(sys, "frozen", False):
        resource_dir = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        resource_dir = Path(__file__).parent

    splash_path = resource_dir / "splash_screen.png"
    splash = None
    if splash_path.exists():
        pixmap = QPixmap(str(splash_path))
        splash = QSplashScreen(pixmap)
        splash.show()
        splash.showMessage(
            "Loading core modules...",
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            Qt.GlobalColor.white,
        )
        app.processEvents()  # Force display update

    print("[Startup] Loading audio engine and processing modules...", flush=True)
    if splash:
        splash.showMessage(
            "Loading audio engine and processing modules...",
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            Qt.GlobalColor.white,
        )
        app.processEvents()
    t_main = time.perf_counter()

    # Import heavy modules (numpy, scipy, pyqtgraph, etc.) after splash
    from main import BREadbeatsWindow

    print(
        f"[Startup] Loaded main module (+{(time.perf_counter() - t_main) * 1000:.0f} ms)",
        flush=True,
    )
    if splash:
        splash.showMessage(
            "Initializing UI...",
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            Qt.GlobalColor.white,
        )
        app.processEvents()

    print("[Startup] Creating main window...", flush=True)

    # Create main window
    window = BREadbeatsWindow()
    if splash:
        splash.showMessage(
            "Starting services...",
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            Qt.GlobalColor.white,
        )
        app.processEvents()

    print("\nInitialization complete. Starting GUI...\n", flush=True)

    # Close splash and show main window
    if splash:
        splash.finish(window)

    window.show()

    return app.exec()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bREadbeats")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile and save stats to --profile-out",
    )
    parser.add_argument(
        "--profile-out",
        default="profile.prof",
        help="Path to save cProfile stats (default: profile.prof)",
    )
    args = parser.parse_args()

    # Keep Qt argument list clean; avoid passing profiling flags downstream
    app_argv = [sys.argv[0]]

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        exit_code = run_app(app_argv)
        profiler.disable()
        profiler.dump_stats(args.profile_out)
    else:
        exit_code = run_app(app_argv)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
