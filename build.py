"""
Build script for bREadbeats with automatic version handling
"""
import subprocess
import sys
import os
from pathlib import Path
from version import get_version_info


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"[BUILD] {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"[BUILD] ✓ {description} completed")
        return result
    except subprocess.CalledProcessError as e:
        print(f"[BUILD] ✗ {description} failed:")
        print(f"  Command: {e.cmd}")
        print(f"  Exit code: {e.returncode}")
        if e.stdout:
            print(f"  Stdout: {e.stdout}")
        if e.stderr:
            print(f"  Stderr: {e.stderr}")
        sys.exit(1)


def main():
    """Main build function"""
    print("bREadbeats Build Script")
    print("=" * 50)
    
    # Show version info
    version_info = get_version_info()
    print(f"Version: {version_info['version']}")
    print(f"Source: {version_info['source']}")
    print(f"Git available: {version_info['git_available']}")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('bREadbeats.spec'):
        print("Error: bREadbeats.spec not found. Run from project root.")
        sys.exit(1)
    
    # Clean previous build
    if os.path.exists('dist'):
        run_command('rmdir /s /q dist', "Cleaning dist directory")
    if os.path.exists('build'): 
        run_command('rmdir /s /q build', "Cleaning build directory")
    
    # Run PyInstaller
    run_command('.venv/Scripts/pyinstaller.exe bREadbeats.spec', "Building with PyInstaller")
    
    # Check if the build was successful
    exe_path = Path('dist/bREadbeats.exe')
    if exe_path.exists():
        exe_size = exe_path.stat().st_size / 1024 / 1024  # MB
        print(f"[BUILD] ✓ Build successful!")
        print(f"[BUILD] Executable: {exe_path}")
        print(f"[BUILD] Size: {exe_size:.1f} MB")
        print(f"[BUILD] Version: {version_info['version']}")
    else:
        print("[BUILD] ✗ Build failed - executable not found")
        sys.exit(1)


if __name__ == "__main__":
    main()