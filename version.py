"""
Version management for bREadbeats
Automatically extracts version from git tags or falls back to default
"""
import os
import subprocess
import sys
from typing import Optional


def get_version() -> str:
    """
    Get application version from git tags or fallback to default.
    
    Returns version in format: v2.1.0 or v3.0 (fallback)
    """
    version = get_git_version()
    if version:
        return version
    
    # Fallback version when git is not available or no tags exist
    return "v3.0"


def get_git_version() -> Optional[str]:
    """
    Extract version from git tags.
    
    Returns the current tag if on a tagged commit,
    or tag-commits-hash if on a development commit.
    """
    try:
        # Check if we're in a git repository
        if not os.path.exists('.git') and not os.environ.get('GIT_DIR'):
            return None
            
        # Get the most recent tag and commit info
        result = subprocess.run(
            ['git', 'describe', '--tags', '--always', '--dirty'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            tag = result.stdout.strip()
            
            # If tag doesn't start with 'v', add it
            if tag and not tag.startswith('v'):
                tag = f'v{tag}'
                
            return tag
            
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        # Git not available or command failed
        pass
    
    return None


def get_version_info() -> dict:
    """
    Get detailed version information for debugging.
    
    Returns:
        dict: Version info including source, git available, etc.
    """
    git_version = get_git_version()
    final_version = get_version()
    
    return {
        'version': final_version,
        'source': 'git' if git_version else 'fallback',
        'git_version': git_version,
        'git_available': git_version is not None,
    }


# Cache the version to avoid repeated git calls
__version__ = get_version()


if __name__ == "__main__":
    # Allow running as script to check version
    import json
    print(json.dumps(get_version_info(), indent=2))