#!/usr/bin/env python3
"""
Development startup script for Akasha.

This script provides a convenient way to start the Akasha development server
with proper configuration and logging.
"""

import sys
import subprocess
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def main():
    """Start the development server."""
    print("🚀 Starting Akasha development server...")
    print(f"📁 Project root: {project_root}")
    
    # Check if virtual environment is activated
    if sys.prefix == sys.base_prefix:
        print("⚠️  Warning: Virtual environment not detected")
        print("   Consider activating the virtual environment first:")
        print(f"   source {project_root}/venv/bin/activate")
        print()
    
    try:
        # Import and run the main application
        from src.api.main import main as run_app
        run_app()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure dependencies are installed:")
        print("   pip install -e .")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Akasha server...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()