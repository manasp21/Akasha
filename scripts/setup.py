#!/usr/bin/env python3
"""
Setup script for Akasha development environment.

This script sets up the complete development environment including
virtual environment, dependencies, and configuration files.
"""

import os
import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"   ✓ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr.strip()}")
        return False

def create_venv():
    """Create virtual environment."""
    venv_path = project_root / "venv"
    if venv_path.exists():
        print("📦 Virtual environment already exists")
        return True
    
    return run_command(
        f"cd {project_root} && python3 -m venv venv",
        "Creating virtual environment"
    )

def install_dependencies():
    """Install project dependencies."""
    venv_python = project_root / "venv" / "bin" / "python"
    
    # Install core dependencies
    success = run_command(
        f"{venv_python} -m pip install --upgrade pip",
        "Upgrading pip"
    )
    
    if success:
        success = run_command(
            f"{venv_python} -m pip install -e .[dev]",
            "Installing project dependencies"
        )
    
    return success

def create_config_files():
    """Create basic configuration files."""
    config_dir = project_root / "config"
    config_dir.mkdir(exist_ok=True)
    
    # Create basic config if it doesn't exist
    config_file = config_dir / "akasha.yaml"
    if not config_file.exists():
        config_content = """# Akasha Configuration
system:
  name: akasha-dev
  environment: development
  debug: true
  max_memory_gb: 40

api:
  host: "127.0.0.1"
  port: 8000
  reload: true

llm:
  backend: mlx
  model_name: gemma-3-27b
  memory_limit_gb: 16

logging:
  level: DEBUG
  output: console
"""
        config_file.write_text(config_content)
        print(f"📝 Created configuration file: {config_file}")
    
    # Create .env file if it doesn't exist
    env_file = project_root / ".env"
    if not env_file.exists():
        env_content = """# Akasha Environment Variables
AKASHA_CONFIG=./config/akasha.yaml
AKASHA_LOG_LEVEL=DEBUG
"""
        env_file.write_text(env_content)
        print(f"📝 Created environment file: {env_file}")
    
    return True

def main():
    """Main setup function."""
    print("🔧 Setting up Akasha development environment")
    print(f"📁 Project root: {project_root}")
    print()
    
    steps = [
        ("Creating virtual environment", create_venv),
        ("Installing dependencies", install_dependencies),
        ("Creating configuration files", create_config_files),
    ]
    
    for description, func in steps:
        if not func():
            print(f"❌ Setup failed at: {description}")
            sys.exit(1)
        print()
    
    print("✅ Setup completed successfully!")
    print()
    print("🚀 To start the development server:")
    print("   source venv/bin/activate")
    print("   python scripts/dev.py")
    print()
    print("🧪 To run tests:")
    print("   source venv/bin/activate")
    print("   pytest tests/")

if __name__ == "__main__":
    main()