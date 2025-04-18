#!/usr/bin/env python
import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible (3.8 or higher)"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")

def check_pip():
    """Check if pip is installed"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✓ pip is installed")
        return True
    except subprocess.CalledProcessError:
        print("Error: pip is not installed")
        return False

def create_virtual_environment():
    """Create a virtual environment"""
    venv_dir = "venv"
    if os.path.exists(venv_dir):
        print(f"Virtual environment directory '{venv_dir}' already exists")
        return venv_dir
    
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_dir], 
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✓ Created virtual environment in '{venv_dir}'")
        return venv_dir
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return None

def install_requirements(venv_dir):
    """Install required packages from requirements.txt"""
    if not os.path.exists("requirements.txt"):
        print("Error: requirements.txt not found")
        return False
    
    # Determine the pip path based on the virtual environment
    if os.name == 'nt':  # Windows
        pip_path = os.path.join(venv_dir, "Scripts", "pip")
    else:  # Unix/Mac
        pip_path =
