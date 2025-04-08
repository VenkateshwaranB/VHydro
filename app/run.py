#!/usr/bin/env python3
"""
VHydro - Hydrocarbon Potential Prediction
Run script for starting the Streamlit application
"""

import os
import sys
import subprocess
import time
import webbrowser
import argparse

def check_requirements():
    """Check if all required dependencies are installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import matplotlib
        import plotly
        import sklearn
        import networkx
        try:
            import stellargraph
        except ImportError:
            print("Warning: StellarGraph is not installed. Installing from GitHub...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/VenkateshwaranB/stellargraph.git"
            ])
        return True
    except ImportError as e:
        print(f"Error: Missing required dependency - {str(e)}")
        print("Please install all required dependencies by running: pip install -r requirements.txt")
        return False

def start_app(port=8501, no_browser=False):
    """Start the Streamlit application"""
    # Check if app.py exists
    if not os.path.exists("app.py"):
        print("Error: app.py not found in the current directory")
        return False
    
    # Start Streamlit server
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port)]
    
    if no_browser:
        cmd.append("--server.headless")
    
    print(f"Starting VHydro application on port {port}...")
    print(f"URL: http://localhost:{port}")
    
    # Open browser if requested
    if not no_browser:
        webbrowser.open(f"http://localhost:{port}")
    
    # Execute Streamlit
    process = subprocess.Popen(cmd)
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down VHydro application...")
        process.terminate()
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run VHydro Hydrocarbon Potential Prediction Application")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the application on (default: 8501)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open a browser window automatically")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies without starting the app")
    
    args = parser.parse_args()
    
    # Check requirements
    if not check_requirements():
        return 1
    
    if args.check_only:
        print("All required dependencies are installed.")
        return 0
    
    # Start application
    if not start_app(port=args.port, no_browser=args.no_browser):
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())