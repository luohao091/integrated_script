#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_exe.py

PyInstaller build script

Used to package integrated_script project as Windows executable
"""

import os
import sys
import shutil
from pathlib import Path

def build_exe():
    """Build executable file"""
    
    # Project root directory - directory where build_exe.py is located
    script_dir = Path(__file__).parent  # Directory where build_exe.py is located
    project_root = script_dir  # Project root is the directory where build_exe.py is located
    src_dir = script_dir / "src"
    main_script = script_dir / "main.py"
    
    # Ensure main script exists
    if not main_script.exists():
        print(f"Error: Main script not found {main_script}")
        return False
    
    # Clean previous builds
    build_dir = project_root / "build"
    dist_dir = project_root / "dist"
    
    # Only clean build directory, let PyInstaller handle dist directory automatically
    if build_dir.exists():
        try:
            shutil.rmtree(build_dir)
        except PermissionError:
            print("Warning: Cannot delete build directory, continuing...")
    
    # PyInstaller command
    cmd_parts = [
        "pyinstaller",
        "--onefile",  # Package as single exe file
        "--console",  # Show console window
        "--name=integrated_script",  # Executable file name
        f"--distpath={project_root / 'dist'}",  # Specify output directory to project root
        f"--workpath={project_root / 'build'}",  # Specify work directory to project root
        f"--add-data={src_dir};src",  # Add source code directory
        f"--add-data={script_dir / 'requirements.txt'};.",  # Add requirements.txt
        f"--add-data={script_dir / 'config'};config",  # Add config directory
        "--hidden-import=integrated_script",
        "--hidden-import=integrated_script.config",
        "--hidden-import=integrated_script.core",
        "--hidden-import=integrated_script.processors",
        "--hidden-import=integrated_script.ui",
        "--hidden-import=PIL",
        "--hidden-import=cv2",
        "--hidden-import=yaml",
        "--hidden-import=tqdm",
        "--hidden-import=logging.handlers",
        "--hidden-import=logging.config",
        "--hidden-import=argparse",
        "--hidden-import=pathlib",
        "--hidden-import=json",
        "--hidden-import=datetime",
        "--hidden-import=typing",
        "--collect-all=integrated_script",
        str(main_script)
    ]
    
    # Execute PyInstaller command
    cmd = " ".join(cmd_parts)
    print(f"Executing command: {cmd}")
    
    result = os.system(cmd)
    
    if result == 0:
        exe_path = project_root / "dist" / "integrated_script.exe"
        if exe_path.exists():
            print(f"\nBuild successful!")
            print(f"Executable location: {exe_path}")
            print(f"File size: {exe_path.stat().st_size / 1024 / 1024:.1f} MB")
            return True
        else:
            print("Build failed: Generated exe file not found")
            print(f"Expected location: {exe_path}")
            return False
    else:
        print("Build failed: PyInstaller execution error")
        return False

if __name__ == "__main__":
    print("Building integrated_script project...")
    success = build_exe()
    if success:
        print("\nBuild completed successfully!")
    else:
        print("\nBuild failed!")
        sys.exit(1)