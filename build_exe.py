#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_exe.py

PyInstaller build script

Used to package integrated_script project as Windows executable
"""

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def build_exe():
    """Build executable file"""

    # Project root directory - directory where build_exe.py is located
    script_dir = Path(__file__).parent  # Directory where build_exe.py is located
    project_root = (
        script_dir  # Project root is the directory where build_exe.py is located
    )
    src_dir = script_dir / "src"
    main_script = script_dir / "main.py"

    # Ensure main script exists
    if not main_script.exists():
        print(f"Error: Main script not found {main_script}")
        return False

    # Clean previous builds
    build_dir = project_root / "build"

    # Only clean build directory, let PyInstaller handle dist directory automatically
    if build_dir.exists():
        try:
            shutil.rmtree(build_dir)
        except PermissionError:
            print("Warning: Cannot delete build directory, continuing...")

    if importlib.util.find_spec("PyInstaller") is None:
        print(
            "Error: PyInstaller is not available in the current Python interpreter."
            " Install it in this environment (for example `python -m pip install pyinstaller`)."
        )
        return False

    data_separator = ";" if os.name == "nt" else ":"
    # PyInstaller command
    cmd_parts = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",  # Package as single exe file
        "--console",  # Show console window
        "--name=integrated_script",  # Executable file name
        f"--distpath={project_root / 'dist'}",  # Specify output directory
        f"--workpath={project_root / 'build'}",  # Specify work directory
        f"--add-data={src_dir}{data_separator}src",  # Add source code directory
        f"--add-data={script_dir / 'requirements.txt'}{data_separator}.",  # Add requirements.txt
        f"--add-data={script_dir / 'config'}{data_separator}config",  # Add config directory
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
        str(main_script),
    ]

    # Execute PyInstaller command
    try:
        command_display = shlex.join(cmd_parts)
    except AttributeError:
        command_display = " ".join(shlex.quote(part) for part in cmd_parts)
    print(f"Executing command: {command_display}")
    try:
        result = subprocess.run(cmd_parts)
    except FileNotFoundError:
        print("Build failed: Python executable disappeared from PATH.")
        return False

    if result.returncode == 0:
        exe_name = "integrated_script.exe" if os.name == "nt" else "integrated_script"
        exe_path = project_root / "dist" / exe_name
        if exe_path.exists():
            print("\nBuild successful!")
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
