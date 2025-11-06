"""
Check system requirements for PyTorch on Windows.
Run with: python check_system.py
"""
import os
import sys
import subprocess
import ctypes
from pathlib import Path


def check_vcredist():
    """Check if Visual C++ Redistributable is installed via registry."""
    try:
        # Common paths for VC++ Redist DLLs
        dll_paths = [
            r"C:\Windows\System32\vcruntime140.dll",
            r"C:\Windows\System32\vcruntime140_1.dll",
            r"C:\Windows\System32\msvcp140.dll",
        ]
        missing = []
        for path in dll_paths:
            if not os.path.exists(path):
                missing.append(path)
        
        if missing:
            print("WARNING: Missing Visual C++ Redistributable DLLs:", missing)
            print("\nPlease install the latest Visual C++ Redistributable from:")
            print("https://aka.ms/vs/17/release/vc_redist.x64.exe")
            return False
        print("✓ Visual C++ Redistributable DLLs found")
        return True
    except Exception as e:
        print(f"Error checking VC++ Redist: {e}")
        return False


def check_path_issues():
    """Check for common PATH-related issues that can affect DLL loading."""
    try:
        # Check if we're in a virtualenv
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        if in_venv:
            venv_path = Path(sys.prefix)
            scripts_dir = venv_path / 'Scripts'
            if str(scripts_dir) not in os.environ.get('PATH', ''):
                print(f"WARNING: virtualenv Scripts directory not in PATH: {scripts_dir}")
                return False
        print("✓ PATH environment looks correct")
        return True
    except Exception as e:
        print(f"Error checking PATH: {e}")
        return False


def main():
    print("Checking system requirements for PyTorch on Windows...\n")
    
    all_ok = True
    
    # Check Python version
    print(f"Python {sys.version}")
    if sys.version_info < (3, 8):
        print("WARNING: Python 3.8 or newer recommended for PyTorch")
        all_ok = False
    
    # Check if running as admin (some DLL issues require admin rights to diagnose)
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        if not is_admin:
            print("\nNote: Not running as administrator (some checks may be limited)")
    except:
        print("\nCouldn't determine admin status")
    
    # Essential checks
    print("\nRunning essential checks...")
    if not check_vcredist():
        all_ok = False
    if not check_path_issues():
        all_ok = False
    
    # Summary
    print("\nSummary:")
    if all_ok:
        print("✓ All basic checks passed")
        print("\nIf you still have issues:")
        print("1. Try the CPU-only build:")
        print("   pip install --index-url https://download.pytorch.org/whl/cpu torch")
        print("2. Or use conda (recommended for GPU support):")
        print("   conda install pytorch cpuonly -c pytorch")
    else:
        print("⚠ Some checks failed - please address the warnings above")
        print("\nAfter fixing the issues, try installing PyTorch via:")
        print("1. CPU only (pip):")
        print("   pip install --index-url https://download.pytorch.org/whl/cpu torch")
        print("2. Or with conda (recommended):")
        print("   conda create -n torch_env python=3.10")
        print("   conda activate torch_env")
        print("   conda install pytorch cpuonly -c pytorch")


if __name__ == '__main__':
    main()