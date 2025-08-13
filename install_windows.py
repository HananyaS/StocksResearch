#!/usr/bin/env python3
"""
Windows Installation Script for Stocks Research Platform
This script handles common Windows-specific installation issues.
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    print("✅ Python version is compatible")
    return True

def check_pip():
    """Check if pip is available and working."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ pip is not available")
        return False

def upgrade_pip():
    """Upgrade pip to latest version."""
    print("🔄 Upgrading pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True)
        print("✅ pip upgraded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to upgrade pip: {e}")
        return False

def install_wheel():
    """Install wheel package for better package compatibility."""
    print("🔄 Installing wheel...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "wheel"], 
                      check=True)
        print("✅ wheel installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install wheel: {e}")
        return False

def install_packages_step_by_step():
    """Install packages one by one to identify issues."""
    
    packages = [
        ("numpy", "numpy==1.24.3"),
        ("pandas", "pandas==1.5.3"),
        ("yfinance", "yfinance>=0.2.28"),
        ("matplotlib", "matplotlib>=3.7.0"),
        ("seaborn", "seaborn>=0.12.0"),
        ("plotly", "plotly>=5.17.0"),
        ("scipy", "scipy>=1.11.0"),
        ("scikit-learn", "scikit-learn>=1.3.0"),
        ("streamlit", "streamlit>=1.28.0"),
        ("requests", "requests>=2.31.0"),
        ("python-dotenv", "python-dotenv>=1.0.0")
    ]
    
    failed_packages = []
    
    for package_name, package_spec in packages:
        print(f"\n🔄 Installing {package_name}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package_spec], 
                          check=True)
            print(f"✅ {package_name} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package_name}: {e}")
            failed_packages.append((package_name, package_spec))
    
    return failed_packages

def try_alternative_installation():
    """Try alternative installation methods for failed packages."""
    
    print("\n🔄 Trying alternative installation methods...")
    
    # Try installing from wheels
    try:
        print("🔄 Installing numpy from wheel...")
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "--only-binary=all", "numpy==1.24.3"], check=True)
        print("✅ numpy installed from wheel")
    except:
        print("❌ Failed to install numpy from wheel")
    
    try:
        print("🔄 Installing pandas from wheel...")
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "--only-binary=all", "pandas==1.5.3"], check=True)
        print("✅ pandas installed from wheel")
    except:
        print("❌ Failed to install pandas from wheel")

def install_visual_cpp_redistributable():
    """Provide instructions for installing Visual C++ Redistributable."""
    print("\n📋 VISUAL C++ REDISTRIBUTABLE INSTALLATION")
    print("=" * 50)
    print("If you're still having issues, you may need to install:")
    print("Microsoft Visual C++ Redistributable for Visual Studio 2015-2022")
    print("\nDownload from:")
    print("https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("\nAfter installing, restart your terminal and try again.")

def check_installation():
    """Check if all required packages are installed."""
    print("\n🔍 Checking installation...")
    
    required_packages = [
        'numpy', 'pandas', 'yfinance', 'matplotlib', 
        'seaborn', 'plotly', 'scipy', 'sklearn', 'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("\n🎉 All packages installed successfully!")
        return True

def main():
    """Main installation function."""
    
    print("🚀 STOCKS RESEARCH PLATFORM - WINDOWS INSTALLATION")
    print("=" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    
    # Check prerequisites
    if not check_python_version():
        return
    
    if not check_pip():
        print("❌ Please install pip first")
        return
    
    # Upgrade pip
    upgrade_pip()
    
    # Install wheel
    install_wheel()
    
    # Try step-by-step installation
    print("\n📦 Installing packages step by step...")
    failed_packages = install_packages_step_by_step()
    
    if failed_packages:
        print(f"\n⚠️  {len(failed_packages)} packages failed to install:")
        for package_name, _ in failed_packages:
            print(f"   - {package_name}")
        
        # Try alternative methods
        try_alternative_installation()
        
        # Check if we can install openpyxl for Excel support
        try:
            print("\n🔄 Installing openpyxl for Excel export...")
            subprocess.run([sys.executable, "-m", "pip", "install", "openpyxl"], 
                          check=True)
            print("✅ openpyxl installed successfully")
        except:
            print("⚠️  openpyxl installation failed - Excel export will not be available")
    
    # Final check
    if check_installation():
        print("\n🎉 Installation completed successfully!")
        print("\nYou can now run:")
        print("   python quick_start.py")
        print("   streamlit run app.py")
    else:
        print("\n❌ Some packages failed to install")
        install_visual_cpp_redistributable()
        
        print("\n💡 Alternative solutions:")
        print("1. Try installing packages individually:")
        print("   pip install numpy==1.24.3")
        print("   pip install pandas==1.5.3")
        print("   pip install yfinance")
        print("\n2. Use conda instead of pip:")
        print("   conda env create -f environment.yml")
        print("   conda activate stocks-research")
        print("\n3. Install from wheels:")
        print("   pip install --only-binary=all numpy==1.24.3 pandas==1.5.3")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Installation interrupted by user")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("💡 Please try the alternative solutions mentioned above") 