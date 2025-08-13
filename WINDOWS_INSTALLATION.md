# Windows Installation Guide for Stocks Research Platform

## 🚨 Common Windows Installation Issues

The error you encountered is a common Windows problem where packages like NumPy and pandas require C++ compilation tools that aren't available by default on Windows.

## 🛠️ Solution 1: Use the Windows Installation Script (Recommended)

Run the Windows-specific installation script:

```bash
python install_windows.py
```

This script will:
- Check your Python version
- Upgrade pip
- Install packages step by step
- Try alternative installation methods
- Provide specific Windows solutions

## 🛠️ Solution 2: Install from Pre-compiled Wheels

Force pip to use pre-compiled wheels instead of building from source:

```bash
pip install --only-binary=all -r requirements.txt
```

Or install packages individually:

```bash
pip install --only-binary=all numpy==1.24.3
pip install --only-binary=all pandas==1.5.3
pip install --only-binary=all scipy
pip install --only-binary=all scikit-learn
pip install yfinance matplotlib seaborn plotly streamlit requests python-dotenv
```

## 🛠️ Solution 3: Use Conda (Most Reliable for Windows)

If you have Anaconda or Miniconda installed, this is often the most reliable method:

### Install Miniconda (if you don't have it):
1. Download from: https://docs.conda.io/en/latest/miniconda.html
2. Choose the Windows 64-bit installer
3. Install with default settings

### Create and activate the environment:
```bash
conda env create -f environment.yml
conda activate stocks-research
```

### Or create manually:
```bash
conda create -n stocks-research python=3.9
conda activate stocks-research
conda install -c conda-forge pandas=1.5.3 numpy=1.24.3 matplotlib seaborn plotly scipy scikit-learn streamlit requests python-dotenv openpyxl
pip install yfinance
```

## 🛠️ Solution 4: Install Visual C++ Build Tools

If you want to build packages from source:

1. Download Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
2. Install "C++ build tools" workload
3. Restart your terminal
4. Try installing again: `pip install -r requirements.txt`

## 🛠️ Solution 5: Use Specific Windows-Compatible Versions

The platform now uses specific versions that are known to work well on Windows:

```bash
pip install numpy==1.24.3
pip install pandas==1.5.3
pip install scipy==1.10.1
pip install scikit-learn==1.3.0
pip install matplotlib==3.7.1
pip install seaborn==0.12.2
pip install plotly==5.15.0
pip install streamlit==1.25.0
pip install yfinance==0.2.18
pip install requests==2.31.0
pip install python-dotenv==1.0.0
```

## 🔍 Troubleshooting Steps

### Step 1: Check Python Version
```bash
python --version
```
Ensure you have Python 3.8 or higher.

### Step 2: Upgrade pip
```bash
python -m pip install --upgrade pip
```

### Step 3: Install wheel
```bash
pip install wheel
```

### Step 4: Clear pip cache
```bash
pip cache purge
```

### Step 5: Try installation with verbose output
```bash
pip install -v numpy==1.24.3
```

## 📋 Package-by-Package Installation

If bulk installation fails, install packages one by one:

```bash
# Core packages (specific versions for Windows compatibility)
pip install numpy==1.24.3
pip install pandas==1.5.3
pip install scipy

# Visualization
pip install matplotlib
pip install seaborn
pip install plotly

# Machine learning
pip install scikit-learn

# Web interface
pip install streamlit

# Data fetching
pip install yfinance

# Utilities
pip install requests
pip install python-dotenv
pip install openpyxl
```

## 🐍 Alternative: Use Python from Microsoft Store

1. Uninstall current Python
2. Install Python from Microsoft Store
3. This version often has better Windows compatibility

## 🔧 Environment Variables

Ensure these environment variables are set:
- `PATH` includes Python and pip
- `PYTHONPATH` is set correctly
- No conflicting Python installations

## 📱 Alternative: Use Google Colab

If all else fails, you can run the platform on Google Colab:
1. Upload your Python files to Google Drive
2. Open Google Colab
3. Mount your Drive and run the analysis there

## ✅ Verification

After installation, verify everything works:

```bash
python -c "import numpy; import pandas; import yfinance; import matplotlib; import seaborn; import plotly; import scipy; import sklearn; import streamlit; print('All packages imported successfully!')"
```

## 🆘 Still Having Issues?

1. **Check Windows version**: Ensure you're on Windows 10/11
2. **Run as Administrator**: Try running terminal as administrator
3. **Antivirus**: Temporarily disable antivirus during installation
4. **Firewall**: Check if firewall is blocking pip
5. **Proxy**: If behind corporate proxy, configure pip accordingly

## 📞 Support

If none of these solutions work:
1. Check the error message carefully
2. Try the Windows installation script: `python install_windows.py`
3. Consider using conda instead of pip
4. Use Google Colab as a last resort

## 🎯 Quick Start After Installation

Once packages are installed:

```bash
# Run the platform
python quick_start.py

# Or start the web interface
streamlit run app.py

# Or run sample analysis
python sample_analysis.py
```

## 💡 Pro Tips for Windows

1. **Use conda**: Generally more reliable on Windows than pip
2. **Prefer wheels**: Always use `--only-binary=all` with pip
3. **Specific versions**: Use numpy==1.24.3 and pandas==1.5.3 for best compatibility
4. **Keep Python updated**: Use the latest stable Python version
5. **Virtual environments**: Use virtual environments to avoid conflicts
6. **Path management**: Ensure only one Python installation is in PATH

## 🔄 Version Compatibility

The platform now uses these specific versions for maximum Windows compatibility:
- **numpy**: 1.24.3 (stable, widely tested on Windows)
- **pandas**: 1.5.3 (last 1.x version, excellent Windows support)
- **Python**: 3.8+ (recommended: 3.9 or 3.10)

---

**Remember**: The Windows installation script (`install_windows.py`) is designed to handle most of these issues automatically. Start there! 