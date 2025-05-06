#!/usr/bin/env python
"""
Setup script for Phase Dominance Analysis

This script creates a virtual environment and installs all necessary packages
for running the phase dominance analysis. It also checks if the phase_dominance.py
module is available and places it in the current directory if needed.

Usage:
    python setup_phase_analysis.py

Author: [Rob Scrutton rms222@cam.ac.uk]
Date: May 2, 2025
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path


def print_section(title: str) -> None:
    """Print a section title with formatting."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_python_version() -> bool:
    """Check if Python version is 3.7 or higher."""
    print_section("Checking Python version")
    
    if sys.version_info < (3, 7):
        print(f"❌ Python version {sys.version.split()[0]} detected.")
        print("This package requires Python 3.7 or higher.")
        return False
    
    print(f"✅ Python version {sys.version.split()[0]} detected.")
    return True


def install_requirements() -> bool:
    """Install required packages."""
    print_section("Installing required packages")
    
    # Define required packages
    required_packages = [
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "jupyterlab>=3.0.0",
        "ipywidgets>=7.6.0"
    ]
    
    # Check which packages are already installed
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    packages_to_install = []
    
    for package in required_packages:
        package_name = package.split('>=')[0]
        if package_name not in installed_packages:
            packages_to_install.append(package)
            print(f"Package {package_name} will be installed.")
        else:
            print(f"Package {package_name} is already installed (version {installed_packages[package_name]}).")
    
    # Install missing packages
    if packages_to_install:
        print("\nInstalling missing packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages_to_install)
            print("✅ All packages installed successfully.")
        except subprocess.CalledProcessError:
            print("❌ Failed to install some packages.")
            return False
    else:
        print("\n✅ All required packages are already installed.")
    
    return True


def check_phase_dominance_module() -> bool:
    """Check if the phase_dominance.py module exists and create it if needed."""
    print_section("Checking for phase_dominance module")
    
    module_path = Path("phase_dominance.py")
    
    if module_path.exists():
        print(f"✅ Found phase_dominance.py at {module_path.absolute()}")
        return True
    
    print(f"❌ Could not find phase_dominance.py")
    print("Please ensure the phase_dominance.py file is in the current directory.")
    print("You can download it from the repository or copy it from another location.")
    
    # Placeholder for auto-download functionality
    # This could be implemented to download the file from a repository
    
    return False


def check_notebook() -> bool:
    """Check if the analysis notebook exists and create it if needed."""
    print_section("Checking for analysis notebook")
    
    notebook_path = Path("phase_dominance_analysis.ipynb")
    
    if notebook_path.exists():
        print(f"✅ Found analysis notebook at {notebook_path.absolute()}")
        return True
    
    print(f"❌ Could not find analysis notebook")
    print("Please ensure the phase_dominance_analysis.ipynb file is in the current directory.")
    print("You can download it from the repository or copy it from another location.")
    
    # Placeholder for auto-download functionality
    
    return False


def create_test_data() -> bool:
    """Create a small test dataset if needed."""
    print_section("Creating test dataset")
    
    test_data_path = Path("test_phase_data.csv")
    
    if test_data_path.exists():
        print(f"✅ Test dataset already exists at {test_data_path.absolute()}")
        return True
    
    try:
        import numpy as np
        import pandas as pd
        
        print("Generating simulated phase separation data...")
        np.random.seed(42)
        
        n_points = 5000
        
        # Generate random concentrations for three components
        comp_A = np.random.uniform(0, 2, n_points)
        comp_B = np.random.uniform(10, 40, n_points)
        comp_C = np.random.uniform(120, 250, n_points)
        
        # Define a simple phase separation model
        phase_sep_prob = 1 / (1 + np.exp(-((comp_A - 1) * 5 + (comp_B - 25) * 0.2 - (comp_C - 180) * 0.05)))
        phase_separated = np.random.random(n_points) < phase_sep_prob
        
        # Calculate dilute phase concentrations
        dilute_factor_A = np.where(phase_separated, 
                                  0.5 + 0.3 * np.random.random(n_points) - 0.01 * comp_B, 
                                  0.95 + 0.05 * np.random.random(n_points))
        
        dilute_A = comp_A * np.clip(dilute_factor_A, 0.1, 0.99)
        normalized_dilute_A = dilute_A / comp_A
        
        # Create DataFrame
        df = pd.DataFrame({
            'comp_A_conc': comp_A,
            'comp_B_conc': comp_B,
            'comp_C_conc': comp_C,
            'PS': phase_separated,
            'dilute_A_conc': dilute_A,
            'Normalised dilute intensity': normalized_dilute_A,
            'Condensate label': 'DefaultCondensate',
            'continuous_frame': np.random.randint(0, 100, n_points)
        })
        
        # Add a 'Phase separated' column as a copy of 'PS'
        df['Phase separated'] = df['PS']
        
        # Save to CSV
        df.to_csv(test_data_path, index=False)
        print(f"✅ Test dataset created at {test_data_path.absolute()}")
        
    except ImportError:
        print("❌ Could not create test data: NumPy or Pandas not installed.")
        return False
    except Exception as e:
        print(f"❌ Failed to create test data: {str(e)}")
        return False
    
    return True


def launch_jupyter() -> None:
    """Ask if the user wants to launch JupyterLab."""
    print_section("Launch JupyterLab")
    
    try:
        response = input("Would you like to launch JupyterLab now? (y/n): ").strip().lower()
        
        if response in ('y', 'yes'):
            print("Launching JupyterLab...")
            subprocess.Popen([sys.executable, "-m", "jupyterlab"])
            print("JupyterLab is starting in a new process. Check your browser or terminal.")
        else:
            print("\nYou can start JupyterLab later with the command:")
            print("    jupyter lab")
    except Exception as e:
        print(f"Could not launch JupyterLab: {str(e)}")


def main() -> None:
    """Run all setup steps."""
    print_section("Phase Dominance Analysis Setup")
    
    # Check Python version
    if not check_python_version():
        print("Setup aborted due to Python version incompatibility.")
        return
    
    # Install required packages
    if not install_requirements():
        print("Setup aborted due to package installation failure.")
        return
    
    # Check for phase_dominance module
    module_exists = check_phase_dominance_module()
    
    # Check for analysis notebook
    notebook_exists = check_notebook()
    
    # Create test dataset
    data_created = create_test_data()
    
    # Summary
    print_section("Setup Summary")
    
    if module_exists and notebook_exists and data_created:
        print("✅ Setup completed successfully!")
        print("\nTo start the analysis:")
        print("1. Launch JupyterLab")
        print("2. Open phase_dominance_analysis.ipynb")
        print("3. Update the configuration section to match your data")
        print("4. Run the notebook cells")
        
        launch_jupyter()
    else:
        print("⚠️ Setup completed with some issues:")
        if not module_exists:
            print("- phase_dominance.py module is missing")
        if not notebook_exists:
            print("- analysis notebook is missing")
        if not data_created:
            print("- test data could not be created")
        
        print("\nPlease resolve these issues before running the analysis.")


if __name__ == "__main__":
    main()