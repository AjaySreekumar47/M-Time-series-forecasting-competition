from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="m1-competition-forecasting",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Time series forecasting toolkit for M1 Competition datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/m1-competition-forecasting",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "statsmodels>=0.13.0",
        "scikit-learn>=1.0.0",
        "xlrd>=2.0.0",
        "openpyxl>=3.0.0",
        "jupyter>=1.0.0",
    ],
)
