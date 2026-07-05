from setuptools import setup, find_packages

setup(
    name="mlb-stats",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "pandas",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "mlb-stats=mlb_stats.cli:main",
        ],
    },
)
