from setuptools import setup, find_packages

setup(
    name="persevera_arbitrage",
    version="0.5.16",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "statsmodels",
        "sklearn",
        "scipy"
    ],
)