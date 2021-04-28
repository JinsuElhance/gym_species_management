from setuptools import setup, find_packages

setup(name="gym_species_management",
    version='0.0.1',
    packages = find_packages(),
    install_requires=["gym", "numpy", "pandas", "matplotlib", "stable-baselines3", "requests"])
