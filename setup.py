"""Setup script to install as a package."""
from setuptools import setup, find_packages

setup(
    name='drug_recommender',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'scikit-learn',
        'tqdm',
        'chembl_webresource_client',
    ],
)