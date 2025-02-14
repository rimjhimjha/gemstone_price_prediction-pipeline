from setuptools import setup, find_packages

setup(
    name="mlops_project",
    version="0.1",
    author="Rimjhim",
    description="An MLOps project setup with necessary dependencies.",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "numpy",
        "seaborn",
        "flask",
        "mlflow==2.2.2",
        "dvc",
        "pytest==7.1.3",
        "tox==3.25.1",
        "black==22.8.0",
        "flake8==5.0.4",
        "mypy==0.971"
    ],
)
