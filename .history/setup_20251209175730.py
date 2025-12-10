"""
Setup script for Universal Embedding Representation (UER)
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="uer",
    version="0.1.0",
    author="UER Contributors",
    author_email="uer@domain.com",
    description="Universal Embedding Representation - Open Source Standard",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/universal-embedding-ir",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pyyaml>=5.1",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "test": ["pytest>=6.0", "pytest-cov"],
        "dev": ["black", "isort", "flake8"],
    },
    include_package_data=True,
    zip_safe=False,
)
