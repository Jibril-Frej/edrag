from setuptools import setup, find_packages

setup(
    name="edrag",
    version="0.1.0",
    author="Jibril Frej",
    author_email="jibril.frej@epfl.ch",
    description="A simple package for RAG in the educational context",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Jibril-Frej/edrag",
    packages=find_packages(),
    install_requires=[
        "torch>=2.5.1",
        "transformers>=4.46.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
