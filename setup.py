from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    dependencies = f.read().splitlines()

setup(
    name="interactive-summarization-model",
    version="0.0.1",
    author="Media Technology Center (ETH Zürich)",
    author_email="mtc@ethz.ch",
    description="A python implementation of Interactive Summarization based on Hugging Face Transformers",
    packages=find_packages(),
    install_requires=dependencies,
)
