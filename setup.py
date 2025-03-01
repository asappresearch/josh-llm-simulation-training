from typing import List

from setuptools import PEP420PackageFinder, setup


def get_requires(requires_filename: str) -> List[str]:
    requirements = []
    with open(requires_filename, "r") as infile:
        for line in infile.readlines():
            line = line.strip()
            requirements.append(line)
    return requirements


setup(
    name="josh_train",
    description="A simulation training framework",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    author="blattimer",
    author_email="blattimer@asapp.com",
    url="https://github.com/josh-llm-simulation-training",
    packages=PEP420PackageFinder.find(exclude=("test*",)),
    python_requires=">=3.8",
    install_requires=get_requires("requirements.txt"),
    include_package_data=True,
    setup_requires=["setuptools_scm"],
    version="0.1.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
