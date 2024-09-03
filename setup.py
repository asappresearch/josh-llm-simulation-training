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
    name="asapp-multiwoz-api",
    description="asapp-multiwoz-api",
    author="blattimer",
    packages=PEP420PackageFinder.find(exclude=("test*",)),
    python_requires=">=3.8",
    install_requires=get_requires("requirements.txt"),
    include_package_data=True,
    extras_require={
        "test": get_requires("test/requirements.txt"),
    },
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
)
