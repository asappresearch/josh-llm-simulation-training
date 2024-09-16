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
    description="josh_train",
    author="blattimer",
    packages=PEP420PackageFinder.find(exclude=("test*",)),
    python_requires=">=3.8",
    install_requires=get_requires("requirements.txt"),
    include_package_data=True,
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
)
