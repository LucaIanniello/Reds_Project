from pathlib import Path

from setuptools import find_packages, setup

long_description = (Path(__file__).parent / "README.md").read_text()

core_requirements = [
    # "gym==0.21.0",
    "gym",
    "joblib",
    "rich",
    "tqdm",
]

setup(
    name="bpref_v2",
    version="0.1",
    author="Anonymous",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["__init__", "tests", "tests.*"]),
    include_package_data=True,
    python_requires=">3.7",
    install_requires=core_requirements,
)
