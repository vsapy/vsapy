import pathlib
from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

print("packages found", find_packages(exclude=("tests",)))  # see what packages are found using deb

# This call to setup() does all the work
setup(
    name="vsapy",
    version="0.4.0",
    description="Vector Symbolic Architecture(VSA) library that allows building VSA apps that use variousl flavours of VSA vectors",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/vsapy/vsapy",
    author="Chris Simpkin",
    author_email="simpkin.chris@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=False,
    install_requires=["numpy", "scipy"],
    entry_points={},
)
