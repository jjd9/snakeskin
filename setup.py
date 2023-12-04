"""
Python setup.py for project_name package

taken from: https://github.com/rochacbruno/python-project-template/blob/main/setup.py

"""

import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="snakeskin",
    version="1.0",
    description="A cli for making beautiful mosaics.",
    url="https://github.com/jjd9/snakeskin/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="John D'Angelo",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["snakeskin = snakeskin.snakeskin:hiss"]
    },
)
