import sys

try:
    import ez_setup
    ez_setup.use_setuptools()
except ImportError:
    pass

from setuptools import setup

setup(
    name="RegexPicker plugin",
    version="0.1",
    author="xxx",
    author_email="xxx@example.com",
    description="Pick test methods based on a regular expression",
    license="Apache Server License 2.0",
    py_modules=['regexpicker'],
    entry_points={
        'nose.plugins': ['regexpicker = regexpicker:RegexPicker']
    }
)


