from setuptools import setup, find_packages

setup(
    name="cram",
    version="0.1",
    packages=find_packages(),  # This will automatically find all packages with __init__.py
    install_requires=[],
    include_package_data=True,
    zip_safe=False,
)