from setuptools import setup, find_packages

setup(
    name='FoKL',  # Replace 'FoKL' with your package name
    version='2.3.2a',  # Specify your package version
    packages=find_packages(where='src'),  # Specify the package's directory
    package_dir={'': 'src'},  # Specify the root directory of packages
    install_requires=[  # Add any required dependencies here
        # list of dependencies
    ],
)
