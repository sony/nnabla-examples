from setuptools import setup

setup(
    name='neu',
    version='1.23.0dev',  # TODO: Must be consistent with nnabla version?
    author='Sony Group Corporation',
    author_email='nnabla@googlegroups.com',
    package_dir={"": "utils"},
    packages=setuptools.find_packages(where="utils"),
    scripts=[],
    license='LICENSE',
    description='Utility package for nnabla enhancement',
    long_description=open('./utils/README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/sony/nnabla-examples/',
    project_urls={
        "nnabla": "https://nnabla.org/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'neu_cli = neu.cli:main',
        ],
    },
    python_requires=">=3.6",
    install_requires=[
        "pyyaml",
        "dominate",
    ],
)
