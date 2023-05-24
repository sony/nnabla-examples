
import setuptools
from setuptools import setup

setup(
    name='nnabla_diffusion',
    version='1.0.0.dev',
    author='Sony Group Corporation',
    author_email='nnabla@googlegroups.com',
    package_dir={"": "src"},
    packages=["nnabla_diffusion.config", "nnabla_diffusion.ddpm_segmentation",
              "nnabla_diffusion.diffusion_model", "nnabla_diffusion.dataset"],
    scripts=[],
    license='LICENSE',
    description='Utility package for nnabla enhancement',
    long_description=open('./README.md').read(),
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
    python_requires=">=3.6",
    install_requires=[
        "pyyaml",
        "dominate",
        "click",
        "moviepy",
        "hydra-core",
        "webdataset",
        "gradio"
    ],
)
