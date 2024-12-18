from setuptools import setup, find_packages

setup(
    name="stable_diffusion_project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'diffusers',
        'accelerate',
    ]
)