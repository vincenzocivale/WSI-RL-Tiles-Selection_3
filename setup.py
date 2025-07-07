from setuptools import setup, find_packages

setup(
    name="hest",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "hestcore",
        "loguru",
        "numpy",
        "pandas",
        "tifffile",
        "packaging",
        "pillow",
        "matplotlib",
        "torch",
        "torchvision",
        "h5py",
        "tqdm",
    ],
    python_requires=">=3.8",
)
