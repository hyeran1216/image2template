from setuptools import setup, find_packages
import os
from pathlib import Path
import shutil


setup(
    name="napari-segment-anything",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "napari",
        "numpy",
        "torch",
        "torchvision",
        "opencv-python",
        "segment-anything",
        "omegaconf",
        "albumentations",
        "pytorch-lightning",
        "hydra-core",
        "easydict",
        "scikit-image",
        "matplotlib",
        "pandas",
        "kornia",
    ],
    package_data={
        "napari_segment_anything": [
            "lama/**/*",
            "weights/**/*",
            "lama/configs/**/*",
            "lama/saicinpainting/**/*",
        ],
    },
    include_package_data=True,
) 