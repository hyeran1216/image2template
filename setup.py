from setuptools import setup, find_packages
import os
from pathlib import Path
import shutil

def copy_lama_files():
    """LAMA 프로젝트 파일을 패키지 디렉토리로 복사"""
    src_lama = Path("C:/Users/kimin/ML/napari-segment-anything-main_1/lama")
    if not src_lama.exists():
        print(f"Warning: LAMA directory not found at {src_lama}")
        return
    
    # LAMA 파일을 패키지 디렉토리로 복사
    dst_lama = Path("src/napari_segment_anything/lama")
    if dst_lama.exists():
        shutil.rmtree(dst_lama)
    print(f"Copying LAMA files from {src_lama} to {dst_lama}")
    shutil.copytree(src_lama, dst_lama)

# LAMA 파일 복사
copy_lama_files()

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