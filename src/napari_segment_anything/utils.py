import urllib.request
import warnings
from pathlib import Path
from typing import Optional

import toolz as tz
from napari.utils import progress

SAM_WEIGHTS_URL = {
    "default": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

LAMA_WEIGHTS_URL = "https://github.com/advimman/lama/releases/download/v1.0/big-lama.pt"

@tz.curry
def _report_hook(
    block_num: int,
    block_size: int,
    total_size: int,
    pbr: "progress" = None,
) -> None:
    downloaded = block_num * block_size
    percent = downloaded * 100 / total_size
    downloaded_mb = downloaded / 1024 / 1024
    total_size_mb = total_size / 1024 / 1024
    increment = int(percent) - pbr.n
    if increment > 1:  # faster than increment at every iteration
        pbr.update(increment)
    print(
        f"Download progress: {percent:.1f}% ({downloaded_mb:.1f}/{total_size_mb:.1f} MB)",
        end="\r",
    )


def download_weights(weight_url: str, weight_path: "Path"):
    print(f"Downloading {weight_url} to {weight_path} ...")
    pbr = progress(total=100)
    try:
        urllib.request.urlretrieve(
            weight_url, weight_path, reporthook=_report_hook(pbr=pbr)
        )
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        urllib.error.ContentTooShortError,
    ) as e:
        warnings.warn(f"Error downloading {weight_url}: {e}")
        return None
    else:
        print("\rDownload complete.                            ")
    pbr.close()


def get_weights_path(model_type: str) -> Optional[Path]:
    """Returns the path to the weight of a given model architecture."""
    weight_url = SAM_WEIGHTS_URL[model_type]

    cache_dir = Path.home() / ".cache/napari-segment-anything"
    cache_dir.mkdir(parents=True, exist_ok=True)

    weight_path = cache_dir / weight_url.split("/")[-1]

    # Download the weights if they don't exist
    if not weight_path.exists():
        download_weights(weight_url, weight_path)

    return weight_path

def get_lama_weights_path():
    """LAMA 모델의 가중치 경로를 반환"""
    print("\n=== LAMA 가중치 경로 확인 ===")
    weights_dir = Path(__file__).parent / "weights"
    print(f"가중치 디렉토리: {weights_dir}")
    print(f"가중치 디렉토리 존재 여부: {weights_dir.exists()}")
    
    weights_path = weights_dir / "big-lama"
    print(f"LAMA 가중치 경로: {weights_path}")
    print(f"LAMA 가중치 경로 존재 여부: {weights_path.exists()}")
    
    if not weights_path.exists():
        print("경고: LAMA 가중치가 없습니다. 다운로드가 필요합니다.")
        # TODO: 가중치 다운로드 로직 추가
        return None
        
    return str(weights_path)
