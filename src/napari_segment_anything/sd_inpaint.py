import os
import site
import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image
from typing import Optional, Tuple, Union, List
from napari_segment_anything.utils import get_lama_weights_path
import torch.nn.functional as F
from kornia.morphology import erosion

# TORCH_HOME 환경 변수 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
torch_home = os.path.join(current_dir, "pretrained_models")
os.environ['TORCH_HOME'] = torch_home
print(f"\n=== TORCH_HOME 설정: {torch_home}")

# 라마 모듈 경로 설정
dirs = Path(__file__).resolve()
lama_dir = dirs.parent / "lama"
print("\n=== LAMA 모듈 import 시도 ===")
print(f"LAMA 경로: {lama_dir}")
print(f"LAMA 경로 존재 여부: {lama_dir.exists()}")
site.addsitedir(str(lama_dir))

try:
    from saicinpainting.training.trainers import load_checkpoint
    from saicinpainting.evaluation.utils import move_to_device
    from saicinpainting.evaluation.data import pad_tensor_to_modulo

    print("LAMA 모듈 import 성공!")
except ImportError as e:
    print(f"Error: Could not import saicinpainting: {str(e)}")
    print("Please install it first.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lama_model = None  # 전역 변수로 선언


def inpaint_with_lama(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    global lama_model
    print("=== [DEBUG] inpaint_with_lama 진입 ===")
    print(f"[DEBUG] image.shape: {image.shape}, mask.shape: {mask.shape}")
    try:
        # 이미지 shape 분기
        if len(image.shape) == 3 and image.shape[2] == 3:
            print("[DEBUG] 3채널 컬러 이미지 분기")
            torch_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        elif len(image.shape) == 2:
            print("[DEBUG] 흑백 이미지 분기")
            torch_image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float() / 255.0
        elif len(image.shape) == 3 and image.shape[0] == 3:
            print("[DEBUG] (3, H, W) 형태 분기")
            torch_image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        else:
            print(f"[DEBUG] 지원하지 않는 image shape: {image.shape}")
            return image

        # 마스크 shape 분기
        if len(mask.shape) == 2:
            print("[DEBUG] 2D 마스크 분기")
            torch_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        elif len(mask.shape) == 3:
            if mask.shape[0] == 1:
                print("[DEBUG] (1, H, W) 마스크 분기")
                torch_mask = torch.from_numpy(mask).unsqueeze(0).float()
            elif mask.shape[2] == 1:
                print("[DEBUG] (H, W, 1) 마스크 분기")
                torch_mask = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).float()
            else:
                print(f"[DEBUG] 지원하지 않는 3D 마스크 shape: {mask.shape}")
                return image
        else:
            print(f"[DEBUG] 지원하지 않는 mask shape: {mask.shape}")
            return image

        print(f"[DEBUG] torch_image.shape: {torch_image.shape}, torch_mask.shape: {torch_mask.shape}")
        orig_h, orig_w = image.shape[0], image.shape[1]
        print(f"[DEBUG] 원본 크기: ({orig_h}, {orig_w})")
        torch_image = pad_tensor_to_modulo(torch_image, 8)
        torch_mask = pad_tensor_to_modulo(torch_mask, 8)
        print(f"[DEBUG] 패딩 후 torch_image.shape: {torch_image.shape}, torch_mask.shape: {torch_mask.shape}")

        print("[DEBUG] 배치 생성 완료")
        batch = {
            'image': torch_image.to(device),
            'mask': torch_mask.to(device)
        }

        # LAMA 모델 로드 (한 번만)
        if lama_model is None:
            print("[DEBUG] LAMA 모델 로드 시작 (최초 1회)")
            config_path = os.path.join(os.path.dirname(__file__), "pretrained_models/big-lama/config.yaml")
            checkpoint_path = os.path.join(os.path.dirname(__file__), "pretrained_models/big-lama/models/best.ckpt")
            print(f"[DEBUG] config_path: {config_path}")
            print(f"[DEBUG] checkpoint_path: {checkpoint_path}")
            config = OmegaConf.load(config_path)
            model = load_checkpoint(config, checkpoint_path, strict=False, map_location=device)
            model.eval()
            model.to(device)
            lama_model = model
            print("[DEBUG] LAMA 모델 로드 완료 및 재사용 준비")
        else:
            print("[DEBUG] LAMA 모델 재사용")
        model = lama_model

        # 인페인팅 수행
        print("[DEBUG] LAMA 모델 forward 시작")
        with torch.no_grad():
            result = model(batch)
        print("[DEBUG] LAMA 모델 forward 성공")

        # 결과 후처리
        if isinstance(result, dict) and 'inpainted' in result:
            result = result['inpainted']
        print(f"[DEBUG] result.shape (torch, pad): {result.shape}")
        # 항상 height, width 모두 원본 크기로 자르기
        result = result[:, :orig_h, :orig_w]
        print(f"[DEBUG] result.shape (torch, unpad): {result.shape}")
        result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
        print(f"[DEBUG] result.shape (numpy): {result.shape}")
        # 혹시라도 shape이 다르면 강제로 resize
        if result.shape[0] != orig_h or result.shape[1] != orig_w:
            print(f"[DEBUG] 강제 resize: result.shape={result.shape}, target=({orig_h},{orig_w})")
            result = cv2.resize(result, (orig_w, orig_h))
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        print("[DEBUG] torch->numpy 변환 및 반환")
        print(f"[DEBUG] 최종 반환 result.shape: {result.shape}, image.shape: {image.shape}")
        return result
    except Exception as e:
        print(f"[DEBUG] LAMA 인페인팅 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return image


def inpaint_image(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "lama",
    edge_padding: Optional[int] = None
) -> np.ndarray:
    """이미지 인페인팅을 수행합니다."""
    # 오직 lama만 지원
    return inpaint_with_lama(image, mask)
