import os
import site

import yaml
import torch
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from napari_segment_anything.utils import get_lama_weights_path
import cv2

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.data import pad_tensor_to_modulo

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# 예시: pretrained_models 폴더를 TORCH_HOME으로 지정
os.environ['TORCH_HOME'] = str(Path(__file__).resolve().parent / "pretrained_models")

# 라마 모듈 경로 설정
dirs = Path(__file__).resolve()
lama_dir = dirs.parent / "lama"
site.addsitedir(str(lama_dir))

def get_lama_config():
    """LAMA 모델의 기본 설정을 반환"""
    config = OmegaConf.create({
        'model': {
            'path': 'big-lama',
            'checkpoint': 'best.ckpt'
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'out_key': 'inpainted'
    })
    return config

def get_lama_weights_path():
    """LAMA 모델의 가중치 경로를 반환"""
    # 가중치 다운로드 또는 경로 설정 로직
    weights_dir = Path(__file__).parent / "weights"
    weights_dir.mkdir(exist_ok=True)
    weights_path = weights_dir / "big-lama"
    # TODO: 가중치 다운로드 로직 추가
    return str(weights_path)

def build_lama_model(device="cuda"):
    try:
        print("\n=== LAMA 모델 빌드 시작 ===")
        from saicinpainting.training.trainers import load_checkpoint
        print("saicinpainting 모듈 import 성공")
    except ImportError as e:
        print(f"Error: Could not import saicinpainting: {str(e)}")
        print("Please install it first.")
        return None

    config = get_lama_config()
    config.device = device
    print(f"LAMA 설정: {config}")
    
    # Get weights path
    weights_path = get_lama_weights_path()
    print(f"LAMA 가중치 경로: {weights_path}")
    if weights_path is None:
        print("Error: Could not download LaMa weights.")
        return None
        
    print("LAMA 모델 로드 시도...")
    model = load_checkpoint(config, weights_path, strict=False, map_location='cpu')
    print("LAMA 모델 로드 성공")
    
    model.to(device)
    model.eval()
    print("LAMA 모델 초기화 완료")
    return model

@torch.no_grad()
def inpaint_img_with_builded_lama(model, image, mask, device="cuda"):
    """
    LAMA 모델을 사용한 인페인팅
    
    Args:
        model: LAMA 모델
        image: numpy array in [0, 255], uint8, shape (H, W, 3)
        mask: numpy array in [0, 1], bool, shape (H, W)
        device: 장치 ('cuda' 또는 'cpu')
    """
    if model is None:
        print("Error: LaMa model is not initialized.")
        return image

    # 마스크 전처리
    if len(mask.shape) == 2:
        mask = mask[..., None]
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    
    # 배치 생성
    batch = {}
    batch['image'] = torch.from_numpy(image).float().div(255.).permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = torch.from_numpy(mask).float()[None, None]
    
    # 패딩 및 디바이스 이동
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], 8)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], 8)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1
    
    # 인페인팅 실행
    with torch.no_grad():
        batch = model(batch)
        cur_res = batch["inpainted"][0].permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()
        
    # 패딩 제거
    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]
        
    # 결과 후처리
    if isinstance(cur_res, dict) and 'inpainted' in cur_res:
        cur_res = cur_res['inpainted']
    print(f"[DEBUG] cur_res.shape (torch, pad): {cur_res.shape}")
    # 항상 원본 크기로 자르기
    cur_res = cur_res[:, :orig_height, :orig_width]
    print(f"[DEBUG] cur_res.shape (torch, unpad): {cur_res.shape}")
    cur_res = cur_res.squeeze(0).permute(1, 2, 0).cpu().numpy()
    print(f"[DEBUG] cur_res.shape (numpy): {cur_res.shape}")
    cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
    print("[DEBUG] torch->numpy 변환 및 반환")
    return cur_res

