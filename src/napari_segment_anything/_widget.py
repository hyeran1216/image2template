from typing import Any, Generator, Optional
import os
import sys
from omegaconf import OmegaConf
from pathlib import Path

# saicinpainting 모듈이 들어있는 lama 폴더를 path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "lama")))

import napari
import numpy as np
import torch
from magicgui.widgets import ComboBox, Container, PushButton, create_widget, Slider
from napari.layers import Image, Points, Shapes
from napari.layers.shapes._shapes_constants import Mode
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QApplication, QMessageBox

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from segment_anything.modeling import Sam

from skimage import color, util
from skimage.measure import find_contours, label, regionprops
from skimage.draw import polygon2mask
from scipy.stats import mode
from PIL import Image as PILImageLib
import cv2

from napari_segment_anything.utils import get_weights_path

from .sd_inpaint import (
    inpaint_image,
    inpaint_with_lama,
    get_lama_weights_path
)

# from segment_anything.utils.transforms import ResizeLongestSide

def inpaint_background(image: np.ndarray, background_mask: np.ndarray) -> np.ndarray:
    """
    배경을 인페인팅하는 함수
    
    Args:
        image: 입력 이미지 [0, 255], uint8, shape (H, W, 3)
        background_mask: 배경 마스크 [True=배경], bool, shape (H, W)
    
    Returns:
        inpainted: 인페인팅된 이미지 [0, 255], uint8, shape (H, W, 3)
    """
    # 마스크 반전 (객체 영역을 1로)
    object_mask = ~background_mask
    
    # 마스크 전처리
    mask = object_mask.astype(np.uint8) * 255
    
    # 마스크 dilate로 경계 부분 보정 (더 큰 커널 사용)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Navier-Stokes 알고리즘으로 인페인팅 (더 큰 반경)
    result = cv2.inpaint(image, mask, 15, cv2.INPAINT_NS)
    
    return result

class SAMWidget(Container):
    _sam: Sam
    _predictor: SamPredictor

    def __init__(self, viewer: napari.Viewer, model_type: str = "default"):
        super().__init__()
        self._viewer = viewer
        if torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        self._model_type_widget = ComboBox(
            value=model_type,
            choices=list(sam_model_registry.keys()),
            label="Model:",
        )
        self._model_type_widget.changed.connect(self._load_model)
        self.append(self._model_type_widget)

        self._im_layer_widget = create_widget(annotation=Image, label="Image:")
        self._im_layer_widget.changed.connect(self._load_image)
        self.append(self._im_layer_widget)

        self._inpaint_method = ComboBox(
            value="combined",
            choices=["combined", "ns", "telea"],
            label="Inpaint Method:",
        )
        self.append(self._inpaint_method)

        self._kernel_size = Slider(
            value=3,
            min=3,
            max=15,
            step=2,
            label="Mask Refine Kernel:",
        )
        self.append(self._kernel_size)

        self._confirm_mask_btn = PushButton(
            text="Confirm Annot.",
            enabled=False,
            tooltip="Press C to confirm annotation.",
        )
        self._confirm_mask_btn.changed.connect(self._on_confirm_mask)
        self.append(self._confirm_mask_btn)

        self._cancel_annot_btn = PushButton(
            text="Cancel Annot.",
            enabled=False,
            tooltip="Press X to cancel annotation.",
        )
        self._cancel_annot_btn.changed.connect(self._cancel_annot)
        self.append(self._cancel_annot_btn)

        self._auto_segm_btn = PushButton(text="Auto. Segm.")
        self._auto_segm_btn.changed.connect(self._on_auto_run)
        self.append(self._auto_segm_btn)

        self._export_btn = PushButton(text="Export Selected Object")
        self._export_btn.changed.connect(self.export_selected_object)
        self.append(self._export_btn)

        self._export_polygon_btn = PushButton(text="Export Polygon as Image")
        self._export_polygon_btn.changed.connect(self.export_selected_polygon)
        self.append(self._export_polygon_btn)

        self._labels_layer = self._viewer.add_labels(
            data=np.zeros((256, 256), dtype=int),
            name="SAM labels",
        )
        
        self._mask_layer = self._viewer.add_labels(
            data=np.zeros((256, 256), dtype=int),
            name="SAM mask",
        )
        self._mask_layer.contour = 2

        self._pts_layer = self._viewer.add_points(name="SAM points")
        self._pts_layer.current_face_color = "blue"
        self._pts_layer.events.data.connect(self._on_interactive_run)
        self._pts_layer.mouse_drag_callbacks.append(self._mouse_button_modifier)

        self._boxes_layer = self._viewer.add_shapes(
            name="SAM box",
            face_color="transparent",
            edge_color="green",
            edge_width=2,
        )
        self._boxes_layer.mouse_drag_callbacks.append(self._on_shape_drag)

        self._image: Optional[np.ndarray] = None
        self._logits: Optional[torch.TensorType] = None

        self._model_type_widget.changed.emit(model_type)
        self._viewer.bind_key("C", self._on_confirm_mask)
        self._viewer.bind_key("X", self._cancel_annot)

    def _load_model(self, model_type: str) -> None:
        self._sam = sam_model_registry[model_type](
            get_weights_path(model_type)
        )
        self._sam.to(self._device)
        self._predictor = SamPredictor(self._sam)
        self._load_image(self._im_layer_widget.value)

    def _load_image(self, im_layer: Optional[Image]) -> None:
        if im_layer is None or not hasattr(self, "_sam"):
            return
        if im_layer.ndim != 2:
            raise ValueError(
                f"Only 2D images supported. Got {im_layer.ndim}-dim image."
            )
            
        image = im_layer.data
        if not im_layer.rgb:
            image = color.gray2rgb(image)
            
        elif image.shape[-1] == 4:
            # images with alpha
            image = color.rgba2rgb(image)
            
        if np.issubdtype(image.dtype, np.floating):
            image = (image - image.min()) / image.max()
            
        self._image = util.img_as_ubyte(image)
        
        self._mask_layer.data = np.zeros(self._image.shape[:2], dtype=int)
        self._labels_layer.data = np.zeros(self._image.shape[:2], dtype=int)
        self._predictor.set_image(self._image)

    def _mouse_button_modifier(self, _: Points, event) -> None:
        self._pts_layer.selected_data = []
        if event.button == Qt.LeftButton:
            self._pts_layer.current_face_color = "blue"
        else:
            self._pts_layer.current_face_color = "red"

    def _on_interactive_run(self, _: Optional[Any] = None) -> None:
        points = self._pts_layer.data
        boxes = self._boxes_layer.data
        
        if len(boxes) > 0:
            box = boxes[-1]
            box = np.stack([box.min(axis=0), box.max(axis=0)], axis=0)
            box = np.flip(box, -1).reshape(-1)[None, ...]
        else:
            box = None

        if len(points) > 0:
            points = np.flip(points, axis=-1)
            colors = self._pts_layer.face_color
            blue = [0, 0, 1, 1]
            labels = np.all(colors == blue, axis=1)
        else:
            points = None
            labels = None
            
        mask, _, self._logits = self._predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            mask_input=self._logits,
            multimask_output=False,
        )
        self._mask_layer.data = mask[0]
        self._confirm_mask_btn.enabled = True
        self._cancel_annot_btn.enabled = True

    def _on_shape_drag(self, _: Shapes, event) -> Generator:
        if self._boxes_layer.mode != Mode.ADD_RECTANGLE:
            return
        # on mouse click
        yield
        # on move
        while event.type == "mouse_move":
            yield
        # on mouse release
        self._on_interactive_run()

    def export_selected_object(self):
        if "Object Polygons" not in self._viewer.layers:
            print("❌ Object Polygons 레이어가 없습니다.")
            return
        
        shapes_layer = self._viewer.layers["Object Polygons"]
        if len(shapes_layer.selected_data) != 1:
            print("❌ polygon 하나만 선택해주세요.")
            return
        
        selected_index = list(shapes_layer.selected_data)[0]
        polygon = shapes_layer.data[selected_index]
        
        # polygon → mask
        mask = polygon2mask(self._image.shape[:2], polygon)
        
        # 원본 이미지에서 마스크 적용
        masked = self._image.copy()
        masked[~mask] = 0
        
        # bounding box로 crop
        y_coords, x_coords = np.where(mask)
        min_y, max_y = y_coords.min(), y_coords.max()
        min_x, max_x = x_coords.min(), x_coords.max()
        cropped = masked[min_y:max_y + 1, min_x:max_x + 1]
        
        # 저장 다이얼로그
        filename, _ = QFileDialog.getSaveFileName(
            caption="Save Selected Object",
            filter="Image Files (*.png *.jpg)"
        )
        if filename:
            PILImageLib.fromarray(cropped).save(filename)
            print(f"✅ 저장 완료: {filename}")

    def export_selected_polygon(self):
        shapes_layer = None
        for layer in self._viewer.layers:
            if isinstance(layer, Shapes) and layer.name == "Object Polygons":
                shapes_layer = layer
                break
            
        if shapes_layer is None or len(shapes_layer.selected_data) != 1:
            print("❌ polygon 하나만 선택해주세요.")
            return
        
        selected_index = list(shapes_layer.selected_data)[0]
        polygon = shapes_layer.data[selected_index]
        face_color = shapes_layer.face_color[selected_index]
        
        # 캔버스 생성 (배경은 흰색)
        canvas = np.ones((*self._image.shape[:2], 3), dtype=np.uint8) * 255
        
        # 마스크 생성
        mask = polygon2mask(canvas.shape[:2], polygon)
        
        # RGB 색 변환 (0~1 → 0~255)
        color_rgb = (np.array(face_color[:3]) * 255).astype(np.uint8)
        for c in range(3):
            canvas[..., c][mask] = color_rgb[c]
            
        # 바운딩 박스로 crop
        y_coords, x_coords = np.where(mask)
        min_y, max_y = y_coords.min(), y_coords.max()
        min_x, max_x = x_coords.min(), x_coords.max()
        cropped = canvas[min_y:max_y + 1, min_x:max_x + 1]
        
        # 저장
        filename, _ = QFileDialog.getSaveFileName(
            caption="Save Polygon as Image",
            filter="Image Files (*.png *.jpg)"
        )
        if filename:
            PILImageLib.fromarray(cropped).save(filename)
            print(f"✅ 단색 Polygon 저장 완료: {filename}")
            
    def _on_auto_run(self) -> None:
        if self._image is None:
            return
        print("\n=== 마스크 생성 시작 ===")

        # SAM 파라미터 조정
        mask_gen = SamAutomaticMaskGenerator(
            self._sam,
            points_per_side=32,
            pred_iou_thresh=0.89,
            stability_score_thresh=0.91,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=200
        )
        preds = mask_gen.generate(self._image)

        # 중복 제거 및 필터링
        preds = self.filter_duplicate_masks(preds, iou_thresh=0.9)
        preds = self.filter_large_uniform_masks(preds, self._image)
        
        # 객체 마스크 생성
        object_mask = np.zeros(self._image.shape[:2], dtype=bool)
        for pred in preds:
            object_mask |= pred["segmentation"]

        # dilate된 마스크 (LaMa용)
        kernel = np.ones((9,9 ), np.uint8)
        dilated_mask = cv2.dilate(object_mask.astype(np.uint8), kernel, iterations=2).astype(bool)

        background_mask = ~object_mask

        # 객체 이미지와 배경 이미지 생성
        object_image = self._image.copy()
        object_image[~object_mask] = 0
        background_image = self._image.copy()
        background_image[~background_mask] = 0

        # 기존 레이어 제거
        for name in ["Objects", "Background", "Inpainted Background", "LAMA Inpainted Background"]:
            if name in self._viewer.layers:
                self._viewer.layers.remove(name)

        print("\n=== 배경 복원 시작 ===")
        try:
            # 배경 복원
            restored_bg = inpaint_image(
                background_image.copy(),
                dilated_mask,
                method="lama"
            )
            
            # 결과 레이어 추가
            self._viewer.add_image(
                restored_bg,
                name="LAMA Inpainted Background",
                blending="additive",
                visible=True
            )
            print("배경 복원 완료!")


            # 기존 레이어 추가
            self._viewer.add_image(
                object_image,
                name="Objects",
                blending="additive",
                visible=True
            )
            self._viewer.add_image(
                background_image,
                name="Background",
                blending="additive",
                visible=True
            )
            
        except Exception as e:
            print(f"배경 복원 실패: {e}")
            import traceback
            print(traceback.format_exc())
            # 실패 시 기존 방식으로 폴백
            self._viewer.add_image(
                object_image,
                name="Objects",
                blending="additive",
                visible=True
            )
            self._viewer.add_image(
                background_image,
                name="Background",
                blending="additive",
                visible=True
            )

        # 객체 분리
        # === 객체 외곽 polygon을 shapes로 시각화 ===
        labels = label(object_mask)
        polygons = []
        face_colors =[]
        edge_colors =[]
        
        for region_label in np.unique(labels):
            if region_label == 0:
                continue

            region_mask = labels == region_label
            contours = find_contours(region_mask.astype(float), level=0.5)

            pixels = self._image[region_mask].reshape(-1, 3)

            r_mode = mode(pixels[:, 0], keepdims=False).mode
            g_mode = mode(pixels[:, 1], keepdims=False).mode
            b_mode = mode(pixels[:, 2], keepdims=False).mode

            dominant_color = (r_mode / 255.0, g_mode / 255.0, b_mode / 255.0)

            for contour in contours:
                polygons.append(contour)
                face_colors.append(dominant_color)
                edge_colors.append(dominant_color)
                
        if "Object Polygons" in self._viewer.layers:
            self._viewer.layers.remove("Object Polygons")

        shapes_layer = self._viewer.add_shapes(
            data=polygons,
            shape_type='polygon',
            edge_width=2,
            edge_color= edge_colors,
            face_color=face_colors,
            name="Object Polygons"
        )
        shapes_layer.mode = 'select'
        
    def filter_large_uniform_masks(self,preds, image, min_area_ratio=0.1):
        h, w = image.shape[:2]
        total_pixels = h * w
        filtered = []
        for pred in preds:
            seg = pred["segmentation"]
            area = seg.sum()
            # # 넓은 마스크는 무조건 배경으로 간주하지 않음 → 객체일 수 있음
            # if area < total_pixels * min_area_ratio:
            #     filtered.append(pred)  # 작은 것들만 객체로 포함
            if area >= total_pixels * min_area_ratio:
                continue  # 넓은 건 제거
            filtered.append(pred)
        return filtered
    
    def filter_duplicate_masks(self, preds, iou_thresh=0.9):
        def compute_iou(mask1, mask2):
            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            return intersection / union if union != 0 else 0.0
        preds_sorted = sorted(preds, key=lambda x: -x["predicted_iou"])
        filtered = []
        for pred in preds_sorted:
            is_duplicate = False
            for kept in filtered:
                if compute_iou(pred["segmentation"], kept["segmentation"]) > iou_thresh:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(pred)

        return filtered
    
    def _on_confirm_mask(self, _: Optional[Any] = None) -> None:
        if not self._confirm_mask_btn.enabled:
            return

        mask = self._mask_layer.data
        if np.any(mask > 0):
            # 객체 레이블링
            labels = label(mask)
            polygons = []
            face_colors = []
            edge_colors = []
            
            # 각 객체별로 처리
            for region_label in np.unique(labels):
                if region_label == 0:  # 배경은 건너뛰기
                    continue
                    
                # 현재 객체의 마스크
                region_mask = labels == region_label
                
                # 마스크를 약간 축소 (edge가 안쪽으로만 보이도록)
                kernel = np.ones((3,3), np.uint8)
                eroded_mask = cv2.erode(region_mask.astype(np.uint8), kernel, iterations=2)
                
                # 축소된 마스크의 contour 찾기
                contours = find_contours(eroded_mask.astype(float), level=0.5)
                
                # 현재 객체의 픽셀 가져오기
                pixels = self._image[region_mask]
                
                # RGB 값을 문자열로 변환하여 최빈값 계산
                pixel_strs = [f"{r},{g},{b}" for r, g, b in pixels]
                most_common = mode(pixel_strs, keepdims=False).mode
                
                # 최빈값 RGB를 파싱
                r_mode, g_mode, b_mode = map(int, most_common.split(','))
                
                # RGB 값을 0~1 범위로 정규화
                face_color = (r_mode / 255.0, g_mode / 255.0, b_mode / 255.0)
                edge_color = (0, 0, 0)  # 검정색
                
                # 현재 객체의 모든 contours 추가
                for contour in contours:
                    polygons.append(contour)
                    face_colors.append(face_color)
                    edge_colors.append(edge_color)
            
            # 기존 shape layer 찾기 또는 새로 생성
            shapes_layer = None
            for layer in self._viewer.layers:
                if isinstance(layer, Shapes) and layer.name == "Object Polygons":
                    shapes_layer = layer
                    break
                    
            if shapes_layer is None:
                # 모든 contours를 한 번에 추가
                shapes_layer = self._viewer.add_shapes(
                    data=polygons,
                    shape_type='polygon',
                    edge_width=4,  # edge_width 증가
                    edge_color=edge_colors,
                    face_color=face_colors,
                    name="Object Polygons"
                )
                # 모드 설정은 layer가 완전히 생성된 후에
                shapes_layer.mode = 'select'
            else:
                # 기존 layer에 새로운 contours 추가
                for i, contour in enumerate(polygons):
                    shapes_layer.add(
                        contour,
                        shape_type='polygon',
                        edge_color=edge_colors[i],
                        face_color=face_colors[i],
                        edge_width=4  # edge_width 증가
                    )
            
            # 마스크 초기화
            self._mask_layer.data = np.zeros_like(mask)

        self._confirm_mask_btn.enabled = False
        self._cancel_annot_btn.enabled = False
        self._pts_layer.data = []
        self._boxes_layer.data = []
        self._logits = None

    def _cancel_annot(self, _: Optional[Any] = None) -> None:
        # boxes must be reset first because of how of points data update signal
        self._boxes_layer.data = []
        self._pts_layer.data = []
        self._mask_layer.data = np.zeros_like(self._mask_layer.data)

        self._confirm_mask_btn.enabled = False
        self._cancel_annot_btn.enabled = False

import atexit
@atexit.register
def cleanup():
    try:
        app = QApplication.instance()
        if app:
            app.quit()
    except Exception as e:
        print(f"Cleanup error: {e}")
