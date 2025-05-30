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
from napari.layers import Image, Points, Shapes, Labels
from napari.layers.shapes._shapes_constants import Mode
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QApplication, QMessageBox
from qtpy.QtGui import QFont
from skimage.draw import polygon2mask
from skimage.measure import find_contours
from scipy.interpolate import splprep, splev

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


def smooth_polygon(contour: np.ndarray, num_points: int = 100) -> np.ndarray:
    """폴리곤을 부드럽게 처리하는 함수"""
    if len(contour) < 3:
        return contour  # 너무 짧으면 스킵

    x = contour[:, 1]
    y = contour[:, 0]
    try:
        tck, _ = splprep([x, y], s=3)
        x_new, y_new = splev(np.linspace(0, 1, num_points), tck)
        return np.vstack([y_new, x_new]).T
    except Exception:
        return contour  # 보간 실패 시 원본 리턴

def close_polygon_if_needed(points: np.ndarray) -> np.ndarray:
    """폴리곤이 닫혀있지 않으면 닫는 함수"""
    if len(points) >= 3 and not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    return points

def mask_to_smoothed_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """마스크에서 부드러운 외곽선을 추출하는 함수"""
    blurred = cv2.GaussianBlur(mask.astype(np.uint8) * 255, (5, 5), 0)
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return approx.reshape(-1, 2)

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

        # 브러시 확장 기능 추가
        self._brush_btn = PushButton(text="Brush Extension")
        self._brush_btn.changed.connect(self._on_brush_extension)
        self.append(self._brush_btn)

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
        selected_shapes = list(shapes_layer.selected_data)
        if not selected_shapes:
            print("❌ export할 객체를 선택해주세요.")
            return
        
        # 저장할 디렉토리 선택
        save_dir = QFileDialog.getExistingDirectory(
            caption="Select Save Directory",
            directory="."
        )
        if not save_dir:
            return
            
        # 각 선택된 객체별로 export
        for idx in selected_shapes:
            polygon = shapes_layer.data[idx]
            
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
            
            # 파일 저장
            filename = os.path.join(save_dir, f"object_{idx}.png")
            PILImageLib.fromarray(cropped).save(filename)
            
        print(f"✅ {len(selected_shapes)}개의 객체가 {save_dir}에 저장되었습니다.")

    def export_selected_polygon(self):
        shapes_layer = None
        for layer in self._viewer.layers:
            if isinstance(layer, Shapes) and layer.name == "Object Polygons":
                shapes_layer = layer
                break
            
        if shapes_layer is None:
            print("❌ Object Polygons 레이어가 없습니다.")
            return
            
        selected_shapes = list(shapes_layer.selected_data)
        if not selected_shapes:
            print("❌ export할 객체를 선택해주세요.")
            return
        
        # 저장할 디렉토리 선택
        save_dir = QFileDialog.getExistingDirectory(
            caption="Select Save Directory",
            directory="."
        )
        if not save_dir:
            return
            
        # 각 선택된 객체별로 export
        for idx in selected_shapes:
            polygon = shapes_layer.data[idx]
            face_color = shapes_layer.face_color[idx]
            
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
            
            # 파일 저장
            filename = os.path.join(save_dir, f"polygon_{idx}.png")
            PILImageLib.fromarray(cropped).save(filename)
            
        print(f"✅ {len(selected_shapes)}개의 polygon이 {save_dir}에 저장되었습니다.")
            
    def _on_auto_run(self) -> None:
        if self._image is None:
            return
        print("\n=== 마스크 생성 시작 ===")

        # SAM 파라미터 조정
        mask_gen = SamAutomaticMaskGenerator(
            self._sam,
            points_per_side=48,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100


            
            # points_per_batch = 64,
            # stability_score_offset = 1.0,
            # box_nms_thresh = 0.7,
            # crop_n_layers  = 0,
            # crop_nms_thresh = 0.7,
            # crop_overlap_ratio = 512 / 1500,
            # crop_n_points_downscale_factor = 1,
            # point_grids = None,
            # min_mask_region_area = 200,
            
        )
        preds = mask_gen.generate(self._image)


         # 필터링
        preds = self.filter_large_uniform_masks(preds, self._image)

        # 객체 마스크 생성
        object_mask = np.zeros(self._image.shape[:2], dtype=bool)
        
        dilated_mask = np.zeros(self._image.shape[:2], dtype=bool)
        for pred in preds:
            object_mask |= pred["segmentation"]
            seg = pred["segmentation"]
            area = seg.sum()
            # if area < 70:  # 작은 도형
            #     kernel = np.ones((2, 2), np.uint8)
            #     dilated = cv2.dilate(seg.astype(np.uint8), kernel, iterations=1).astype(bool)
            # if area < 1000:  # 중간 크기 도형
            #     kernel = np.ones((3, 3), np.uint8)
            #     dilated = cv2.dilate(seg.astype(np.uint8), kernel, iterations=1).astype(bool)
            # else:  # 큰 도형
            kernel = np.ones((12, 12), np.uint8)
            dilated = cv2.dilate(seg.astype(np.uint8), kernel, iterations=3).astype(bool)
            dilated_mask |= dilated  # or 연산으로 합치기

        background_mask = ~object_mask

        # 객체 이미지와 배경 이미지 생성
        object_image = self._image.copy()
        object_image[~object_mask] = 0
        background_image = self._image.copy()
        background_image[~background_mask] = 0

        # 기존 레이어 제거
        for name in ["Objects", "LAMA Inpainted Background"]:
            if name in self._viewer.layers:
                self._viewer.layers.remove(name)

        print("\n=== 배경 복원 시작 ===")
        try:
            # 정사각형이 아니면 lama에 넣기 전에 정사각형으로 리사이즈 후 인페인팅
            h, w = self._image.shape[:2]
            if h != w:
                print("⚠️ 정사각형이 아닌 이미지는 인페인팅 품질이 저하될 수 있습니다. 임시로 정사각형으로 리사이즈 후 복원합니다.")
                new_size = (max(h, w), max(h, w))
                resized_bg = cv2.resize(background_image, new_size)
                resized_mask = cv2.resize(dilated_mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST).astype(bool)
                restored_bg = inpaint_image(
                    resized_bg,
                    resized_mask,
                    method="lama"
                )
                # 결과를 다시 원본 크기로 리사이즈
                restored_bg = cv2.resize(restored_bg, (w, h))
            else:
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

            # 기존 레이어 추가 (Background 레이어는 추가하지 않음)
            self._viewer.add_image(
                object_image,
                name="Objects",
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
            contours = find_contours(region_mask.astype(float), level=0.1)

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
                
                # 각 contour마다 개별적으로 처리
                for contour in contours:
                    # contour 좌표를 정수로 변환
                    contour_int = contour.astype(int)
                    
                    # 현재 contour의 경계선 색상 수집
                    boundary_colors = []
                    for y, x in contour_int:
                        if 0 <= y < self._image.shape[0] and 0 <= x < self._image.shape[1]:
                            boundary_colors.append(self._image[y, x])
                    
                    # 현재 contour의 경계선 색상 최빈값 계산
                    if boundary_colors:
                        boundary_strs = [f"{r},{g},{b}" for r, g, b in boundary_colors]
                        most_common_boundary = mode(boundary_strs, keepdims=False).mode
                        r_edge, g_edge, b_edge = map(int, most_common_boundary.split(','))
                        edge_color = (r_edge / 255.0, g_edge / 255.0, b_edge / 255.0)
                    else:
                        edge_color = face_color  # 경계선 색상을 찾지 못한 경우 face_color 사용
                    
                    # 현재 contour 추가
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

    def _on_brush_extension(self) -> None:
        """polygon 또는 브러시 도구를 사용하여 영역을 확장하는 기능"""
        if "Object Polygons" not in self._viewer.layers:
            print("❌ Object Polygons 레이어가 없습니다.")
            return

        shapes_layer = self._viewer.layers["Object Polygons"]
        selected_shapes = list(shapes_layer.selected_data)
        if not selected_shapes:
            print("❌ 확장할 polygon을 선택해주세요.")
            return

        # 기존 확장 레이어 제거
        for name in ["Extension Polygon", "Brush Extension"]:
            if name in self._viewer.layers:
                self._viewer.layers.remove(name)

        h, w = self._image.shape[:2]

        # 브러시 레이어 생성
        brush_layer = self._viewer.add_labels(
            np.zeros((h, w), dtype=np.uint8),
            name="Brush Extension"
        )
        brush_layer.brush_size = 20
        brush_layer.mode = 'paint'  # 브러시 모드로 설정

        # 폴리곤 레이어 생성
        polygon_layer = self._viewer.add_shapes(
            name="Extension Polygon",
            shape_type='polygon',
            edge_color='yellow',
            face_color='yellow',
            edge_width=2
        )
        polygon_layer.mode = 'add_polygon'

        # 브러시 크기 조절 슬라이더
        brush_slider = Slider(
            value=20,
            min=1,
            max=100,
            step=1,
            label="Brush Size"
        )
        brush_slider.changed.connect(lambda size: setattr(brush_layer, 'brush_size', size))
        self.append(brush_slider)

        # 적용 버튼
        apply_btn = PushButton(text="Apply Extension")
        apply_btn.changed.connect(lambda: self._apply_polygon_extension(
            shapes_layer,
            selected_shapes,
            polygon_layer if len(polygon_layer.data) > 0 else brush_layer
        ))
        self.append(apply_btn)

        # Enter 키로 폴리곤 확정
        def on_enter(event):
            if polygon_layer.mode == 'add_polygon':
                polygon_layer.mode = 'select'
                print("✅ Extension polygon 그리기 완료. Apply 버튼을 눌러 병합하세요.")
                return True
            return False

        self._viewer.bind_key('Enter', on_enter)

    def _apply_polygon_extension(self, shapes_layer, selected_shapes, extension_layer) -> None:
        """그려진 polygon 또는 브러시를 마스크로 변환하여 기존 객체와 병합"""
        try:
            h, w = self._image.shape[:2]

            # 확장 마스크 추출
            if hasattr(extension_layer, "shape_type"):  # polygon (Shapes 레이어)
                if len(extension_layer.data) == 0:
                    print("❌ 확장할 polygon이 없습니다.")
                    return
                extension_polygon = extension_layer.data[0]
                extension_mask = polygon2mask((h, w), extension_polygon).astype(bool)
                print("polygon 기반 확장 적용됨")
            elif hasattr(extension_layer, "brush_size"):  # 브러시 (Labels 레이어)
                extension_mask = extension_layer.data.astype(bool)
                if extension_mask.sum() == 0:
                    print("❌ 브러시로 색칠된 영역이 없습니다.")
                    return
                print("브러시 기반 확장 적용됨")

            else:
                print("❌ 알 수 없는 레이어 타입입니다.")
                return
            
            # 선택된 polygon마다 병합 수행
            for idx in selected_shapes:
                print(f"\n▶ 처리 중인 polygon 인덱스: {idx}")
                polygon = shapes_layer.data[idx]
                original_mask = polygon2mask((h, w), polygon).astype(bool)

                # 마스크 병합
                combined_mask = np.logical_or(original_mask, extension_mask)
                print(f"  병합된 마스크 True 개수: {combined_mask.sum()}")
                
                # 마스크에서 contour 추출
                contours = find_contours(combined_mask.astype(float), level=0.5)
                if not contours:
                    print("❌ contour 추출 실패")
                    continue
                
                # 가장 큰 contour 선택
                new_contour = max(contours, key=lambda x: len(x))
                new_contour = close_polygon_if_needed(new_contour)
                print(f"  최종 polygon 포인트 수: {len(new_contour)}")
                
                # 스타일 유지 + polygon 갱신
                face_color = shapes_layer.face_color[idx]
                edge_color = shapes_layer.edge_color[idx]
                shapes_layer.remove_selected()
                
                # new_contour를 올바른 형태로 변환
                if len(new_contour.shape) == 2 and new_contour.shape[1] == 2:
                    new_contour = new_contour.reshape(1, -1, 2)  # (1, N, 2) 형태로 변환
                
                shapes_layer.add(
                    new_contour,
                    shape_type='polygon',
                    edge_color=edge_color,
                    face_color=face_color,
                    edge_width=4
                )
                print("polygon 업데이트 완료")
                
            shapes_layer.refresh()
            
            # 선택 상태 초기화
            shapes_layer._value = (None, None)
            shapes_layer._selected_data = set()
            shapes_layer._selected_box = None
            
            # 레이어 제거 및 키 해제
            if "Extension Polygon" in self._viewer.layers:
                self._viewer.layers.remove("Extension Polygon")
            if "Brush Extension" in self._viewer.layers:
                self._viewer.layers.remove("Brush Extension")
            
            self._viewer.bind_key('Enter', None)
            
            # Apply 버튼과 Brush Size 슬라이더 제거
            for widget in self:
                if isinstance(widget, PushButton) and widget.text == "Apply Extension":
                    self.remove(widget)
                elif isinstance(widget, Slider) and widget.label == "Brush Size":
                    self.remove(widget)
                    
            print("polygon 확장이 적용되었습니다.")

        except Exception as e:
            print(f"❌ polygon 확장 중 오류 발생: {e}")
            import traceback
            print(traceback.format_exc())

    def smooth_polygon(self, contour: np.ndarray, num_points: int = 100) -> np.ndarray:
        if len(contour) < 3:
            return contour  # 너무 짧으면 스킵

        x = contour[:, 1]
        y = contour[:, 0]
        try:
            tck, _ = splprep([x, y], s=3)
            x_new, y_new = splev(np.linspace(0, 1, num_points), tck)
            return np.vstack([y_new, x_new]).T
        except Exception:
            return contour  # 보간 실패 시 원본 리턴

def close_polygon_if_needed( points: np.ndarray) -> np.ndarray:
    if len(points) >= 3 and not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    return points

def mask_to_smoothed_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    h, w = mask.shape
    blurred = cv2.GaussianBlur(mask.astype(np.uint8) * 255, (5, 5), 0)
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    coords = approx.reshape(-1, 2)
    return approx.reshape(-1, 2)



import atexit
@atexit.register
def cleanup():
    try:
        app = QApplication.instance()
        if app:
            app.quit()
    except Exception as e:
        print(f"Cleanup error: {e}")



















































