from qtpy.QtWidgets import (
    QLineEdit, QWidget, QGridLayout, QLabel, QSpinBox, 
    QHBoxLayout, QPushButton, QColorDialog
)
from qtpy.QtGui import QColor, QCursor
from qtpy.QtCore import Qt
import numpy as np
import napari
from napari_plugin_engine import napari_hook_implementation
from napari.layers import Shapes
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit

_TEXT_SYMBOL = "t"
_INITIAL_TEXT_COLOR = "white"
_INITIAL_FONT_SIZE = 6
_MIN_FONT_SIZE = 2
_MAX_FONT_SIZE = 48
_MIN_SHAPE_X_SIZE = 16
_MIN_SHAPE_Y_SIZE = 16

class TextLayerOverview(QWidget):
    def __init__(self, napari_viewer):
        try:
            super().__init__()
            self.setMinimumWidth(320)  # 전체 패널 최소 너비 더 크게
            self.viewer = napari_viewer
            self.layer = None  # 초기에는 None으로 설정
            
            # UI 초기화
            self.setLayout(QGridLayout())
            self.layout().setAlignment(Qt.AlignTop)
            self.layout().setContentsMargins(10, 10, 10, 10)
            self.layout().setSpacing(12)
            
            # 공통 폰트 설정
            common_font = self.font()
            common_font.setPointSize(14)
            self.setFont(common_font)
            
            # 색상 선택 버튼
            self.color_btn = QPushButton("색상 선택", self)
            self.color_btn.setMinimumHeight(40)
            self.color_btn.setMinimumWidth(120)
            self.color_btn.setFont(common_font)
            self.color_btn.clicked.connect(self._choose_color)
            self.layout().addWidget(self.color_btn, 0, 0)
            
            # 크기 조절
            frame = QWidget(self)
            frame.setLayout(QHBoxLayout())
            frame.layout().setContentsMargins(0, 0, 0, 0)
            frame.layout().setSpacing(10)
            frame.setMinimumWidth(180)  # 프레임 최소 너비 추가
            
            self._font_size_spinbox = QSpinBox(frame)
            self._font_size_spinbox.setRange(_MIN_FONT_SIZE, _MAX_FONT_SIZE)
            self._font_size_spinbox.setValue(_INITIAL_FONT_SIZE)
            self._font_size_spinbox.setMinimumHeight(40)
            self._font_size_spinbox.setMinimumWidth(100)
            self._font_size_spinbox.setMaximumWidth(120)
            self._font_size_spinbox.setFont(common_font)
            
            @self._font_size_spinbox.valueChanged.connect
            def _(e):
                if self.layer is not None:
                    size = self._font_size_spinbox.value()
                    self.layer.text.size = size
            
            label_size = QLabel("크기:", frame)
            label_size.setFont(common_font)
            frame.layout().addWidget(label_size)
            frame.layout().addWidget(self._font_size_spinbox)
            self.layout().addWidget(frame, 1, 0)
            
            # 회전 조절 프레임
            rot_frame = QWidget(self)
            rot_frame.setLayout(QHBoxLayout())
            rot_frame.layout().setContentsMargins(0, 0, 0, 0)
            rot_frame.layout().setSpacing(10)
            rot_frame.setMinimumWidth(180)

            self._rot_spin_box = QSpinBox(rot_frame)
            self._rot_spin_box.setRange(-180, 180)
            self._rot_spin_box.setValue(0)
            self._rot_spin_box.setSingleStep(5)
            self._rot_spin_box.setMinimumHeight(40)
            self._rot_spin_box.setMinimumWidth(100)
            self._rot_spin_box.setMaximumWidth(120)
            self._rot_spin_box.setFont(common_font)

            @self._rot_spin_box.valueChanged.connect
            def _(e):
                if self.layer is not None:
                    deg = self._rot_spin_box.value()
                    self.layer.text.rotation = deg

            label_rot = QLabel("text rotation", rot_frame)
            label_rot.setFont(common_font)
            rot_frame.layout().addWidget(label_rot)
            rot_frame.layout().addWidget(self._rot_spin_box)
            self.layout().addWidget(rot_frame, 2, 0)
            
            # 앵커 버튼들
            frame = QWidget(self)
            frame.setLayout(QGridLayout())
            frame.layout().setContentsMargins(0, 0, 0, 0)
            frame.layout().setSpacing(8)
            frame.setFixedWidth(140)
            
            self._button_ul = QPushButton("◤", frame)
            self._button_ur = QPushButton("◥", frame)
            self._button_ll = QPushButton("◣", frame)
            self._button_lr = QPushButton("◢", frame)
            self._button_ct = QPushButton("●", frame)
            for btn in [self._button_ul, self._button_ur, self._button_ll, self._button_lr, self._button_ct]:
                btn.setMinimumHeight(40)
                btn.setMinimumWidth(40)
                btn.setFont(common_font)
            
            frame.layout().addWidget(self._button_ul, 0, 0, 2, 2)
            frame.layout().addWidget(self._button_ur, 0, 2, 2, 2)
            frame.layout().addWidget(self._button_ll, 2, 0, 2, 2)
            frame.layout().addWidget(self._button_lr, 2, 2, 2, 2)
            frame.layout().addWidget(self._button_ct, 1, 1, 2, 2)
            
            @self._button_ul.clicked.connect
            def _(e):
                if self.layer is not None:
                    self.layer.text.anchor = "upper_left"
                
            @self._button_ur.clicked.connect
            def _(e):
                if self.layer is not None:
                    self.layer.text.anchor = "upper_right"
                
            @self._button_ll.clicked.connect
            def _(e):
                if self.layer is not None:
                    self.layer.text.anchor = "lower_left"
                
            @self._button_lr.clicked.connect
            def _(e):
                if self.layer is not None:
                    self.layer.text.anchor = "lower_right"
                
            @self._button_ct.clicked.connect
            def _(e):
                if self.layer is not None:
                    self.layer.text.anchor = "center"
            
            frame.setToolTip("텍스트 앵커")
            self.layout().addWidget(frame, 3, 0)
            
            # 새 텍스트 레이어 버튼
            button = QPushButton("새 텍스트 레이어", self)
            button.setMinimumHeight(40)
            button.setMinimumWidth(160)
            button.setFont(common_font)
            self.layout().addWidget(button, 4, 0)
            button.clicked.connect(self._add_text_layer)
            
            # 초기 텍스트 레이어 생성
            self._add_text_layer()
            
        except Exception as e:
            print(f"TextLayerOverview 초기화 중 오류 발생: {str(e)}")
            raise

    def _choose_color(self):
        try:
            color = QColorDialog.getColor()
            if color.isValid() and self.layer is not None:
                self.layer.text.color = color.name()
        except Exception as e:
            print(f"색상 선택 중 오류 발생: {str(e)}")

    def _add_text_layer(self, e=None):
        try:
            # 새 텍스트 레이어 추가
            layer = Shapes(
                ndim=2,
                shape_type="rectangle",
                name="Text Layer",
                properties={_TEXT_SYMBOL: np.array([], dtype="<U32")},
                face_color="transparent",
                edge_color="transparent",
                blending="additive",
                opacity=1,
                text=dict(
                    text="{" + _TEXT_SYMBOL + "}", 
                    size=_INITIAL_FONT_SIZE,
                    color=_INITIAL_TEXT_COLOR,
                    anchor="center"
                )
            )
            layer.mode = "add_rectangle"
            self.layer = layer
            
            # F2 키로 텍스트 편집
            @layer.bind_key("F2", overwrite=True)
            def edit_selected(layer):
                try:
                    selected = list(layer.selected_data)
                    if layer.nshapes == 0:
                        return
                    elif len(selected) == 0:
                        i = -1
                    else:
                        i = selected[-1]
                    data = layer.data[i]
                    center = np.mean(data, axis=0)
                    screen_coords = _get_data_coords_in_screen(center, self.viewer)
                    self._enter_editing_mode(i, screen_coords)
                except Exception as e:
                    print(f"텍스트 편집 중 오류 발생: {str(e)}")
            
            # Enter 키로 새 텍스트 추가
            @layer.bind_key("Enter", overwrite=True)
            def add(layer):
                try:
                    self.layer.selected_data = {}
                    
                    if layer.nshapes == 0:
                        next_data = np.array([
                            [0, 0],
                            [0, _MIN_SHAPE_X_SIZE],
                            [_MIN_SHAPE_Y_SIZE, _MIN_SHAPE_X_SIZE],
                            [_MIN_SHAPE_Y_SIZE, 0]
                        ])
                        layer.add_rectangles(next_data)
                    elif layer.nshapes == 1:
                        next_data = layer.data[-1].copy()
                        next_data[:, -2] += _MIN_SHAPE_Y_SIZE
                        next_data[:, -1] += _MIN_SHAPE_X_SIZE
                        layer.add(next_data, shape_type=layer.shape_type[-1])
                    else:
                        dr = np.mean(layer.data[-1], axis=0) - np.mean(layer.data[-2], axis=0)
                        next_data = layer.data[-1] + dr
                        layer.add(next_data, shape_type=layer.shape_type[-1])
                    
                    center = np.mean(next_data, axis=0)
                    screen_coords = _get_data_coords_in_screen(center, self.viewer)
                    self._enter_editing_mode(-1, screen_coords)
                except Exception as e:
                    print(f"새 텍스트 추가 중 오류 발생: {str(e)}")
            
            # 방향키로 이동
            @layer.bind_key("Left", overwrite=True)
            def left(layer):
                try:
                    _translate_shape(layer, -1, -1)
                except Exception as e:
                    print(f"왼쪽 이동 중 오류 발생: {str(e)}")
                
            @layer.bind_key("Right", overwrite=True)
            def right(layer):
                try:
                    _translate_shape(layer, -1, 1)
                except Exception as e:
                    print(f"오른쪽 이동 중 오류 발생: {str(e)}")
                
            @layer.bind_key("Up", overwrite=True)
            def up(layer):
                try:
                    _translate_shape(layer, -2, -1)
                except Exception as e:
                    print(f"위로 이동 중 오류 발생: {str(e)}")
                
            @layer.bind_key("Down", overwrite=True)
            def down(layer):
                try:
                    _translate_shape(layer, -2, 1)
                except Exception as e:
                    print(f"아래로 이동 중 오류 발생: {str(e)}")
            
            # 크기 조절 단축키
            @layer.bind_key("Control-Shift-<", overwrite=True)
            def size_down(layer):
                try:
                    layer.text.size = max(_MIN_FONT_SIZE, layer.text.size - 1)
                except Exception as e:
                    print(f"크기 감소 중 오류 발생: {str(e)}")
            
            @layer.bind_key("Control-Shift->", overwrite=True)
            def size_up(layer):
                try:
                    layer.text.size = min(_MAX_FONT_SIZE, layer.text.size + 1)
                except Exception as e:
                    print(f"크기 증가 중 오류 발생: {str(e)}")
            
            # 더블클릭으로 편집
            @layer.mouse_double_click_callbacks.append
            def double_clicked(layer, event):
                try:
                    i, _ = layer.get_value(
                        event.position,
                        view_direction=event.view_direction,
                        dims_displayed=event.dims_displayed,
                        world=True
                    )
                    if i is None:
                        return None
                    self._enter_editing_mode(i)
                except Exception as e:
                    print(f"더블클릭 편집 중 오류 발생: {str(e)}")
            
            # 드래그로 새 텍스트 추가
            @layer.mouse_drag_callbacks.append
            def _(layer, e):
                try:
                    if layer.mode not in ("add_rectangle", "add_ellipse", "add_line"):
                        return
                    x0, y0 = _get_mouse_coords_in_screen(self.viewer)
                    yield
                    while e.type == "mouse_move":
                        yield
                    x1, y1 = _get_mouse_coords_in_screen(self.viewer)
                    
                    # 최소 크기 보장
                    data = layer.data
                    dx = abs(x1 - x0)
                    if dx <= _MIN_SHAPE_X_SIZE:
                        center = np.mean(layer.data[-1][:, -1])
                        xsmall = center - _MIN_SHAPE_X_SIZE/2
                        xlarge = center + _MIN_SHAPE_X_SIZE/2
                        data[-1][:, -1] = [xsmall, xlarge, xlarge, xsmall]
                    
                    dy = abs(y1 - y0)
                    if dy <= _MIN_SHAPE_Y_SIZE:
                        center = np.mean(layer.data[-1][:, -2])
                        ysmall = center - _MIN_SHAPE_Y_SIZE/2
                        ylarge = center + _MIN_SHAPE_Y_SIZE/2
                        data[-1][:, -2] = [ysmall, ysmall, ylarge, ylarge]
                    
                    layer.data = data
                    self._enter_editing_mode(-1, ((x0+x1)/2, (y0+y1)/2))
                except Exception as e:
                    print(f"드래그 추가 중 오류 발생: {str(e)}")
            
            self.viewer.add_layer(layer)
            return None
        except Exception as e:
            print(f"텍스트 레이어 추가 중 오류 발생: {str(e)}")
            raise

    def _enter_editing_mode(self, i, position=None):
        try:
            if self.layer is None:
                return None
                
            # 텍스트 편집 모드 진입
            self.layer.current_properties = {_TEXT_SYMBOL: np.array([""], dtype="<U32")}
            if position is not None:
                x, y = position
            else:
                x, y = _get_mouse_coords_in_screen(self.viewer)
            
            # 편집 위젯 생성
            line = QLineEdit(self.viewer.window._qt_window)
            edit_geometry = line.geometry()
            edit_geometry.setWidth(140)
            edit_geometry.moveLeft(x)
            edit_geometry.moveTop(y)
            line.setGeometry(edit_geometry)
            f = line.font()
            f.setPointSize(20)
            line.setFont(f)
            line.setText(self.layer.text.values[i])
            line.setHidden(False)
            line.setFocus()
            line.selectAll()
            
            @line.textChanged.connect
            def _():
                try:
                    old = self.layer.properties.get(_TEXT_SYMBOL, [""]*len(self.layer.data))
                    old[i] = line.text().strip()
                    self.layer.text.refresh_text({_TEXT_SYMBOL: old})
                except Exception as e:
                    print(f"텍스트 변경 중 오류 발생: {str(e)}")
            
            @line.editingFinished.connect
            def _():
                try:
                    line.setHidden(True)
                    line.deleteLater()
                except Exception as e:
                    print(f"편집 종료 중 오류 발생: {str(e)}")
            
            return None
        except Exception as e:
            print(f"편집 모드 진입 중 오류 발생: {str(e)}")
            return None

def _get_mouse_coords_in_screen(viewer):
    try:
        window_geo = viewer.window._qt_window.geometry()
        pos = QCursor.pos()
        x = pos.x() - window_geo.x()
        y = pos.y() - window_geo.y()
        return x, y
    except Exception as e:
        print(f"마우스 좌표 변환 중 오류 발생: {str(e)}")
        return 0, 0

def _translate_shape(layer, ind, direction):
    try:
        data = layer.data
        selected = layer.selected_data
        for i in selected:
            data[i][:, ind] += direction
        layer.data = data
        layer.selected_data = selected
        layer._set_highlight()
        return None
    except Exception as e:
        print(f"도형 이동 중 오류 발생: {str(e)}")
        return None

def _get_data_coords_in_screen(coords, viewer):
    try:
        dr = viewer.window._qt_window.centralWidget().geometry()
        w = dr.width()
        h = dr.height()
        canvas_center = np.array([dr.y(), dr.x()]) + np.array([h, w])/2
        crds = canvas_center + (coords - viewer.camera.center[-2:])* viewer.camera.zoom
        return crds.astype(int)[::-1]
    except Exception as e:
        print(f"데이터 좌표 변환 중 오류 발생: {str(e)}")
        return np.array([0, 0])

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return TextLayerOverview