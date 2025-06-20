# image2template

**image2template**는 이미지를 업로드하면 객체를 자동 분리하고, 배경을 복원하며 편집 가능한 템플릿 형태로 재구성할 수 있는 napari 기반 플러그인입니다.

이 플러그인은 포스터, 카드뉴스 등 디자인 결과물을 쉽게 편집하고 재활용할 수 있도록 돕습니다.

> 본 프로젝트는 **경희대학교 소프트웨어융합학과의 데이터분석캡스톤디자인 수업**의 결과물입니다.

---

## 주요 기능

- 객체 자동 분리 (Segment Anything Model 기반)
- 객체 제거 후 배경 자동 복원 (inpainting)
- 개별 객체 편집 (위치 이동, 색상 변경, 삭제 등)
- 브러시 및 도형을 활용한 객체 확장
- 레이어 기반 저장 및 템플릿 재사용

---

## 설치 방법

```bash
# (선택) 가상환경 생성 및 활성화 권장
python -m venv venv
source venv/bin/activate  # Windows는 venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# napari 실행 -> Plugin에서 sam 선택
napari

# 바로 napari sam 실행
napari <your image path> -w napari-segment-anything 'Segment Anything'

```

---

## 사용법 요약


1.	이미지를 불러온 후 우측의 "Auto. Segm."를 누르면 모든 객체 자동 분리를 시작합니다.
2.	Editable polygon에서 분리할 객체와 배경으로 남길 객체를 선택할 수 있습니다.
3.	분리가 끝나면 "Final polygons" layer에서 개별 객체의 위치/크기/색상을 조정할 수 있습니다.
4.	"Final polygons" layer에서 객체 선택후 Brush Extension 또는 Polygon Extension 버튼을 클릭하여 원하는 객체를 확장하거나 수정할 수 있습니다.
5.	객체들이 제거된 배경은 "LAMA Inpainted Background" layer에 저장됩니다.
6.	개별 객체 선택 후 "Export Selected Object"을 누르면 단색의 다각형으로, "Export Polygon as Image"를 누르면 원본 객체 그대로가 저장됩니다.
7.  Text Layer 플러그인을 사용하여 텍스트의 크기·각도·색상을 조정할 수 있습니다.
8.	결과는 File > Save > Save screenshot으로 png/jpg로 내보내거나 File > Save > Save all layers로 svg/ai 형태로 저장할 수 있습니다.

https://www.youtube.com/watch?v=LHBGCCF6A94

🔍 추가 사용 영상은 docs/ 폴더 참고

---

## 프로젝트 구조

```
image2template/
│
├── src/
│   └── napari_segment_anything/
│       ├── _widget.py           # 주요 위젯 및 UI 동작
│       ├── utils/               # 도우미 함수들
│       ├── lama_inpaint.py      # LaMa 모델을 사용한 이미지 인페인팅 구현
│       ├── sd_inpaint.py        # Stable Diffusion을 사용한 이미지 인페인팅 구현
│       └── ...
├── requirements.txt
├── README.md
└── docs/
    ├── usage.gif
    └── example_result.png
    
```

---

## 활용 예시

- 기존 이미지 기반 포스터를 쉽게 편집해야 할 때
- 텍스트만 바꿔서 유사 디자인을 제작해야 할 때

---

## 🚨 콘텐츠 사용 시 유의사항

**image2template는 학술 연구 및 교육 목적을 위해 개발된 도구**로, 사용자가 직접 업로드한 이미지를 기반으로 작업이 이루어집니다.

사용자는 업로드하는 이미지에 대해 **직접 저작권을 보유하거나, 정당한 사용 권한을 확보한 경우에 한해** 본 도구를 사용해 주시기 바랍니다.   
반드시 이미지 및 콘텐츠 사용 시 저작권 등 관련 사항을 **사전에 확인해주시길 권장**드리며 이를 확인하지 않고 사용함으로써 발생할 수 있는 이슈는 프로젝트 외부의 사항입니다.

---

## 라이선스
[Apache Software License 2.0](https://www.apache.org/licenses/LICENSE-2.0) 라이선스 조건에 따라 배포되는 "napari-segment-anything"은 무료 오픈 소스 소프트웨어입니다.
