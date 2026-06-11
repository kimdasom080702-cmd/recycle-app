# recycle_all_in_one_complete_v9.py
# ------------------------------------------------------------
# 재활용 쓰레기 분류 + 오염도 판정 올인원 버전 v9
# 개선 사항:
#   1. 이미지 업로드 결과 유지: st.session_state를 사용하여 버튼 클릭 후 결과가 사라지지 않도록 수정
#   2. 배포 최적화: Streamlit Community Cloud 등 클라우드 환경에서 안정적으로 작동하도록 경로 및 설정 조정
#   3. 분석 영역 시각화 및 분석 버튼 기능 강화
# ------------------------------------------------------------

import argparse
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

def get_work_dir() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "RecycleClassifier" / "auto_train_output"
    return BASE_DIR / "auto_train_output"

WORK_DIR = get_work_dir()
FEEDBACK_DIR = BASE_DIR / "feedback_samples"

MATERIAL_MODEL_PATH = MODELS_DIR / "material_cls.pt"
CONTAMINATION_MODEL_PATH = MODELS_DIR / "contamination_cls.pt"
TRAIN_INFO_PATH = MODELS_DIR / "train_info.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MATERIAL_MAP = {
    "유리": "glass", "종이": "paper", "캔": "can",
    "비닐": "vinyl", "스티로폼": "styrofoam",
    "플라스틱": "plastic", "페트": "plastic",
}

MATERIAL_LABELS_KOR = {
    "glass": "유리", "paper": "종이", "can": "캔",
    "vinyl": "비닐", "styrofoam": "스티로폼",
    "plastic": "플라스틱", "unknown": "판정불가",
}

CONTAMINATION_LABELS_KOR = {
    "clean": "깨끗함", "dirty": "오염됨", "uncertain": "판정불가",
}

RECYCLE_EXCHANGE_INFO = {
    "paper": {
        "title": "📄 종이류 / 종이팩 교환",
        "content": "우유팩, 두유팩 등 종이팩을 깨끗이 씻어 말려 거주지 주민센터로 가져가시면 **화장지** 또는 **종량제 봉투**로 교환해 드립니다.",
        "tip": "일반 폐지와 종이팩은 분리해서 배출해야 재활용률이 높습니다!"
    },
    "can": {
        "title": "🥫 캔류 / 폐건전지 교환",
        "content": "다 쓴 건전지를 주민센터나 일부 대형마트의 수거함에 가져가시면 **새 건전지**로 교환해 주는 사업이 활발합니다.",
        "tip": "캔은 내용물을 비우고 압착하여 배출하면 부피를 줄일 수 있습니다."
    },
    "plastic": {
        "title": "🧴 투명 페트병 보상",
        "content": "일부 지자체에서는 '순환자원 회수로봇(네프론 등)'을 운영합니다. 투명 페트병을 넣으면 **현금 포인트**를 적립해 드립니다.",
        "tip": "라벨을 반드시 제거하고 압착해서 넣어주세요!"
    },
    "glass": {
        "title": "🍾 빈 병 보증금 반환",
        "content": "소주병, 맥주병 등 '보증금 환불 문구'가 있는 병은 편의점이나 대형마트에 반납하고 **보증금**을 돌려받을 수 있습니다.",
        "tip": "이물질이 들어간 병은 반환이 거부될 수 있습니다."
    },
    "vinyl": {
        "title": "🛍️ 비닐류 배출",
        "content": "깨끗하게 모은 비닐은 고형연료 등으로 재활용됩니다. 오염된 비닐은 종량제 봉투에 버려주세요.",
        "tip": "색상에 상관없이 깨끗한 비닐은 모두 재활용 대상입니다."
    },
    "styrofoam": {
        "title": "📦 스티로폼 배출",
        "content": "흰색 스티로폼은 테이프와 운송장을 완전히 제거한 후 깨끗한 상태로 배출해 주세요.",
        "tip": "코팅되거나 색깔이 있는 스티로폼은 재활용이 어렵습니다."
    }
}

REQUIRED_PACKAGES = [
    ("cv2", "opencv-python-headless"), # 서버 배포용은 headless 권장
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("PIL", "pillow"),
    ("streamlit", "streamlit"),
    ("ultralytics", "ultralytics"),
]

def ensure_packages() -> None:
    for module_name, package_name in REQUIRED_PACKAGES:
        if importlib.util.find_spec(module_name) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

ensure_packages()

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ============================================================
# 이미지 분석 및 시각화 로직
# ============================================================

def find_object_bbox(image_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        margin = 0.1
        return int(w * margin), int(h * margin), int(w * (1 - margin)), int(h * (1 - margin))
    
    c = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    if bw * bh < (w * h * 0.01):
        margin = 0.1
        return int(w * margin), int(h * margin), int(w * (1 - margin)), int(h * (1 - margin))
        
    pad = int(max(bw, bh) * 0.1)
    return max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)

def draw_analysis_box(image_bgr: np.ndarray, bbox: Tuple[int, int, int, int], label: str) -> np.ndarray:
    out = image_bgr.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
    mask = np.zeros_like(out)
    cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
    out_overlay = cv2.addWeighted(out, 0.7, np.zeros_like(out), 0.3, 0)
    out = np.where(mask == 255, out, out_overlay)
    text = f"Analyzing: {label}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (x1, y1 - th - 15), (x1 + tw + 10, y1), (0, 255, 0), -1)
    cv2.putText(out, text, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return out

# ============================================================
# 예측 및 Streamlit UI
# ============================================================

def predict(model, img_bgr):
    if model is None: return "unknown", 0.0
    results = model.predict(img_bgr, verbose=False)
    if not results or not results[0].probs: return "unknown", 0.0
    idx = int(results[0].probs.top1)
    conf = float(results[0].probs.top1conf)
    label = results[0].names[idx]
    return label, conf

def main():
    st.set_page_config(page_title="스마트 재활용 분류기 v9", layout="wide")
    st.title("♻️ 스마트 재활용 분류기")
    
    # 상태 유지를 위한 session_state 초기화
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "last_uploaded" not in st.session_state:
        st.session_state.last_uploaded = None

    @st.cache_resource
    def load_models():
        m = YOLO(str(MATERIAL_MODEL_PATH)) if MATERIAL_MODEL_PATH.exists() else None
        c = YOLO(str(CONTAMINATION_MODEL_PATH)) if CONTAMINATION_MODEL_PATH.exists() else None
        return m, c

    m_model, c_model = load_models()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        tab1, tab2 = st.tabs(["📁 이미지 업로드", "📷 카메라 촬영"])
        img_input = None
        
        with tab1:
            uploaded = st.file_uploader("재활용품 이미지를 선택하세요", type=["jpg", "jpeg", "png", "webp"])
            if uploaded:
                img_input = Image.open(uploaded)
                st.image(img_input, caption="업로드된 이미지", use_container_width=True)
                
                # 새 이미지가 업로드되면 이전 결과 초기화
                if st.session_state.last_uploaded != uploaded.name:
                    st.session_state.analysis_result = None
                    st.session_state.last_uploaded = uploaded.name
                
                if st.button("🔍 분석하기", key="btn_upload"):
                    # 분석 실행 및 결과 저장
                    with st.spinner("이미지를 분석 중입니다..."):
                        img_np = np.array(img_input)
                        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        bbox = find_object_bbox(img_bgr)
                        x1, y1, x2, y2 = bbox
                        crop = img_bgr[y1:y2, x1:x2]
                        m_label, m_conf = predict(m_model, crop if crop.size > 0 else img_bgr)
                        c_label, c_conf = predict(c_model, crop if crop.size > 0 else img_bgr)
                        
                        st.session_state.analysis_result = {
                            "m_label": m_label, "m_conf": m_conf,
                            "c_label": c_label, "c_conf": c_conf,
                            "bbox": bbox, "img_bgr": img_bgr
                        }
                    
        with tab2:
            camera = st.camera_input("카메라로 재활용품을 촬영하세요")
            if camera:
                img_input = Image.open(camera)
                if st.button("🔍 분석하기", key="btn_camera"):
                    with st.spinner("분석 중..."):
                        img_np = np.array(img_input)
                        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        bbox = find_object_bbox(img_bgr)
                        x1, y1, x2, y2 = bbox
                        crop = img_bgr[y1:y2, x1:x2]
                        m_label, m_conf = predict(m_model, crop if crop.size > 0 else img_bgr)
                        c_label, c_conf = predict(c_model, crop if crop.size > 0 else img_bgr)
                        st.session_state.analysis_result = {
                            "m_label": m_label, "m_conf": m_conf,
                            "c_label": c_label, "c_conf": c_conf,
                            "bbox": bbox, "img_bgr": img_bgr
                        }

    # 결과 출력 (session_state에 결과가 있을 때만)
    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        with col2:
            st.subheader("🔍 판정 결과")
            res_m_kor = MATERIAL_LABELS_KOR.get(res["m_label"], res["m_label"])
            res_c_kor = CONTAMINATION_LABELS_KOR.get(res["c_label"], res["c_label"])
            
            m_col, c_col = st.columns(2)
            m_col.metric("품목", res_m_kor, f"{res['m_conf']*100:.1f}%")
            c_col.metric("상태", res_c_kor, f"{res['c_conf']*100:.1f}%")
            
            if res["c_label"] == "dirty":
                st.warning(f"⚠️ **{res_m_kor}**이(가) 오염되었습니다. 세척 후 배출해 주세요.")
            else:
                st.success(f"✅ 깨끗한 **{res_m_kor}**입니다. 바로 재활용이 가능합니다.")

            info = RECYCLE_EXCHANGE_INFO.get(res["m_label"])
            if info:
                with st.expander(f"💡 {info['title']} 정보 확인", expanded=True):
                    st.write(info['content'])
                    st.info(f"**Tip:** {info['tip']}")
            
            viz_img = draw_analysis_box(res["img_bgr"], res["bbox"], res_m_kor)
            st.image(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB), caption="분석 결과 시각화", use_container_width=True)

if __name__ == "__main__":
    if "streamlit" in sys.modules and st.runtime.exists():
        main()
    else:
        subprocess.run(["streamlit", "run", __file__])
