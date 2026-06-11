# recycle_all_in_one_complete_v10.py
# ------------------------------------------------------------
# 재활용 쓰레기 분류 + 오염도 판정 올인원 버전 v10
# 개선 사항:
#   1. 물체 감지 정확도 향상: 배경(전등)이나 손가락을 피하고 중앙 물체에 집중하도록 로직 강화
#   2. 한글 깨짐 해결: PIL을 사용하여 결과 화면에 한글 라벨이 정상적으로 표시되도록 수정
#   3. 분석 영역 시각화 개선: 박스 가독성 및 포커스 효과 강화
#   4. 상태 유지 로직 최적화: Streamlit 세션 상태를 통한 안정적인 결과 출력
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
# 패키지 및 환경 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MATERIAL_MODEL_PATH = MODELS_DIR / "material_cls.pt"
CONTAMINATION_MODEL_PATH = MODELS_DIR / "contamination_cls.pt"

REQUIRED_PACKAGES = [
    ("cv2", "opencv-python-headless"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("streamlit", "streamlit"),
    ("ultralytics", "ultralytics"),
]

def ensure_packages():
    for module_name, package_name in REQUIRED_PACKAGES:
        if importlib.util.find_spec(module_name) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

ensure_packages()

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 상수 및 데이터
# ============================================================

MATERIAL_LABELS_KOR = {
    "glass": "유리", "paper": "종이", "can": "캔",
    "vinyl": "비닐", "styrofoam": "스티로폼",
    "plastic": "플라스틱", "unknown": "판정불가",
}

CONTAMINATION_LABELS_KOR = {
    "clean": "깨끗함", "dirty": "오염됨", "uncertain": "판정불가",
}

RECYCLE_EXCHANGE_INFO = {
    "paper": {"title": "📄 종이류 / 종이팩 교환", "content": "우유팩 등을 깨끗이 씻어 주민센터로 가져가시면 화장지나 종량제 봉투로 교환해 드립니다."},
    "can": {"title": "🥫 캔류 / 폐건전지 교환", "content": "다 쓴 건전지를 주민센터 수거함에 가져가시면 새 건전지로 교환해 줍니다."},
    "plastic": {"title": "🧴 투명 페트병 보상", "content": "순환자원 회수로봇(네프론 등)에 투명 페트병을 넣으면 포인트를 적립해 드립니다."},
    "glass": {"title": "🍾 빈 병 보증금 반환", "content": "보증금 문구가 있는 병은 마트/편의점에서 보증금을 돌려받을 수 있습니다."},
    "vinyl": {"title": "🛍️ 비닐류 배출", "content": "깨끗한 비닐은 분리 배출하면 고형연료 등으로 재활용됩니다."},
    "styrofoam": {"title": "📦 스티로폼 배출", "content": "흰색 스티로폼은 테이프 제거 후 깨끗하게 배출해 주세요."},
}

# ============================================================
# 분석 영역 감지 로직 (개선됨)
# ============================================================

def get_refined_bbox(image_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """배경 노이즈를 피하고 실제 물체가 있을 법한 중앙 영역을 정밀하게 추출합니다."""
    h, w = image_bgr.shape[:2]
    
    # 1. 이미지 중앙부만 집중 (가장자리 전등 등 노이즈 제거)
    center_y, center_x = h // 2, w // 2
    roi_h, roi_w = int(h * 0.7), int(w * 0.7)
    y1_roi, x1_roi = max(0, center_y - roi_h // 2), max(0, center_x - roi_w // 2)
    roi = image_bgr[y1_roi:y1_roi+roi_h, x1_roi:x1_roi+roi_w]
    
    # 2. ROI 내에서 물체 찾기
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 적응형 임계값 처리를 통해 조명 영향 최소화
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 중앙과 가장 가까운 큰 컨투어 찾기
        best_cnt = None
        min_dist = float('inf')
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (roi_h * roi_w * 0.05): continue # 너무 작은 건 무시
            
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            dist = ((cx - roi_w//2)**2 + (cy - roi_h//2)**2)**0.5
            
            if dist < min_dist:
                min_dist = dist
                best_cnt = cnt
        
        if best_cnt is not None:
            x, y, bw, bh = cv2.boundingRect(best_cnt)
            # 전체 이미지 좌표로 변환
            pad = 20
            return (max(0, x + x1_roi - pad), max(0, y + y1_roi - pad), 
                    min(w, x + x1_roi + bw + pad), min(h, y + y1_roi + bh + pad))

    # 물체를 못 찾으면 기본 중앙 영역 반환
    m = 0.2
    return int(w*m), int(h*m), int(w*(1-m)), int(h*(1-m))

# ============================================================
# 시각화 로직 (한글 지원)
# ============================================================

def draw_info_pil(image_bgr: np.ndarray, bbox: Tuple[int, int, int, int], label: str) -> np.ndarray:
    """PIL을 사용하여 한글이 포함된 분석 결과를 이미지에 그립니다."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img, "RGBA")
    
    x1, y1, x2, y2 = bbox
    
    # 1. 배경 어둡게 처리 (박스 외 영역)
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    draw_ov.rectangle([0, 0, pil_img.width, pil_img.height], fill=(0, 0, 0, 100))
    draw_ov.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 0)) # 박스 영역은 투명하게
    pil_img.paste(Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB"))
    
    # 2. 박스 그리기
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=5)
    
    # 3. 한글 텍스트 쓰기
    try:
        # 나눔고딕 등 시스템 폰트 시도, 없으면 기본 폰트
        font_paths = ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "arial.ttf"]
        font = None
        for p in font_paths:
            if os.path.exists(p):
                font = ImageFont.truetype(p, 25)
                break
        if font is None: font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
        
    text = f"분석 대상: {label}"
    # 텍스트 배경
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    draw.rectangle([x1, y1 - th - 10, x1 + tw + 10, y1], fill=(0, 255, 0))
    draw.text((x1 + 5, y1 - th - 5), text, font=font, fill=(0, 0, 0))
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ============================================================
# Streamlit 앱 메인
# ============================================================

try:
    from ultralytics import YOLO
except:
    YOLO = None

def predict(model, img_bgr):
    if model is None: return "unknown", 0.0
    results = model.predict(img_bgr, verbose=False)
    if not results or not results[0].probs: return "unknown", 0.0
    idx = int(results[0].probs.top1)
    conf = float(results[0].probs.top1conf)
    return results[0].names[idx], conf

def main():
    st.set_page_config(page_title="스마트 재활용 분류기 v10", layout="wide")
    st.title("♻️ 스마트 재활용 분류기 v10")
    st.info("💡 중앙에 물체를 크게 위치시키면 더 정확하게 분석됩니다.")

    if "result" not in st.session_state: st.session_state.result = None

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
            uploaded = st.file_uploader("이미지 선택", type=["jpg", "png", "jpeg"])
            if uploaded:
                img_input = Image.open(uploaded)
                st.image(img_input, use_container_width=True)
                if st.button("🔍 분석 시작", key="up_btn"):
                    st.session_state.btn_trigger = True
        with tab2:
            camera = st.camera_input("물체를 중앙에 맞추고 촬영하세요")
            if camera:
                img_input = Image.open(camera)
                if st.button("🔍 분석 시작", key="cam_btn"):
                    st.session_state.btn_trigger = True

    if img_input and st.session_state.get("btn_trigger"):
        with st.spinner("정밀 분석 중..."):
            img_np = np.array(img_input)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            bbox = get_refined_bbox(img_bgr)
            x1, y1, x2, y2 = bbox
            crop = img_bgr[y1:y2, x1:x2]
            
            m_label, m_conf = predict(m_model, crop if crop.size > 0 else img_bgr)
            c_label, c_conf = predict(c_model, crop if crop.size > 0 else img_bgr)
            
            st.session_state.result = {
                "m_label": m_label, "m_conf": m_conf,
                "c_label": c_label, "c_conf": c_conf,
                "bbox": bbox, "img_bgr": img_bgr
            }
            st.session_state.btn_trigger = False

    if st.session_state.result:
        res = st.session_state.result
        with col2:
            st.subheader("📊 분석 결과")
            m_kor = MATERIAL_LABELS_KOR.get(res["m_label"], res["m_label"])
            c_kor = CONTAMINATION_LABELS_KOR.get(res["c_label"], res["c_label"])
            
            c1, c2 = st.columns(2)
            c1.metric("품목", m_kor, f"{res['m_conf']*100:.1f}%")
            c2.metric("상태", c_kor, f"{res['c_conf']*100:.1f}%")
            
            if res["c_label"] == "dirty": st.warning(f"⚠️ **{m_kor}**이(가) 오염되었습니다. 세척해 주세요.")
            else: st.success(f"✅ 깨끗한 **{m_kor}**입니다.")
            
            viz = draw_info_pil(res["img_bgr"], res["bbox"], m_kor)
            st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), caption="분석 영역 확인", use_container_width=True)
            
            info = RECYCLE_EXCHANGE_INFO.get(res["m_label"])
            if info:
                with st.expander("💡 분리배출 꿀팁", expanded=True):
                    st.write(f"**{info['title']}**")
                    st.write(info['content'])

if __name__ == "__main__":
    main()
