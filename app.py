# recycle_all_in_one_complete_v11.py
# ------------------------------------------------------------
# 재활용 쓰레기 분류 + 오염도 판정 + AI 직접 학습 올인원 버전 v11
# 개선 사항:
#   1. 물체 포커싱 대폭 강화: 에지 밀도 분석을 통해 배경(천장/전등) 및 손가락 영역 오인식 차단
#   2. AI 모델 학습 기능 내장: 사이드바를 통해 누끼(배경 제거) 데이터셋(ZIP) 업로드 및 즉시 학습 가능
#   3. 시각화 개선: PIL 기반 한글 출력 최적화 및 타이트한 포커싱 박스 제공
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# 패키지 및 환경 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MATERIAL_MODEL_PATH = MODELS_DIR / "material_cls.pt"
CONTAMINATION_MODEL_PATH = MODELS_DIR / "contamination_cls.pt"
DATASET_DIR = BASE_DIR / "dataset"

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

try:
    from ultralytics import YOLO
except:
    YOLO = None

# ============================================================
# 데이터셋 라벨 정의
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
# 핵심 개선: 에지 밀도 기반 분석 영역 추출 (정밀 줌인)
# ============================================================

def get_refined_bbox(image_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """배경 노이즈를 배제하고 물체의 실루엣 에지가 밀집된 구역을 정확하게 도출합니다."""
    h, w = image_bgr.shape[:2]
    
    # 1. 주변부 강제 제외 (중앙 75% 영역을 타깃 ROI로 설정)
    cy, cx = h // 2, w // 2
    rh, rw = int(h * 0.75), int(w * 0.75)
    y1_r, x1_r = max(0, cy - rh // 2), max(0, cx - rw // 2)
    roi = image_bgr[y1_r:y1_r+rh, x1_r:x1_r+rw]
    
    # 2. 에지 추출을 위한 전처리 (그레이스케일 -> 블러 -> 가우시안 에지)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 130)
    
    # 3. 가로/세로 축으로 에지 픽셀 분포(프로젝션) 계산
    x_counts = np.sum(edges > 0, axis=0)
    y_counts = np.sum(edges > 0, axis=1)
    
    # 4. 에지 밀도가 일정 수준(평균의 60%) 이상인 유효 구간 검출
    x_indices = np.where(x_counts > np.mean(x_counts) * 0.6)[0]
    y_indices = np.where(y_counts > np.mean(y_counts) * 0.6)[0]
    
    if len(x_indices) > 0 and len(y_indices) > 0:
        bx1 = x1_r + x_indices[0]
        by1 = y1_r + y_indices[0]
        bx2 = x1_r + x_indices[-1]
        by2 = y1_r + y_indices[-1]
        
        # 물체 주변에 살짝 마진(Padding) 부여
        pad = 15
        return (max(0, bx1 - pad), max(0, by1 - pad), min(w, bx2 + pad), min(h, by2 + pad))
        
    # 에지 검출이 어려울 경우 기본 중앙 박스 반환
    return int(w * 0.25), int(h * 0.2), int(w * 0.75), int(h * 0.8)

# ============================================================
# 시각화 및 한글 깨짐 방지
# ============================================================

def draw_info_pil(image_bgr: np.ndarray, bbox: Tuple[int, int, int, int], label: str) -> np.ndarray:
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img, "RGBA")
    
    x1, y1, x2, y2 = bbox
    
    # 1. 주변부 딤(Dim) 처리하여 포커싱 효과 극대화
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    draw_ov.rectangle([0, 0, pil_img.width, pil_img.height], fill=(0, 0, 0, 110))
    draw_ov.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 0))
    pil_img.paste(Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB"))
    
    # 2. 바운딩 박스 테두리 드로잉
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 120), width=4)
    
    # 3. 라벨 텍스트 처리
    font = ImageFont.load_default()
    text = f" 감지 대상: {label} "
    
    try:
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    except:
        tw, th = 110, 20
        
    draw.rectangle([x1, max(0, y1 - th - 6), x1 + tw, y1], fill=(0, 255, 120))
    draw.text((x1, max(0, y1 - th - 4)), text, font=font, fill=(0, 0, 0))
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ============================================================
# 신규 기능: 배경 없는 이미지 데이터셋(ZIP) 학습 로직
# ============================================================

def train_with_zip(zip_file, target_type="material"):
    """업로드된 데이터셋 압축파일을 풀어 YOLOv8 Classification 모델을 파인튜닝합니다."""
    target_dir = DATASET_DIR / target_type
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # ZIP 해제
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
        
    # 폴더 기본 검증 (train 폴더 존재 여부 확인)
    if not (target_dir / "train").exists():
        st.error("❌ 압축파일 내부에 'train' 폴더가 확인되지 않습니다. 올바른 구조로 압축해 주세요.")
        return False
        
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MATERIAL_MODEL_PATH if target_type == "material" else CONTAMINATION_MODEL_PATH
    
    # 기존 가중치가 있으면 불러오고, 없으면 기본 기본 모델 사용
    if model_path.exists():
        model = YOLO(str(model_path))
    else:
        model = YOLO("yolov8n-cls.pt")
        
    st.write(f"🔄 배경 제거 데이터셋으로 {target_type} 모델 학습을 시작합니다...")
    
    # 모델 학습 가동 (분류 모델용 파라미터 세팅)
    model.train(data=str(target_dir), epochs=10, imgsz=224, verbose=True)
    
    # 학습이 끝난 가중치 갱신 저장
    model.save(str(model_path))
    return True

# ============================================================
# Streamlit 서비스 메인 루프
# ============================================================

def predict(model, img_bgr):
    if model is None: return "unknown", 0.0
    results = model.predict(img_bgr, verbose=False)
    if not results or not results[0].probs: return "unknown", 0.0
    idx = int(results[0].probs.top1)
    conf = float(results[0].probs.top1conf)
    return results[0].names[idx], conf

def main():
    st.set_page_config(page_title="스마트 재활용 분류기 v11", layout="wide")
    
    # 사이드바 관리자용 AI 직접 학습 대시보드
    st.sidebar.title("⚙️ AI 모델 학습 연구소")
    st.sidebar.info("전처리(배경 제거) 완료된 ZIP 파일을 활용하여 모델 정확도를 향상시킬 수 있습니다.")
    
    select_task = st.sidebar.selectbox("학습 타깃 선택", ["품목 분류 (Material)", "오염도 판정 (Contamination)"])
    zip_upload = st.sidebar.file_uploader("데이터셋 묶음(.zip) 업로드", type=["zip"])
    
    if zip_upload is not None:
        if st.sidebar.button("🚀 업로드 데이터로 학습 실행"):
            task_key = "material" if "품목" in select_task else "contamination"
            with st.sidebar.spinner("AI 모델 파인튜닝 진행 중..."):
                if train_with_zip(zip_upload, task_key):
                    st.sidebar.success("🎉 학습 완료! 신규 가중치가 탑재되었습니다.")
                    st.rerun()

    # 메인 앱 대시보드
    st.title("♻️ 스마트 재활용 분류기 v11")
    st.caption("새로운 에지 트래킹 알고리즘이 도입되어 주변 사물 노이즈를 획기적으로 차단합니다.")

    if "result" not in st.session_state: 
        st.session_state.result = None

    @st.cache_resource
    def load_models():
        m = YOLO(str(MATERIAL_MODEL_PATH)) if MATERIAL_MODEL_PATH.exists() else None
        c = YOLO(str(CONTAMINATION_MODEL_PATH)) if CONTAMINATION_MODEL_PATH.exists() else None
        return m, c

    m_model, c_model = load_models()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        tab1, tab2 = st.tabs(["📁 이미지 파일 분석", "📷 실시간 카메라 촬영"])
        img_input = None
        with tab1:
            uploaded = st.file_uploader("파일을 선택하세요", type=["jpg", "png", "jpeg"])
            if uploaded:
                img_input = Image.open(uploaded)
                st.image(img_input, use_container_width=True)
                if st.button("🔍 재활용 판별 시작", key="up_btn"):
                    st.session_state.btn_trigger = True
        with tab2:
            camera = st.camera_input("카메라 정중앙에 물체를 가깝게 비춰주세요")
            if camera:
                img_input = Image.open(camera)
                if st.button("🔍 재활용 판별 시작", key="cam_btn"):
                    st.session_state.btn_trigger = True

    if img_input and st.session_state.get("btn_trigger"):
        with st.spinner("에지 투영 분석 및 AI 연산 중..."):
            img_np = np.array(img_input)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # 콤팩트 크롭 영역 확보
            bbox = get_refined_bbox(img_bgr)
            x1, y1, x2, y2 = bbox
            crop = img_bgr[y1:y2, x1:x2]
            
            # 크롭 영역을 기반으로 최종 추론 수행
            input_img = crop if crop.size > 0 else img_bgr
            m_label, m_conf = predict(m_model, input_img)
            c_label, c_conf = predict(c_model, input_img)
            
            st.session_state.result = {
                "m_label": m_label, "m_conf": m_conf,
                "c_label": c_label, "c_conf": c_conf,
                "bbox": bbox, "img_bgr": img_bgr
            }
            st.session_state.btn_trigger = False

    if st.session_state.result:
        res = st.session_state.result
        with col2:
            st.subheader("📊 분석 및 리포트")
            m_kor = MATERIAL_LABELS_KOR.get(res["m_label"], res["m_label"])
            c_kor = CONTAMINATION_LABELS_KOR.get(res["c_label"], res["c_label"])
            
            c1, c2 = st.columns(2)
            c1.metric("판정 품목", m_kor, f"{res['m_conf']*100:.1f}%")
            c2.metric("위생 상태", c_kor, f"{res['c_conf']*100:.1f}%")
            
            if res["c_label"] == "dirty": 
                st.warning(f"⚠️ **{m_kor}** 품목에 오염이 확인되었습니다. 깨끗이 세척 후 배출하세요.")
            else: 
                st.success(f"✅ 상태가 깨끗한 **{m_kor}**입니다. 정상 분리배출이 가능합니다.")
            
            viz = draw_info_pil(res["img_bgr"], res["bbox"], m_kor)
            st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), caption="시스템 인식 분석 범위(ROI)", use_container_width=True)
            
            info = RECYCLE_EXCHANGE_INFO.get(res["m_label"])
            if info:
                with st.expander("💡 알아두면 유용한 분리배출 팁", expanded=True):
                    st.write(f"**{info['title']}**")
                    st.write(info['content'])

if __name__ == "__main__":
    main()
