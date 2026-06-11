# app.py
# ------------------------------------------------------------
# 재활용 쓰레기 분류 + 오염도 판정 + 로컬/Git ZIP 자동 학습 v12
# 개선 사항:
#   1. 자동 데이터셋 빌드: 실행 시 주변의 모든 ZIP 파일을 찾아 압축 해제 후 품목/오염도 데이터셋으로 분류
#   2. 자동 AI 파인튜닝: 압축 해제된 누끼(배경 제거) 이미지로 YOLO 분류 모델을 자동 학습 및 정확도 극대화
#   3. 정밀 에지 밀도 검출: 자(Ruler)나 투명 물체 측정 시 주변 천장/손가락 노이즈를 차단하는 줌인 기능
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
# 패키지 및 환경 설정 (자동 설치)
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
# 상수 및 데이터 구조 정의
# ============================================================

MATERIAL_LABELS_KOR = {
    "glass": "유리", "paper": "종이", "can": "캔",
    "vinyl": "비닐", "styrofoam": "스티로폼",
    "plastic": "플라스틱", "unknown": "판정불가",
}

CONTAMINATION_LABELS_KOR = {
    "clean": "깨끗함", "dirty": "오염됨", "uncertain": "판정불가",
}

# ============================================================
# 핵심 기능: Git에 올라온 ZIP 파일 자동 압축해제 및 통합 학습 로직
# ============================================================

def auto_extract_and_train():
    """레포지토리 내의 모든 ZIP 파일을 찾아 압축을 풀고 AI 모델을 자동 학습시킵니다."""
    zip_files = list(BASE_DIR.glob("*.zip"))
    if not zip_files:
        return
        
    # 안내 팝업 및 진행 상황 표시 준비
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.info("📦 Git 레포지토리 내의 ZIP 데이터셋을 감지했습니다. 압축 해제 및 이미지 분류 중...")
    
    # 품목(Material) 및 오염도(Contamination)용 학습 폴더 초기화 생성
    for target in ["material", "contamination"]:
        for phase in ["train", "val"]:
            (DATASET_DIR / target / phase).mkdir(parents=True, exist_ok=True)
            
    # 1. 발견된 모든 ZIP 파일 압축 해제 및 데이터 배치
    for idx, zip_path in enumerate(zip_files):
        status_text.text(f"🔄 압축 해제 중: {zip_path.name} ({idx+1}/{len(zip_files)})")
        
        # 파일 이름 분석 (예: '종이_오염o.zip', '플라스틱_오염x.zip' 등)
        name = zip_path.stem
        
        # 한국어 파일명에 따른 클래스명 매핑 매칭
        # 품목 매핑
        m_class = "unknown"
        if "종이" in name: m_class = "paper"
        elif "플라스틱" in name: m_class = "plastic"
        elif "유리" in name: m_class = "glass"
        elif "비닐" in name: m_class = "vinyl"
        elif "캔" in name: m_class = "can"
        
        # 오염도 매핑
        c_class = "uncertain"
        if "오염o" in name or "오염O" in name or "오염됨" in name: c_class = "dirty"
        elif "오염x" in name or "오염X" in name or "깨끗" in name: c_class = "clean"
        
        # 임시 해제 구역
        temp_dir = BASE_DIR / f"temp_{name}"
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
            # 임시 폴더 내의 이미지들을 8:2 비율로 train과 val로 나누어 목적지 폴더로 이동
            all_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                all_images.extend(list(temp_dir.rglob(ext)))
                
            random.seed(42)
            random.shuffle(all_images)
            
            split_idx = int(len(all_images) * 0.8)
            train_imgs = all_images[:split_idx]
            val_imgs = all_images[split_idx:]
            
            # 파일 복사 함수 정의
            def copy_images(img_list, target_type, class_name, phase):
                dest_dir = DATASET_DIR / target_type / phase / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                for img in img_list:
                    # 파일명 충돌 방지를 위해 unique 고유 해시명 부여
                    shutil.copy(img, dest_dir / f"{zip_path.stem}_{img.name}")
                    
            # 매핑이 유효할 경우 데이터셋 배치 작동
            if m_class != "unknown":
                copy_images(train_imgs, "material", m_class, "train")
                copy_images(val_imgs, "material", m_class, "val")
            if c_class != "uncertain":
                copy_images(train_imgs, "contamination", c_class, "train")
                copy_images(val_imgs, "contamination", c_class, "val")
                
        except Exception as e:
            st.warning(f"⚠️ {zip_path.name} 처리 중 오류 발생: {e}")
        finally:
            if temp_dir.exists(): shutil.rmtree(temp_dir)
            
    progress_bar.progress(30)
    
    # 2. 통합 데이터셋 기반 YOLOv8 Classification 파인튜닝 시작
    MODELS_DIR.mkdir(exist_ok=True)
    
    # [파트 A] 품목 분류 모델 학습
    if (DATASET_DIR / "material" / "train").exists() and any((DATASET_DIR / "material" / "train").iterdir()):
        status_text.text("🚀 [1/2] 품목 분류 AI 모델(Material) 파인튜닝 가동 중...")
        model_m = YOLO(str(MATERIAL_MODEL_PATH)) if MATERIAL_MODEL_PATH.exists() else YOLO("yolov8n-cls.pt")
        model_m.train(data=str(DATASET_DIR / "material"), epochs=10, imgsz=224, verbose=False)
        model_m.save(str(MATERIAL_MODEL_PATH))
        
    progress_bar.progress(75)
    
    # [파트 B] 오염도 판정 모델 학습
    if (DATASET_DIR / "contamination" / "train").exists() and any((DATASET_DIR / "contamination" / "train").iterdir()):
        status_text.text("🚀 [2/2] 오염도 판단 AI 모델(Contamination) 파인튜닝 가동 중...")
        model_c = YOLO(str(CONTAMINATION_MODEL_PATH)) if CONTAMINATION_MODEL_PATH.exists() else YOLO("yolov8n-cls.pt")
        model_c.train(data=str(DATASET_DIR / "contamination"), epochs=10, imgsz=224, verbose=False)
        model_c.save(str(CONTAMINATION_MODEL_PATH))
        
    progress_bar.progress(100)
    status_text.success("🎉 모든 ZIP 파일 분석 및 AI 가중치 파인튜닝이 완료되었습니다! 서비스를 시작합니다.")
    time.sleep(2)
    status_text.empty()
    progress_bar.empty()

# ============================================================
# 정밀 에지 프로젝션 (손가락/배경 노이즈 무력화 기술)
# ============================================================

def get_refined_bbox(image_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = image_bgr.shape[:2]
    
    # 중앙 집중 ROI 추출 (가장자리 전등/천장 노이즈 차단)
    cy, cx = h // 2, w // 2
    rh, rw = int(h * 0.75), int(w * 0.75)
    y1_r, x1_r = max(0, cy - rh // 2), max(0, cx - rw // 2)
    roi = image_bgr[y1_r:y1_r+rh, x1_r:x1_r+rw]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 130)
    
    x_counts = np.sum(edges > 0, axis=0)
    y_counts = np.sum(edges > 0, axis=1)
    
    x_indices = np.where(x_counts > np.mean(x_counts) * 0.6)[0]
    y_indices = np.where(y_counts > np.mean(y_counts) * 0.6)[0]
    
    if len(x_indices) > 0 and len(y_indices) > 0:
        bx1 = x1_r + x_indices[0]
        by1 = y1_r + y_indices[0]
        bx2 = x1_r + x_indices[-1]
        by2 = y1_r + y_indices[-1]
        
        pad = 15
        return (max(0, bx1 - pad), max(0, by1 - pad), min(w, bx2 + pad), min(h, by2 + pad))
        
    return int(w * 0.25), int(h * 0.2), int(w * 0.75), int(h * 0.8)

# ============================================================
# 시각화 렌더링
# ============================================================

def draw_info_pil(image_bgr: np.ndarray, bbox: Tuple[int, int, int, int], label: str) -> np.ndarray:
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img, "RGBA")
    
    x1, y1, x2, y2 = bbox
    
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    draw_ov.rectangle([0, 0, pil_img.width, pil_img.height], fill=(0, 0, 0, 110))
    draw_ov.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 0))
    pil_img.paste(Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB"))
    
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 120), width=4)
    
    font = ImageFont.load_default()
    text = f" 인식 대상: {label} "
    
    try:
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    except:
        tw, th = 110, 20
        
    draw.rectangle([x1, max(0, y1 - th - 6), x1 + tw, y1], fill=(0, 255, 120))
    draw.text((x1, max(0, y1 - th - 4)), text, font=font, fill=(0, 0, 0))
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ============================================================
# 메인 어플리케이션 구동부
# ============================================================

def predict(model, img_bgr):
    if model is None: return "unknown", 0.0
    results = model.predict(img_bgr, verbose=False)
    if not results or not results[0].probs: return "unknown", 0.0
    idx = int(results[0].probs.top1)
    conf = float(results[0].probs.top1conf)
    return results[0].names[idx], conf

def main():
    st.set_page_config(page_title="스마트 재활용 분류 시스템 v12", layout="wide")
    st.title("♻️ 스마트 재활용 분류 시스템 v12")
    
    # 최초 구동 시 단 한 번만 로컬 데이터셋 자동 빌드 및 AI 학습 프로세스 트리거
    if "trained" not in st.session_state:
        auto_extract_and_train()
        st.session_state.trained = True

    @st.cache_resource
    def load_models():
        m = YOLO(str(MATERIAL_MODEL_PATH)) if MATERIAL_MODEL_PATH.exists() else None
        c = YOLO(str(CONTAMINATION_MODEL_PATH)) if CONTAMINATION_MODEL_PATH.exists() else None
        return m, c

    m_model, c_model = load_models()
    
    if m_model is None:
        st.warning("⚠️ 학습 데이터가 없거나 모델 파일이 생성되지 않았습니다. 메인 폴더에 한글 이름 정보가 담긴 ZIP 파일을 업로드해 주세요.")
        
    if "result" not in st.session_state:
        st.session_state.result = None

    col1, col2 = st.columns([1, 1])
    
    with col1:
        tab1, tab2 = st.tabs(["📁 이미지 분석", "📷 카메라 촬영"])
        img_input = None
        with tab1:
            uploaded = st.file_uploader("재활용 쓰레기 사진을 올려주세요", type=["jpg", "png", "jpeg"])
            if uploaded:
                img_input = Image.open(uploaded)
                st.image(img_input, use_container_width=True)
                if st.button("🔍 분류 분석 시작", key="up_btn"):
                    st.session_state.btn_trigger = True
        with tab2:
            camera = st.camera_input("카메라에 물체가 꽉 차도록 대어주세요")
            if camera:
                img_input = Image.open(camera)
                if st.button("🔍 분류 분석 시작", key="cam_btn"):
                    st.session_state.btn_trigger = True

    if img_input and st.session_state.get("btn_trigger"):
        with st.spinner("에지 기반 타이트 크롭 및 인공지능 분류 중..."):
            img_np = np.array(img_input)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            bbox = get_refined_bbox(img_bgr)
            x1, y1, x2, y2 = bbox
            crop = img_bgr[y1:y2, x1:x2]
            
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
            st.subheader("📊 인공지능 분석 리포트")
            m_kor = MATERIAL_LABELS_KOR.get(res["m_label"], res["m_label"])
            c_kor = CONTAMINATION_LABELS_KOR.get(res["c_label"], res["c_label"])
            
            c1, c2 = st.columns(2)
            c1.metric("분류 품목", m_kor, f"{res['m_conf']*100:.1f}%")
            c2.metric("위생 상태", c_kor, f"{res['c_conf']*100:.1f}%")
            
            if res["c_label"] == "dirty":
                st.warning(f"⚠️ **{m_kor}** 내부에 잔여물이 있거나 오염 상태입니다. 물에 깨끗이 씻어서 배출해 주세요.")
            else:
                st.success(f"✅ 오염되지 않은 깨끗한 **{m_kor}**입니다. 정상 분리수거가 가능합니다.")
            
            viz = draw_info_pil(res["img_bgr"], res["bbox"], m_kor)
            st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), caption="물체 포커싱 추적 영역(BBox)", use_container_width=True)

if __name__ == "__main__":
    main()
