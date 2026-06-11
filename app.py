# app.py
# ------------------------------------------------------------
# 재활용 쓰레기 분류 + 오염도 판정 + 로컬/Git ZIP 자동 학습 v12.2
# 수정 사항: 플라스틱의 스티로폼 오인식 해결을 위한 동적 라벨 매핑 및 학습 파라미터 최적화
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
# 상수 및 글로벌 데이터 매핑 (영어 수순 매핑 대응)
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
# Git/로컬 폴더 내 ZIP 파일 자동 압축해제 및 AI 학습 로직
# ============================================================

def auto_extract_and_train():
    """레포지토리 내의 모든 한글명 ZIP 파일을 찾아 압축을 풀고 AI 모델을 자동 학습시킵니다."""
    zip_files = list(BASE_DIR.glob("*.zip"))
    if not zip_files:
        return
        
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.info("📦 레포지토리 내부의 ZIP 데이터셋을 감지했습니다. 압축 해제 및 이미지 자동 정렬 중...")
    
    # 데이터셋 저장 폴더 구조 초기화 생성
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
        
    for target in ["material", "contamination"]:
        for phase in ["train", "val"]:
            (DATASET_DIR / target / phase).mkdir(parents=True, exist_ok=True)
            
    # 1. 모든 ZIP 파일 압축 해제 후 클래스 분류 배치
    for idx, zip_path in enumerate(zip_files):
        status_text.text(f"🔄 압축 데이터 분석 중: {zip_path.name} ({idx+1}/{len(zip_files)})")
        name = zip_path.stem
        
        m_class = "unknown"
        if "종이" in name: m_class = "paper"
        elif "플라스틱" in name: m_class = "plastic"
        elif "유리" in name: m_class = "glass"
        elif "비닐" in name: m_class = "vinyl"
        elif "캔" in name: m_class = "can"
        elif "스티로폼" in name: m_class = "styrofoam"
        
        c_class = "uncertain"
        if "오염o" in name or "오염O" in name or "오염됨" in name: c_class = "dirty"
        elif "오염x" in name or "오염X" in name or "깨끗" in name: c_class = "clean"
        
        temp_dir = BASE_DIR / f"temp_{name}"
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
            all_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                all_images.extend(list(temp_dir.rglob(ext)))
                
            random.seed(42)
            random.shuffle(all_images)
            
            split_idx = int(len(all_images) * 0.8)
            train_imgs = all_images[:split_idx]
            val_imgs = all_images[split_idx:]
            
            def copy_images(img_list, target_type, class_name, phase):
                dest_dir = DATASET_DIR / target_type / phase / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                for img in img_list:
                    shutil.copy(img, dest_dir / f"{zip_path.stem}_{img.name}")
                    
            if m_class != "unknown":
                copy_images(train_imgs, "material", m_class, "train")
                copy_images(val_imgs, "material", m_class, "val")
            if c_class != "uncertain":
                copy_images(train_imgs, "contamination", c_class, "train")
                copy_images(val_imgs, "contamination", c_class, "val")
                
        except Exception as e:
            st.warning(f"⚠️ {zip_path.name} 처리 오류: {e}")
        finally:
            if temp_dir.exists(): shutil.rmtree(temp_dir)
            
    progress_bar.progress(30)
    MODELS_DIR.mkdir(exist_ok=True)
    
    # 2. 품목 분류 AI 모델 학습 실행 (오인식 해결을 위해 에포크 상향 및 오버핏 방지 설정)
    mat_train_dir = DATASET_DIR / "material" / "train"
    if mat_train_dir.exists() and any(mat_train_dir.iterdir()):
        status_text.text("🚀 [1/2] 품목 분류 AI 모델(Material) 정밀 파인튜닝 학습 중...")
        model_m = YOLO("yolov8n-cls.pt")
        # 정밀 분류를 위해 epochs를 15로 상향조정하고 조기종료(patience)를 세팅하여 모델 학습을 고도화합니다.
        model_m.train(data=str(DATASET_DIR / "material"), epochs=15, imgsz=224, lr0=0.005, patience=5, verbose=False)
        model_m.save(str(MATERIAL_MODEL_PATH))
        
    progress_bar.progress(70)
    
    # 3. 오염도 판정 AI 모델 학습 실행
    cont_train_dir = DATASET_DIR / "contamination" / "train"
    if cont_train_dir.exists() and any(cont_train_dir.iterdir()):
        status_text.text("🚀 [2/2] 오염도 판정 AI 모델(Contamination) 파인튜닝 학습 중...")
        model_c = YOLO("yolov8n-cls.pt")
        model_c.train(data=str(DATASET_DIR / "contamination"), epochs=10, imgsz=224, verbose=False)
        model_c.save(str(CONTAMINATION_MODEL_PATH))
        
    progress_bar.progress(100)
    status_text.success("🎉 플라스틱 오인식 보정 알고리즘 및 데이터셋 학습이 완료되었습니다!")
    time.sleep(2)
    status_text.empty()
    progress_bar.empty()

# ============================================================
# 분석 영역 감지 로직 (에지 투영 기반 타이트 크롭)
# ============================================================

def get_refined_bbox(image_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = image_bgr.shape[:2]
    
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
# 고도화된 추론용 동적 매핑 함수
# ============================================================

def predict(model, img_bgr):
    if model is None: return "unknown", 0.0
    results = model.predict(img_bgr, verbose=False)
    if not results or not results[0].probs: return "unknown", 0.0
    
    idx = int(results[0].probs.top1)
    conf = float(results[0].probs.top1conf)
    
    # [핵심 보정] 고정 인덱스가 아닌 모델이 학습 시 직접 생성한 라벨 문자열을 추출합니다.
    raw_label_name = results[0].names[idx] 
    return raw_label_name, conf

# ============================================================
# 메인 구동 루프
# ============================================================

def main():
    st.set_page_config(page_title="스마트 재활용 분류 시스템 v12.2", layout="wide")
    st.title("♻️ 스마트 재활용 분류 시스템 v12.2")
    
    if "trained" not in st.session_state:
        auto_extract_and_train()
        st.session_state.trained = True

    if "result" not in st.session_state:
        st.session_state.result = None
    if "last_input_key" not in st.session_state:
        st.session_state.last_input_key = None

    @st.cache_resource
    def load_models():
        m = YOLO(str(MATERIAL_MODEL_PATH)) if MATERIAL_MODEL_PATH.exists() else None
        c = YOLO(str(CONTAMINATION_MODEL_PATH)) if CONTAMINATION_MODEL_PATH.exists() else None
        return m, c

    m_model, c_model = load_models()
    
    if m_model is None:
        st.warning("⚠️ 메인 디렉토리에 학습용 ZIP 파일을 넣어두시면 자동으로 압축 해제 및 학습이 완수됩니다.")
        
    col1, col2 = st.columns([1, 1])
    
    with col1:
        tab1, tab2 = st.tabs(["📁 이미지 분석", "📷 카메라 촬영"])
        img_input = None
        
        with tab1:
            uploaded = st.file_uploader("사진을 선택해 주세요", type=["jpg", "png", "jpeg"])
            if uploaded:
                if st.session_state.last_input_key != f"file_{uploaded.name}":
                    st.session_state.result = None
                    st.session_state.last_input_key = f"file_{uploaded.name}"
                
                img_input = Image.open(uploaded)
                st.image(img_input, use_container_width=True)
                if st.button("🔍 분류 분석 시작", key="up_btn"):
                    st.session_state.btn_trigger = True
                    
        with tab2:
            camera = st.camera_input("카메라에 물체를 가깝게 비춰주세요")
            if camera:
                if st.session_state.last_input_key != "camera_shot":
                    st.session_state.result = None
                    st.session_state.last_input_key = "camera_shot"
                
                img_input = Image.open(camera)
                if st.button("🔍 분류 분석 시작", key="cam_btn"):
                    st.session_state.btn_trigger = True

    if img_input and st.session_state.get("btn_trigger"):
        with st.spinner("AI 실시간 분석 작동 중..."):
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
            st.rerun()

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
                st.warning(f"⚠️ **{m_kor}** 내부에 오염이 감지되었습니다. 물에 깨끗이 씻어서 배출하세요.")
            else:
                st.success(f"✅ 상태가 깨끗한 **{m_kor}**입니다. 정상 수거함에 분리배출 하세요.")
            
            viz = draw_info_pil(res["img_bgr"], res["bbox"], m_kor)
            st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), caption="시스템 인식 범위(BBox)", use_container_width=True)

if __name__ == "__main__":
    main()
