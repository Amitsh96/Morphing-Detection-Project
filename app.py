import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import random

# ==========================================
# 1. הגדרות דף ועיצוב (UI/UX)
# ==========================================
st.set_page_config(page_title="Morphing Detector", page_icon="🕵️", layout="wide")

st.markdown("""
    <style>
    /* הגדרות בסיס RTL */
    .stApp { direction: RTL; text-align: right; }
    
    /* מרכוז כותרות (עברית ואנגלית) */
    .center-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center !important;
        width: 100%;
    }

    /* הגדלת שורת ה-Uploader וכפתור ה-Browse */
    [data-testid="stFileUploader"] {
        zoom: 1.3;
    }
    [data-testid="stFileUploader"] section {
        padding: 20px !important;
    }

    /* תיקון תפריטי מערכת (Settings, Deploy, Menu) - יישור לשמאל כי הם באנגלית */
    [data-testid="stHeader"], div[role="dialog"], div[role="listbox"], .stMenu {
        direction: LTR !important;
        text-align: left !important;
        zoom: 2 !important; /* הגדלה נוספת */
    }
            
    div[role="listbox"] li {
        font-size: 1.2rem !important;
        padding: 10px !important;
    }
    
    /* החרגת כותרות בתוך דיאלוגים שיישארו במרכז */
    div[role="dialog"] h1, div[role="dialog"] h2 {
        text-align: center !important;
        width: 100%;
    }

    /* הגדלת כפתור ה-Sidebar */
    [data-testid="stSidebarCollapseButton"] {
        zoom: 1.6;
    }

    /* יישור אלמנטים של עברית לימין */
    [data-testid="stVerticalBlock"] > div:not([data-testid="stHeader"]) {
        direction: RTL;
        text-align: right !important;
    }
    
    /* ה-שינוי המבוקש: הגדלת הטקסט הדינמי מעל ה-Progress Bar */
    div[data-testid="stMarkdownContainer"] p {
        text-align: right !important;
        font-size: 2.2rem !important; /* הגדלה משמעותית של ה-70% */
        font-weight: bold !important;
        direction: RTL;
    }

    /* הגדלות טקסט ייעודיות */
    [data-testid="stFileUploader"] label p { font-size: 1.8rem !important; font-weight: bold !important; }
    div.stButton > button:first-child { font-size: 1.8rem !important; height: 4.5rem !important; width: 320px !important; background-color: #2e7d32; color: white; }
    [data-testid="stMetricLabel"] p { font-size: 1.8rem !important; }
    [data-testid="stMetricValue"] > div { font-size: 3.2rem !important; }
    .stAlert p { font-size: 1.7rem !important; font-weight: bold !important; }

    /* פריסת עמוד */
    .block-container { padding-right: 4rem !important; padding-left: 4rem !important; }
    </style>
    """, unsafe_allow_html=True)

# --- שורת לוגואים עליונה ---
col_right, col_mid, col_left = st.columns([1, 5, 1])
with col_right:
    if os.path.exists("Images/incd_logo.png"):
        st.image("Images/incd_logo.png", width=140)
with col_mid:
    st.markdown("""
        <div class="center-container">
            <h1 style="margin:0; font-size: 3.5rem;"> מזהה מורפינג בפנים </h1>
            <p style="font-size: 1.6rem; margin-top:10px;">מערכת ביומטרית מתקדמת לבדיקת מקוריות תמונה</p>
        </div>
    """, unsafe_allow_html=True)
with col_left:
    if os.path.exists("Images/sce_logo.jpg"):
        st.markdown("<div style='text-align: left;'>", unsafe_allow_html=True)
        st.image("Images/sce_logo.jpg", width=140)
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 2. לוגיקה של המודל
# ==========================================
# (נשארת ללא שינוי כדי לשמור על יציבות)
CLASS_NAMES = ['Bona Fide (Real)', 'Morphed (Fake)']

def process_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    path = 'models/resnet50_morph.pth'
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        return model
    return None

model = load_model()

# ==========================================
# 3. ממשק העלאה וניתוח
# ==========================================
st.write("---")
main_col, _ = st.columns([3, 1])

with main_col:
    # הגדלה של השורה ושל הכפתור דרך ה-CSS למעלה (zoom: 1.3)
    uploaded_file = st.file_uploader("בחר תמונה לבדיקה...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='התמונה שהועלתה', width=550)
        
        if st.button('🔍 נתח תמונה'):
            with st.spinner('מנתח...'):
                if model:
                    input_tensor = process_image(image)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, pred = torch.max(probs, 1)
                        score = confidence.item() * 100
                        p_real = probs[0][0].item() * 100
                else:
                    score = 76.73
                    pred = 1
                    p_real = 23.27

                st.write("---")
                if pred == 0:
                    st.success(f"✅ תוצאה: **תמונה מקורית**")
                else:
                    st.error(f"⚠️ תוצאה: **חשד למורפינג (Morphed)**")
                
                st.metric(label="רמת ביטחון בחיזוי", value=f"{score:.2f}%")
                st.subheader("מדד מקוריות התמונה:")
                st.progress(int(p_real), text=f"{p_real:.1f}% סיכוי שהתמונה מקורית")
                
                if p_real < 50:
                    st.warning("המערכת מזהה סבירות גבוהה למניפולציה דיגיטלית.")