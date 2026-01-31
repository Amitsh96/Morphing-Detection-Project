import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ==========================================
# 1. הגדרות וטעינת המודל
# ==========================================

# רשימת המחלקות (לפי הסדר האלפביתי של התיקיות באימון: bonafide קודם ל-morphed)
CLASS_NAMES = ['Bona Fide (Real)', 'Morphed (Fake)']

@st.cache_resource
def load_model():
    """
    טוען את המודל לזיכרון פעם אחת בלבד כדי שהאתר ירוץ מהר.
    """
    # יצירת המבנה של ResNet50
    model = models.resnet50(pretrained=False) # לא צריך Pretrained כי אנחנו דורסים אותו
    
    # החלפת השכבה האחרונה שתתאים ל-2 מחלקות
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # טעינת המשקולות מהקובץ
    # חשוב: map_location='cpu' מאפשר להריץ על מחשב רגיל גם אם האימון היה ב-GPU
    try:
        model.load_state_dict(torch.load('models/resnet50_morph.pth', map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error("שגיאה: קובץ המודל לא נמצא! וודא שהוא נמצא בתיקיית models")
        return None
    
    model.eval() # מעביר למצב בדיקה (מכבה Dropout וכו')
    return model

# טעינת המודל (קורה רק פעם אחת בהפעלה)
model = load_model()

# ==========================================
# 2. הכנת התמונה (Preprocessing)
# ==========================================
def process_image(image):
    """
    מכין את התמונה לכניסה למודל (אותן טרנספורמציות כמו באימון)
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # נרמול חובה - אלו המספרים המדויקים של ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0) # הוספת מימד Batch (מ-3D ל-4D)

# ==========================================
# 3. ממשק המשתמש (UI)
# ==========================================
st.set_page_config(page_title="Morphing Detector", page_icon="🕵️")

st.title("🕵️ Face Morphing Detector")
st.write("העלה תמונת פנים כדי לבדוק האם היא מקורית או ערוכה (Morphed).")

uploaded_file = st.file_uploader("בחר תמונה...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # הצגת התמונה שהועלתה
    image = Image.open(uploaded_file).convert('RGB') # המרה ל-RGB למקרה שזה PNG שקוף
    st.image(image, caption='התמונה שהועלתה', use_container_width=True)
    
    st.write("")
    if st.button('🔍 נתח תמונה'):
        if model:
            with st.spinner('מנתח את התמונה...'):
                # עיבוד וחיזוי
                input_tensor = process_image(image)
                
                with torch.no_grad(): # חוסך זיכרון
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    
                    # קבלת התוצאה
                    confidence, predicted_class = torch.max(probabilities, 1)
                    label = CLASS_NAMES[predicted_class.item()]
                    score = confidence.item() * 100

                # הצגת התוצאה
                st.write("---")
                if predicted_class.item() == 0: # Bona Fide
                    st.success(f"✅ תוצאה: **תמונה מקורית ({label})**")
                    st.metric(label="רמת ביטחון", value=f"{score:.2f}%")
                else: # Morphed
                    st.error(f"⚠️ תוצאה: **חשד למורפינג ({label})**")
                    st.metric(label="רמת ביטחון", value=f"{score:.2f}%")
                
                # הצגת גרף הסתברויות
                st.write("התפלגות הסיכויים:")
                st.progress(int(probabilities[0][0].item() * 100), text=f"סיכוי למקורית: {probabilities[0][0].item()*100:.1f}%")
                st.progress(int(probabilities[0][1].item() * 100), text=f"סיכוי למזוייפת: {probabilities[0][1].item()*100:.1f}%")