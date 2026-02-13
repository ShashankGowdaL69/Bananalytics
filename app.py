import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
from banana_brain import BananaIntelligence

st.set_page_config(page_title="Banana Intelligence", page_icon="🍌")

@st.cache_resource
def load_model():
    # Load the ResNet-18 architecture and our saved weights
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 4)
    m.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    m.eval() # Set to evaluation mode
    return m

model = load_model()
brain = BananaIntelligence()

st.title("🍌 Banana Intelligence (Pro PyTorch Edition)")
file = st.file_uploader("Upload banana photo...", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert('RGB')
    st.image(img, use_container_width=True)
    
    # --- PYTORCH PREPROCESSING ---
    # This automatically resizes (squishes) and scales pixels to 0-1!
    tr = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Apply transform and add batch dimension [1, 3, 224, 224]
    data = tr(img).unsqueeze(0)
    
    # --- PREDICT ---
    with torch.no_grad():
        out = model(data)
        # Convert raw output to percentages (0.0 to 1.0)
        probs = torch.nn.functional.softmax(out[0], dim=0).numpy()
    
    # --- GET INTELLIGENCE ---
    # Wrap probs in a list so it matches the format the brain expects
    result = brain.analyze_prediction([probs])
    
    # --- DISPLAY RESULTS ---
    st.markdown("---")
    st.header(f"Status: {result['label']}")
    st.progress(result['score'] / 100) 
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Ripeness Score", f"{result['score']}/100")
    c2.metric("Days Left", result['days_remaining'])
    c3.metric("Risk Level", result['risk'])
    
    st.subheader("💡 Recommendation")
    st.info(result['action'])
    
    st.subheader("🥗 Nutritional Note")
    st.success(result['nutrition'])
    
    with st.expander("Raw Confidences"):
        labels = ["Overripe", "Ripe", "Rotten", "Unripe"]
        st.write({l: f"{p*100:.2f}%" for l, p in zip(labels, probs)})