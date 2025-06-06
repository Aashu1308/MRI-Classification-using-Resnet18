from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch
from torchvision.models import resnet18, ResNet18_Weights
import streamlit as st

class_dict = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load('resnet18_brain_tumor.pth', map_location=device))
model = model.to(device)
model.eval()

v_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_image(img):
    # img = Image.open(img_path).convert("RGB")
    img = v_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        op = model(img)
        _, pred = torch.max(op,1)
    return class_dict[pred.item()]

if __name__ == '__main__':
    # img_path_1  = 'data\Validation\meningioma\Tr-me_0284.jpg'
    # img_path_2 = 'data\Validation\/notumor\Te-no_0012.jpg'
    # print(f"prediction 1: {predict_image(img_path_1)}") #should be meningioma
    # print(f"prediction 2: {predict_image(img_path_2)}") #should be notumor


    st.title("Brain Tumor Classifier")
    img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if img is not None:
        image = Image.open(img).convert("RGB")
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        pred = predict_image(image)
        st.success(f"Prediction: {pred}")

