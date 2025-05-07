import streamlit as st
from PIL import Image
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
import torch

def load_model():
    detection_model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    detection_model.eval()
    return detection_model

def load_coco_data():
    coco = COCO('instances_val2017.json')  
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {cat['id']: cat['name'] for cat in cats}
    return cat_id_to_name

def detect_objects(image, model, cat_id_to_name):
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image).unsqueeze(0)  

    with torch.no_grad():
        predictions = model(img_tensor)

    return predictions

def display_results(image, predictions, cat_id_to_name):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for element in range(len(predictions[0]['boxes'])):
        box = predictions[0]['boxes'][element].cpu().numpy()
        score = predictions[0]['scores'][element].cpu().numpy()
        label_id = predictions[0]['labels'][element].cpu().numpy().item()
        label = cat_id_to_name.get(label_id, 'Unknown')

        threshold_score = 0.85
        if score >= threshold_score:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(box[0], box[1] - 10, f'{label}: {score:.2f}', color='red', fontsize=12, weight='bold')

    plt.axis('off')
    st.pyplot(fig)

def main():
    st.title("Image Classification with Mask R-CNN")
    st.write("Upload an image to detect objects.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        cat_id_to_name = load_coco_data()
        model = load_model()
        predictions = detect_objects(image, model, cat_id_to_name)

        display_results(image, predictions, cat_id_to_name)

if __name__ == "__main__":
    main()