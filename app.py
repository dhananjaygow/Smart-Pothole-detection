import gradio as gr
from ultralytics import YOLO

# Load your trained model weights
model = YOLO(r"D:\SmartBus-Pothole-Demo\runs\detect\train\weights\best.pt")

def detect_potholes(image):
    # Run inference using GPU (device=0)
    results = model.predict(source=image, device=0, conf=0.50)
    # Render the annotated image array
    annotated_img = results[0].plot()
    return annotated_img

# Build UI layout
demo = gr.Interface(
    fn=detect_potholes,
    inputs=gr.Image(type="numpy", label="Upload Road Image"),
    outputs=gr.Image(type="numpy", label="Pothole Detections"),
    title="SmartBus - Automated Pothole Detection System",
    description="Upload a road photo to detect potholes using custom-trained YOLOv8."
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)