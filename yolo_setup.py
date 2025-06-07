import cv2
import torch
import time                                                                                                                                                                                         
import threading
from ultralytics import YOLO
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

# Load YOLO model (object detection)
yolo_model = YOLO("yolov8n.pt")  # Make sure this model is in your directory

# Load pre-trained ViT model
vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Vehicle class IDs (from COCO dataset) that YOLO will detect
VEHICLE_CLASSES = {
    2: "Car", 
    3: "Motorcycle",  # Two-wheeler class for motorcycles
    5: "Bus", 
    7: "Truck", 
    8: "Train", 
    9: "Boat", 
    37: "Bicycle",  # Two-wheeler class for bicycles (from COCO dataset)
    14: "Airplane", 
    15: "Helicopter",
}

# --- Webcam detection for external webcam ---
def open_webcam():
    print("🔍 Searching for external webcam...")
    for i in range(1, 5):  # Start from index 1 to skip built-in webcam
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use CAP_DSHOW on Windows to avoid warnings
        if cap.isOpened():
            print(f"✅ External webcam detected at index: {i}")
            cap.set(3, 640)  # width
            cap.set(4, 480)  # height
            return cap
        else:
            print(f"❌ Webcam at index {i} not accessible.")
    raise Exception("❌ No external webcam found.")

cap = open_webcam()

# --- Traffic signal control ---
current_signal_time = 20
last_update_time = time.time()
green_light = True
lock = threading.Lock()

def get_signal_time(vehicle_count):
    if vehicle_count <= 2:
        return 10
    elif 3 <= vehicle_count <= 5:
        return 20
    else:
        return 30

def traffic_signal_control():
    global current_signal_time, last_update_time, green_light
    while True:
        time.sleep(1)
        with lock:
            if time.time() - last_update_time >= current_signal_time:
                green_light = not green_light
                last_update_time = time.time()
                if green_light:
                    print(f"🚦 GREEN Light for {current_signal_time} seconds")
                else:
                    print("🔴 RED Light for 10 seconds")
                    time.sleep(10)

signal_thread = threading.Thread(target=traffic_signal_control, daemon=True)
signal_thread.start()

# --- Main detection loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame capture failed.")
        break

    results = yolo_model(frame)
    vehicle_count = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = box.conf[0].item()  # Get the confidence score of the detection
            
            # Lower confidence threshold to detect more vehicles
            if class_id in VEHICLE_CLASSES and confidence > 0.3:  # Lowered to 0.3
                vehicle_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Crop and classify using ViT (if needed for further classification)
                vehicle_img = frame[y1:y2, x1:x2]
                try:
                    pil_img = Image.fromarray(cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB))
                    inputs = vit_processor(images=pil_img, return_tensors="pt")
                    with torch.no_grad():
                        outputs = vit_model(**inputs)
                        pred_class = outputs.logits.argmax(-1).item()
                        pred_label = vit_model.config.id2label[pred_class]
                except Exception as e:
                    pred_label = "Unknown"

                # Draw bounding box and label
                label = VEHICLE_CLASSES.get(class_id, "Unknown")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence*100:.2f}%)", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Update signal time if green
    with lock:
        if green_light:
            current_signal_time = get_signal_time(vehicle_count)

    # Display info
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Signal Time: {current_signal_time} sec", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, "GREEN" if green_light else "RED", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if green_light else (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("🛣️ Traffic Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
