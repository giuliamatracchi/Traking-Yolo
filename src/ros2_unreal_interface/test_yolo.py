
from ultralytics import YOLO

# se NON hai ancora un tuo modello custom .pt usa quello base COCO
model = YOLO("yolov8n.pt")

# usa una immagine che hai nel pc
img = "immagine1.jpg"          # <--- cambia questo nome con la tua immagine

# fai prediction
results = model.predict(source=img, conf=0.25, save=True)

print(f"Detections fatte: {len(results[0].boxes)}")
print("Immagine salvata in -->", results[0].save_dir)


