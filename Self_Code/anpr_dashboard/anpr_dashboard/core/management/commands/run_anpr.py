import os
import cv2
import numpy as np
import torch
import re
from collections import defaultdict, deque
from datetime import datetime
from django.core.management.base import BaseCommand
from django.conf import settings
from ultralytics import YOLO
from paddleocr import PaddleOCR
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from core.models import VehicleRegistry, SightingLog

class Command(BaseCommand):
    help = "Run Real-time ANPR with Dashboard Integration"

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("🚀 ANPR Engine Started – Webcam Live!"))

        # ====================== YOUR FULL CONFIG ======================
        VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_type = YOLO("yolov10m.pt")

        model_plate = YOLO(
            r"D:\Automatic ANPR\Self_Code\runs\detect\lp_yolov8s_stage2_final3\weights\best.pt"
        )

        model_color = models.efficientnet_b0()
        model_color.classifier[1] = nn.Linear(model_color.classifier[1].in_features, 15)
        model_color.load_state_dict(torch.load(r"D:\Automatic ANPR\Self_Code\vehicle_color_best.pth", map_location=device))
        model_color.to(device)
        model_color.eval()

        class_names = ['beige','black','blue','brown','gold','green','grey','orange',
                       'pink','purple','red','silver','tan','white','yellow']

        color_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

        # ====================== YOUR HELPERS ======================
        def inside_roi(bbox, frame_h):
            x1, y1, x2, y2 = bbox
            cy = (y1 + y2) // 2
            ROI_Y_MIN = int(frame_h * 0.35)
            ROI_Y_MAX = int(frame_h * 0.85)
            return ROI_Y_MIN <= cy <= ROI_Y_MAX

        def predict_vehicle_color(crop):
            image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = color_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model_color(image)
                _, pred = torch.max(outputs, 1)
            return class_names[pred.item()]

        def preprocess_plate_for_paddle(plate):
            h, w = plate.shape[:2]
            plate = cv2.resize(plate, (int(w * 2), int(h * 2)))
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            return clahe.apply(gray)

        def ocr_plate_paddle(img):
            result = ocr_engine.ocr(img, cls=True)
            if not result or not result[0]:
                return ""
            texts = []
            for line in result[0]:
                t, conf = line[1]
                if conf > 0.6:
                    texts.append(t)
            text = "".join(texts).upper()
            return re.sub(r'[^A-Z0-9]', '', text)

        def validate_indian_plate(text):
            p1 = r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$'
            p2 = r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$'
            return bool(re.match(p1, text) or re.match(p2, text))

        def broadcast_detection(data):
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                "dashboard_group",
                {"type": "send_vehicle_update", "message": data}
            )

        # ====================== YOUR VIDEO PIPELINE ======================
        # video_path = r"C:\Users\100ra\Downloads\ANPR India Detection Demo - SmartCow.mp4"
        cap = cv2.VideoCapture(0)  # Webcam

        track_memory = defaultdict(lambda: {
            "state": "idle",
            "plate_buffer": deque(maxlen=7),
            "final_plate": "",
            "color": "",
            "type": ""
        })

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            H = frame.shape[0]
            results = model_type.track(frame, persist=True, verbose=False)

            if results[0].boxes.id is None:
                cv2.imshow("ANPR System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box, cls, track_id in zip(boxes, classes, ids):
                cls = int(cls)
                track_id = int(track_id)
                if cls not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box)
                bbox = [x1, y1, x2, y2]

                mem = track_memory[track_id]

                if not inside_roi(bbox, H):
                    continue

                if mem["state"] == "done":
                    label = f"ID:{track_id} {mem['type']} {mem['color']} {mem['final_plate']}"
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,label,(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                    continue

                mem["state"] = "active"
                vehicle_crop = frame[y1:y2, x1:x2]
                if vehicle_crop.size == 0:
                    continue

                mem["type"] = VEHICLE_CLASSES[cls]
                if mem["color"] == "":
                    mem["color"] = predict_vehicle_color(vehicle_crop)

                plate_results = model_plate(vehicle_crop, conf=0.4, verbose=False)
                for pr in plate_results:
                    for pbox in pr.boxes:
                        px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                        plate_crop = vehicle_crop[py1:py2, px1:px2]
                        if plate_crop.size == 0:
                            continue

                        proc = preprocess_plate_for_paddle(plate_crop)
                        raw = ocr_plate_paddle(proc)

                        if validate_indian_plate(raw):
                            mem["plate_buffer"].append(raw)

                buf = mem["plate_buffer"]
                if len(buf) >= 5:
                    final_plate = max(set(buf), key=buf.count)
                    mem["final_plate"] = final_plate
                    mem["state"] = "done"

                    # ====================== SAVE TO DB + BROADCAST ======================
                    vehicle, _ = VehicleRegistry.objects.get_or_create(
                        plate_number=final_plate,
                        defaults={"predicted_color": mem["color"], "predicted_type": mem["type"]}
                    )
                    vehicle.predicted_color = mem["color"]
                    vehicle.predicted_type = mem["type"]
                    vehicle.save()

                    snapshot_name = f"{final_plate}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    snapshot_path = os.path.join(settings.MEDIA_ROOT, "snapshots", snapshot_name)
                    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
                    cv2.imwrite(snapshot_path, vehicle_crop)

                    sighting = SightingLog.objects.create(
                        vehicle=vehicle,
                        camera_id="CAM_01_TOLL_GATE",
                        confidence=98.5,
                        snapshot_path=f"snapshots/{snapshot_name}"
                    )

                    data = {
                        "plate": final_plate,
                        "color": mem["color"],
                        "type": mem["type"],
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "img_path": f"/media/snapshots/{snapshot_name}",
                        "confidence": 98.5
                    }
                    broadcast_detection(data)

                label = f"ID:{track_id} {mem['type']} {mem['color']}"
                if mem["final_plate"]:
                    label += f" {mem['final_plate']}"

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            cv2.imshow("ANPR System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()