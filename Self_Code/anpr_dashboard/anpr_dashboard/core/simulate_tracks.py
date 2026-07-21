from django.utils import timezone
from datetime import timedelta
import random

from core.models import VehicleRegistry, SightingLog

PLATES = [
    "MH20EE7602",
    "JK08H0088",
    "MH20DV2366",
]

CAMERA_PATHS = [
    ["CAM_01_TOLL_GATE", "CAM_02_HIGHWAY", "CAM_04_CITY", "CAM_06_BORDER"],
    ["CAM_01_TOLL_GATE", "CAM_03_MARKET", "CAM_05_TUNNEL"],
    ["CAM_02_HIGHWAY", "CAM_04_CITY", "CAM_06_BORDER"],
]


def simulate():
    base_time = timezone.now() - timedelta(hours=1)

    for plate in PLATES:
        vehicle, _ = VehicleRegistry.objects.get_or_create(
            plate_number=plate,
            defaults={
                "predicted_color": random.choice(["white","black","silver"]),
                "predicted_type": random.choice(["car","truck","bus"])
            }
        )

        path = random.choice(CAMERA_PATHS)

        for i, cam in enumerate(path):
            SightingLog.objects.create(
                vehicle=vehicle,
                camera_id=cam,
                timestamp=base_time + timedelta(minutes=5*i),
                confidence=random.uniform(85,98)
            )

    print("✅ Multi-camera tracking data inserted")