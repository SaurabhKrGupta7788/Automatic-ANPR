from django.db import models
from django.utils import timezone

class VehicleRegistry(models.Model):
    plate_number = models.CharField(max_length=20, primary_key=True)
    predicted_color = models.CharField(max_length=20, blank=True)
    predicted_type = models.CharField(max_length=20, blank=True)
    owner_status = models.CharField(
        max_length=20,
        choices=[('Clean', 'Clean'), ('Stolen', 'Stolen'), ('Watchlist', 'Watchlist')],
        default='Clean'
    )
    first_seen = models.DateTimeField(default=timezone.now)
    last_seen = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.plate_number


class SightingLog(models.Model):
    vehicle = models.ForeignKey(
        'VehicleRegistry',
        on_delete=models.CASCADE,
        related_name='sightings'
    )
    camera_id       = models.CharField(max_length=50, default='CAM_01_TOLL_GATE')
    timestamp       = models.DateTimeField(default=timezone.now)
    snapshot_path   = models.ImageField(upload_to='snapshots/', null=True, blank=True)
    confidence      = models.FloatField(default=0.0)          # real OCR confidence
    weather_condition = models.CharField(max_length=30, default='Clear')  # will be updated
    lat             = models.FloatField(null=True, blank=True)
    lon             = models.FloatField(null=True, blank=True)

    # Simple camera → coordinate mapping (expand as needed)
    CAMERA_COORDS = {
        'CAM_01_TOLL_GATE': (16.3067, 80.4365),
        'CAM_02_HIGHWAY':   (16.3100, 80.4400),
        'CAM_03_MARKET':    (16.2950, 80.4200),
        'CAM_04_CITY':      (16.3000, 80.4550),
        'CAM_05_TUNNEL':    (16.3200, 80.4700),
        'CAM_06_BORDER':    (16.3400, 80.4900),
    }

    def save(self, *args, **kwargs):
        coords = self.CAMERA_COORDS.get(self.camera_id)
        if coords:
            self.lat, self.lon = coords
        super().save(*args, **kwargs)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.vehicle.plate_number} @ {self.timestamp}"


class Camera(models.Model):
    camera_id = models.CharField(max_length=50, primary_key=True)
    name = models.CharField(max_length=100)
    location = models.CharField(max_length=200, blank=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name