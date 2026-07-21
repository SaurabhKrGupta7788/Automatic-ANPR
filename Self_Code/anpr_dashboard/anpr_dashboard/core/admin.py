from django.contrib import admin
from django.utils.html import format_html
from .models import VehicleRegistry, SightingLog


class SightingInline(admin.TabularInline):
    model = SightingLog
    extra = 0
    readonly_fields = ("timestamp", "camera_id", "confidence", "snapshot_preview")
    fields = ("timestamp", "camera_id", "confidence", "snapshot_preview")
    can_delete = False

    def snapshot_preview(self, obj):
        if obj.snapshot_path:
            return format_html(
                '<img src="{}" width="120" style="border-radius:4px;" />',
                obj.snapshot_path.url
            )
        return "-"
    snapshot_preview.short_description = "Snapshot"


@admin.register(VehicleRegistry)
class VehicleRegistryAdmin(admin.ModelAdmin):
    list_display = (
        "plate_number",
        "predicted_type",
        "predicted_color",
        "owner_status",
        "first_seen",
        "last_seen",
    )
    list_filter = ("owner_status", "predicted_type", "predicted_color")
    search_fields = ("plate_number",)
    ordering = ("-last_seen",)
    inlines = [SightingInline]
    readonly_fields = ("first_seen", "last_seen")


@admin.register(SightingLog)
class SightingLogAdmin(admin.ModelAdmin):
    list_display = (
        "vehicle",
        "camera_id",
        "timestamp",
        "confidence",
        "snapshot_preview",
    )
    list_filter = ("camera_id", "timestamp")
    search_fields = ("vehicle__plate_number",)
    ordering = ("-timestamp",)
    readonly_fields = ("timestamp", "snapshot_preview")

    def snapshot_preview(self, obj):
        if obj.snapshot_path:
            return format_html(
                '<img src="{}" width="150" style="border-radius:4px;" />',
                obj.snapshot_path.url
            )
        return "-"
    snapshot_preview.short_description = "Snapshot"