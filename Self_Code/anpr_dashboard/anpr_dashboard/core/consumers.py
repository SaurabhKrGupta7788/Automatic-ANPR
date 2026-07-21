import json
from urllib.parse import parse_qs
from channels.generic.websocket import AsyncWebsocketConsumer


class DashboardConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        # Get camera id from URL query ?cam=CAM_01
        query_string = self.scope["query_string"].decode()
        params = parse_qs(query_string)

        self.camera_id = params.get("cam", ["CAM_01_TOLL_GATE"])[0]

        # Create camera-specific group
        self.group_name = f"dashboard_{self.camera_id}"

        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )

        await self.accept()

        print(f"WS CONNECT → {self.camera_id}")


    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

        print(f"WS DISCONNECT → {self.camera_id}")


    async def send_vehicle_update(self, event):
        await self.send(text_data=json.dumps(event["message"]))