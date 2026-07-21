from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import numpy as np
import torch
import re
from collections import defaultdict, deque
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.conf import settings
from .models import VehicleRegistry, SightingLog
import os
import os
import cv2
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte

import torchvision.transforms.functional as TF
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
from tqdm import tqdm
import argparse
import numpy as np


from django.http import JsonResponse
from .models import SightingLog


from django.http import JsonResponse


from .models import Camera





from django.utils.timezone import now




NO_VEHICLE_RULE = {
    0: "Friday",
    9: "Friday",
    1: "Monday",
    2: "Monday",
    3: "Tuesday",
    4: "Tuesday",
    5: "Wednesday",
    6: "Wednesday",
    7: "Thursday",
    8: "Thursday",
}



from django.utils import timezone

def check_no_vehicle_violation(plate):
    if not plate or not plate[-1].isdigit():
        return False, None, None

    last_digit = int(plate[-1])
    banned_day = NO_VEHICLE_RULE.get(last_digit)

    today = timezone.localtime().strftime("%A")

    return today == banned_day, banned_day, today



def today_stats(request):
    today = now().date()

    total = SightingLog.objects.filter(timestamp__date=today).count()

    # count violations
    alerts = 0
    logs = SightingLog.objects.filter(timestamp__date=today).select_related("vehicle")

    for s in logs:
        plate = s.vehicle.plate_number
        is_violation, _, _ = check_no_vehicle_violation(plate)
        if is_violation:
            alerts += 1

    return JsonResponse({
        "total": total,
        "alerts": alerts,
        "day": today.strftime("%A")
    })



# ================= GLOBAL CAMERA =================
GLOBAL_CAMERA = None
CURRENT_SOURCE = None

def get_camera(source=0):
    global GLOBAL_CAMERA, CURRENT_SOURCE

    # If already opened with same source → reuse
    if GLOBAL_CAMERA is not None and CURRENT_SOURCE == source:
        return GLOBAL_CAMERA

    # If different source → release old
    if GLOBAL_CAMERA is not None:
        GLOBAL_CAMERA.release()

    GLOBAL_CAMERA = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    CURRENT_SOURCE = source

    return GLOBAL_CAMERA


def multicamera_dashboard(request):
    cameras = [
        {"id":"CAM_01_TOLL_GATE","online":True},
        {"id":"CAM_02_HIGHWAY","online":False},
        {"id":"CAM_03_MARKET","online":False},
        {"id":"CAM_04_CITY","online":False},
        {"id":"CAM_05_TUNNEL","online":False},
        {"id":"CAM_06_BORDER","online":False},
    ]
    return render(request,"multicamera_dashboard.html",{"cameras":cameras})


def forensic_dashboard(request):
    return render(request, 'forensic_dashboard.html')

from django.http import JsonResponse
from datetime import datetime
from .models import SightingLog

def search_sightings(request):
    try:
        qs = SightingLog.objects.select_related('vehicle').order_by('-timestamp')

        plate = request.GET.get('plate_search')
        if plate:
            qs = qs.filter(vehicle__plate_number__icontains=plate)

        start = request.GET.get('start_time')
        if start:
            try:
                qs = qs.filter(timestamp__gte=datetime.fromisoformat(start))
            except:
                pass

        end = request.GET.get('end_time')
        if end:
            try:
                qs = qs.filter(timestamp__lte=datetime.fromisoformat(end))
            except:
                pass

        color = request.GET.get('color_filter')
        if color:
            qs = qs.filter(vehicle__predicted_color__iexact=color)

        vtype = request.GET.get('type_filter')
        if vtype:
            qs = qs.filter(vehicle__predicted_type__iexact=vtype)

        qs = qs[:50]

        sightings = list(qs.values(
            "id",
            "timestamp",
            "camera_id",
            "lat",
            "lon",
            "confidence",
            "weather_condition",
            "vehicle__plate_number",
            "vehicle__predicted_color",
            "vehicle__predicted_type"
        ))

        return JsonResponse({"sightings": sightings}, safe=False)

    except Exception as e:
        print("SEARCH API ERROR:", e)
        return JsonResponse({"sightings": []})


def latest_sightings(request):
    sightings = SightingLog.objects.select_related('vehicle') \
                                   .order_by('-timestamp')[:10]

    data = []
    for s in sightings:
        plate = s.vehicle.plate_number
        is_violation, banned_day, today = check_no_vehicle_violation(plate)

        data.append({
            "id": s.id,
            "plate": plate,
            "color": s.vehicle.predicted_color,
            "vehicle_type": s.vehicle.predicted_type,
            "confidence": s.confidence,
            "timestamp": s.timestamp.isoformat(),
            "snapshot_url": s.snapshot_path.url if s.snapshot_path else None,

            # ADD THESE
            "is_violation": is_violation,
            "no_vehicle_day": banned_day,
            "today": today
        })

    return JsonResponse({"vehicles": data})

# ... (imports same as before)

# def master_dashboard(request):
#     return render(request, 'master_dashboard.html')

def master_dashboard(request):
    cam = request.GET.get("cam", "CAM_01_TOLL_GATE")
    return render(request, 'master_dashboard.html', {
        "camera_id": cam
    })



task = 'Single_Image_Defocus_Deblurring'

def get_weights_and_parameters(task, parameters):
    if task == 'Motion_Deblurring':
        weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    elif task == 'Single_Image_Defocus_Deblurring':
        weights = os.path.join('core', 'Restormer','Defocus_Deblurring', 'pretrained_models', 'net_g_2000 (2).pth.zip')
    elif task == 'Deraining':
        weights = os.path.join('Deraining', 'pretrained_models', 'deraining.pth')
    elif task == 'Real_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'real_denoising.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
    return weights, parameters


# Get model weights and parameters
parameters = {'inp_channels':3, 'out_channels':3, 'dim':12, 'num_blocks':[2,2,2,2], 'num_refinement_blocks':2, 'heads':[1,2,2,4], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
weights, parameters = get_weights_and_parameters(task, parameters)

load_arch = run_path(os.path.join('core','Restormer', 'basicsr', 'models', 'archs', 'restormer_arch.py'))
model_deblur = load_arch['Restormer'](**parameters)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_deblur.to(device)

# model_deblur.cuda()

checkpoint = torch.load(weights)
model_deblur.load_state_dict(checkpoint['params'])
model_deblur.eval()


def restore_image(img, model, img_multiple_of=8):
    """
    img: numpy array (BGR or RGB)
    returns: restored numpy image (same size)
    """

    model.eval()

    # If BGR → RGB
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).cuda()

        h, w = input_.shape[2], input_.shape[3]
        H = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of
        W = ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - h if h % img_multiple_of != 0 else 0
        padw = W - w if w % img_multiple_of != 0 else 0

        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        restored = model(input_)
        restored = torch.clamp(restored, 0, 1)

        restored = restored[:, :, :h, :w]
        restored = restored.permute(0,2,3,1).cpu().numpy()[0]
        restored = img_as_ubyte(restored)

        # Back to BGR for OpenCV pipeline
        restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)

    return restored


import torch
import cv2
import numpy as np
import sys
sys.path.append('.')

# from .ESRGAN.RRDBNet_arch import RRDBNet

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load model once
# model = RRDBNet(3, 3, 64, 23, gc=32)
# model.load_state_dict(torch.load(r'D:\Automatic ANPR\Self_Code\anpr_dashboard\anpr_dashboard\core\ESRGAN\models\RRDB_ESRGAN_x4.pth'), strict=True)
# model.eval()
# model = model.to(device)


# def esrgan_super_resolve_cv(img_bgr):
#     """
#     img_bgr: OpenCV image (numpy array, BGR)
#     Returns: Super-resolved image (numpy array, BGR)
#     """

#     img = img_bgr.astype(np.float32) / 255.0

#     # BGR → RGB and HWC → CHW
#     img = torch.from_numpy(np.transpose(img[:, :, [2,1,0]], (2,0,1))).float()
#     img = img.unsqueeze(0).to(device)

#     with torch.no_grad():
#         output = model(img).data.squeeze().float().cpu().clamp_(0, 1).numpy()

#     # CHW → HWC and RGB → BGR
#     output = np.transpose(output[[2,1,0], :, :], (1,2,0))
#     output = (output * 255.0).round().astype(np.uint8)

#     return output

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from runpy import run_path


class RestormerDerain:
    def __init__(self, weights_path, device=None):

        # Auto device selection
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Check Restormer directory
        arch_path = os.path.join(
            "core",
            "Restormer",
            "basicsr",
            "models",
            "archs",
            "restormer_arch.py"
        )

        if not os.path.exists(arch_path):
            raise FileNotFoundError(
                f"Restormer architecture not found at:\n{arch_path}\n"
                "Make sure you cloned the Restormer repo."
            )

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weights file not found at:\n{weights_path}"
            )

        # Model parameters
        parameters = {
            'inp_channels': 3,
            'out_channels': 3,
            'dim': 48,
            'num_blocks': [4, 6, 6, 8],
            'num_refinement_blocks': 4,
            'heads': [1, 2, 4, 8],
            'ffn_expansion_factor': 2.66,
            'bias': False,
            'LayerNorm_type': 'WithBias',
            'dual_pixel_task': False
        }

        # Load architecture
        load_arch = run_path(os.path.join('core','Restormer', 'basicsr', 'models', 'archs', 'restormer_arch.py'))
        model = load_arch['Restormer'](**parameters)

        # Load weights
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['params'])

        self.model = model.to(device)
        self.model.eval()

        self.img_multiple_of = 8

        print(f"✅ Derain model loaded on {self.device}")


    def __call__(self, img):
        """
        img: numpy RGB image (H,W,3), uint8
        returns: restored RGB image (H,W,3), uint8
        """

        if not isinstance(img, np.ndarray):
            raise ValueError("Input must be numpy array (H,W,3)")

        with torch.no_grad():

            input_ = torch.from_numpy(img).float().div(255.0)
            input_ = input_.permute(2, 0, 1).unsqueeze(0).to(self.device)

            h, w = input_.shape[2], input_.shape[3]

            # Pad to multiple of 8
            H = ((h + 7) // 8) * 8
            W = ((w + 7) // 8) * 8

            padh = H - h
            padw = W - w

            if padh != 0 or padw != 0:
                input_ = F.pad(input_, (0, padw, 0, padh), mode='reflect')

            # Forward
            restored = self.model(input_)
            restored = torch.clamp(restored, 0, 1)

            # Remove padding
            restored = restored[:, :, :h, :w]

            restored = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
            restored = (restored * 255).astype(np.uint8)

        return restored

derain_model = RestormerDerain(
    weights_path=r"D:\Automatic ANPR\Self_Code\anpr_dashboard\anpr_dashboard\core\Restormer\Deraining\pretrained_models\deraining.pth")


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from runpy import run_path
import yaml

class MBTaylorFormerDehaze:
    def __init__(self, weights_path,
                 yaml_path=None,
                 device=None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load YAML if given (for network params)
        if yaml_path is not None and os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                cfg = yaml.load(f, Loader=yaml.Loader)
            net_params = cfg['network_g']
            net_params.pop('type', None)
        else:
            # Default fallback: empty dict
            net_params = {}

        # Load architecture
        arch_path = os.path.join('core','ICCV-2023-MB-TaylorFormer', 'basicsr', 'models', 'archs', 'MB_TaylorFormer.py')
        if not os.path.exists(arch_path):
            raise FileNotFoundError(f"Cannot find MB_TaylorFormer.py at {arch_path}")

        load_arch = run_path(arch_path)
        if 'MB_TaylorFormer' not in load_arch:
            raise KeyError("MB_TaylorFormer not found in loaded architecture")
        model = load_arch['MB_TaylorFormer'](**net_params)

        # Load weights
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['params'])

        # Move to device & eval
        model = model.to(device)
        model.eval()

        self.model = model
        self.img_multiple_of = 8

        print(f"✅ Dehaze model loaded on {device}")

    def __call__(self, img):
        """
        img: numpy RGB (H,W,3)
        returns: dehazed RGB (H,W,3)
        """

        if not isinstance(img, np.ndarray):
            raise ValueError("Input must be numpy array (H,W,3)")

        with torch.no_grad():
            input_ = torch.from_numpy(img).float().div(255.0).permute(2,0,1).unsqueeze(0).to(self.device)

            h, w = input_.shape[2], input_.shape[3]

            # Pad to multiple of 8
            H = ((h + 7)//8)*8
            W = ((w + 7)//8)*8
            padh = H - h
            padw = W - w
            if padh !=0 or padw !=0:
                input_ = F.pad(input_, (0,padw,0,padh), mode='reflect')

            # Forward
            output = self.model(input_)
            output = torch.clamp(output,0,1)
            output = output[:, :, :h, :w]
            output = output.squeeze(0).permute(1,2,0).cpu().numpy()
            output = (output*255).astype(np.uint8)

        return output

dehaze_model = MBTaylorFormerDehaze(
    weights_path=r"D:\Automatic ANPR\Self_Code\anpr_dashboard\anpr_dashboard\core\ICCV-2023-MB-TaylorFormer\Dehazing\pretrained_models\ohaze-MB-TaylorFormer-B.pth.zip",
    yaml_path=r"D:\Automatic ANPR\Self_Code\anpr_dashboard\core\ICCV-2023-MB-TaylorFormer\Dehazing\Options\MB-TaylorFormer-B.yml"
)

class WeatherRestorationPipeline:
    def __init__(self,
                 derain_model,
                 dehaze_model,
                 label_name = "clear",
                 device=None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.derain_model = derain_model
        self.dehaze_model = dehaze_model
        self.label_name = label_name

        # Adjust if your labels differ
        self.label_map = {
            0: "clean",
            1: "hazy",
            2: "rainy"
        }

        print("✅ Weather Restoration Pipeline Ready")


    def __call__(self, img):
        """
        img: numpy RGB image
        returns: restored image + predicted label
        """

        if self.label_name == "rainy":
            restored = self.derain_model(img)

        elif self.label_name == "hazy":
            restored = self.dehaze_model(img)

        else:
            restored = img

        return restored, self.label_name









import cv2
import numpy as np
import torch
from core.SwinIR.models.network_swinir import SwinIR as net

# -------- MODEL LOADER (from define_model) --------
def load_swinir(model_path, scale=4, large_model=False, device='cuda'):

    if not large_model:
        model = net(
            upscale=scale,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.,
            depths=[6,6,6,6,6,6],
            embed_dim=180,
            num_heads=[6,6,6,6,6,6],
            mlp_ratio=2,
            upsampler='nearest+conv',
            resi_connection='1conv'
        )
    else:
        model = net(
            upscale=scale,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.,
            depths=[6,6,6,6,6,6,6,6,6],
            embed_dim=240,
            num_heads=[8,8,8,8,8,8,8,8,8],
            mlp_ratio=2,
            upsampler='nearest+conv',
            resi_connection='3conv'
        )

    pretrained = torch.load(model_path)

    # 🔥 FIX HERE
    if 'params_ema' in pretrained:
        model.load_state_dict(pretrained['params_ema'], strict=True)
    elif 'params' in pretrained:
        model.load_state_dict(pretrained['params'], strict=True)
    else:
        model.load_state_dict(pretrained, strict=True)

    model.eval()
    model = model.to(device)

    return model


# -------- INFERENCE FUNCTION --------
def swinir_process(image, model, scale=4, tile=None, tile_overlap=32, device='cuda'):
    """
    image: input BGR image (cv2 format)
    returns: output BGR image
    """

    window_size = 8

    # Preprocess (same as original)
    img = image.astype(np.float32) / 255.
    img = img[:, :, [2,1,0]]  # BGR → RGB
    img = np.transpose(img, (2,0,1))
    img = torch.from_numpy(img).float().unsqueeze(0).to(device)

    # Padding (IMPORTANT)
    _, _, h_old, w_old = img.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old

    img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
    img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]

    # Inference
    with torch.no_grad():
        if tile is None:
            output = model(img)
        else:
            output = tile_inference(img, model, scale, tile, tile_overlap)

    output = output[..., :h_old*scale, :w_old*scale]

    # Postprocess
    output = output.squeeze().float().cpu().clamp_(0,1).numpy()
    output = np.transpose(output[[2,1,0], :, :], (1,2,0))  # RGB → BGR
    output = (output * 255.0).round().astype(np.uint8)

    return output


# -------- TILE INFERENCE (from original test()) --------
def tile_inference(img, model, scale, tile, tile_overlap):
    b, c, h, w = img.size()
    tile = min(tile, h, w)
    stride = tile - tile_overlap

    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]

    E = torch.zeros(b, c, h*scale, w*scale).type_as(img)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            out_patch = model(in_patch)
            mask = torch.ones_like(out_patch)

            E[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale] += out_patch
            W[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale] += mask

    return E / W


model_swinir = load_swinir(r"D:\Automatic ANPR\Self_Code\anpr_dashboard\anpr_dashboard\core\SwinIR\003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth.zip")
model_swinir = model_swinir.to("cuda" if torch.cuda.is_available() else "cpu")
    
##################################################################################
########################### weather choose ######################################

pipeline = WeatherRestorationPipeline(
    derain_model=derain_model,
    dehaze_model=dehaze_model,
    label_name="clear"  # Change as needed for testing (e.g., "rainy", "hazy", "clear"
)
############################################################################
############################################################################
############################################################################


import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
import cv2

# ---- NAFNet root ----
nafnet_root = r"D:\Automatic ANPR\Self_Code\anpr_dashboard\anpr_dashboard\core\NAFNet"
if nafnet_root not in sys.path:
    sys.path.insert(0, nafnet_root)

from basicsr.models.archs.NAFNet_arch import NAFNet


class NAFNetDeblur:
    def __init__(self, weights_path, device=None, tile=256, tile_overlap=32):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Official REDS config
        self.net = NAFNet(
            img_channel=3,
            width=64,
            middle_blk_num=1,
            enc_blk_nums=[1,1,1,28],
            dec_blk_nums=[1,1,1,1],
        )

        checkpoint = torch.load(weights_path, map_location=device)
        state = checkpoint.get("params_ema",
                checkpoint.get("params",
                checkpoint.get("state_dict",
                checkpoint)))

        self.net.load_state_dict(state)
        self.net.to(device).eval()

        # grid inference params (same idea as basicsr)
        self.tile = tile
        self.tile_overlap = tile_overlap

        print("NAFNet REDS (official-style) loaded")

    def _forward_tile(self, img):
        b, c, h, w = img.shape
        tile = self.tile
        overlap = self.tile_overlap

        stride = tile - overlap
        h_idx = list(range(0, h-tile, stride)) + [h-tile]
        w_idx = list(range(0, w-tile, stride)) + [w-tile]

        output = torch.zeros_like(img)
        weight = torch.zeros_like(img)

        for hi in h_idx:
            for wi in w_idx:
                patch = img[:, :, hi:hi+tile, wi:wi+tile]
                out_patch = self.net(patch)

                output[:, :, hi:hi+tile, wi:wi+tile] += out_patch
                weight[:, :, hi:hi+tile, wi:wi+tile] += 1

        return output / weight

    def __call__(self, img_rgb):

        img = torch.from_numpy(img_rgb).float()/255.0
        img = img.permute(2,0,1).unsqueeze(0).to(self.device)

        h, w = img.shape[2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            img = F.pad(img, (0,pad_w,0,pad_h), mode="reflect")

        with torch.no_grad():
            if self.tile is None:
                out = self.net(img)
            else:
                out = self._forward_tile(img)

        out = out[:, :, :h, :w]
        out = torch.clamp(out,0,1)
        out = out.squeeze(0).permute(1,2,0).cpu().numpy()
        out = (out*255).astype(np.uint8)

        return out
    
model_naf = NAFNetDeblur(
    weights_path=r"C:\Users\100ra\OneDrive\Desktop\NAFNet\NAFNet\experiments\pretrained_models\NAFNet-REDS-width64.pth.zip"
)

import cv2
import numpy as np


def generate_rain_layer(shape,
                        rain_density=800,
                        drop_length=20,
                        drop_width=1,
                        angle=-20):
    """
    Creates rain streak layer.
    angle: negative = slant left, positive = slant right
    """

    h, w = shape[:2]
    rain = np.zeros((h, w), dtype=np.uint8)

    for _ in range(rain_density):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        x_end = int(x + drop_length * np.sin(np.radians(angle)))
        y_end = int(y + drop_length * np.cos(np.radians(angle)))

        cv2.line(rain,
                 (x, y),
                 (x_end, y_end),
                 255,
                 drop_width)

    return rain

def apply_blur(image):
    print(f"Applying blur with kernel size: {image.shape}")

    # Strong blur
    blurred = cv2.GaussianBlur(image, (11, 11), 0)

    return blurred


def apply_rain_effect(image,
                      rain_density=1200,
                      drop_length=25,
                      drop_width=1,
                      angle=-25,
                      brightness_reduction=0.7,
                      rain_intensity=0.5):
    """
    Takes an image (numpy array BGR) and returns rainy image.
    """

    # Convert to float
    img = image.astype(np.float32) / 255.0

    # 1️⃣ Generate rain streaks
    rain_layer = generate_rain_layer(img.shape,
                                     rain_density,
                                     drop_length,
                                     drop_width,
                                     angle)

    # 2️⃣ Motion blur kernel (rain direction)
    ksize = 15
    kernel = np.zeros((ksize, ksize))
    kernel[:, ksize // 2] = np.ones(ksize)
    kernel = kernel / ksize

    rain_layer = cv2.filter2D(rain_layer, -1, kernel)

    # 3️⃣ Normalize rain
    rain_layer = rain_layer.astype(np.float32) / 255.0
    rain_layer = cv2.merge([rain_layer] * 3)

    # 4️⃣ Darken scene (rainy atmosphere)
    img = img * brightness_reduction

    # 5️⃣ Blend rain with image
    output = img + rain_layer * rain_intensity

    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)

    return output

import cv2
import torch
import numpy as np

import torch
import cv2
import numpy as np


def landing(request):
    return render(request, "landing.html")
    

# class DepthSmogger:
#     def __init__(self, model_type="DPT_Large"):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Load MiDaS once
#         self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
#         self.midas.to(self.device)
#         self.midas.eval()
        
#         transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
#         self.transform = transforms.dpt_transform

#     def apply(self, img_bgr,
#               haze_strength=1.5,
#               atmospheric_light=0.85):
#         """
#         img_bgr : numpy array (OpenCV image)
#         returns : smogged BGR image
#         """

#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         input_batch = self.transform(img_rgb).to(self.device)

#         # Depth prediction
#         with torch.no_grad():
#             prediction = self.midas(input_batch)
#             prediction = torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=img_rgb.shape[:2],
#                 mode="bicubic",
#                 align_corners=False,
#             ).squeeze()

#         depth = prediction.cpu().numpy()

#         # Normalize depth (far=1)
#         depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
#         depth = 1 - depth
#         depth = depth[..., None]

#         # Atmospheric scattering
#         img_float = img_rgb.astype(np.float32) / 255.0
#         transmission = np.exp(-haze_strength * depth)

#         A = np.ones_like(img_float) * atmospheric_light
#         smog = img_float * transmission + A * (1 - transmission)

#         smog = cv2.GaussianBlur(smog, (7, 7), 0)
#         smog = np.clip(smog, 0, 1)

#         smog = (smog * 255).astype(np.uint8)
#         smog = cv2.cvtColor(smog, cv2.COLOR_RGB2BGR)

#         return smog

# smogger = DepthSmogger()





def video_feed(request):
    return StreamingHttpResponse(gen_processed_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def gen_processed_frames():
    # ====================== YOUR COMPLETE CODE ======================
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

    def broadcast_detection(data, camera_id):
        channel_layer = get_channel_layer()

        group = f"dashboard_{camera_id}"

        async_to_sync(channel_layer.group_send)(
            group,
            {
                "type": "send_vehicle_update",
                "message": data
            }
        )

    # ====================== VIDEO SOURCE ======================
    video_path = r"C:\Users\100ra\Downloads\4K Road traffic video for object detection and tracking - free download now!.mp4"
    cap = get_camera(0)  # Or 0 for webcam

    track_memory = defaultdict(lambda: {
        "state": "idle",
        "plate_buffer": deque(maxlen=7),
        "final_plate": "",
        "color": "",
        "type": ""
    })

    while True:  # Infinite loop
        if cap is None or not cap.isOpened():
            break

        ret, frame = cap.read()
        if not ret:
            continue
################################################333333333333333333333333333333333
#################################  Rain Effect Apply ###################################################


        H = frame.shape[0]


#########################################Apply Rain Effect##########################################################


        # frame = apply_rain_effect(
        #         frame,
        #         rain_density=1600,
        #         drop_length=40,
        #         drop_width=1,
        #         angle=-30
        #     )
        
#########################################################################################
##########################################################################################



        results = model_type.track(frame, persist=True, verbose=False)

        if results[0].boxes.id is not None:
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

#####################################################################################
#####################################################################################

               


############################## Apply Rain Model For deraining ###################################

                # SAVE_DIR = "rainy_vehicle_crops"
                # os.makedirs(SAVE_DIR, exist_ok=True)
                # alpha = cv2.resize(vehicle_crop,(256,256))

                # debug_path = os.path.join(SAVE_DIR, f"vehicle_rainy_crop{track_id}.jpg")
                # cv2.imwrite(debug_path, alpha)

                # vehicle_crop = cv2.resize(vehicle_crop,(256,256))

                # vehicle_crop, weather = pipeline(vehicle_crop)

                # vehicle_crop = np.ascontiguousarray(vehicle_crop)

                # debug_path = os.path.join(SAVE_DIR, f"vehicle_derain_done{track_id}.jpg")
                # cv2.imwrite(debug_path, vehicle_crop)

################################################################################################


############################# Apply NAFNet for deblurring ########################################
                # SAVE_DIR = "Blurring_vehicle_crops"
                # os.makedirs(SAVE_DIR, exist_ok=True)

                # debug_path = os.path.join(SAVE_DIR, f"vehicle_Blur_crop{track_id}.jpg")
                # cv2.imwrite(debug_path, vehicle_crop)

                # # applying blur effect in vehicle crop
                # vehicle_crop = model_naf(vehicle_crop)

                # vehicle_crop = np.ascontiguousarray(vehicle_crop)

                # debug_path = os.path.join(SAVE_DIR, f"vehicle_deblur_done{track_id}.jpg")
                # cv2.imwrite(debug_path, vehicle_crop)

################################################################################################


                

####################################################################################################
###############################################################################################


                        
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
                        


###################################### swinir before ocr ########################################

                        # plate_crop = apply_blur(plate_crop)  # Add blur to plate crop to simulate bad conditions
                        # H, W, C = plate_crop.shape

                        # SAVE_DIR = "no_plate_crops_swinir"
                        # os.makedirs(SAVE_DIR, exist_ok=True)

                        # debug_path = os.path.join(SAVE_DIR, f"plate_crop{track_id}.jpg")
                        # cv2.imwrite(debug_path, plate_crop)

                        # now = datetime.now()
                        # print(now.strftime("%H:%M:%S"))
                        
                        # plate_crop = swinir_process(plate_crop, model_swinir)
                        
                        # now = datetime.now()
                        # print(now.strftime("%H:%M:%S"))

                        # plate_crop = cv2.resize(plate_crop, (int(1*W), int(1*H)), interpolation=cv2.INTER_CUBIC)
                        

                        # debug_path = os.path.join(SAVE_DIR, f"plate_crop_blur_done{track_id}.jpg")
                        # cv2.imwrite(debug_path, plate_crop)

###################################################################################################

                        if plate_crop.size == 0:
                            continue

                        proc = preprocess_plate_for_paddle(plate_crop)
                        raw = ocr_plate_paddle(proc)

                        if validate_indian_plate(raw):
                            mem["plate_buffer"].append(raw)

                buf = mem["plate_buffer"]
                if len(buf) >= 4:  # Need at least 3 readings to be confident
                    final_plate = max(set(buf), key=buf.count)
                    mem["final_plate"] = final_plate

                    is_violation, banned_day, today = check_no_vehicle_violation(final_plate)

                    mem["no_vehicle_day"] = banned_day
                    mem["is_violation"] = is_violation



                    mem["state"] = "done"

                    # Optional: calculate average confidence of all readings that match final_plate
                    matching_confs = []
                    for i, candidate in enumerate(buf):
                        if candidate == final_plate:
                            # You would need to store confidence together with plate text
                            # For simplicity we use the last OCR confidence here
                            # Better solution = store list of (text, conf) tuples
                            matching_confs.append(0.92)  # placeholder – see improvement below

                    avg_conf = sum(matching_confs) / len(matching_confs) if matching_confs else 0.95


                    # SAVE TO DB + BROADCAST
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

                    SightingLog.objects.create(
                        vehicle=vehicle,
                        camera_id="CAM_01_TOLL_GATE",
                        confidence=avg_conf * 100,
                        snapshot_path=f"snapshots/{snapshot_name}"
                    )

                    is_violation, banned_day, today = check_no_vehicle_violation(final_plate)

                    data = {
                        "type": "vehicle_detected",
                        "plate": final_plate,
                        "color": mem["color"],
                        "vehicle_type": mem["type"],
                        "confidence": round(avg_conf * 100, 1),
                        "timestamp": datetime.now().isoformat(),
                        "snapshot_url": f"/media/snapshots/{snapshot_name}",

                        # NEW
                        "no_vehicle_day": banned_day,
                        "is_violation": is_violation,
                        "today": today
                    }
                    camera_id = "CAM_01_TOLL_GATE"
                    broadcast_detection(data, camera_id)

                label = f"ID:{track_id} {mem['type']} {mem['color']}"
                if mem["final_plate"]:
                    label += f" {mem['final_plate']}"

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        # Stream frame to browser (no imshow)
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()  # Cleanup (though loop is infinite)












