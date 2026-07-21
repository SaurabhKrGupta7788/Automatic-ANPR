"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# Example usage:
# python references/deploy/rtdetrv2_torch.py \
#   -c path/to/model_config.yml \
#   -r path/to/model.pth \
#   --im-dir path/to/images_dir \
#   -d cuda \
#   -o path/to/output.json

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw
import sys

# Ensure repository root is on sys.path so `src` package can be imported
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core import YAMLConfig

def save_coco_format(results, output_file='detections.json'):
    """Save detection results in COCO format
    
    Args:
        results: List of detection dictionaries
        output_file: Path to save JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved COCO format results to {output_file}')


def get_image_files(path):
    """Get all image files from a path (file or directory)
    
    Args:
        path: Path to image file or directory
        
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    path = Path(path)
    
    if path.is_file():
        return [path]
    elif path.is_dir():
        image_files = []
        for ext in image_extensions:
            image_files.extend(path.glob(f'*{ext}'))
            image_files.extend(path.glob(f'*{ext.upper()}'))
        return sorted(image_files)
    else:
        raise ValueError(f"Path {path} is neither a file nor a directory")


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)
    model.eval()  # Ensure model is in eval mode
    
    # Get image files from either single file or directory
    if args.im_dir:
        image_files = get_image_files(args.im_dir)
    elif args.im_file:
        image_files = get_image_files(args.im_file)
    else:
        raise ValueError("Either --im-file or --im-dir must be provided")
    
    print(f'Processing {len(image_files)} image(s)...')
    
    # Prepare transforms
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    # Store results for COCO format
    coco_results = []
    
    # Process each image with memory-efficient approach
    with torch.no_grad():  # Disable gradient computation to save memory
        for idx, image_path in enumerate(image_files):
            image_name = image_path.name
            print(f'Processing {image_name} ({idx+1}/{len(image_files)})...')
            
            # Load and prepare image
            im_pil = Image.open(image_path).convert('RGB')
            w, h = im_pil.size
            orig_size = torch.tensor([w, h], dtype=torch.int64)[None].to(args.device)
            
            # Transform and run inference
            im_data = transforms(im_pil)[None].to(args.device)
            output = model(im_data, orig_size)
            labels, boxes, scores = output
            
            # Move to CPU immediately to free GPU memory
            labels_cpu = labels[0].cpu()
            boxes_cpu = boxes[0].cpu()
            scores_cpu = scores[0].cpu()
            
            # Delete GPU tensors immediately
            del im_data, orig_size, output, labels, boxes, scores
            if args.device != 'cpu':
                torch.cuda.empty_cache()  # Clear CUDA cache after each image
            
            # Convert to COCO format
            for label, box, score in zip(labels_cpu, boxes_cpu, scores_cpu):
                # bbox format in COCO: [x, y, width, height]
                x1, y1, x2, y2 = box.tolist()
                bbox = [x1, y1, x2 - x1, y2 - y1]
                
                coco_result = {
                    "image_id": image_name,
                    "category_id": int(label.item()),
                    "bbox": bbox,
                    "score": float(score.item())
                }
                coco_results.append(coco_result)
            
            # Delete CPU tensors (they're already converted to Python objects)
            del labels_cpu, boxes_cpu, scores_cpu
            
            # Periodically clear cache for large batches
            if (idx + 1) % 50 == 0 and args.device != 'cpu':
                torch.cuda.empty_cache()
                print(f'  Cleared GPU cache after {idx+1} images')
    
    # Save COCO format JSON
    save_coco_format(coco_results, args.output_json)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RT-DETR PyTorch Inference')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
    parser.add_argument('-r', '--resume', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('-f', '--im-file', type=str, default=None, help='Path to single image file')
    parser.add_argument('--im-dir', type=str, default=None, help='Path to directory containing images')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device to run inference on (cpu/cuda)')
    parser.add_argument('-o', '--output-json', type=str, default='detections.json', help='Path to save COCO format JSON')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save visualization images')
    args = parser.parse_args()
    main(args)

