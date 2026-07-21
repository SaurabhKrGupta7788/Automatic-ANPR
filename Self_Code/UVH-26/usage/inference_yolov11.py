#!/usr/bin/env python3

"""
Example usage:
python inference.py \
--model_path path/to/model.pth \
--image_dir path/to/image_dir \
--output_json path/to/output.json \
"""

import argparse
import json
import os
from pathlib import Path
from ultralytics import YOLO


def convert_bbox_to_coco_format(bbox):
    """
    Convert YOLO bbox format [x_min, y_min, x_max, y_max] to COCO format [x_min, y_min, width, height].
    
    Args:
        bbox: List or tensor [x_min, y_min, x_max, y_max]
    
    Returns:
        List [x_min, y_min, width, height]
    """
    x_min, y_min, x_max, y_max = bbox[:4]
    width = x_max - x_min
    height = y_max - y_min
    return [float(x_min), float(y_min), float(width), float(height)]


def run_inference(model_path, image_dir, output_json_path):
    """
    Run YOLOv11 inference on images in a directory and save results in COCO format.
    
    Args:
        model_path: Path to the YOLOv11 model file (.pt)
        image_dir: Directory containing images to process
        output_json_path: Path where output JSON will be saved
    """
    # Load the model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Get all image files from the directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_dir_path = Path(image_dir)
    image_files = [
        f for f in image_dir_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    image_files.sort()  # Sort for consistent ordering
    
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")
    
    print(f"Found {len(image_files)} images to process...")
    
    # Run inference
    coco_results = []
    image_id_map = {}  # Map filename to image_id
    
    for idx, image_file in enumerate(image_files):
        image_id = image_file.stem  # Use filename without extension as image_id
        image_id_map[str(image_file)] = image_id
        
        # Run inference on the image (use GPU if available)
        results = model(str(image_file), device='cuda', verbose=False)
        
        # Process results for this image
        result = results[0]  # Get first (and only) result
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for i in range(len(boxes)):
                # Get box coordinates, class, and confidence
                box = boxes.xyxy[i].cpu().numpy()  # [x_min, y_min, x_max, y_max]
                cls = int(boxes.cls[i].cpu().numpy())  # class_id
                conf = float(boxes.conf[i].cpu().numpy())  # confidence score
                
                # Convert to COCO bbox format
                coco_bbox = convert_bbox_to_coco_format(box)
                
                # Add to results
                coco_results.append({
                    "image_id": image_id,
                    "category_id": cls,
                    "bbox": coco_bbox,
                    "score": conf
                })
        else:
            # No detections for this image
            print(f"No detections in {image_file.name}")
    
    # Save results to JSON file
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(coco_results, f, indent=2)
    
    print(f"\nInference complete!")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total detections: {len(coco_results)}")
    print(f"Results saved to: {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run YOLOv11 inference on images and output results in COCO format"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the YOLOv11 model file (.pt)"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing images to process"
    )
    parser.add_argument(
        "--output_json_path",
        type=str,
        required=True,
        help="Path where output JSON file will be saved"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    if not os.path.isdir(args.image_dir):
        raise NotADirectoryError(f"Image directory not found: {args.image_dir}")
    
    run_inference(args.model_path, args.image_dir, args.output_json_path)


