#!/usr/bin/env python3

# Example usage:
# python tools/inference_dir.py \
#     --model_path path/to/model.pth \
#     --config path/to/config.py \
#     --image_dir path/to/image_dir \
#     --output_json path/to/output.json \
#     --infer_size 640 640 \
#     --device cuda

import argparse
import json
import os
from pathlib import Path

if 'PYTORCH_CUDA_ALLOC_CONF' in os.environ:
    alloc_conf = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    if 'expandable_segments' in alloc_conf:
        # Remove expandable_segments option
        new_conf = ','.join([opt for opt in alloc_conf.split(',') if 'expandable_segments' not in opt])
        if new_conf:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = new_conf
        else:
            os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm

from damo.base_models.core.ops import RepConv
from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.utils import postprocess
from damo.utils.demo_utils import transform_img
from damo.structures.image_list import ImageList
from damo.structures.bounding_box import BoxList


def pad_image(img, target_size):
    """Pad image to target size."""
    n, c, h, w = img.shape
    assert n == 1
    assert h <= target_size[0] and w <= target_size[1]
    target_size = [n, c, target_size[0], target_size[1]]
    pad_imgs = torch.zeros(*target_size)
    pad_imgs[:, :c, :h, :w].copy_(img)

    img_sizes = [img.shape[-2:]]
    pad_sizes = [pad_imgs.shape[-2:]]

    return ImageList(pad_imgs, img_sizes, pad_sizes)


def get_image_files(image_dir):
    """Get all image files from directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_dir = Path(image_dir)
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))
    return sorted(image_files)


class BatchInfer:
    def __init__(self, config, infer_size=[640, 640], device='cuda', ckpt=None):
        """Initialize inference engine."""
        self.ckpt_path = ckpt
        suffix = ckpt.split('.')[-1]
        if suffix == 'onnx':
            self.engine_type = 'onnx'
        elif suffix == 'trt':
            self.engine_type = 'tensorRT'
        elif suffix in ['pt', 'pth']:
            self.engine_type = 'torch'
        else:
            raise ValueError(f'Unknown checkpoint format: {suffix}')
        
        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            logger.warning('CUDA not available, using CPU')

        if "class_names" in config.dataset:
            self.class_names = config.dataset.class_names
        else:
            self.class_names = []
            for i in range(config.model.head.num_classes):
                self.class_names.append(str(i))
            self.class_names = tuple(self.class_names)

        self.infer_size = infer_size
        config.dataset.size_divisibility = 0
        self.config = config
        self.model = self._build_engine(self.config, self.engine_type)

    def _build_engine(self, config, engine_type):
        """Build inference engine."""
        logger.info(f'Inference with {engine_type} engine!')
        if engine_type == 'torch':
            model = build_local_model(config, self.device)
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt['model'], strict=True)
            for layer in model.modules():
                if isinstance(layer, RepConv):
                    layer.switch_to_deploy()
            model.eval()
            return model
        elif engine_type == 'tensorRT':
            raise NotImplementedError('TensorRT inference not implemented in this script. Use demo.py instead.')
        elif engine_type == 'onnx':
            raise NotImplementedError('ONNX inference not implemented in this script. Use demo.py instead.')
        else:
            raise NotImplementedError(f'{engine_type} is not supported yet! Please use one of [onnx, torch, tensorRT]')

    def preprocess(self, origin_img):
        """Preprocess image for inference."""
        img = transform_img(origin_img, 0,
                            **self.config.test.augment.transform,
                            infer_size=self.infer_size)
        oh, ow, _ = origin_img.shape
        img = pad_image(img.tensors, self.infer_size)
        img = img.to(self.device)
        return img, (ow, oh)

    def forward(self, origin_image):
        """Run inference on image."""
        image, origin_shape = self.preprocess(origin_image)
        with torch.no_grad():
            output = self.model(image)
        return output, image, origin_shape

    def postprocess_to_coco(self, preds, image, origin_shape):
        """Postprocess predictions to COCO format."""
        output = preds[0]
        output = output.resize(origin_shape)
        output = output.convert('xywh')  # Convert to xywh format for COCO
        
        # Handle empty predictions
        if len(output) == 0:
            return []
        
        bboxes = output.bbox.cpu().detach().numpy()
        scores = output.get_field('scores').cpu().detach().numpy()
        labels = output.get_field('labels').cpu().detach().numpy()
        
        # Model outputs 0-indexed labels (0 to num_classes-1)
        # COCO category_id is 1-indexed (1 to num_classes)
        category_ids = labels + 1
        
        coco_results = []
        for k in range(len(bboxes)):
            coco_results.append({
                'image_id': None,  # Will be set later
                'category_id': int(category_ids[k]),
                'bbox': bboxes[k].tolist(),  # [x, y, width, height]
                'score': float(scores[k]),
            })
        
        return coco_results


def main():
    parser = argparse.ArgumentParser('DAMO-YOLO Directory Inference')
    parser.add_argument(
        '--model_path',
        required=True,
        type=str,
        help='Path to model checkpoint (.pth, .pt)'
    )
    parser.add_argument(
        '--config',
        required=True,
        type=str,
        help='Path to config file'
    )
    parser.add_argument(
        '--image_dir',
        required=True,
        type=str,
        help='Path to directory containing images'
    )
    parser.add_argument(
        '--output_json',
        required=True,
        type=str,
        help='Path to output JSON file (COCO format)'
    )
    parser.add_argument(
        '--infer_size',
        nargs='+',
        type=int,
        default=[640, 640],
        help='Inference image size [height width]'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        type=str,
        help='Device for inference (cuda or cpu)'
    )
    parser.add_argument(
        '--conf_threshold',
        default=None,
        type=float,
        help='Confidence threshold (uses config default if not specified)'
    )
    
    args = parser.parse_args()
    
    # Parse config
    config = parse_config(args.config)
    
    # Override confidence threshold if provided
    if args.conf_threshold is not None:
        config.model.head.nms_conf_thre = args.conf_threshold
    
    # Parse inference size
    if len(args.infer_size) == 1:
        infer_size = [args.infer_size[0], args.infer_size[0]]
    elif len(args.infer_size) == 2:
        infer_size = args.infer_size
    else:
        raise ValueError('infer_size should be 1 or 2 values')
    
    # Initialize inference engine
    logger.info(f'Loading model from {args.model_path}')
    infer_engine = BatchInfer(
        config,
        infer_size=infer_size,
        device=args.device,
        ckpt=args.model_path
    )
    
    # Get all image files
    image_files = get_image_files(args.image_dir)
    if len(image_files) == 0:
        logger.error(f'No image files found in {args.image_dir}')
        return
    
    logger.info(f'Found {len(image_files)} images')
    
    # Process images
    all_results = []
    
    for img_id, image_path in enumerate(tqdm(image_files, desc='Processing images')):
        # Load image
        origin_img = cv2.imread(str(image_path))
        if origin_img is None:
            logger.warning(f'Failed to load image: {image_path}')
            continue
        
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        
        # Run inference
        preds, image, origin_shape = infer_engine.forward(origin_img)
        
        # Postprocess to COCO format
        coco_results = infer_engine.postprocess_to_coco(preds, image, origin_shape)
        
        # Use image filename (without extension) as image_id
        image_id = image_path.stem
        
        for result in coco_results:
            result['image_id'] = image_id
            all_results.append(result)
    
    # Save results
    output_dir = Path(args.output_json).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f'Saved {len(all_results)} detections to {args.output_json}')
    logger.info(f'Processed {len(image_files)} images')


if __name__ == '__main__':
    main()


