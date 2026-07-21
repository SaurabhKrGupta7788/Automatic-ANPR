---
license: apache-2.0
language: en
tags:
  - object-detection
  - yolo
  - yolo11
  - rtdetr
  - damo-yolo
  - pytorch
  - urban-traffic
datasets:
  - UVH-26
pipeline_tag: object-detection
pretty_name: UVH-26
---

<div align="center">

  <!-- Banner Image -->
  <img width="50%" src="banner.png" alt="UVH-26 Banner">

  <div align="center">
    <a href="https://arxiv.org/abs/2511.02563" ><img src="arxiv-logomark-small.svg" height="16" width="11.96" style="display: inline-block; vertical-align: middle; margin: 2px;"> <b style="display: inline-block;"> ArXiv </b></a>  |  
    <a href="https://huggingface.co/datasets/iisc-aim/UVH-26"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="16" width="16" style="display: inline-block; vertical-align: middle; margin: 2px;"><b style="display: inline-block;"> Dataset </b></a>
  </div>

  <!-- Performance Graphic -->
  <img width="95%" src="uvh26_model_perf.png" alt="Performance on UVH-26">

  <p><em>
    Models trained on UVH-26 deliver up to <b>31.5% higher mAP</b> than COCO-pretrained baselines,
    demonstrating significant gains in real-world performance for Indian traffic scenarios.
  </em></p>

</div>

# UVH-26 Vehicle Detection Models — AIM@IISc

High-quality object detection models built for **Indian road traffic** — where vehicle appearance, traffic density, and scene complexity differ significantly from Western datasets like COCO.

These models are trained on the **UVH-26 dataset**, featuring:

- 14 road-relevant vehicle categories
- Real urban environments across India
- Diverse viewpoints, lighting, occlusion & density variations
- Multi-user labeled data with consensus filtering (MV / ST variants)

We currently release **six SOTA detector variants** trained on the dataset:

| Model Family  | Sizes | Strengths                                               |
| ------------- | ----- | ------------------------------------------------------- |
| **YOLOv11**   | S, X  | Fast + lightweight deployment                           |
| **RT-DETRv2** | S, X  | High-accuracy, transformer-based real-time detection    |
| **DAMO-YOLO** | T, L  | Efficient architectures, strong large-scale performance |

> Designed for Indian mobility — adaptable to real city surveillance, roadside cameras, safety monitoring, and ITS applications.

Model Dataset -> [https://huggingface.co/datasets/iisc-aim/UVH-26](https://huggingface.co/datasets/iisc-aim/UVH-26)

---

## Attribution

More technical details about the dataset and models are available in our [Technical Report available on arXiv](https://arxiv.org/abs/2511.02563). 
If you use these datasets or models, kindly cite the following:
**The Urban Vision Hackathon Dataset and Models: Towards Image Annotations and Accurate Vision Models for Indian Traffic, Preliminary Dataset Release, UVH-26-v1.0**, 
Akash Sharma, Chinmay Mhatre, Sankalp Gawali, Ruthvik Bokkasam, Brij Kishore, Vishwajeet Pattanaik, Tarun Rambha, Abdul R. Pinjari, Vijay Kovvali, Anirban Chakraborty, Punit Rathore, Raghu Krishnapuram and Yogesh Simmhan, 
*Technical Report, Indian Institute of Science*, [arXiv:2511.02563](https://arxiv.org/abs/2511.02563), Nov, 2025.


```bibtex
@techreport{sharma2025uvh26,
  title        = {Towards Image Annotations and Accurate Vision Models for Indian Traffic, Preliminary Dataset Release, UVH-26-v1.0},
  author       = {Akash Sharma and Chinmay Mhatre and Sankalp Gawali and Ruthvik Bokkasam and Brij Kishore and Vishwajeet Pattanaik and Tarun Rambha and Abdul R. Pinjari and Vijay Kovvali and Anirban Chakraborty and Punit Rathore and Raghu Krishnapuram and Yogesh Simmhan},
  institution  = {Indian Institute of Science},
  type         = {Technical Report},
  number       = {arXiv:2511.02563},
  year         = {2025},
  month        = {November},
  doi          = {10.48550/arXiv.2511.02563}
}
```

---

### Repository Structure

- **LICENSE** – Apache-2.0 license
- **uvh_classes.txt** – 14 object classes (one per line)
- **configs/** – Model configuration files
  - `yolo11_s.yaml`
  - `yolo11_x.yaml`
  - `rtdetr_s.yaml`
  - `rtdetr_x.yaml`
  - `damo_yolo_t.yaml`
  - `damo_yolo_l.yaml`
- **weights/** – Trained model weights

  > **Naming convention**
  >
  > - `MV` → Majority Voting
  > - `ST` → STAPLE (expected future update)

  - **YOLOv11-S/**
    - `UVH-26-MV-YOLOv11-S.pt`
    - (ST version coming soon)
  - **YOLOv11-X/**
    - `UVH-26-MV-YOLOv11-X.pt`
    - (ST version coming soon)
  - **RT-DETRv2-S/**
    - `UVH-26-MV-RT-DETRv2-S.pth`
    - (ST version coming soon)
  - **RT-DETRv2-X/**
    - `UVH-26-MV-RT-DETRv2-X.pth`
    - (ST version coming soon)
  - **DAMO-YOLO-T/**
    - `UVH-26-MV-DAMO-YOLO-T.pth`
    - (ST version coming soon)
  - **DAMO-YOLO-L/**
    - `UVH-26-MV-DAMO-YOLO-L.pth`
    - (ST version coming soon)

---

### Note — Future Releases

STAPLE (`ST`) variants for all models will be released soon in a follow-up version.

## Model Families

| **Model Family** | **Variants** | **Consensus Versions** | **Format** |
| ---------------- | ------------ | ---------------------- | ---------- |
| YOLOv11          | S, X         | ST, MV                 | `.pt`      |
| RT-DETRv2        | S, X         | ST, MV                 | `.pth`     |
| DAMO-YOLO        | T, L         | ST, MV                 | `.pth`     |

- **MV (Majority Voting):** Uses strict label agreement for conservative labeling.
- **ST (STAPLE):** Uses probabilistic label fusion for reduced annotation noise.

---

## Classes

The file `uvh_classes.txt` lists all **14 object categories**, one per line:

| ID  | Class Name      | Description                                                                                                                                            |
| --- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Hatchback       | Small passenger cars without a protruding rear boot (“dickey”).                                                                                        |
| 2   | Sedan           | Passenger cars with a low-slung design and a separate protruding rear boot (“dickey”).                                                                 |
| 3   | SUV             | Car-like vehicles with high ground clearance, a sturdy body, and no protruding boot.                                                                   |
| 4   | MUV             | Large vehicles with three seating rows, combining passenger and cargo functionality.                                                                   |
| 5   | Bus             | Large passenger vehicles used for public or private transport, including office shuttles and intercity buses.                                          |
| 6   | Truck           | Heavy goods carriers with a front cabin and a rear cargo compartment.                                                                                  |
| 7   | Three-wheeler   | Compact vehicles with one front wheel and two rear wheels, featuring a covered passenger cabin.                                                        |
| 8   | Two-wheeler     | Motorbikes and scooters for single or double riders. Bounding boxes include both vehicle and rider.                                                    |
| 9   | LCV             | Lightweight goods carriers used for short- to medium-distance transport.                                                                               |
| 10  | Mini-bus        | Shorter, compact buses with fewer seats; larger than a Tempo Traveller, often featuring a flat front.                                                  |
| 11  | Tempo-traveller | Medium-sized passenger vans with tall roofs and side windows; larger than vans but smaller than minibuses, with a protruding front.                    |
| 12  | Bicycle         | Non-motorized, manually pedalled vehicles including geared, non-geared, women’s, and children’s cycles. Bounding boxes include both vehicle and rider. |
| 13  | Van             | Medium-sized vehicles for transporting goods or people, typically with a flat front and sliding side doors; smaller than Tempo Travellers.             |

---

## Training Hyperparameters and Architecture

| **Setting**                   | **DAMO-YOLO-S**                  | **DAMO-YOLO-X**                  | **YOLOv11-S**       | **YOLOv11-X**       | **RT-DETRv2-S**                                    | **RT-DETRv2-X**                                    |
| ----------------------------- | -------------------------------- | -------------------------------- | ------------------- | ------------------- | -------------------------------------------------- | -------------------------------------------------- |
| **Batch Size**                | 16                               | 16                               | 16                  | 16                  | 16                                                 | 16                                                 |
| **Best Epoch**                | 47                               | 18                               | 30                  | 13                  | 66                                                 | 68                                                 |
| **Learning Rate / Optimizer** | 0.01 / SGD                       | 0.01 / SGD                       | auto / AdamW        | auto / AdamW        | 1e-4 / AdamW                                       | 1e-4 / AdamW                                       |
| **Weight Decay / Momentum**   | 5e-4 / 0.9                       | 5e-4 / 0.9                       | 5e-4 / (0.9, 0.999) | 5e-4 / (0.9, 0.999) | 1e-4 / (0.9, 0.999)                                | 1e-4 / (0.9, 0.999)                                |
| **LR Policy**                 | Constant                         | Constant                         | Cosine              | Cosine              | Linear                                             | Linear                                             |
| **Warm-up / No-Aug Phase**    | 5 epochs / 16 epochs             | 5 / 16                           | patience = 150      | patience = 150      | – / 70 epochs                                      | – / 70 epochs                                      |
| **Augmentation**              | mixup 0.15, shear 2°, rotate 10° | mixup 0.15, shear 2°, rotate 10° | no mosaic/mixup     | no mosaic/mixup     | flip, color-distort, zoom, IoU-crop (to 70 epochs) | flip, color-distort, zoom, IoU-crop (to 70 epochs) |
| **Backbone**                  | TinyNAS-L20 + GiraffeNeck V2     | TinyNAS-L45 + GiraffeNeck V2     | YOLOv11-S + CSP     | YOLOv11-X + CSP     | PResNet-18 + HybridEnc                             | PResNet-101 + HybridEnc                            |

_All models were trained on the UVH-26 dataset with identical batch sizes and consistent augmentation settings for fair comparison._

---

## License

- This repository (models, weights, configs) is released under the **Apache License 2.0**.
- _Note:_ The underlying YOLO-family models (e.g., YOLOv11) from Ultralytics are distributed under the **GNU AGPL v3.0** (or newer) license.

---
