# Automatic ANPR

A repository for an Automatic Number Plate Recognition system (YOLO/EasyOCR pipeline).

## Architecture Overview

```text
Camera Feed
    │
    ▼
┌──────────────────────────┐
│  YOLO Object Detection   │  ← Detects License Plates
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  EasyOCR / Tesseract     │  ← Extracts Text
└──────────┬───────────────┘
           │
           ▼
        Output Data
```

## Directory Structure

```text
Automatic-ANPR/
└── README.md
```

*(This repository is currently pending code upload).*
