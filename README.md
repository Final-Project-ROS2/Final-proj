# YOLOv11 + SAM Object Detection Pipeline

A professional-grade computer vision pipeline combining YOLOv11 object detection with Segment Anything Model (SAM) for precise tool detection and segmentation.

## 🏗️ Project Structure

```
Final-proj/
├── config/                    # Configuration files
│   ├── __init__.py
│   ├── config.py             # Main configuration
│   ├── requirements.txt      # Production dependencies
│   └── requirements-minimal.txt
├── data/                     # Data and assets
│   ├── images/              # Input images
│   └── test_images/         # Test dataset
├── docs/                    # Documentation
│   ├── README.md            # Detailed documentation
│   └── INSTALL.md           # Installation guide
├── models/                  # Pre-trained models
│   ├── yolo/               # YOLO model weights
│   └── sam/                # SAM model weights
├── outputs/                 # Generated outputs
│   ├── visualizations/     # Output images
│   ├── metrics/            # JSON results
│   └── logs/               # Log files
├── scripts/                 # Executable scripts
│   ├── demos/              # Demo scripts
│   ├── tests/              # Test scripts
│   └── sam_mvp/            # Experimental code
└── src/                     # Core source code
    └── pipeline/            # Main pipeline modules
```

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r config/requirements.txt
   ```

2. **Configure Pipeline**:
   ```python
   from config import config
   config.set_image('tools')
   config.validate_config()
   ```

3. **Run Detection Pipeline**:
   ```bash
   python scripts/tests/configurable_smart_pipeline.py
   ```

## 🎯 Features

- **Multi-Model Ensemble**: YOLOv8n, YOLOv11s, YOLOv11m
- **Advanced Segmentation**: SAM integration for precise masks
- **Weighted Boxes Fusion**: Smart detection merging
- **Configurable Pipeline**: Easy image and model switching
- **Professional Architecture**: Clean, maintainable codebase

## 📊 Pipeline Capabilities

- Object detection with confidence scoring
- Instance segmentation with SAM refinement
- Ensemble learning with multiple YOLO models
- Automatic visualization generation
- Comprehensive metrics reporting

## 🔧 Configuration

The pipeline uses a centralized configuration system:

- **Images**: Add new test images in `data/test_images/`
- **Models**: Manage model weights in `models/yolo/` and `models/sam/`
- **Outputs**: Results saved to `outputs/visualizations/` and `outputs/metrics/`

## 📈 Results

All pipeline runs generate:
- Visualization images with masks and bounding boxes
- JSON metrics with confidence scores and object counts
- Detailed performance analytics

See `docs/README.md` for detailed documentation.