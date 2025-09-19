"""
YOLOv11 + SAM Object Detection and Segmentation Pipeline
A comprehensive computer vision pipeline for tool detection and segmentation.
"""

__version__ = "1.0.0"
__author__ = "Final Project Team"

# Package exports
from .pipeline.object_detection_segmentation import ObjectDetectionSegmentation

__all__ = [
    "ObjectDetectionSegmentation"
]