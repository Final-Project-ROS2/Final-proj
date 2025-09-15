# Unit Tests for Robotic Vision Pipeline

This directory contains comprehensive unit tests for each pipeline module.

## Test Structure

### 🧪 Main Test Files

- **`test_pipeline_units.py`** - Complete unit test suite for all modules
- **`quick_test.py`** - Fast individual module testing
- **`test_config.py`** - Test configuration and setup

### 📝 Test Modules

1. **Object Detection & Segmentation Test**
   - Input: RGB image (`src/tools.png`)
   - Output: Masks + bounding boxes
   - Validation: Visual alignment with real tools
   - Pass criteria: Detects ≥1 object with reasonable masks

2. **Object Classification Test**
   - Input: Cropped object regions from masks
   - Output: Class labels + confidence scores
   - Validation: Labels match tool categories
   - Pass criteria: Detects tool-related classes

3. **Grasp Synthesis Test**
   - Input: Object masks + dummy depth
   - Output: Grasp candidates with quality scores
   - Validation: Grasps align with object geometry
   - Pass criteria: Quality scores >0.3, collision-free

4. **Scene Understanding Test**
   - Input: Full image + object labels + boxes
   - Output: Spatial relations graph + scene description
   - Validation: Relations match visual reality
   - Pass criteria: Meaningful spatial relationships

## 🚀 Running Tests

### Option 1: Full Test Suite
```bash
cd tests
python test_pipeline_units.py
```

### Option 2: Quick Individual Tests
```bash
cd tests
python quick_test.py
```

## 📋 Prerequisites

1. **Test Image**: Place a test image with tools at `src/tools.png`
2. **SAM Checkpoint**: Download `sam_vit_b_01ec64.pth` to project root
3. **Dependencies**: Install pipeline requirements

## 📊 Test Outputs

- **Visual Results**: 
  - `tests/test_detection_output.jpg` - Detection visualization
  - `tests/test_grasp_output.jpg` - Grasp visualization
  
- **Data**: 
  - `tests/test_results.json` - Structured test results

## ✅ Pass/Fail Criteria

| Module | Pass Criteria | Fail Criteria |
|--------|---------------|---------------|
| Detection | ≥1 object detected, masks align with tools | No objects or poor segmentation |
| Classification | Tool-related labels (hammer, screwdriver, etc.) | Random/irrelevant labels |
| Grasp Synthesis | Quality >0.3, reasonable grasp poses | Low quality or floating grasps |
| Scene Understanding | Meaningful spatial relations | Nonsense descriptions |

## 🔧 Troubleshooting

- **No objects detected**: Check SAM checkpoint and test image quality
- **Classification errors**: Verify CLIP embeddings and tool classes
- **Grasp failures**: Check depth processing and collision detection
- **Scene errors**: Verify spatial relation calculations

## 📈 Future Enhancements

- [ ] Real depth camera integration
- [ ] Ground truth annotation tools
- [ ] Quantitative metrics (IoU, precision, recall)
- [ ] Benchmark dataset integration
- [ ] Automated pass/fail evaluation
