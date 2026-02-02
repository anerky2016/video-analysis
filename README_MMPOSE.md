# Swimmer Posture Analysis with MMPose

This project uses MMPose for advanced body pose tracking and analysis of swimmer posture from video footage. MMPose provides better accuracy and more sophisticated pose estimation compared to MediaPipe.

## Features

- **Advanced pose estimation** using MMPose (state-of-the-art accuracy)
- **Customizable body part tracking** - Track only the parts you need: head, arms, body, or legs
- **Multiple model options** - Choose from various pre-trained models
- **GPU acceleration support** - Fast processing with CUDA
- Analysis of key swimming posture metrics:
  - Head position and tilt
  - Body alignment
  - Elbow angles (for stroke efficiency)
  - Knee angles (for kick technique)
  - Hip and shoulder level (for rotation and balance)
- Visual feedback overlay on video
- Support for video file processing with optional output saving

## Installation

### Step 1: Install PyTorch

Install PyTorch based on your system. Visit https://pytorch.org/get-started/locally/

For CPU:
```bash
pip install torch torchvision
```

For GPU (CUDA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Install MMPose and dependencies

```bash
pip install -U openmim
mim install mmcv
mim install mmpose
mim install mmdet
pip install opencv-python numpy
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import mmpose; print(mmpose.__version__)"
```

## Usage

### Basic Usage

```bash
python analyze_swimmer_mmpose.py input_video.mp4
```

### Save Output Video

```bash
python analyze_swimmer_mmpose.py input_video.mp4 -o output_analyzed.mp4
```

### Use GPU for Faster Processing

```bash
python analyze_swimmer_mmpose.py input_video.mp4 -d cuda
```

### Track Only Specific Body Parts

Track only arms and legs:
```bash
python analyze_swimmer_mmpose.py input_video.mp4 --no-head --no-body
```

Track only head and body:
```bash
python analyze_swimmer_mmpose.py input_video.mp4 --no-arms --no-legs
```

### Use Different Model

```bash
python analyze_swimmer_mmpose.py input_video.mp4 -m td-hm_hrnet-w32_8xb64-210e_coco-256x192
```

## Command-line Options

**Required:**
- `input_video`: Path to input video file

**Optional:**
- `-o, --output`: Path to save output video
- `-m, --model`: MMPose model config name (default: td-hm_hrnet-w48_8xb32-210e_coco-256x192)
- `-c, --checkpoint`: Path to model checkpoint (downloads automatically if not provided)
- `-d, --device`: Device to run on ('cpu' or 'cuda'). Default: cpu
- `-t, --threshold`: Keypoint detection threshold (0.0-1.0). Default: 0.3

**Body Part Tracking (all enabled by default):**
- `--no-head`: Disable head tracking
- `--no-arms`: Disable arms tracking
- `--no-body`: Disable body/torso tracking
- `--no-legs`: Disable legs tracking

## Available Models

MMPose supports various pre-trained models. Popular options:

1. **HRNet (High Resolution)** - Best accuracy
   - `td-hm_hrnet-w48_8xb32-210e_coco-256x192` (default, most accurate)
   - `td-hm_hrnet-w32_8xb64-210e_coco-256x192` (balanced)

2. **ResNet** - Faster processing
   - `td-hm_res50_8xb64-210e_coco-256x192`
   - `td-hm_res101_8xb64-210e_coco-256x192`

3. **MobileNet** - Lightweight for real-time
   - `td-hm_mobilenetv2_8xb64-210e_coco-256x192`

## Posture Metrics

The analyzer tracks the following metrics (depending on which body parts are enabled):

1. **Head Tilt**: Ear level difference (minimal is better, <15px ideal)
2. **Body Alignment**: Overall straightness of the body (ideal: 160-180°)
3. **Elbow Angles**: Left and right elbow bend during stroke (optimal catch: 90-120°)
4. **Knee Angles**: Leg straightness for kick efficiency (flutter kick: >160°)
5. **Hip Angles**: Hip flexion for body position
6. **Shoulder Level**: Difference in shoulder height (minimal is better)
7. **Hip Level**: Difference in hip height (minimal is better)

## Python API

### Using the detector in your own code:

```python
from swimmer_pose_mmpose import SwimmerPoseMMPose

# Initialize detector
detector = SwimmerPoseMMPose(
    model_config='td-hm_hrnet-w48_8xb32-210e_coco-256x192',
    device='cuda',  # or 'cpu'
    track_head=True,
    track_arms=True,
    track_body=True,
    track_legs=True,
    detection_threshold=0.3
)

# Process video
detector.process_video('swimmer.mp4', 'output.mp4')
```

### Track only specific parts:

```python
detector = SwimmerPoseMMPose(
    track_head=False,
    track_arms=True,
    track_body=False,
    track_legs=True,
    device='cpu'
)
```

### Process individual frames:

```python
import cv2
from swimmer_pose_mmpose import SwimmerPoseMMPose

detector = SwimmerPoseMMPose(device='cpu')
frame = cv2.imread('swimmer_frame.jpg')

processed_frame, analysis = detector.process_frame(frame)

print(analysis['metrics'])
print(analysis['feedback'])

cv2.imwrite('analyzed_frame.jpg', processed_frame)
```

## Performance Tips

1. **Use GPU**: Add `-d cuda` for 5-10x faster processing
2. **Lower resolution models**: Use models with 128x96 input for faster processing
3. **Adjust threshold**: Increase `-t` value (e.g., 0.5) to reduce false detections
4. **Track only needed parts**: Use `--no-*` flags to disable unnecessary tracking

## Comparison: MMPose vs MediaPipe

| Feature | MMPose | MediaPipe |
|---------|--------|-----------|
| Accuracy | Higher (COCO dataset) | Good |
| Model Options | Multiple (HRNet, ResNet, etc.) | Fixed |
| GPU Support | Yes | Limited |
| Customization | Extensive | Limited |
| Speed (CPU) | Moderate | Fast |
| Speed (GPU) | Very Fast | N/A |
| Installation | More complex | Simple |

## Troubleshooting

### CUDA Out of Memory
```bash
# Use smaller model or CPU
python analyze_swimmer_mmpose.py video.mp4 -d cpu
```

### Model Download Issues
```bash
# Download models manually using mim
mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192
```

### Slow Performance
```bash
# Use GPU or lighter model
python analyze_swimmer_mmpose.py video.mp4 -d cuda -m td-hm_mobilenetv2_8xb64-210e_coco-256x192
```

## Requirements

- Python 3.7+
- PyTorch 2.0+
- MMPose 1.3+
- MMCV 2.0+
- MMDetection 3.0+
- OpenCV
- NumPy

## References

- MMPose: https://github.com/open-mmlab/mmpose
- MMCV: https://github.com/open-mmlab/mmcv
- PyTorch: https://pytorch.org/

## License

This project is provided as-is for educational and analysis purposes.
