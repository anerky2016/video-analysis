# Swimmer Posture Analysis with MediaPipe

This project uses MediaPipe Pose to analyze swimmer posture from video footage, providing real-time feedback on body alignment, arm angles, leg position, and more.

## Features

- Real-time pose detection and tracking
- **Customizable body part tracking** - Track only the parts you need: head, arms, body, or legs
- Analysis of key swimming posture metrics:
  - Head position and tilt
  - Body alignment
  - Elbow angles (for stroke efficiency)
  - Knee angles (for kick technique)
  - Hip and shoulder level (for rotation and balance)
- Visual feedback overlay on video
- Support for video file processing with optional output saving

## Installation

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python analyze_swimmer.py input_video.mp4
```

Save the analyzed video:

```bash
python analyze_swimmer.py input_video.mp4 -o output_analyzed.mp4
```

Adjust detection parameters:

```bash
python analyze_swimmer.py input_video.mp4 -c 2 -d 0.7 -t 0.7
```

Track only specific body parts (arms and legs only):

```bash
python analyze_swimmer.py input_video.mp4 --no-head --no-body
```

Track only head and body:

```bash
python analyze_swimmer.py input_video.mp4 --no-arms --no-legs
```

### Command-line Options

**Required:**
- `input_video`: Path to input video file

**Optional:**
- `-o, --output`: Path to save output video
- `-c, --complexity`: Model complexity (0=Lite, 1=Full, 2=Heavy). Default: 2
- `-d, --detection-confidence`: Minimum detection confidence (0.0-1.0). Default: 0.5
- `-t, --tracking-confidence`: Minimum tracking confidence (0.0-1.0). Default: 0.5

**Body Part Tracking (all enabled by default):**
- `--no-head`: Disable head tracking
- `--no-arms`: Disable arms tracking
- `--no-body`: Disable body/torso tracking
- `--no-legs`: Disable legs tracking

### Controls

- Press `q` to quit the video playback early

## Posture Metrics

The analyzer tracks the following metrics (depending on which body parts are enabled):

1. **Head Tilt**: Ear level difference (minimal is better, <15px ideal)
2. **Body Alignment**: Overall straightness of the body (ideal: 160-180°)
3. **Elbow Angles**: Left and right elbow bend during stroke (optimal catch: 90-120°)
4. **Knee Angles**: Leg straightness for kick efficiency (flutter kick: >160°)
5. **Hip Angles**: Hip flexion for body position
6. **Shoulder Level**: Difference in shoulder height (minimal is better)
7. **Hip Level**: Difference in hip height (minimal is better)

## Example Code

Using the detector in your own code:

```python
from swimmer_pose_detector import SwimmerPoseDetector

detector = SwimmerPoseDetector(
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    track_head=True,
    track_arms=True,
    track_body=True,
    track_legs=True
)

detector.process_video('swimmer.mp4', 'output.mp4')

detector.close()
```

Track only arms and legs:

```python
detector = SwimmerPoseDetector(
    track_head=False,
    track_arms=True,
    track_body=False,
    track_legs=True
)
```

Process individual frames:

```python
import cv2
from swimmer_pose_detector import SwimmerPoseDetector

detector = SwimmerPoseDetector()
frame = cv2.imread('swimmer_frame.jpg')

processed_frame, analysis = detector.process_frame(frame)

print(analysis['metrics'])
print(analysis['feedback'])

cv2.imwrite('analyzed_frame.jpg', processed_frame)
detector.close()
```

## Tips for Best Results

1. Use high-quality video with good lighting
2. Ensure the swimmer is clearly visible in the frame
3. Side views work best for analyzing body alignment and stroke mechanics
4. For underwater footage, ensure water clarity is good
5. Higher model complexity (2) provides better accuracy but slower processing

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

## License

This project is provided as-is for educational and analysis purposes.
