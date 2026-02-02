#!/usr/bin/env python3
import argparse
from swimmer_pose_yolo import SwimmerPoseYOLO


def main():
    parser = argparse.ArgumentParser(description='Analyze swimmer posture from video using YOLOv8-Pose')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Path to save output video (optional)')
    parser.add_argument('-m', '--model', type=str, default='x',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv8-Pose model size (n=nano, s=small, m=medium, l=large, x=xlarge). Default: x')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                       help='Device to run inference on (cpu, cuda, or mps for Apple Silicon). Default: cpu')
    parser.add_argument('-t', '--threshold', type=float, default=0.3,
                       help='Keypoint detection threshold (0.0-1.0). Default: 0.3')
    parser.add_argument('--no-head', action='store_true',
                       help='Disable head tracking')
    parser.add_argument('--no-arms', action='store_true',
                       help='Disable arms tracking')
    parser.add_argument('--no-body', action='store_true',
                       help='Disable body/torso tracking')
    parser.add_argument('--no-legs', action='store_true',
                       help='Disable legs tracking')

    args = parser.parse_args()

    track_parts = {
        'head': not args.no_head,
        'arms': not args.no_arms,
        'body': not args.no_body,
        'legs': not args.no_legs
    }

    tracking_info = [part for part, enabled in track_parts.items() if enabled]
    print(f"Tracking body parts: {', '.join(tracking_info)}")

    print("Initializing YOLOv8-Pose swimmer detector...")
    print(f"Model: yolov8{args.model}-pose")
    print(f"Device: {args.device}")

    detector = SwimmerPoseYOLO(
        model_size=args.model,
        device=args.device,
        track_head=track_parts['head'],
        track_arms=track_parts['arms'],
        track_body=track_parts['body'],
        track_legs=track_parts['legs'],
        detection_threshold=args.threshold
    )

    print(f"\nProcessing video: {args.input_video}")
    if args.output:
        print(f"Output will be saved to: {args.output}")

    print("\nControls:")
    print("  Press 'q' to quit early")
    print("\nStarting analysis...")

    try:
        detector.process_video(args.input_video, args.output)
        print("\nAnalysis complete!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
