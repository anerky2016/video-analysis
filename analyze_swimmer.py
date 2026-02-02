#!/usr/bin/env python3
import argparse
from swimmer_pose_detector import SwimmerPoseDetector


def main():
    parser = argparse.ArgumentParser(description='Analyze swimmer posture from video')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Path to save output video (optional)')
    parser.add_argument('-c', '--complexity', type=int, default=2, choices=[0, 1, 2],
                       help='Model complexity (0=Lite, 1=Full, 2=Heavy). Default: 2')
    parser.add_argument('-d', '--detection-confidence', type=float, default=0.5,
                       help='Minimum detection confidence (0.0-1.0). Default: 0.5')
    parser.add_argument('-t', '--tracking-confidence', type=float, default=0.5,
                       help='Minimum tracking confidence (0.0-1.0). Default: 0.5')
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

    print("Initializing swimmer pose detector...")
    detector = SwimmerPoseDetector(
        model_complexity=args.complexity,
        min_detection_confidence=args.detection_confidence,
        min_tracking_confidence=args.tracking_confidence,
        track_head=track_parts['head'],
        track_arms=track_parts['arms'],
        track_body=track_parts['body'],
        track_legs=track_parts['legs']
    )

    print(f"Processing video: {args.input_video}")
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
    finally:
        detector.close()


if __name__ == "__main__":
    main()
