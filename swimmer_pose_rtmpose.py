import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
import torch
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.structures import merge_data_samples


class SwimmerPoseRTMPose:
    def __init__(self,
                 model_name: str = 'rtmpose-l',
                 device: str = 'cpu',
                 track_head: bool = True,
                 track_arms: bool = True,
                 track_body: bool = True,
                 track_legs: bool = True,
                 detection_threshold: float = 0.3):
        """
        Initialize RTMPose-based swimmer pose detector.

        Args:
            model_name: 'rtmpose-l', 'rtmpose-m', or 'rtmpose-s'
            device: 'cpu' or 'cuda'
            track_head: Enable head tracking
            track_arms: Enable arms tracking
            track_body: Enable body/torso tracking
            track_legs: Enable legs tracking
            detection_threshold: Confidence threshold for keypoint detection
        """
        self.device = device
        self.track_head = track_head
        self.track_arms = track_arms
        self.track_body = track_body
        self.track_legs = track_legs
        self.detection_threshold = detection_threshold

        # RTMPose model configurations
        model_configs = {
            'rtmpose-l': {
                'config': 'rtmpose-l_8xb256-420e_coco-256x192.py',
                'checkpoint': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth'
            },
            'rtmpose-m': {
                'config': 'rtmpose-m_8xb256-420e_coco-256x192.py',
                'checkpoint': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'
            },
            'rtmpose-s': {
                'config': 'rtmpose-s_8xb256-420e_coco-256x192.py',
                'checkpoint': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth'
            }
        }

        if model_name not in model_configs:
            raise ValueError(f"Model {model_name} not supported. Choose from: {list(model_configs.keys())}")

        config_file = model_configs[model_name]['config']
        checkpoint_file = model_configs[model_name]['checkpoint']

        print(f"Loading RTMPose model: {model_name}")
        print(f"Config: {config_file}")
        print(f"Checkpoint: {checkpoint_file}")

        try:
            self.model = init_pose_estimator(
                config_file,
                checkpoint_file,
                device=device,
                cfg_options=dict(
                    model=dict(test_cfg=dict(output_heatmaps=False))
                )
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting to download and load model...")
            self.model = init_pose_estimator(
                config_file,
                checkpoint_file,
                device=device
            )

        self._setup_keypoint_groups()

    def _setup_keypoint_groups(self):
        """Define COCO keypoint indices for each body part."""
        # COCO keypoint format (17 keypoints):
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
        # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
        # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

        self.head_indices = [0, 1, 2, 3, 4]
        self.arm_indices = [5, 6, 7, 8, 9, 10]
        self.body_indices = [5, 6, 11, 12]
        self.leg_indices = [11, 12, 13, 14, 15, 16]

        # Skeleton connections
        self.skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # legs
            [5, 11], [6, 12],  # torso
            [5, 6],  # shoulders
            [5, 7], [7, 9], [6, 8], [8, 10],  # arms
            [0, 1], [0, 2], [1, 3], [2, 4], [5, 0], [6, 0]  # head
        ]

        self._filter_skeleton_connections()

    def _filter_skeleton_connections(self):
        """Filter skeleton connections based on enabled body parts."""
        filtered = []

        for connection in self.skeleton:
            start_idx, end_idx = connection
            include = False

            if self.track_head and (start_idx in self.head_indices or end_idx in self.head_indices):
                include = True
            if self.track_arms and (start_idx in self.arm_indices or end_idx in self.arm_indices):
                include = True
            if self.track_body and (start_idx in self.body_indices or end_idx in self.body_indices):
                include = True
            if self.track_legs and (start_idx in self.leg_indices or end_idx in self.leg_indices):
                include = True

            if include:
                filtered.append(connection)

        self.filtered_skeleton = filtered

    def calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculate angle between three points."""
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def analyze_swimmer_posture(self, keypoints: np.ndarray,
                               keypoint_scores: np.ndarray) -> Dict[str, float]:
        """Analyze swimmer posture from keypoints."""
        metrics = {}

        try:
            # Head analysis
            if self.track_head:
                left_ear = keypoints[3]
                right_ear = keypoints[4]
                left_ear_score = keypoint_scores[3]
                right_ear_score = keypoint_scores[4]

                if left_ear_score > self.detection_threshold and right_ear_score > self.detection_threshold:
                    head_tilt = abs(left_ear[1] - right_ear[1])
                    metrics['head_tilt'] = head_tilt

            # Body and shoulder analysis
            if self.track_body or self.track_arms:
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]
                left_shoulder_score = keypoint_scores[5]
                right_shoulder_score = keypoint_scores[6]

                if left_shoulder_score > self.detection_threshold and right_shoulder_score > self.detection_threshold:
                    if self.track_body:
                        shoulder_level_diff = abs(left_shoulder[1] - right_shoulder[1])
                        metrics['shoulder_level_diff'] = shoulder_level_diff

            # Arm analysis
            if self.track_arms:
                left_elbow = keypoints[7]
                left_wrist = keypoints[9]
                right_elbow = keypoints[8]
                right_wrist = keypoints[10]

                if (keypoint_scores[5] > self.detection_threshold and
                    keypoint_scores[7] > self.detection_threshold and
                    keypoint_scores[9] > self.detection_threshold):
                    left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                    metrics['left_elbow_angle'] = left_elbow_angle

                if (keypoint_scores[6] > self.detection_threshold and
                    keypoint_scores[8] > self.detection_threshold and
                    keypoint_scores[10] > self.detection_threshold):
                    right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
                    metrics['right_elbow_angle'] = right_elbow_angle

            # Hip and body analysis
            if self.track_body or self.track_legs:
                left_hip = keypoints[11]
                right_hip = keypoints[12]
                left_hip_score = keypoint_scores[11]
                right_hip_score = keypoint_scores[12]

                if left_hip_score > self.detection_threshold and right_hip_score > self.detection_threshold:
                    if self.track_body:
                        hip_level_diff = abs(left_hip[1] - right_hip[1])
                        metrics['hip_level_diff'] = hip_level_diff

            # Leg analysis
            if self.track_legs:
                left_knee = keypoints[13]
                left_ankle = keypoints[15]
                right_knee = keypoints[14]
                right_ankle = keypoints[16]

                if (keypoint_scores[11] > self.detection_threshold and
                    keypoint_scores[13] > self.detection_threshold and
                    keypoint_scores[15] > self.detection_threshold):
                    left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
                    metrics['left_knee_angle'] = left_knee_angle

                if (keypoint_scores[12] > self.detection_threshold and
                    keypoint_scores[14] > self.detection_threshold and
                    keypoint_scores[16] > self.detection_threshold):
                    right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
                    metrics['right_knee_angle'] = right_knee_angle

            # Body alignment
            if (self.track_body and self.track_legs and self.track_arms and
                keypoint_scores[5] > self.detection_threshold and
                keypoint_scores[11] > self.detection_threshold and
                keypoint_scores[15] > self.detection_threshold):
                body_alignment = self.calculate_angle(left_shoulder, left_hip, left_ankle)
                metrics['body_alignment'] = body_alignment

        except Exception as e:
            print(f"Error analyzing posture: {e}")

        return metrics

    def evaluate_posture(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Generate feedback based on posture metrics."""
        feedback = {}

        if 'head_tilt' in metrics:
            if metrics['head_tilt'] < 15:
                feedback['head_position'] = "Good - Head is level"
            else:
                feedback['head_position'] = "Keep head level - Minimize head tilt"

        if 'body_alignment' in metrics:
            if 160 <= metrics['body_alignment'] <= 180:
                feedback['body_alignment'] = "Good - Body is well aligned"
            else:
                feedback['body_alignment'] = "Needs improvement - Body should be straighter"

        if 'left_elbow_angle' in metrics and 'right_elbow_angle' in metrics:
            avg_elbow = (metrics['left_elbow_angle'] + metrics['right_elbow_angle']) / 2
            if 90 <= avg_elbow <= 120:
                feedback['elbow_position'] = "Good - Elbow angle is optimal for catch phase"
            elif avg_elbow > 150:
                feedback['elbow_position'] = "Straighten less - Bend elbows more during catch"
            else:
                feedback['elbow_position'] = "Elbow angle tracked"

        if 'left_knee_angle' in metrics and 'right_knee_angle' in metrics:
            avg_knee = (metrics['left_knee_angle'] + metrics['right_knee_angle']) / 2
            if avg_knee > 160:
                feedback['kick'] = "Good - Legs are straight for efficient kick"
            else:
                feedback['kick'] = "Straighten legs - Minimize knee bend for flutter kick"

        if 'shoulder_level_diff' in metrics:
            if metrics['shoulder_level_diff'] < 20:
                feedback['shoulder_level'] = "Good - Shoulders are level"
            else:
                feedback['shoulder_level'] = "Check rotation - Shoulders showing tilt"

        if 'hip_level_diff' in metrics:
            if metrics['hip_level_diff'] < 20:
                feedback['hip_level'] = "Good - Hips are level"
            else:
                feedback['hip_level'] = "Check hip position - Hips showing tilt"

        return feedback

    def draw_skeleton(self, image: np.ndarray, keypoints: np.ndarray,
                     keypoint_scores: np.ndarray) -> np.ndarray:
        """Draw skeleton on image."""
        img = image.copy()

        # Draw connections
        for connection in self.filtered_skeleton:
            start_idx, end_idx = connection
            if (keypoint_scores[start_idx] > self.detection_threshold and
                keypoint_scores[end_idx] > self.detection_threshold):
                start_point = tuple(keypoints[start_idx].astype(int))
                end_point = tuple(keypoints[end_idx].astype(int))
                cv2.line(img, start_point, end_point, (0, 255, 0), 2)

        # Draw keypoints
        enabled_indices = set()
        if self.track_head:
            enabled_indices.update(self.head_indices)
        if self.track_arms:
            enabled_indices.update(self.arm_indices)
        if self.track_body:
            enabled_indices.update(self.body_indices)
        if self.track_legs:
            enabled_indices.update(self.leg_indices)

        for idx in enabled_indices:
            if keypoint_scores[idx] > self.detection_threshold:
                point = tuple(keypoints[idx].astype(int))
                cv2.circle(img, point, 4, (0, 0, 255), -1)

        return img

    def draw_posture_info(self, image: np.ndarray, metrics: Dict[str, float],
                         feedback: Dict[str, str]) -> np.ndarray:
        """Draw posture metrics and feedback on image."""
        y_offset = 30

        cv2.putText(image, "Swimmer Posture Analysis (RTMPose)", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30

        for key, value in metrics.items():
            text = f"{key.replace('_', ' ').title()}: {value:.1f}"
            cv2.putText(image, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

        y_offset += 10
        cv2.putText(image, "Feedback:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25

        for key, value in feedback.items():
            color = (0, 255, 0) if "Good" in value else (0, 165, 255)
            cv2.putText(image, value, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25

        return image

    def process_frame(self, frame: np.ndarray,
                     draw_skeleton_flag: bool = True,
                     draw_info: bool = True) -> Tuple[np.ndarray, Optional[Dict]]:
        """Process a single frame."""
        image = frame.copy()

        # Run inference
        results = inference_topdown(self.model, image)

        metrics = None
        feedback = None

        if len(results) > 0:
            # Get the first person detected
            result = results[0]
            keypoints = result.pred_instances.keypoints[0]  # (17, 2)
            keypoint_scores = result.pred_instances.keypoint_scores[0]  # (17,)

            # Draw skeleton
            if draw_skeleton_flag:
                image = self.draw_skeleton(image, keypoints, keypoint_scores)

            # Analyze posture
            metrics = self.analyze_swimmer_posture(keypoints, keypoint_scores)
            feedback = self.evaluate_posture(metrics)

            # Draw info
            if draw_info:
                image = self.draw_posture_info(image, metrics, feedback)

        return image, {'metrics': metrics, 'feedback': feedback}

    def process_video(self, video_path: str, output_path: Optional[str] = None):
        """Process a video file."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        print("Processing video...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, analysis = self.process_frame(frame)

            if writer:
                writer.write(processed_frame)

            cv2.imshow('Swimmer Pose Analysis (RTMPose)', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        print(f"Video processing complete. Total frames: {frame_count}")
