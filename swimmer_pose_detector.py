import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List, Tuple


class SwimmerPoseDetector:
    def __init__(self,
                 model_complexity: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 track_head: bool = True,
                 track_arms: bool = True,
                 track_body: bool = True,
                 track_legs: bool = True):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.track_head = track_head
        self.track_arms = track_arms
        self.track_body = track_body
        self.track_legs = track_legs

        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self._setup_body_part_connections()

    def _setup_body_part_connections(self):
        """Define which landmarks and connections belong to each body part."""
        self.head_landmarks = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_EYE,
            self.mp_pose.PoseLandmark.RIGHT_EYE,
            self.mp_pose.PoseLandmark.LEFT_EAR,
            self.mp_pose.PoseLandmark.RIGHT_EAR,
            self.mp_pose.PoseLandmark.MOUTH_LEFT,
            self.mp_pose.PoseLandmark.MOUTH_RIGHT
        ]

        self.arm_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_PINKY,
            self.mp_pose.PoseLandmark.RIGHT_PINKY,
            self.mp_pose.PoseLandmark.LEFT_INDEX,
            self.mp_pose.PoseLandmark.RIGHT_INDEX,
            self.mp_pose.PoseLandmark.LEFT_THUMB,
            self.mp_pose.PoseLandmark.RIGHT_THUMB
        ]

        self.body_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP
        ]

        self.leg_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.LEFT_HEEL,
            self.mp_pose.PoseLandmark.RIGHT_HEEL,
            self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        ]

        all_connections = list(self.mp_pose.POSE_CONNECTIONS)
        self.filtered_connections = []

        for connection in all_connections:
            start_idx, end_idx = connection
            include = False

            if self.track_head and (start_idx in [l.value for l in self.head_landmarks] or
                                   end_idx in [l.value for l in self.head_landmarks]):
                include = True
            if self.track_arms and (start_idx in [l.value for l in self.arm_landmarks] or
                                   end_idx in [l.value for l in self.arm_landmarks]):
                include = True
            if self.track_body and (start_idx in [l.value for l in self.body_landmarks] or
                                   end_idx in [l.value for l in self.body_landmarks]):
                include = True
            if self.track_legs and (start_idx in [l.value for l in self.leg_landmarks] or
                                   end_idx in [l.value for l in self.leg_landmarks]):
                include = True

            if include:
                self.filtered_connections.append(connection)

    def calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def get_landmark_coords(self, landmarks, landmark_name: str,
                           image_width: int, image_height: int) -> np.ndarray:
        landmark = landmarks[self.mp_pose.PoseLandmark[landmark_name].value]
        return np.array([landmark.x * image_width, landmark.y * image_height])

    def analyze_swimmer_posture(self, landmarks, image_width: int,
                                image_height: int) -> Dict[str, float]:
        metrics = {}

        try:
            if self.track_head:
                nose = self.get_landmark_coords(landmarks, 'NOSE', image_width, image_height)
                left_ear = self.get_landmark_coords(landmarks, 'LEFT_EAR', image_width, image_height)
                right_ear = self.get_landmark_coords(landmarks, 'RIGHT_EAR', image_width, image_height)

                head_tilt = abs(left_ear[1] - right_ear[1])
                metrics['head_tilt'] = head_tilt

            if self.track_arms or self.track_body:
                left_shoulder = self.get_landmark_coords(landmarks, 'LEFT_SHOULDER',
                                                        image_width, image_height)
                right_shoulder = self.get_landmark_coords(landmarks, 'RIGHT_SHOULDER',
                                                         image_width, image_height)

                if self.track_body:
                    shoulder_level_diff = abs(left_shoulder[1] - right_shoulder[1])
                    metrics['shoulder_level_diff'] = shoulder_level_diff

            if self.track_arms:
                left_elbow = self.get_landmark_coords(landmarks, 'LEFT_ELBOW',
                                                     image_width, image_height)
                left_wrist = self.get_landmark_coords(landmarks, 'LEFT_WRIST',
                                                     image_width, image_height)
                right_elbow = self.get_landmark_coords(landmarks, 'RIGHT_ELBOW',
                                                      image_width, image_height)
                right_wrist = self.get_landmark_coords(landmarks, 'RIGHT_WRIST',
                                                      image_width, image_height)

                left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

                metrics['left_elbow_angle'] = left_elbow_angle
                metrics['right_elbow_angle'] = right_elbow_angle

            if self.track_body or self.track_legs:
                left_hip = self.get_landmark_coords(landmarks, 'LEFT_HIP',
                                                   image_width, image_height)
                right_hip = self.get_landmark_coords(landmarks, 'RIGHT_HIP',
                                                    image_width, image_height)

                if self.track_body:
                    hip_level_diff = abs(left_hip[1] - right_hip[1])
                    metrics['hip_level_diff'] = hip_level_diff

                    if self.track_arms:
                        left_hip_angle = self.calculate_angle(left_shoulder, left_hip,
                                                             self.get_landmark_coords(landmarks, 'LEFT_KNEE',
                                                                                    image_width, image_height))
                        right_hip_angle = self.calculate_angle(right_shoulder, right_hip,
                                                              self.get_landmark_coords(landmarks, 'RIGHT_KNEE',
                                                                                     image_width, image_height))
                        metrics['left_hip_angle'] = left_hip_angle
                        metrics['right_hip_angle'] = right_hip_angle

            if self.track_legs:
                left_knee = self.get_landmark_coords(landmarks, 'LEFT_KNEE',
                                                    image_width, image_height)
                left_ankle = self.get_landmark_coords(landmarks, 'LEFT_ANKLE',
                                                     image_width, image_height)
                right_knee = self.get_landmark_coords(landmarks, 'RIGHT_KNEE',
                                                     image_width, image_height)
                right_ankle = self.get_landmark_coords(landmarks, 'RIGHT_ANKLE',
                                                      image_width, image_height)

                left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

                metrics['left_knee_angle'] = left_knee_angle
                metrics['right_knee_angle'] = right_knee_angle

            if self.track_body and self.track_legs and self.track_arms:
                body_alignment = self.calculate_angle(left_shoulder, left_hip, left_ankle)
                metrics['body_alignment'] = body_alignment

            return metrics
        except Exception as e:
            print(f"Error analyzing posture: {e}")
            return metrics

    def evaluate_posture(self, metrics: Dict[str, float]) -> Dict[str, str]:
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

    def draw_posture_info(self, image: np.ndarray, metrics: Dict[str, float],
                         feedback: Dict[str, str]) -> np.ndarray:
        """Draw posture metrics and feedback on image - bottom right corner."""
        img_height, img_width = image.shape[:2]

        # Calculate total height needed for all text
        line_height = 35
        num_metrics = len(metrics)
        num_feedback = len(feedback)
        total_lines = 1 + num_metrics + 1 + num_feedback  # title + metrics + "Feedback:" + feedback items

        # Start from bottom and work up
        y_start = img_height - (total_lines * line_height) - 20
        y_offset = y_start

        # Title
        title = "Swimmer Posture Analysis"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)[0]
        x_pos = img_width - title_size[0] - 20
        cv2.putText(image, title, (x_pos, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        y_offset += line_height

        # Metrics
        for key, value in metrics.items():
            text = f"{key.replace('_', ' ').title()}: {value:.1f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            x_pos = img_width - text_size[0] - 20
            cv2.putText(image, text, (x_pos, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height

        # Feedback header
        y_offset += 10
        feedback_title = "Feedback:"
        feedback_size = cv2.getTextSize(feedback_title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 3)[0]
        x_pos = img_width - feedback_size[0] - 20
        cv2.putText(image, feedback_title, (x_pos, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)
        y_offset += line_height

        # Feedback items
        for key, value in feedback.items():
            color = (0, 255, 0) if "Good" in value else (0, 165, 255)
            text_size = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            x_pos = img_width - text_size[0] - 20
            cv2.putText(image, value, (x_pos, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += line_height

        return image

    def process_frame(self, frame: np.ndarray,
                     draw_landmarks: bool = True,
                     draw_info: bool = True) -> Tuple[np.ndarray, Optional[Dict]]:
        image = frame.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = self.pose.process(image_rgb)

        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        metrics = None
        feedback = None

        if results.pose_landmarks:
            if draw_landmarks:
                # Custom drawing specs for bold lines and larger keypoints
                landmark_spec = self.mp_drawing.DrawingSpec(
                    color=(0, 0, 255),      # Red keypoints
                    thickness=3,             # Thicker keypoint circles
                    circle_radius=5          # Larger keypoint circles
                )
                connection_spec = self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0),      # Green connections
                    thickness=5              # Bold connection lines
                )

                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    frozenset(self.filtered_connections),
                    landmark_drawing_spec=landmark_spec,
                    connection_drawing_spec=connection_spec
                )

            h, w = image.shape[:2]
            metrics = self.analyze_swimmer_posture(results.pose_landmarks.landmark, w, h)
            feedback = self.evaluate_posture(metrics)

            if draw_info:
                image = self.draw_posture_info(image, metrics, feedback)

        return image, {'metrics': metrics, 'feedback': feedback}

    def process_video(self, video_path: str, output_path: Optional[str] = None):
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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, analysis = self.process_frame(frame)

            if writer:
                writer.write(processed_frame)

            cv2.imshow('Swimmer Pose Analysis', processed_frame)

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

    def close(self):
        self.pose.close()
