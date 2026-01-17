"""
AI Multi-Gesture & Hand-Face Interaction Detector

Real-time detection of:
- Facial expressions: Smirk, Wink, Surprise
- Hand-face interaction: "Thinking" gesture (finger touching mouth corner)

Uses MediaPipe Tasks API with FaceLandmarker and HandLandmarker.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
from pathlib import Path


# =============================================================================
# MODEL DOWNLOAD CONFIGURATION
# =============================================================================

MODEL_DIR = Path(__file__).parent / "models"
FACE_MODEL_PATH = MODEL_DIR / "face_landmarker.task"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def download_model(url: str, path: Path) -> None:
    """Download a model file if it doesn't exist."""
    if path.exists():
        print(f"✓ Model already exists: {path.name}")
        return
    
    print(f"⬇ Downloading {path.name}...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    print(f"✓ Downloaded: {path.name}")


def ensure_models() -> None:
    """Ensure all required models are downloaded."""
    download_model(FACE_MODEL_URL, FACE_MODEL_PATH)
    download_model(HAND_MODEL_URL, HAND_MODEL_PATH)


# =============================================================================
# MEME PLAYER CLASS
# =============================================================================

# Gesture to filename mapping
GESTURE_MEME_MAP = {
    "smirk": "smirk-meme.jpg",
    "wink": "monkey-wink.jpg",
    "shaq_t": "shaq-t.jpg",
    "patrick": "patrick-meme.jpg",
    "speed": "my-mom-is-kinda-homeless-ishowspeed.gif",
    "shock": "shock-guy-meme.jpg",
    "cut_it_out": "cut-it.gif",
    "shush": "dog-shush.jpg",
    "thinking": "monkey-thinking.jpg",
    "lebron": "lebron-james-lebron-screaming.gif",
    "giggle": "baby-meme-giggle.gif",
    "surprise": "surprise.jpg",
    "idle": "idle.jpg"
}

# Delay before switching memes (in seconds)
GESTURE_SWITCH_DELAY = 0.15


class MemePlayer:
    """Handles loading and playback of meme images and GIFs."""
    
    def __init__(self, images_folder: str = "./images"):
        self.images_folder = Path(images_folder)
        self.current_gesture = None
        self.media_type = None  # 'gif' or 'image'
        self.video_capture = None
        self.static_image = None
        self.target_height = 480  # Will be set based on webcam
        self.target_width = 640
        
        # Create images folder if it doesn't exist
        self.images_folder.mkdir(parents=True, exist_ok=True)
        
        # Create a black "idle" frame
        self.idle_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        cv2.putText(
            self.idle_frame, "Waiting for gesture...",
            (50, self.target_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2
        )
    
    def set_dimensions(self, height: int, width: int):
        """Set target dimensions based on webcam frame size."""
        self.target_height = height
        self.target_width = width
        self.idle_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            self.idle_frame, "Waiting for gesture...",
            (50, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2
        )
    
    def load_media(self, gesture_name: str) -> bool:
        """Load media file for the given gesture."""
        if gesture_name == self.current_gesture:
            return True  # Already loaded
        
        # Cleanup previous media
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.static_image = None
        
        self.current_gesture = gesture_name
        
        if gesture_name is None or gesture_name not in GESTURE_MEME_MAP:
            self.media_type = None
            return False
        
        filename = GESTURE_MEME_MAP[gesture_name]
        filepath = self.images_folder / filename
        
        if not filepath.exists():
            print(f"⚠ Meme not found: {filename}")
            self.media_type = None
            return False
        
        # Determine media type based on extension
        if filename.lower().endswith('.gif'):
            self.media_type = 'gif'
            self.video_capture = cv2.VideoCapture(str(filepath))
            if not self.video_capture.isOpened():
                print(f"⚠ Failed to open GIF: {filename}")
                self.media_type = None
                return False
        else:
            self.media_type = 'image'
            self.static_image = cv2.imread(str(filepath))
            if self.static_image is None:
                print(f"⚠ Failed to load image: {filename}")
                self.media_type = None
                return False
        
        return True
    
    def get_frame(self) -> np.ndarray:
        """Get the current frame (handles GIF looping)."""
        frame = None
        
        if self.media_type == 'gif' and self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if not ret:
                # Reset to beginning for looping
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_capture.read()
                if not ret:
                    frame = None
        elif self.media_type == 'image' and self.static_image is not None:
            frame = self.static_image.copy()
        
        if frame is None:
            return self.idle_frame.copy()
        
        # Resize to match target dimensions
        frame = self._resize_to_fit(frame)
        return frame
    
    def _resize_to_fit(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to fit target dimensions while maintaining aspect ratio."""
        h, w = frame.shape[:2]
        
        # Calculate scaling factor to fit height
        scale = self.target_height / h
        new_w = int(w * scale)
        new_h = self.target_height
        
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create canvas and center the image
        canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        # Center horizontally
        x_offset = max(0, (self.target_width - new_w) // 2)
        
        # Handle case where image is wider than target
        if new_w > self.target_width:
            # Crop from center
            crop_x = (new_w - self.target_width) // 2
            resized = resized[:, crop_x:crop_x + self.target_width]
            x_offset = 0
            new_w = self.target_width
        
        canvas[:, x_offset:x_offset + new_w] = resized
        return canvas
    
    def release(self):
        """Release any open resources."""
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None


# =============================================================================
# DETECTOR INITIALIZATION
# =============================================================================

def create_face_landmarker() -> vision.FaceLandmarker:
    """Create FaceLandmarker with blendshapes enabled."""
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)


def create_hand_landmarker() -> vision.HandLandmarker:
    """Create HandLandmarker for hand skeletal tracking."""
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2
    )
    return vision.HandLandmarker.create_from_options(options)


# =============================================================================
# GESTURE DETECTION LOGIC
# =============================================================================

def get_blendshape_value(blendshapes: list, name: str) -> float:
    """Extract a specific blendshape value by name."""
    for bs in blendshapes:
        if bs.category_name == name:
            return bs.score
    return 0.0


def detect_smirk(blendshapes: list) -> bool:
    """Detect smirk: asymmetry between left and right mouth smile."""
    smile_left = get_blendshape_value(blendshapes, "mouthSmileLeft")
    smile_right = get_blendshape_value(blendshapes, "mouthSmileRight")
    
    # Relaxed thresholds for easier detection
    diff = abs(smile_left - smile_right)
    return diff > 0.15 and max(smile_left, smile_right) > 0.25


def detect_wink(blendshapes: list) -> tuple[bool, str]:
    """Detect wink: one eye closed, the other open."""
    blink_left = get_blendshape_value(blendshapes, "eyeBlinkLeft")
    blink_right = get_blendshape_value(blendshapes, "eyeBlinkRight")
    
    # Left eye wink (left closed, right open)
    if blink_left > 0.5 and blink_right < 0.3:
        return True, "Left"
    # Right eye wink (right closed, left open)
    if blink_right > 0.5 and blink_left < 0.3:
        return True, "Right"
    
    return False, ""


def detect_surprise(blendshapes: list) -> bool:
    """Detect surprise: raised eyebrows."""
    brow_up_left = get_blendshape_value(blendshapes, "browOuterUpLeft")
    brow_up_right = get_blendshape_value(blendshapes, "browOuterUpRight")
    
    return brow_up_left > 0.4 or brow_up_right > 0.4


def detect_speed(blendshapes: list) -> bool:
    """
    Detect "Speed" expression: squinting while pursing lips.
    Mimics IShowSpeed's signature face.
    """
    # Condition 1: Lips (pucker or funnel)
    mouth_pucker = get_blendshape_value(blendshapes, "mouthPucker")
    mouth_funnel = get_blendshape_value(blendshapes, "mouthFunnel")
    lips_condition = mouth_pucker > 0.45 or mouth_funnel > 0.45
    
    # Condition 2: Eyes (both squinting)
    eye_squint_left = get_blendshape_value(blendshapes, "eyeSquintLeft")
    eye_squint_right = get_blendshape_value(blendshapes, "eyeSquintRight")
    eyes_condition = eye_squint_left > 0.4 and eye_squint_right > 0.4
    
    return lips_condition and eyes_condition


def detect_patrick_expression(blendshapes: list) -> bool:
    """
    Detect "Patrick" expression: jaw drop / shocked face (no hands).
    
    Requirements:
    - Mouth wide open (jawOpen > 0.4)
    - This is checked separately to ensure no hand gestures are active
    """
    jaw_open = get_blendshape_value(blendshapes, "jawOpen")
    return jaw_open > 0.4


def detect_thinking_gesture(
    face_landmarks: list,
    hand_landmarks: list,
    blendshapes: list,
    threshold: float = 0.05
) -> bool:
    """
    Detect "Thinking" gesture: index finger touching mouth corner with mouth OPEN.
    
    Face landmarks for mouth corners:
    - Left mouth corner: index 61
    - Right mouth corner: index 291
    
    Hand landmark for index finger tip: index 8
    """
    if not face_landmarks or not hand_landmarks:
        return False
    
    # Only trigger with exactly one hand (to avoid confusion with two-hand gestures)
    if len(hand_landmarks) != 1:
        return False
    
    # Require mouth to be OPEN for thinking gesture
    if blendshapes:
        jaw_open = get_blendshape_value(blendshapes, "jawOpen")
        if jaw_open < 0.1:  # Mouth must be open
            return False
    
    # Get mouth corner positions (normalized coordinates)
    left_mouth = face_landmarks[61]
    right_mouth = face_landmarks[291]
    mouth_corners = [
        (left_mouth.x, left_mouth.y),
        (right_mouth.x, right_mouth.y)
    ]
    
    # Check each hand's index finger tip
    for hand in hand_landmarks:
        index_tip = hand[8]  # Index finger tip
        finger_pos = (index_tip.x, index_tip.y)
        
        # Check distance to each mouth corner
        for corner in mouth_corners:
            distance = np.sqrt(
                (finger_pos[0] - corner[0])**2 +
                (finger_pos[1] - corner[1])**2
            )
            if distance < threshold:
                return True
    
    return False


def detect_shush_gesture(
    face_landmarks: list,
    hand_landmarks: list,
    blendshapes: list,
    threshold: float = 0.05
) -> bool:
    """
    Detect "Shush" gesture: index finger placed vertically over closed lips.
    Only triggers when face is turned sideways (to differentiate from thinking gesture).
    
    Face landmarks for lip center:
    - Upper lip bottom: index 13
    - Lower lip top: index 14
    
    Hand landmark for index finger tip: index 8
    """
    if not face_landmarks or not hand_landmarks:
        return False
    
    # Check if face is sideways (head turned)
    # Compare nose tip position to face center (midpoint of left/right face edges)
    nose_tip = face_landmarks[1]  # Nose tip
    left_face = face_landmarks[234]  # Left face edge
    right_face = face_landmarks[454]  # Right face edge
    
    face_center_x = (left_face.x + right_face.x) / 2
    nose_offset = abs(nose_tip.x - face_center_x)
    face_width = abs(right_face.x - left_face.x)
    
    # Face must be turned at least 15% off-center to count as "sideways"
    is_sideways = (nose_offset / face_width) > 0.15 if face_width > 0 else False
    
    if not is_sideways:
        return False
    
    # Check if mouth is closed using blendshapes
    if blendshapes:
        mouth_funnel = get_blendshape_value(blendshapes, "mouthFunnel")
        mouth_pucker = get_blendshape_value(blendshapes, "mouthPucker")
        jaw_open = get_blendshape_value(blendshapes, "jawOpen")
        
        mouth_closed = mouth_funnel < 0.1 and mouth_pucker < 0.1 and jaw_open < 0.1
    else:
        # Fallback: geometric check - lips close together
        upper_lip = face_landmarks[13]
        lower_lip = face_landmarks[14]
        lip_distance = abs(upper_lip.y - lower_lip.y)
        mouth_closed = lip_distance < 0.02
    
    if not mouth_closed:
        return False
    
    # Get lip center position (average of landmarks 13 and 14)
    upper_lip = face_landmarks[13]
    lower_lip = face_landmarks[14]
    lip_center = (
        (upper_lip.x + lower_lip.x) / 2,
        (upper_lip.y + lower_lip.y) / 2
    )
    
    # Check each hand's index finger tip
    for hand in hand_landmarks:
        index_tip = hand[8]  # Index finger tip
        finger_pos = (index_tip.x, index_tip.y)
        
        # Check distance to lip center
        distance = np.sqrt(
            (finger_pos[0] - lip_center[0])**2 +
            (finger_pos[1] - lip_center[1])**2
        )
        if distance < threshold:
            return True
    
    return False


def detect_shaq_t_gesture(hand_landmarks_list: list, proximity_threshold: float = 0.15) -> bool:
    """
    Detect "Shaq T" gesture: two hands forming a perpendicular 'T' shape (timeout).
    
    Requirements:
    - Two hands detected
    - One hand roughly horizontal (top of T)
    - One hand roughly vertical (stem of T)
    - Hands reasonably close together
    """
    if len(hand_landmarks_list) != 2:
        return False
    
    def get_hand_angle(hand):
        """Calculate angle of hand from wrist to middle finger tip."""
        wrist = hand[0]
        middle_tip = hand[12]  # Middle finger tip
        
        dx = middle_tip.x - wrist.x
        dy = middle_tip.y - wrist.y
        
        # Calculate angle in degrees (0° = right, 90° = down, -90° = up)
        angle = np.degrees(np.arctan2(dy, dx))
        return angle
    
    def get_palm_center(hand):
        """Get palm center (midpoint of wrist and middle finger base)."""
        wrist = hand[0]
        middle_base = hand[9]
        return ((wrist.x + middle_base.x) / 2, (wrist.y + middle_base.y) / 2)
    
    def get_fingertips_center(hand):
        """Get center of fingertips (index and middle)."""
        index_tip = hand[8]
        middle_tip = hand[12]
        return ((index_tip.x + middle_tip.x) / 2, (index_tip.y + middle_tip.y) / 2)
    
    hand1, hand2 = hand_landmarks_list[0], hand_landmarks_list[1]
    angle1 = get_hand_angle(hand1)
    angle2 = get_hand_angle(hand2)
    
    # Relaxed: Check for horizontal hand (angle ~0° or ~180°, within ±40°)
    def is_horizontal(angle):
        return abs(angle) < 40 or abs(angle) > 140
    
    # Relaxed: Check for vertical hand (angle ~90° or ~-90°, within ±40°)
    def is_vertical(angle):
        return 50 < abs(angle) < 130
    
    horizontal_hand = None
    vertical_hand = None
    
    # Determine which hand is horizontal and which is vertical
    if is_horizontal(angle1) and is_vertical(angle2):
        horizontal_hand = hand1
        vertical_hand = hand2
    elif is_horizontal(angle2) and is_vertical(angle1):
        horizontal_hand = hand2
        vertical_hand = hand1
    else:
        return False  # No valid T configuration
    
    # Check proximity: horizontal palm center should be close to vertical fingertips
    horiz_palm = get_palm_center(horizontal_hand)
    vert_tips = get_fingertips_center(vertical_hand)
    
    distance = np.sqrt(
        (horiz_palm[0] - vert_tips[0])**2 +
        (horiz_palm[1] - vert_tips[1])**2
    )
    
    return distance < proximity_threshold


def detect_shock_gesture(
    face_landmarks: list,
    hand_landmarks_list: list,
    blendshapes: list
) -> bool:
    """
    Detect "Shock" gesture: both hands on head with shocked expression.
    
    Requirements:
    - Two hands detected
    - Both wrists above nose level (smaller Y value)
    - Wrists within horizontal face bounds
    - Mouth open (jawOpen > 0.3)
    """
    if not face_landmarks or len(hand_landmarks_list) != 2:
        return False
    
    # Check for shocked expression (mouth open)
    if blendshapes:
        jaw_open = get_blendshape_value(blendshapes, "jawOpen")
        if jaw_open < 0.2:
            return False
    else:
        return False  # Need blendshapes to detect shock expression
    
    # Get face reference points
    nose_tip = face_landmarks[1]
    left_ear = face_landmarks[234]   # Left face edge
    right_ear = face_landmarks[454]  # Right face edge
    
    # Check both hands
    for hand in hand_landmarks_list:
        wrist = hand[0]
        
        # Wrist must be above nose (smaller Y = higher on screen)
        if wrist.y > nose_tip.y:
            return False
        
        # Wrist X must be within face horizontal bounds (with some margin)
        margin = 0.1  # Allow some margin outside face
        if wrist.x < (left_ear.x - margin) or wrist.x > (right_ear.x + margin):
            return False
    
    return True


def detect_giggle_gesture(
    face_landmarks: list,
    hand_landmarks_list: list,
    blendshapes: list,
    threshold: float = 0.07
) -> bool:
    """
    Detect "Giggle" gesture: hand covering mouth.
    
    Requirements:
    - Hand middle finger MCP close to lips center
    """
    if not face_landmarks or not hand_landmarks_list:
        return False
    
    # Get lips center position
    lips_center = face_landmarks[13]  # Upper lip center
    
    # Check if any hand's middle finger MCP is close to lips
    for hand in hand_landmarks_list:
        middle_mcp = hand[9]  # Middle finger MCP
        
        distance = np.sqrt(
            (middle_mcp.x - lips_center.x)**2 +
            (middle_mcp.y - lips_center.y)**2
        )
        
        if distance < threshold:
            return True
    
    return False


def detect_cut_it_out_gesture(
    face_landmarks: list,
    hand_landmarks_list: list
) -> bool:
    """
    Detect "Cut It Out" gesture: flat hand positioned horizontally at neck level.
    
    Requirements:
    - Only ONE hand detected (to differentiate from Shaq T)
    - Hand at neck level (below chin, above chest)
    - Hand is horizontal (wrist and fingertip at similar Y)
    """
    if not face_landmarks or not hand_landmarks_list:
        return False
    
    # Only trigger with exactly one hand (to avoid confusion with Shaq T)
    if len(hand_landmarks_list) != 1:
        return False
    
    # Get chin position
    chin = face_landmarks[152]
    
    for hand in hand_landmarks_list:
        wrist = hand[0]
        index_tip = hand[8]
        
        # Check if hand is at neck level (below chin but not too far)
        # Y increases downward, so neck is > chin_y
        at_neck_level = (index_tip.y > chin.y) and (index_tip.y < chin.y + 0.35)
        
        if not at_neck_level:
            continue
        
        # Check if hand is horizontal (wrist and index tip at similar Y)
        is_horizontal = abs(wrist.y - index_tip.y) < 0.05
        
        if is_horizontal:
            return True
    
    return False


def detect_lebron_scream_gesture(
    face_landmarks: list,
    hand_landmarks_list: list,
    blendshapes: list
) -> bool:
    """
    Detect "LeBron Scream" gesture: screaming with hands down/flexing.
    
    Requirements:
    - Mouth wide open (screaming)
    - Two hands detected
    - Both hands below shoulder level (not on head)
    """
    if not face_landmarks or len(hand_landmarks_list) != 2 or not blendshapes:
        return False
    
    # Check for screaming (mouth wide open)
    jaw_open = get_blendshape_value(blendshapes, "jawOpen")
    if jaw_open < 0.4:
        return False
    
    # Get chin position for reference
    chin = face_landmarks[152]
    nose_tip = face_landmarks[1]
    
    # Both hands must be below shoulder level (below chin + 0.3)
    # AND not on head (to differentiate from Shock gesture)
    for hand in hand_landmarks_list:
        wrist = hand[0]
        
        # Wrist must be below shoulder level
        if wrist.y < chin.y + 0.2:
            return False
        
        # Wrist must NOT be above nose (would be Shock gesture)
        if wrist.y < nose_tip.y:
            return False
    
    return True


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def draw_face_landmarks(frame: np.ndarray, face_landmarks: list) -> None:
    """Draw face mesh landmarks on the frame."""
    h, w = frame.shape[:2]
    
    for landmark in face_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def draw_hand_landmarks(frame: np.ndarray, hand_landmarks_list: list) -> None:
    """Draw hand skeleton on the frame."""
    h, w = frame.shape[:2]
    
    # Hand connections for skeleton
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17)              # Palm
    ]
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        color = colors[idx % len(colors)]
        
        # Draw connections
        for start_idx, end_idx in connections:
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            
            cv2.line(frame, start_point, end_point, color, 2)
        
        # Draw landmarks
        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, color, -1)


def draw_detection_text(frame: np.ndarray, detections: list[tuple[str, tuple]]) -> None:
    """Draw detection labels on the frame."""
    y_offset = 40
    
    for text, color in detections:
        # Draw background rectangle for better visibility
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame, (10, y_offset - text_h - 5), (20 + text_w, y_offset + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        y_offset += 45


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application loop."""
    print("=" * 60)
    print("AI Multi-Gesture & Hand-Face Interaction Detector")
    print("          with Parallel Meme Display")
    print("=" * 60)
    
    # Ensure models are downloaded
    ensure_models()
    
    # Initialize detectors
    print("\n⚙ Initializing detectors...")
    face_landmarker = create_face_landmarker()
    hand_landmarker = create_hand_landmarker()
    print("✓ Detectors ready!")
    
    # Initialize MemePlayer
    print("\n🎭 Initializing Meme Player...")
    meme_player = MemePlayer("./images")
    print("✓ Meme Player ready!")
    
    # Open webcam
    print("\n📷 Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam!")
        return
    
    # Get webcam dimensions and set meme player dimensions
    ret, test_frame = cap.read()
    if ret:
        h, w = test_frame.shape[:2]
        meme_player.set_dimensions(h, w)
    
    print("✓ Webcam opened successfully!")
    print("\n🎯 Detection active! Press 'q' to quit.\n")
    
    current_gesture = None
    pending_gesture = None  # Gesture waiting to be switched to
    gesture_change_time = 0  # Time when gesture change was requested
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame")
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run detections
        face_result = face_landmarker.detect(mp_image)
        hand_result = hand_landmarker.detect(mp_image)
        
        detections = []
        active_gesture = None  # Track the primary detected gesture
        
        # Process face detection
        if face_result.face_landmarks:
            face_landmarks = face_result.face_landmarks[0]
            # draw_face_landmarks(frame, face_landmarks)  # Disabled: no tracking dots
            
            # Check facial expressions (if blendshapes available)
            if face_result.face_blendshapes:
                blendshapes = face_result.face_blendshapes[0]
                
                # Check for smirk
                if detect_smirk(blendshapes):
                    detections.append(("Smirking! 😏", (0, 165, 255)))
                    if active_gesture is None:
                        active_gesture = "smirk"
                
                # Check for wink
                is_wink, wink_side = detect_wink(blendshapes)
                if is_wink:
                    detections.append((f"{wink_side} Wink! 😉", (255, 0, 255)))
                    if active_gesture is None:
                        active_gesture = "wink"
                
                # Check for Speed expression
                if detect_speed(blendshapes):
                    detections.append(("SPEED! ⚡", (0, 100, 255)))
                    if active_gesture is None:
                        active_gesture = "speed"
                
                # Check for Patrick expression (jaw drop, only when no hands detected)
                if not hand_result.hand_landmarks and detect_patrick_expression(blendshapes):
                    detections.append(("PATRICK! ⭐", (128, 128, 255)))
                    if active_gesture is None:
                        active_gesture = "patrick"
            
            # Check for hand-face interactions (requires both face and hand)
            if hand_result.hand_landmarks:
                hand_landmarks_list = hand_result.hand_landmarks
                # draw_hand_landmarks(frame, hand_landmarks_list)  # Disabled: no tracking dots
                
                # Check for Shaq T gesture (two hands forming T shape)
                if detect_shaq_t_gesture(hand_landmarks_list):
                    detections.append(("SHAQ T! ⏱️", (128, 0, 128)))
                    if active_gesture is None:
                        active_gesture = "shaq_t"
                
                # Check for Shock gesture (hands on head with shocked expression)
                if face_result.face_blendshapes and detect_shock_gesture(face_landmarks, hand_landmarks_list, blendshapes):
                    detections.append(("SHOCK! 😱", (0, 165, 255)))
                    if active_gesture is None:
                        active_gesture = "shock"
                
                # Check for LeBron Scream gesture (screaming with hands down)
                if face_result.face_blendshapes and detect_lebron_scream_gesture(face_landmarks, hand_landmarks_list, blendshapes):
                    detections.append(("LEBRON SCREAM! 👑", (0, 215, 255)))
                    if active_gesture is None:
                        active_gesture = "lebron"
                
                # Check for Giggle gesture (hand over mouth)
                if face_result.face_blendshapes and detect_giggle_gesture(face_landmarks, hand_landmarks_list, blendshapes):
                    detections.append(("GIGGLE! 🤭", (255, 105, 180)))
                    if active_gesture is None:
                        active_gesture = "giggle"
                
                # Check for Cut It Out gesture (flat hand at neck level)
                if detect_cut_it_out_gesture(face_landmarks, hand_landmarks_list):
                    detections.append(("CUT IT OUT! ✋", (0, 0, 139)))
                    if active_gesture is None:
                        active_gesture = "cut_it_out"
                
                if face_result.face_blendshapes:
                    # Check for shush gesture (finger over CLOSED lips, face sideways)
                    if detect_shush_gesture(face_landmarks, hand_landmarks_list, blendshapes):
                        detections.append(("SHUSH! 🤫", (255, 255, 255)))
                        if active_gesture is None:
                            active_gesture = "shush"
                    # Check for thinking gesture (finger at mouth corner, mouth OPEN)
                    elif detect_thinking_gesture(face_landmarks, hand_landmarks_list, blendshapes):
                        detections.append(("Hmm... Thinking 🤔", (255, 200, 0)))
                        if active_gesture is None:
                            active_gesture = "thinking"
        
        # Draw hand landmarks even if no face detected
        elif hand_result.hand_landmarks:
            pass  # draw_hand_landmarks disabled
        
        # Draw detection labels
        if detections:
            draw_detection_text(frame, detections)
        
        # Update meme player with active gesture (with delay)
        current_time = time.time()
        
        if active_gesture != current_gesture:
            # New gesture detected - start the delay timer
            if pending_gesture != active_gesture:
                pending_gesture = active_gesture
                gesture_change_time = current_time
        
        # Check if delay has passed and we should switch
        if pending_gesture is not None and pending_gesture != current_gesture:
            if current_time - gesture_change_time >= GESTURE_SWITCH_DELAY:
                meme_player.load_media(pending_gesture)
                current_gesture = pending_gesture
        
        # Get meme frame
        meme_frame = meme_player.get_frame()
        
        # Show info overlay on webcam frame
        cv2.putText(
            frame, "Press 'q' to quit",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )
        
        # Create split-screen display
        combined_frame = np.hstack((frame, meme_frame))
        
        # Display combined frame
        cv2.imshow("Meme Mirror", combined_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    meme_player.release()
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
