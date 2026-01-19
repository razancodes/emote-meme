# AI Meme Emote Detector

Real-time gesture detection using MediaPipe with parallel meme display.

# Demo:

![IMG_0909 (1)](https://github.com/user-attachments/assets/016b0a84-6fc3-4b99-8523-998320e2bf44)

## Features

- **12 Gesture Detection**: Smirk, Wink, Speed, Patrick, Thinking, Shush, Giggle, Cut It Out, Shock, LeBron Scream, Shaq T, Surprise
- **Split-Screen Display**:  Webcam feed (left) + corresponding meme (right)
- **GIF Support**: Animated memes loop automatically

## Requirements

- Python 3.11+
- Webcam
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/razancodes/emote-meme.git
cd emote-meme
```

### 2. Create a Virtual Environment (Recommended)

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Meme Images

The `./images/` folder should already exist with default memes. If not, create it:

```bash
mkdir images
```

## Required Meme Files

Add these files to the `./images/` folder:

| Gesture | Filename |
|---------|----------|
| Smirk | `smirk-meme.jpg` |
| Wink | `monkey-wink.jpg` |
| Shaq T | `shaq.jpg` |
| Patrick | `patrick-meme.jpg` |
| Speed | `speed.gif` |
| Shock | `shock-guy-meme.jpg` |
| Cut It Out | `cut-it.gif` |
| Shush | `dog-shush.jpg` |
| Thinking | `monkey-thinking.jpg` |
| LeBron | `lebron-scream.jpg` |
| Giggle | `baby-meme-giggle.gif` |
| Idle | `idle.jpg` |

## Usage

### Running the Application

```bash
python main.py
```

- The application will automatically download required MediaPipe models on first run
- Your webcam feed will appear on the left, with the corresponding meme on the right
- Perform gestures to trigger different memes
- Press **'q'** to quit

### Troubleshooting

**Webcam not detected:**
- Ensure your webcam is connected and not in use by another application
- Try changing the camera index in `main.py` (line ~750): `cap = cv2.VideoCapture(0)` → `cap = cv2.VideoCapture(1)`

**Model download fails:**
- Check your internet connection
- Manually download models from:
  - Face: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  - Hand: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
- Place them in `./models/` folder

**Gestures not detecting:**
- Ensure good lighting
- Keep your face and hands visible in the frame
- Adjust detection thresholds in the corresponding gesture functions

## Gestures

### Face-Only
- **Smirk** 😏 - Asymmetric smile
- **Wink** 😉 - One eye closed
- **Speed** ⚡ - Squint + pursed lips
- **Patrick** ⭐ - Jaw drop (no hands)

### Hand-Face (1 hand)
- **Thinking** 🤔 - Finger at mouth corner + mouth open
- **Shush** 🤫 - Finger on lips + face sideways
- **Giggle** 🤭 - Hand covering mouth
- **Cut It Out** ✋ - Flat hand at neck level

### Two-Hand
- **Shock** 😱 - Hands on head + mouth open
- **LeBron** 👑 - Scream + hands down
- **Shaq T** ⏱️ - T-shape timeout gesture

## Contributing

### Pull Requests for Meme Updates

We welcome contributions to improve or replace memes! Follow these guidelines when submitting meme updates:

#### Before You Start
- Check existing memes in the `./images/` folder to avoid duplicates
- Ensure your meme aligns with its corresponding gesture
- Verify the meme is appropriate and high-quality

#### Submission Process
1. **Fork the repository** and create a new branch for your meme update: 
   ```bash
   git checkout -b update/gesture-name-meme
   ```

2. **Add or replace your meme file** in the `./images/` folder: 
   - Use the exact filename from the "Required Meme Files" table
   - Supported formats: `.jpg`, `.png`, `.gif`
   - For GIFs:  Ensure they loop smoothly and aren't excessively large
   - Image resolution:  Recommended 500x500px or larger (will be resized to fit)

3. **Test locally** before submitting:
   ```bash
   python main.py
   # Verify the gesture displays your new meme correctly
   ```

4. **Commit your changes**:
   ```bash
   git add images/your-meme-file
   git commit -m "Update:  Replace [gesture] meme with [brief description]"
   ```

5. **Push and open a Pull Request** with: 
   - **Title**: `Update [Gesture Name] meme`
   - **Description**: Explain why you're updating the meme (better quality, funnier, more relevant, etc.)
   - **PR Body**: Include a screenshot or preview if possible

#### Meme Quality Checklist
- [ ] Image is clear and high-quality
- [ ] Filename matches the table above exactly
- [ ] File size is reasonable (GIFs < 5MB, JPGs < 2MB)
- [ ] Tested with the gesture detection locally
- [ ] No copyright/licensing issues

---

### Adding New Gestures & Updating Detection Functions

Want to add a new gesture or improve existing detection? Here's how to update the gesture detection functions in `main.py`:

#### 1. Understanding the Code Structure

Gesture detection functions are located in `main.py` around lines 280-650. Each function typically:
- Takes face landmarks, hand landmarks, and/or blendshapes as parameters
- Returns a boolean indicating if the gesture is detected
- Uses distance calculations and threshold values

#### 2. Enabling Visual Debugging (Tracking Dots & Skeleton)

By default, tracking dots and hand skeletons are **disabled** for a cleaner UI. To enable them for debugging:

**For Face Landmarks (tracking dots on face):**

Find this line in the `main()` function (around line 815):
```python
# draw_face_landmarks(frame, face_landmarks)  # Disabled: no tracking dots
```

**Uncomment it to:**
```python
draw_face_landmarks(frame, face_landmarks)  # Enabled: shows face tracking dots
```

**For Hand Skeleton (bone structure visualization):**

Find this line in the `main()` function (around line 837):
```python
# draw_hand_landmarks(frame, hand_landmarks_list)  # Disabled: no tracking dots
```

**Uncomment it to:**
```python
draw_hand_landmarks(frame, hand_landmarks_list)  # Enabled: shows hand skeleton
```

#### 3. Example: Creating a "Thumbs Up" Gesture

Let's create a new thumbs-up gesture as an example:

**Step A: Add the meme to the mapping** (around line 60):
```python
GESTURE_MEME_MAP = {
    # ... existing gestures ...
    "thumbs_up": "thumbs-up-meme.jpg",  # Add this line
}
```

**Step B: Create the detection function** (add around line 600):
```python
def detect_thumbs_up_gesture(hand_landmarks_list: list) -> bool:
    """
    Detect "Thumbs Up" gesture: thumb extended upward, other fingers curled.
    
    Hand landmarks:
    - Thumb tip: index 4
    - Thumb IP: index 3
    - Index finger tip: index 8
    - Index finger MCP: index 5
    
    Requirements:
    - Thumb tip higher than thumb IP (thumb pointing up)
    - Index finger tip lower than MCP (finger curled)
    """
    if not hand_landmarks_list:
        return False
    
    for hand in hand_landmarks_list:
        thumb_tip = hand[4]
        thumb_ip = hand[3]
        index_tip = hand[8]
        index_mcp = hand[5]
        
        # Check if thumb is extended upward
        thumb_extended = thumb_tip.y < thumb_ip.y
        
        # Check if index finger is curled (tip below knuckle)
        index_curled = index_tip.y > index_mcp.y
        
        if thumb_extended and index_curled:
            return True
    
    return False
```

**Step C: Add detection to main loop** (around line 880):
```python
# Inside the main() function, after other gesture checks:
if detect_thumbs_up_gesture(hand_landmarks_list):
    detections.append(("Thumbs Up! 👍", (0, 255, 0)))
    if active_gesture is None:
        active_gesture = "thumbs_up"
```

**Step D: Test with visual debugging enabled**:
```python
# Enable hand skeleton to see landmark positions
draw_hand_landmarks(frame, hand_landmarks_list)
```

Run the app and perform the thumbs-up gesture. You'll see:
- Hand skeleton with numbered landmark points
- Detection label appears when successful

**Step E: Fine-tune threshold values**:
If detection is too sensitive/insensitive, adjust the conditions:
```python
# Make thumb requirement stricter
thumb_extended = thumb_tip.y < (thumb_ip.y - 0.05)

# Add middle finger check
middle_tip = hand[12]
middle_mcp = hand[9]
middle_curled = middle_tip.y > middle_mcp.y

if thumb_extended and index_curled and middle_curled:
    return True
```

#### 4. Key Landmark Indices Reference

**Face Landmarks** (468 total points):
- Nose tip: `1`
- Upper lip: `13`
- Lower lip: `14`
- Left mouth corner: `61`
- Right mouth corner: `291`
- Left eye outer: `33`
- Right eye outer: `263`
- Left face edge: `234`
- Right face edge: `454`

**Hand Landmarks** (21 points per hand):
- Wrist: `0`
- Thumb: `1, 2, 3, 4` (tip at 4)
- Index: `5, 6, 7, 8` (tip at 8)
- Middle: `9, 10, 11, 12` (tip at 12)
- Ring: `13, 14, 15, 16` (tip at 16)
- Pinky: `17, 18, 19, 20` (tip at 20)

**Blendshapes** (52 facial expressions):
- `"jawOpen"` - Mouth openness (0-1)
- `"mouthSmileLeft"` / `"mouthSmileRight"` - Smile asymmetry
- `"eyeBlinkLeft"` / `"eyeBlinkRight"` - Eye closure
- `"browInnerUp"` - Eyebrow raise

Full reference: https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_blendshapes.png

#### 5. Pull Request Checklist for New Gestures

When submitting a PR for a new gesture or detection improvement:

- [ ] Gesture function is well-documented with clear requirements
- [ ] Function includes parameter types and return type hints
- [ ] Tested with visual debugging enabled (dots/skeleton)
- [ ] Meme file added to `./images/` folder
- [ ] Meme mapping updated in `GESTURE_MEME_MAP`
- [ ] Detection call added to main loop with appropriate priority
- [ ] False positive rate is acceptable (doesn't trigger on similar gestures)
- [ ] Threshold values are commented and explained
- [ ] PR includes before/after video demonstration

#### 6. Common Debugging Tips

**Gesture not triggering:**
- Enable `draw_hand_landmarks()` or `draw_face_landmarks()`
- Print landmark values to console:
  ```python
  print(f"Thumb Y: {thumb_tip.y}, IP Y: {thumb_ip.y}")
  ```
- Lower threshold values

**False positives:**
- Add more restrictive conditions
- Require multiple landmarks to meet criteria
- Increase threshold values
- Add blendshape requirements

**Gesture conflicts with others:**
- Check gesture priority order in main loop
- Add exclusion conditions (e.g., "only trigger if X gesture is NOT active")

---

Thank you for helping improve the AI Meme Emote Detector! 🎬