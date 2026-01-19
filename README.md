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

## Installation

```bash
# Install dependencies
pip install -r requirements. txt

# Create images folder and add memes
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

```bash
python main.py
```

Press **'q'** to quit. 

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

Thank you for helping improve the meme collection!  🎬
