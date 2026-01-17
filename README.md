# AI Multi-Gesture Detector with Meme Display

Real-time gesture detection using MediaPipe with parallel meme display.

## Features

- **12 Gesture Detection**: Smirk, Wink, Speed, Patrick, Thinking, Shush, Giggle, Cut It Out, Shock, LeBron Scream, Shaq T, Surprise
- **Split-Screen Display**: Webcam feed (left) + corresponding meme (right)
- **GIF Support**: Animated memes loop automatically
- **1.5s Switch Delay**: Prevents rapid meme switching

## Requirements

- Python 3.11+
- Webcam

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

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
