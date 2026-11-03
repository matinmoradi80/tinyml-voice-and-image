"""
Serial image viewer for Arduino TFLite inference.

Reads from the Arduino's serial port, captures the 32x32 grayscale images sent
between <IMG> and </IMG> markers, and displays them alongside the inference
result in a live matplotlib window.

Setup:
    pip install pyserial matplotlib numpy pillow

Usage:
    1. Close the Arduino IDE Serial Monitor (only one program can use the port)
    2. Edit PORT below to match your COM port (e.g. 'COM12' on Windows,
       '/dev/cu.usbmodem...' on Mac, '/dev/ttyACM0' on Linux)
    3. Run:  python serial_image_viewer.py
"""

import serial
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PORT      = "COM12"     # change if your port differs
BAUD      = 115200
IMG_W     = 32
IMG_H     = 32
SAVE_DIR  = "captured_frames"  # folder where images will be saved

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Opening {PORT} at {BAUD} baud...")
    ser = serial.Serial(PORT, BAUD, timeout=5)

    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 5))
    img_display = ax.imshow(np.zeros((IMG_H, IMG_W), dtype=np.uint8),
                            cmap="gray", vmin=0, vmax=255)
    ax.set_title("Waiting for image...")
    ax.axis("off")
    plt.show()

    capture_mode = False
    hex_buffer   = ""
    last_scores  = {}
    frame_count  = 0

    try:
        while True:
            try:
                line = ser.readline().decode("utf-8", errors="replace").strip()
            except Exception as e:
                print(f"[serial error] {e}")
                continue

            if not line:
                continue

            if line == "<IMG>":
                capture_mode = True
                hex_buffer   = ""
                continue

            if line == "</IMG>":
                capture_mode = False
                try:
                    raw = bytes.fromhex(hex_buffer)
                    if len(raw) == IMG_W * IMG_H:
                        img = np.frombuffer(raw, dtype=np.uint8).reshape(IMG_H, IMG_W)

                        # Update display
                        img_display.set_data(img)
                        title = " | ".join(f"{c}:{s}" for c, s in last_scores.items())
                        ax.set_title(title or "Image captured")
                        fig.canvas.draw_idle()
                        plt.pause(0.001)

                        # Save to disk with counter
                        filename = os.path.join(SAVE_DIR, f"captured_{frame_count:04d}.png")
                        Image.fromarray(img).save(filename)
                        print(f"[saved] {filename}")
                        frame_count += 1

                    else:
                        print(f"[warn] got {len(raw)} bytes, expected {IMG_W*IMG_H}")
                except ValueError as e:
                    print(f"[hex parse error] {e}")
                continue

            if capture_mode:
                hex_buffer += line
                continue

            # Echo non-image lines to console
            print(line)

            # Parse score lines like "  rock: 89"
            for cls in ("rock", "paper", "scissors"):
                if line.startswith(cls + ":") or line.startswith("  " + cls + ":"):
                    try:
                        score = int(line.split(":")[1].strip())
                        last_scores[cls] = score
                    except (ValueError, IndexError):
                        pass

    except KeyboardInterrupt:
        print(f"\nExiting. {frame_count} frames saved to '{SAVE_DIR}/'")
    finally:
        ser.close()


if __name__ == "__main__":
    main()