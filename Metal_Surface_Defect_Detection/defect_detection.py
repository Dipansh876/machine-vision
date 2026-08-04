import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# -----------------------------
# Paths and Read Image
# -----------------------------
# Use paths relative to this script so the script can be run from any CWD
BASE_DIR = Path(__file__).parent
image_path = BASE_DIR / "dataset" / "img1.jpg"

image = cv2.imread(str(image_path))

if image is None:
    print(f"Error: Image not found at {image_path}")
    sys.exit(1)

# Copy original image
result = image.copy()

# -----------------------------
# Convert to Grayscale
# -----------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -----------------------------
# Gaussian Blur
# -----------------------------
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# -----------------------------
# Thresholding (Otsu)
# -----------------------------
_, thresh = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# -----------------------------
# Edge Detection
# -----------------------------
edges = cv2.Canny(blur, 50, 150)

# -----------------------------
# Morphological Operations
# -----------------------------
kernel = np.ones((3,3), np.uint8)

opening = cv2.morphologyEx(
    thresh,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)

closing = cv2.morphologyEx(
    opening,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

# -----------------------------
# Find Contours
# -----------------------------
contours, _ = cv2.findContours(
    closing,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

count = 0

for contour in contours:

    area = cv2.contourArea(contour)

    # Ignore very small contours (noise)
    if area > 30:

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            result,
            (x, y),
            (x+w, y+h),
            (0, 0, 255),
            2
        )

        cv2.putText(
            result,
            "Defect",
            (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            1
        )

        count += 1

print("Total Defects Detected:", count)

# -----------------------------
# Save Results
# -----------------------------

# Save outputs next to the script
cv2.imwrite(str(BASE_DIR / "output_original.jpg"), image)
cv2.imwrite(str(BASE_DIR / "output_gray.jpg"), gray)
cv2.imwrite(str(BASE_DIR / "output_threshold.jpg"), thresh)
cv2.imwrite(str(BASE_DIR / "output_edges.jpg"), edges)
cv2.imwrite(str(BASE_DIR / "output_morphological.jpg"), closing)
cv2.imwrite(str(BASE_DIR / "output_detected_defects.jpg"), result)

print("\n✓ Results saved successfully!")
print("\nOutput files created in:", BASE_DIR)
print("  - output_original.jpg")
print("  - output_gray.jpg")
print("  - output_threshold.jpg")
print("  - output_edges.jpg")
print("  - output_morphological.jpg")
print("  - output_detected_defects.jpg (with defects highlighted in red boxes)")
print("\nScript completed successfully!")