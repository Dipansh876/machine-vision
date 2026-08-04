import cv2
from deepface import DeepFace
import argparse
import os


def annotate_frame(frame, result):
    res = result[0] if isinstance(result, list) else result
    emotion = res.get('dominant_emotion', 'Unknown')
    face = res.get('region', {}) or {}
    x = int(face.get('x', 0))
    y = int(face.get('y', 0))
    w = int(face.get('w', 0))
    h = int(face.get('h', 0))

    if w > 0 and h > 0:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Emotion: {emotion}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )
    else:
        cv2.putText(
            frame,
            f"Emotion: {emotion}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )

    return frame


def analyze_image(image_path, output_path=None):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to read image: {image_path}")
        return

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        annotated = annotate_frame(frame, result)
        cv2.imshow("Emotion Result", annotated)
        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"Annotated image saved to: {output_path}")
        print("Press any key on the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error analyzing image: {e}")


def run_webcam():
    # Try multiple capture backends and device indices for better Windows compatibility
    def try_open_camera():
        # Prefer DirectShow on Windows which is often more reliable than MSMF
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for backend in backends:
            for idx in range(0, 3):
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    print(f"Opened camera index {idx} with backend {backend}")
                    return cap
                cap.release()
        return None

    cap = try_open_camera()
    if cap is None or not cap.isOpened():
        print("Error: Could not open webcam. Make sure no other app is using the camera and Windows permissions allow camera access.")
        return

    print("Emotion Detection Started...")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            annotated = annotate_frame(frame, result)
            cv2.imshow("Customer Emotion Classification", annotated)
        except Exception as e:
            print(f"Emotion analysis error: {e}")
            cv2.imshow("Customer Emotion Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Emotion detection from webcam or image')
    parser.add_argument('-i', '--image', help='Path to an input image to analyze')
    parser.add_argument('-o', '--output', help='Path to save annotated image (optional)')
    args = parser.parse_args()

    if args.image:
        analyze_image(args.image, args.output)
    else:
        run_webcam()


if __name__ == '__main__':
    main()
