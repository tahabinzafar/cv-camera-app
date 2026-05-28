import cv2

from .utils import FPS, draw_hud, open_camera


def run(source=0, model_path="yolov8n.pt", conf=0.4, classes=None):
    """
    Detect objects with YOLOv8.

    The weights file (yolov8n.pt) downloads automatically on first run,
    so you need an internet connection the first time. After that it's
    cached locally. Press 'q' to quit.

    source      camera index (0) or path to a video file
    model_path  any ultralytics model; yolov8n is the small/fast default
    conf        confidence threshold (0-1)
    classes     optional list of class ids to keep, e.g. [0] for people only
    """
    # imported here so motion mode works without ultralytics installed
    from ultralytics import YOLO

    model = YOLO(model_path)
    cap = open_camera(source)
    fps = FPS()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=conf, classes=classes, verbose=False)[0]
        annotated = results.plot()  # draws boxes + labels for us

        n = len(results.boxes)
        draw_hud(annotated, [f"FPS: {fps.tick():.1f}", f"Objects: {n}"])

        cv2.imshow("YOLO object detector  (press q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
