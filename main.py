import argparse

from src import motion_detector, object_detector


def parse_source(value):
    """A webcam index like '0' becomes int 0; anything else stays a path."""
    return int(value) if str(value).isdigit() else value


def main():
    p = argparse.ArgumentParser(
        description="Simple computer vision camera app: motion or YOLO detection."
    )
    p.add_argument(
        "mode",
        choices=["motion", "detect"],
        help="motion = movement detection, detect = YOLO object detection",
    )
    p.add_argument(
        "--source", default="0",
        help="camera index (default 0) or path to a video file",
    )
    # motion options
    p.add_argument(
        "--min-area", type=int, default=800,
        help="[motion] ignore moving blobs smaller than this many pixels",
    )
    p.add_argument(
        "--save-dir", default=None,
        help="[motion] folder to save snapshots when motion is detected",
    )
    # detect options
    p.add_argument(
        "--model", default="yolov8n.pt",
        help="[detect] ultralytics weights file (auto-downloads if missing)",
    )
    p.add_argument(
        "--conf", type=float, default=0.4,
        help="[detect] confidence threshold, 0-1",
    )
    p.add_argument(
        "--people-only", action="store_true",
        help="[detect] keep only the 'person' class",
    )

    args = p.parse_args()
    source = parse_source(args.source)

    if args.mode == "motion":
        motion_detector.run(
            source, min_area=args.min_area, save_dir=args.save_dir
        )
    else:
        classes = [0] if args.people_only else None
        object_detector.run(
            source, model_path=args.model, conf=args.conf, classes=classes
        )


if __name__ == "__main__":
    main()
