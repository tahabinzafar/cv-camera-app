import os
import time

import cv2

from .utils import FPS, draw_hud, open_camera


def run(source=0, min_area=800, save_dir=None, cooldown=2.0):
    """
    Detect movement using background subtraction (MOG2).

    Anything that moves against the learned background gets boxed.
    Press 'q' to quit.

    source    camera index (0) or path to a video file
    min_area  ignore moving blobs smaller than this (pixels)
    save_dir  if set, save a snapshot whenever motion appears
    cooldown  min seconds between saved snapshots
    """
    cap = open_camera(source)
    backsub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=40, detectShadows=True
    )
    fps = FPS()
    last_save = 0.0

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        mask = backsub.apply(frame)
        mask = cv2.medianBlur(mask, 5)
        # drop the grey "shadow" pixels MOG2 marks as 127
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        moving = 0
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            moving += 1

        status = "MOTION" if moving else "idle"
        draw_hud(
            frame,
            [f"FPS: {fps.tick():.1f}", f"Status: {status} ({moving} regions)"],
            color=(0, 0, 255) if moving else (0, 255, 0),
        )

        if save_dir and moving and (time.time() - last_save) > cooldown:
            fname = os.path.join(save_dir, f"motion_{int(time.time())}.jpg")
            cv2.imwrite(fname, frame)
            last_save = time.time()

        cv2.imshow("Motion detector  (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
