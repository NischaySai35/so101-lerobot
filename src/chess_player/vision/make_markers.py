"""Print sheet for the four ArUco markers: one A4 PNG at 300 dpi, markers MARKER_MM wide with a white border.

  python -m chess_player.vision.make_markers      -> data/aruco_markers_A4.png

Print at 100% scale (no "fit to page"), then measure the black square and put it in config.MARKER_MM.
Tape them flat outside the board corners, all the same way up, ids 0..3 next to a1, h1, h8, a8.
"""
import cv2
import numpy as np

from chess_player import config as C

DPI = 300


def main():
    px_per_mm = DPI / 25.4
    side = int(round(C.MARKER_MM * px_per_mm))
    page = np.full((int(297 * px_per_mm), int(210 * px_per_mm)), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, C.ARUCO_DICT))
    gap = int(25 * px_per_mm)
    for k, mid in enumerate(sorted(C.MARKER_CENTERS)):
        img = cv2.aruco.generateImageMarker(d, mid, side)
        r, c = divmod(k, 2)
        y, x = gap + r * (side + 2 * gap), gap + c * (side + 2 * gap)
        page[y : y + side, x : x + side] = img
        cv2.putText(page, f"id {mid}", (x, y + side + int(8 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
    out = C.DATA / "aruco_markers_A4.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), page)
    print(f"wrote {out} ({C.MARKER_MM:.0f} mm markers, print at 100%)")


if __name__ == "__main__":
    main()
