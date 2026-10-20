"""HUD gian trại: neon, tiếng Việt, thanh điểm, ghost skeleton."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from poses import POSE_CONNECTIONS

MAGENTA = (255, 45, 149)
CYAN = (0, 240, 255)
YELLOW = (255, 230, 0)
LIME = (57, 255, 20)
WHITE = (255, 255, 255)
DARK = (11, 6, 24)

_FONT_PATHS = [
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\calibri.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = []
    if bold:
        names = [
            Path(r"C:\Windows\Fonts\segoeuib.ttf"),
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
        ]
    for p in names + _FONT_PATHS:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def draw_text(
    canvas: Image.Image,
    text: str,
    xy: tuple[int, int],
    size: int,
    fill: tuple[int, int, int] = WHITE,
    bold: bool = False,
    anchor: str = "lt",
    stroke: int = 0,
    stroke_fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.text(
        xy,
        text,
        font=_font(size, bold=bold),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=stroke_fill,
    )


def rounded_rect(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int] | tuple[int, int, int, int],
    radius: int = 18,
    outline: tuple[int, int, int] | None = None,
    width: int = 3,
) -> None:
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    canvas.alpha_composite(overlay)


def draw_skeleton(
    frame: np.ndarray,
    landmarks_xy: np.ndarray,
    color_bgr: tuple[int, int, int],
    thickness: int = 4,
    point_r: int = 6,
    alpha: float = 1.0,
) -> np.ndarray:
    h, w = frame.shape[:2]
    layer = frame.copy()
    pts = [(int(p[0] * w), int(p[1] * h)) for p in landmarks_xy]
    for a, b in POSE_CONNECTIONS:
        pa, pb = pts[a], pts[b]
        cv2.line(layer, pa, pb, color_bgr, thickness, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if i > 10 or i == 0:
            cv2.circle(layer, p, point_r, color_bgr, -1, cv2.LINE_AA)
            cv2.circle(layer, p, point_r + 2, (255, 255, 255), 1, cv2.LINE_AA)
    if alpha >= 0.99:
        return layer
    return cv2.addWeighted(layer, alpha, frame, 1 - alpha, 0)


def draw_floor_guide(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.ellipse(overlay, (w // 2, int(h * 0.90)), (int(w * 0.18), int(h * 0.05)), 0, 0, 360, (0, 240, 255), 3)
    cv2.ellipse(overlay, (w // 2, int(h * 0.90)), (int(w * 0.12), int(h * 0.035)), 0, 0, 360, (255, 45, 149), 2)
    return cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)


def makeup_hud(
    frame: np.ndarray,
    *,
    title: str,
    tagline: str,
    score: float,
    hold_t: float,
    hold_need: float,
    remain: float,
    game_seconds: int,
    pose_i: int,
    pose_n: int,
    pose_name: str,
    vibe: str,
    passed: int,
    need_pass: int,
    threshold: float,
    hint: str,
    flash_pass: bool,
    state: str,
) -> np.ndarray:
    h, w = frame.shape[:2]
    img = bgr_to_pil(frame).convert("RGBA")

    # top bar
    rounded_rect(img, (24, 18, w - 24, 118), (12, 8, 28, 210), radius=22, outline=CYAN, width=3)
    draw_text(img, title, (48, 38), 28, YELLOW, bold=True, stroke=2)
    draw_text(img, tagline, (48, 76), 22, CYAN, stroke=1)

    timer_col = LIME if remain > 12 else YELLOW if remain > 6 else MAGENTA
    draw_text(img, f"{max(0, remain):.1f}s", (w - 48, 68), 42, timer_col, bold=True, anchor="rm", stroke=2)

    # score panel
    rounded_rect(img, (24, 140, 430, 300), (12, 8, 28, 200), radius=20, outline=MAGENTA, width=3)
    draw_text(img, "SIMILARITY", (48, 156), 18, WHITE)
    col = LIME if score >= threshold else CYAN if score >= threshold * 0.65 else WHITE
    draw_text(img, f"{score:.0f}%", (48, 186), 54, col, bold=True, stroke=2)
    draw_text(img, f"Ngưỡng Pass: {threshold:.0f}%  •  Giữ {hold_need:.0f}s", (48, 250), 16, WHITE)

    bar_x0, bar_x1, bar_y = 48, 406, 278
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((bar_x0, bar_y, bar_x1, bar_y + 14), 7, outline=WHITE, width=1)
    fill_w = int((bar_x1 - bar_x0) * min(score / 100.0, 1.0))
    if fill_w > 8:
        draw.rounded_rectangle((bar_x0, bar_y, bar_x0 + fill_w, bar_y + 14), 7, fill=col)

    # hold bar
    if score >= threshold and state == "PLAYING":
        rounded_rect(img, (24, 316, 430, 368), (12, 8, 28, 200), radius=16, outline=LIME, width=2)
        draw_text(img, "GIỮ DÁNG...", (48, 324), 16, LIME, bold=True)
        hx0, hx1, hy = 48, 406, 348
        draw.rounded_rectangle((hx0, hy, hx1, hy + 10), 5, outline=LIME, width=1)
        hw = int((hx1 - hx0) * min(hold_t / hold_need, 1.0))
        if hw > 6:
            draw.rounded_rectangle((hx0, hy, hx0 + hw, hy + 10), 5, fill=LIME)

    # pose card
    rounded_rect(img, (w - 430, 140, w - 24, 300), (12, 8, 28, 200), radius=20, outline=YELLOW, width=3)
    draw_text(img, f"TƯ THẾ {pose_i}/{pose_n}", (w - 406, 156), 18, YELLOW, bold=True)
    draw_text(img, pose_name, (w - 406, 192), 26, WHITE, bold=True, stroke=1)
    draw_text(img, vibe, (w - 406, 236), 18, CYAN)
    draw_text(img, f"Pass {passed}/{need_pass} để thắng", (w - 406, 264), 18, WHITE)

    # bottom hint
    rounded_rect(img, (24, h - 92, w - 24, h - 24), (12, 8, 28, 210), radius=18, outline=CYAN, width=2)
    draw_text(img, hint, (w // 2, h - 58), 22, WHITE, bold=True, anchor="mm", stroke=1)

    if flash_pass:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.rectangle((0, 0, w, h), outline=LIME + (220,), width=18)
        img.alpha_composite(overlay)
        draw_text(img, "PASS!", (w // 2, h // 2 - 20), 96, LIME, bold=True, anchor="mm", stroke=4)

    return pil_to_bgr(img)


def attract_overlay(frame: np.ndarray, pose_name: str) -> np.ndarray:
    h, w = frame.shape[:2]
    img = bgr_to_pil(frame).convert("RGBA")
    rounded_rect(img, (60, 40, w - 60, 210), (12, 8, 28, 225), radius=28, outline=MAGENTA, width=4)
    draw_text(img, "POSE MATCHING", (w // 2, 88), 56, YELLOW, bold=True, anchor="mm", stroke=3)
    draw_text(
        img,
        "Đọ dáng cực lầy – Chấm điểm bằng AI ngay tức thì!",
        (w // 2, 150),
        26,
        CYAN,
        bold=True,
        anchor="mm",
        stroke=2,
    )
    rounded_rect(img, (w // 2 - 280, h // 2 + 80, w // 2 + 280, h // 2 + 170), (12, 8, 28, 220), radius=20, outline=CYAN, width=3)
    draw_text(img, "NHẤN SPACE ĐỂ BẮT ĐẦU", (w // 2, h // 2 + 112), 28, WHITE, bold=True, anchor="mm")
    draw_text(img, "3/5 tư thế  •  45 giây  •  Giữ 80% trong 2s", (w // 2, h // 2 + 148), 18, YELLOW, anchor="mm")
    draw_text(img, f"Demo: {pose_name}", (w // 2, h - 48), 20, WHITE, anchor="mm", stroke=1)
    draw_text(img, "Q / ESC = thoát   F = toàn màn hình", (w // 2, h - 22), 16, (200, 200, 200), anchor="mm")
    return pil_to_bgr(img)


def countdown_overlay(frame: np.ndarray, n: int) -> np.ndarray:
    h, w = frame.shape[:2]
    img = bgr_to_pil(frame).convert("RGBA")
    draw_text(img, "SẴN SÀNG", (w // 2, h // 2 - 90), 32, CYAN, bold=True, anchor="mm", stroke=2)
    draw_text(img, str(n), (w // 2, h // 2 + 10), 160, YELLOW, bold=True, anchor="mm", stroke=6)
    return pil_to_bgr(img)


def result_overlay(frame: np.ndarray, win: bool, passed: int, need: int, snapshots: int) -> np.ndarray:
    h, w = frame.shape[:2]
    img = bgr_to_pil(frame).convert("RGBA")
    title = "CHIẾN THẮNG!" if win else "HẾT GIỜ / CHƯA ĐỦ DÁNG"
    color = LIME if win else MAGENTA
    rounded_rect(img, (80, h // 2 - 160, w - 80, h // 2 + 160), (12, 8, 28, 230), radius=28, outline=color, width=4)
    draw_text(img, title, (w // 2, h // 2 - 90), 48, color, bold=True, anchor="mm", stroke=3)
    draw_text(img, f"Pass {passed}/{need} tư thế", (w // 2, h // 2 - 20), 28, WHITE, bold=True, anchor="mm")
    draw_text(img, f"Đã lưu {snapshots} ảnh trong thư mục snapshots/", (w // 2, h // 2 + 28), 20, CYAN, anchor="mm")
    draw_text(img, "SPACE chơi lại  •  Q thoát", (w // 2, h // 2 + 88), 22, YELLOW, bold=True, anchor="mm")
    return pil_to_bgr(img)
