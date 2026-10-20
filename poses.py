"""Thư viện tư thế mẫu — dáng meme, anime, thể thao."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# MediaPipe Pose: 33 điểm
N_LANDMARKS = 33
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

BODY_INDEXES = [
    L_SHOULDER,
    R_SHOULDER,
    L_ELBOW,
    R_ELBOW,
    L_WRIST,
    R_WRIST,
    L_HIP,
    R_HIP,
    L_KNEE,
    R_KNEE,
    L_ANKLE,
    R_ANKLE,
]

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (15, 17),
    (15, 19),
    (15, 21),
    (16, 18),
    (16, 20),
    (16, 22),
    (27, 29),
    (27, 31),
    (28, 30),
    (28, 32),
    (0, 11),
    (0, 12),
]


def _polar(origin: tuple[float, float], length: float, angle_deg: float) -> tuple[float, float]:
    rad = math.radians(angle_deg)
    return origin[0] + length * math.cos(rad), origin[1] + length * math.sin(rad)


def _fill_face_and_hands(lm: np.ndarray) -> np.ndarray:
    mid_sh = (lm[L_SHOULDER] + lm[R_SHOULDER]) / 2
    hip = (lm[L_HIP] + lm[R_HIP]) / 2
    torso = np.linalg.norm(mid_sh - hip) + 1e-6
    up = (mid_sh - hip) / torso

    nose = mid_sh + up * 0.38 * torso
    lm[0] = nose
    ear_w = 0.18 * torso
    eye_w = 0.10 * torso
    side = np.array([1.0, 0.0])
    lm[1] = nose - side * eye_w * 0.6 - np.array([0, 0.012])
    lm[2] = nose - side * eye_w - np.array([0, 0.01])
    lm[3] = nose - side * eye_w * 1.3 - np.array([0, 0.008])
    lm[4] = nose + side * eye_w * 0.6 - np.array([0, 0.012])
    lm[5] = nose + side * eye_w - np.array([0, 0.01])
    lm[6] = nose + side * eye_w * 1.3 - np.array([0, 0.008])
    lm[7] = nose - side * ear_w
    lm[8] = nose + side * ear_w
    lm[9] = nose - side * 0.06 - np.array([0.0, -0.02])
    lm[10] = nose + side * 0.06 - np.array([0.0, -0.02])

    def hand(wrist_i: int, elbow_i: int, pinky_i: int, index_i: int, thumb_i: int) -> None:
        d = lm[wrist_i] - lm[elbow_i]
        n = np.linalg.norm(d) + 1e-6
        fwd = d / n
        perp = np.array([-fwd[1], fwd[0]])
        lm[pinky_i] = lm[wrist_i] + fwd * 0.035 + perp * 0.02
        lm[index_i] = lm[wrist_i] + fwd * 0.04 - perp * 0.01
        lm[thumb_i] = lm[wrist_i] + fwd * 0.02 - perp * 0.03

    hand(L_WRIST, L_ELBOW, L_PINKY, L_INDEX, L_THUMB)
    hand(R_WRIST, R_ELBOW, R_PINKY, R_INDEX, R_THUMB)

    def foot(ankle_i: int, knee_i: int, heel_i: int, foot_i: int) -> None:
        d = lm[ankle_i] - lm[knee_i]
        n = np.linalg.norm(d) + 1e-6
        fwd = d / n
        lm[heel_i] = lm[ankle_i] - fwd * 0.03
        lm[foot_i] = lm[ankle_i] + np.array([fwd[0], abs(fwd[1])]) * 0.05 + np.array([0.0, 0.01])

    foot(L_ANKLE, L_KNEE, L_HEEL, L_FOOT)
    foot(R_ANKLE, R_KNEE, R_HEEL, R_FOOT)
    return lm


def build_pose(
    l_upper: float,
    l_lower: float,
    r_upper: float,
    r_lower: float,
    l_thigh: float,
    l_shin: float,
    r_thigh: float,
    r_shin: float,
    lean_x: float = 0.0,
    hip_y: float = 0.54,
) -> np.ndarray:
    """Góc theo hệ màn hình: 0° = phải, 90° = xuống, 180° = trái, 270° = lên."""
    torso = 0.18
    sh_half = 0.11
    hip_half = 0.07
    upper_arm = 0.13
    forearm = 0.12
    thigh = 0.17
    shin = 0.16

    hip_mid = np.array([0.50 + lean_x * 0.15, hip_y])
    l_hip = hip_mid + np.array([-hip_half, 0.0])
    r_hip = hip_mid + np.array([hip_half, 0.0])
    sh_mid = hip_mid + np.array([lean_x * 0.22, -torso])
    l_sh = sh_mid + np.array([-sh_half, 0.0])
    r_sh = sh_mid + np.array([sh_half, 0.0])

    l_elb = np.array(_polar(tuple(l_sh), upper_arm, l_upper))
    l_wri = np.array(_polar(tuple(l_elb), forearm, l_lower))
    r_elb = np.array(_polar(tuple(r_sh), upper_arm, r_upper))
    r_wri = np.array(_polar(tuple(r_elb), forearm, r_lower))
    l_kn = np.array(_polar(tuple(l_hip), thigh, l_thigh))
    l_an = np.array(_polar(tuple(l_kn), shin, l_shin))
    r_kn = np.array(_polar(tuple(r_hip), thigh, r_thigh))
    r_an = np.array(_polar(tuple(r_kn), shin, r_shin))

    lm = np.zeros((N_LANDMARKS, 2), dtype=np.float32)
    lm[L_SHOULDER], lm[R_SHOULDER] = l_sh, r_sh
    lm[L_ELBOW], lm[R_ELBOW] = l_elb, r_elb
    lm[L_WRIST], lm[R_WRIST] = l_wri, r_wri
    lm[L_HIP], lm[R_HIP] = l_hip, r_hip
    lm[L_KNEE], lm[R_KNEE] = l_kn, r_kn
    lm[L_ANKLE], lm[R_ANKLE] = l_an, r_an
    return _fill_face_and_hands(lm)


@dataclass(frozen=True)
class PoseTemplate:
    id: str
    name: str
    vibe: str
    landmarks: np.ndarray


def catalog() -> list[PoseTemplate]:
    return [
        PoseTemplate(
            "tpose",
            "T-POSE SIÊU NGẦU",
            "Thể thao",
            build_pose(180, 180, 0, 0, 90, 90, 90, 90),
        ),
        PoseTemplate(
            "yay",
            "GIƠ TAY ĂN MỪNG",
            "Meme",
            build_pose(230, 250, 310, 290, 90, 90, 90, 90),
        ),
        PoseTemplate(
            "hero",
            "SIÊU NHÂN CHỐNG HÔNG",
            "Anime",
            build_pose(145, 55, 35, 125, 95, 90, 85, 90),
        ),
        PoseTemplate(
            "dab",
            "DAB CỰC LẦY",
            "Meme",
            build_pose(200, 210, 195, 200, 100, 95, 80, 85, lean_x=-0.25),
        ),
        PoseTemplate(
            "tree",
            "CÂY YOGA LẬT ĐẬT",
            "Thể thao",
            build_pose(230, 240, 310, 300, 90, 90, 20, 200),
        ),
        PoseTemplate(
            "box",
            "ĐẤM BỐC",
            "Thể thao",
            build_pose(250, 200, 320, 10, 105, 95, 75, 80, lean_x=0.12),
        ),
        PoseTemplate(
            "naruto",
            "NARUTO CHẠY",
            "Anime",
            build_pose(70, 50, 110, 130, 55, 40, 115, 100, lean_x=0.35, hip_y=0.56),
        ),
        PoseTemplate(
            "heart",
            "TIM TO ĐÙNG",
            "Meme",
            build_pose(250, 300, 290, 240, 90, 90, 90, 90),
        ),
        PoseTemplate(
            "squat",
            "SQUAT GOKU",
            "Anime",
            build_pose(150, 90, 30, 90, 55, 95, 125, 85, hip_y=0.62),
        ),
        PoseTemplate(
            "kick",
            "ĐÁ CAO",
            "Thể thao",
            build_pose(200, 210, 340, 300, 90, 90, 300, 300, lean_x=-0.15),
        ),
        PoseTemplate(
            "point",
            "CHỈ TRỜI ANIME",
            "Anime",
            build_pose(90, 90, 270, 270, 90, 90, 90, 90),
        ),
        PoseTemplate(
            "wave",
            "CHÀO VẪY LẦY LỘI",
            "Meme",
            build_pose(90, 90, 300, 250, 90, 90, 90, 90),
        ),
    ]
