"""Chấm điểm tương đồng khung xương (bất biến tịnh tiến + tỉ lệ)."""

from __future__ import annotations

import math

import numpy as np

from poses import BODY_INDEXES, L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST
from poses import R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST

ANGLE_TRIPLETS = [
    (L_SHOULDER, L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW, R_WRIST),
    (L_ELBOW, L_SHOULDER, L_HIP),
    (R_ELBOW, R_SHOULDER, R_HIP),
    (L_SHOULDER, L_HIP, L_KNEE),
    (R_SHOULDER, R_HIP, R_KNEE),
    (L_HIP, L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE, R_ANKLE),
]


def landmarks_from_result(landmarks) -> np.ndarray | None:
    if not landmarks:
        return None
    pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    if pts.shape[0] < 29:
        return None
    return pts


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


def normalize_pose(lm: np.ndarray) -> np.ndarray | None:
    hip = _mid(lm[L_HIP], lm[R_HIP])
    sh = _mid(lm[L_SHOULDER], lm[R_SHOULDER])
    scale = float(np.linalg.norm(sh - hip))
    if scale < 1e-4:
        return None
    return (lm - hip) / scale


def _joint_angle(lm: np.ndarray, a: int, b: int, c: int) -> float | None:
    ba = lm[a] - lm[b]
    bc = lm[c] - lm[b]
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return None
    cos = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def similarity_score(player: np.ndarray, target: np.ndarray) -> float:
    """Trả về 0–100. Kết hợp khoảng cách landmark đã chuẩn hoá + góc khớp."""
    pn = normalize_pose(player)
    tn = normalize_pose(target)
    if pn is None or tn is None:
        return 0.0

    dists = [float(np.linalg.norm(pn[i] - tn[i])) for i in BODY_INDEXES]
    mean_d = float(np.mean(dists))
    landmark_score = max(0.0, 1.0 - mean_d / 1.15) * 100.0

    diffs: list[float] = []
    for a, b, c in ANGLE_TRIPLETS:
        pa = _joint_angle(player, a, b, c)
        ta = _joint_angle(target, a, b, c)
        if pa is None or ta is None:
            continue
        diffs.append(abs(pa - ta))
    if diffs:
        angle_score = max(0.0, 1.0 - (float(np.mean(diffs)) / 90.0)) * 100.0
    else:
        angle_score = 0.0

    return float(np.clip(0.55 * landmark_score + 0.45 * angle_score, 0.0, 100.0))


def align_template_to_player(template: np.ndarray, player: np.ndarray) -> np.ndarray:
    """Đặt ghost tư thế mẫu lên người chơi (cùng hông + tỉ lệ thân)."""
    p_hip = _mid(player[L_HIP], player[R_HIP])
    t_hip = _mid(template[L_HIP], template[R_HIP])
    p_sh = _mid(player[L_SHOULDER], player[R_SHOULDER])
    t_sh = _mid(template[L_SHOULDER], template[R_SHOULDER])
    p_scale = float(np.linalg.norm(p_sh - p_hip))
    t_scale = float(np.linalg.norm(t_sh - t_hip))
    if t_scale < 1e-6:
        return template.copy()
    scale = p_scale / t_scale
    return (template - t_hip) * scale + p_hip
