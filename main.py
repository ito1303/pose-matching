"""
POSE MATCHING — Đọ dáng cực lầy, chấm điểm bằng AI ngay tức thì.
Python + OpenCV + MediaPipe Pose. Webcam realtime.
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pose_engine import create_landmarker, detect_pose
from poses import PoseTemplate, catalog
from scoring import align_template_to_player, landmarks_from_result, similarity_score
from ui import (
    attract_overlay,
    countdown_overlay,
    draw_floor_guide,
    draw_skeleton,
    makeup_hud,
    result_overlay,
)

TAGLINE = "Đọ dáng cực lầy – Chấm điểm bằng AI ngay tức thì!"
WINDOW = "POSE MATCHING"
PASS_THRESHOLD = 80.0
HOLD_SECONDS = 2.0
GAME_SECONDS = 45.0
POSES_PER_ROUND = 5
PASSES_TO_WIN = 3
CAM_W, CAM_H = 1280, 720
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"

def beep_pass() -> None:
    try:
        import winsound

        winsound.Beep(880, 90)
        winsound.Beep(1174, 160)
    except Exception:
        pass


def beep_win() -> None:
    try:
        import winsound

        for f in (784, 988, 1174, 1568):
            winsound.Beep(f, 110)
    except Exception:
        pass


@dataclass
class Round:
    poses: list[PoseTemplate]
    index: int = 0
    passed: int = 0
    hold: float = 0.0
    deadline: float = 0.0
    flash_until: float = 0.0
    snapshots: int = 0
    last_score: float = 0.0

    @property
    def current(self) -> PoseTemplate:
        return self.poses[self.index]

    @property
    def done_poses(self) -> bool:
        return self.index >= len(self.poses)


def new_round() -> Round:
    pool = catalog()
    n = min(POSES_PER_ROUND, len(pool))
    picks = random.sample(pool, n)
    return Round(poses=picks, deadline=time.time() + GAME_SECONDS)


@dataclass
class Game:
    state: str = "ATTRACT"
    round: Round | None = None
    countdown_end: float = 0.0
    fullscreen: bool = True
    demo_i: int = 0
    demo_t: float = 0.0
    last: float = field(default_factory=time.time)
    win: bool = False
    ts_ms: int = 0

    def start_countdown(self) -> None:
        self.state = "COUNTDOWN"
        self.countdown_end = time.time() + 3.2
        self.round = new_round()

    def begin_play(self) -> None:
        assert self.round is not None
        self.state = "PLAYING"
        self.round.deadline = time.time() + GAME_SECONDS
        self.round.hold = 0.0

    def next_pose_or_end(self, now: float) -> None:
        assert self.round is not None
        r = self.round
        r.index += 1
        r.hold = 0.0
        if r.passed >= PASSES_TO_WIN:
            self.finish(True)
            return
        remain_poses = len(r.poses) - r.index
        if r.passed + remain_poses < PASSES_TO_WIN:
            self.finish(False)
            return
        if r.done_poses:
            self.finish(r.passed >= PASSES_TO_WIN)
            return
        if now >= r.deadline:
            self.finish(r.passed >= PASSES_TO_WIN)

    def finish(self, win: bool) -> None:
        self.win = win
        self.state = "RESULT"
        if win:
            beep_win()


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def save_snapshot(frame: np.ndarray, pose_name: str, score: float) -> None:
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() else "_" for ch in pose_name)
    path = SNAPSHOT_DIR / f"PASS_{stamp}_{safe}_{int(score)}.jpg"
    cv2.imwrite(str(path), frame)


def setup_window() -> None:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, CAM_W, CAM_H)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def toggle_fullscreen(game: Game) -> None:
    game.fullscreen = not game.fullscreen
    mode = cv2.WINDOW_FULLSCREEN if game.fullscreen else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, mode)
    if not game.fullscreen:
        cv2.resizeWindow(WINDOW, CAM_W, CAM_H)


def main() -> int:
    cap = open_camera()
    if not cap.isOpened():
        print("Cannot open webcam. Check the camera, then run: python main.py")
        return 1

    setup_window()
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    game = Game()
    poses_all = catalog()

    landmarker = create_landmarker()
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                print("Webcam signal lost.")
                break

            frame = cv2.flip(raw, 1)
            frame = cv2.resize(frame, (CAM_W, CAM_H))
            now = time.time()
            dt = max(0.0, min(0.1, now - game.last))
            game.last = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            game.ts_ms += max(1, int(dt * 1000))
            result = detect_pose(landmarker, rgb, game.ts_ms)
            player = landmarks_from_result(result.pose_landmarks[0] if result.pose_landmarks else None)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("f"), ord("F")):
                toggle_fullscreen(game)

            if game.state == "ATTRACT":
                if now - game.demo_t > 1.6:
                    game.demo_t = now
                    game.demo_i = (game.demo_i + 1) % len(poses_all)
                demo = poses_all[game.demo_i]
                vis = draw_floor_guide(frame)
                ghost = demo.landmarks.copy()
                if player is not None:
                    ghost = align_template_to_player(demo.landmarks, player)
                    vis = draw_skeleton(vis, player, (255, 45, 149), 3, 5)
                vis = draw_skeleton(vis, ghost, (0, 240, 255), 4, 6, alpha=0.85)
                vis = attract_overlay(vis, demo.name)
                if key in (ord(" "), 13):
                    game.start_countdown()
                cv2.imshow(WINDOW, vis)
                continue

            if game.state == "COUNTDOWN":
                left = game.countdown_end - now
                n = max(1, int(np.ceil(left)))
                vis = draw_floor_guide(frame)
                if player is not None:
                    vis = draw_skeleton(vis, player, (255, 45, 149), 3, 5)
                vis = countdown_overlay(vis, n)
                if left <= 0:
                    game.begin_play()
                cv2.imshow(WINDOW, vis)
                continue

            if game.state == "RESULT":
                vis = frame
                r = game.round
                passed = r.passed if r else 0
                snaps = r.snapshots if r else 0
                vis = result_overlay(vis, game.win, passed, PASSES_TO_WIN, snaps)
                if key in (ord(" "), 13):
                    game.state = "ATTRACT"
                cv2.imshow(WINDOW, vis)
                continue

            # PLAYING
            r = game.round
            assert r is not None
            remain = r.deadline - now
            if remain <= 0:
                game.finish(r.passed >= PASSES_TO_WIN)
                continue
            if r.done_poses:
                game.finish(r.passed >= PASSES_TO_WIN)
                continue

            target = r.current
            score = 0.0
            vis = draw_floor_guide(frame)

            if player is not None:
                vis = draw_skeleton(vis, player, (255, 45, 149), 4, 6)
                ghost = align_template_to_player(target.landmarks, player)
                vis = draw_skeleton(vis, ghost, (0, 240, 255), 3, 5, alpha=0.7)
                score = similarity_score(player, target.landmarks)
            else:
                vis = draw_skeleton(vis, target.landmarks, (0, 240, 255), 4, 6, alpha=0.85)

            r.last_score = score
            flashing = now < r.flash_until

            just_passed = False
            if flashing:
                pass
            elif score >= PASS_THRESHOLD and player is not None:
                r.hold += dt
                if r.hold >= HOLD_SECONDS:
                    r.snapshots += 1
                    r.passed += 1
                    r.hold = 0.0
                    r.flash_until = now + 0.85
                    beep_pass()
                    flashing = True
                    just_passed = True
            else:
                r.hold = 0.0

            if flashing and now >= r.flash_until:
                r.flash_until = 0.0
                game.next_pose_or_end(now)
                if game.state != "PLAYING":
                    continue

            hint = "Đứng vào vòng tròn (1.5m – 2m) • Khớp khung xương xanh với dáng mẫu"
            if player is None:
                hint = "Chưa thấy người — hãy đứng vào khung, cách camera 1.5m đến 2m"
            elif score >= PASS_THRESHOLD:
                hint = "Đỉnh! Giữ nguyên dáng 2 giây để PASS"
            elif score >= 55:
                hint = "Gần đúng rồi — chỉnh tay và chân theo skeleton xanh"

            vis = makeup_hud(
                vis,
                title="POSE MATCHING",
                tagline=TAGLINE,
                score=score,
                hold_t=r.hold,
                hold_need=HOLD_SECONDS,
                remain=remain,
                game_seconds=int(GAME_SECONDS),
                pose_i=min(r.index + 1, len(r.poses)),
                pose_n=len(r.poses),
                pose_name=target.name,
                vibe=target.vibe,
                passed=r.passed,
                need_pass=PASSES_TO_WIN,
                threshold=PASS_THRESHOLD,
                hint=hint,
                flash_pass=flashing,
                state=game.state,
            )
            if just_passed:
                save_snapshot(vis, target.name, score)
            cv2.imshow(WINDOW, vis)
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
