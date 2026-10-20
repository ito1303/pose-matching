# POSE MATCHING

**Đọ dáng cực lầy – Chấm điểm bằng AI ngay tức thì!**

Trò chơi gian trại: webcam + MediaPipe Pose, khớp khung xương với tư thế mẫu. Pass khi similarity ≥ 80% và giữ 2 giây. Thắng khi **3/5 tư thế** trong **45 giây**.

## Chạy nhanh

Chạy được trên Python 3.14 (MediaPipe Tasks PoseLandmarker). Lần đầu sẽ tự tải `models/pose_landmarker_lite.task`.

```powershell
cd C:\Users\tcuon\pose-matching
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Đứng cách camera khoảng **1.5m – 2m**, đủ sáng, người nằm gọn trong khung.

## Phím

| Phím | Việc |
|------|------|
| `Space` / `Enter` | Bắt đầu / chơi lại |
| `F` | Bật/tắt toàn màn hình |
| `Q` / `Esc` | Thoát |

## Luật (code)

Trong `main.py`:

- `PASS_THRESHOLD = 80`
- `HOLD_SECONDS = 2`
- `GAME_SECONDS = 45`
- `POSES_PER_ROUND = 5`
- `PASSES_TO_WIN = 3`

Mỗi lần Pass, game lưu ảnh vào `snapshots/`.

## Thêm tư thế mẫu

Sửa `poses.py` — hàm `build_pose(...)` dùng góc khớp (0° = phải, 90° = xuống, 180° = trái, 270° = lên). Thêm một `PoseTemplate` vào `catalog()`.

## Gợi ý booth

- Màn hình lớn, fullscreen (`F`).
- Vạch đứng dưới đất trùng với vòng tròn cyan trên màn hình.
- Nền tối / đèn phía trước mặt giúp Pose ổn hơn.
