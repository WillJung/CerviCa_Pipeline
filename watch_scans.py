"""scans_input 폴더를 감시해 .stl 파일 저장이 끝나면 MediaPipe 기반 분석을 실행한다."""

from __future__ import annotations

import os
import struct
import threading
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc
from scipy.spatial import cKDTree
from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
from watchdog.observers import Observer

WATCH_DIR = Path(__file__).resolve().parent / "scans_input"
STABLE_INTERVAL_SEC = 1.0
STABLE_CHECKS = 3
READY_TIMEOUT_SEC = 600

_in_progress: set[str] = set()
_lock = threading.Lock()
_analyze_lock = threading.Lock()

LM_NOSE = 1
LM_GLABELLA = 9
LM_PHILTRUM = 164
LM_MENTON = 152
LM_TRAGUS_L = 234
LM_TRAGUS_R = 454
LM_NASION = 168
LM_GONION_L = 132
LM_GONION_R = 361
LM_EYE_L = 33
LM_EYE_R = 263
LM_ALAR_L = 129
LM_ALAR_R = 358
LM_MOUTH_L = 61
LM_MOUTH_R = 291

FACE_LMS = {
    "nose": LM_NOSE,
    "glabella": LM_GLABELLA,
    "philtrum": LM_PHILTRUM,
    "menton": LM_MENTON,
    "nasion": LM_NASION,
    "gonion_l": LM_GONION_L,
    "gonion_r": LM_GONION_R,
    "eye_l": LM_EYE_L,
    "eye_r": LM_EYE_R,
    "alar_l": LM_ALAR_L,
    "alar_r": LM_ALAR_R,
    "mouth_l": LM_MOUTH_L,
    "mouth_r": LM_MOUTH_R,
}

TILT_WARN_DEG = 2.0
DEV_WARN_MM = 3.0
CVA_WARN_DEG = 50.0
DEPTH_WARN_MM = 35.0

C_CVA = "green"
C_CRANIAL = "darkorange"
C_DEPTH = "c"
C_MIDLINE = "navy"
C_PUPIL = "blue"
C_ALAR = "goldenrod"
C_COMMISSURE = "purple"
C_GONION = "red"
C_ACROMION = "olive"
C_MENTON = "m"

_FACE_LANDMARKER = None
_FACE_LANDMARKER_LOCK = threading.Lock()
_FACE_LANDMARKER_MODEL = Path(__file__).resolve().parent / "face_landmarker.task"


def read_stl_binary(filename: str) -> np.ndarray:
    if not os.path.exists(filename):
        return np.array([])
    with open(filename, "rb") as f:
        f.seek(80)
        num_triangles_data = f.read(4)
        if not num_triangles_data:
            return np.array([])
        num_triangles = struct.unpack("<I", num_triangles_data)[0]
        dtype = np.dtype(
            [("normal", "f4", (3,)), ("vertices", "f4", (3, 3)), ("attr", "u2")]
        )
        data = np.fromfile(f, dtype=dtype, count=num_triangles)
        return data["vertices"].reshape(-1, 3)


def _point_to_line_foot(point: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    denom = float(np.dot(direction, direction))
    if denom == 0:
        return origin.copy()
    t = float(np.dot(point - origin, direction) / denom)
    return origin + t * direction


def _project_front_depth(pts: np.ndarray, target_width: int = 720) -> tuple[np.ndarray, dict]:
    """X-Z 정면 투영. 앞면(작은 Y)이 밝게 보이도록 grayscale depth를 만든다."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())
    x_span = max(x_max - x_min, 1e-6)
    z_span = max(z_max - z_min, 1e-6)

    width = int(target_width)
    height = max(int(round(target_width * z_span / x_span)), 64)
    col = np.clip(((x - x_min) / x_span * (width - 1)).astype(np.int32), 0, width - 1)
    row = np.clip(((z_max - z) / z_span * (height - 1)).astype(np.int32), 0, height - 1)

    y_buf = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(y_buf, row * width + col, y)
    finite = np.isfinite(y_buf)
    pixels = np.zeros(height * width, dtype=np.uint8)
    if np.any(finite):
        y_valid = y_buf[finite]
        y0, y1 = float(y_valid.min()), float(y_valid.max())
        pixels[finite] = np.clip(
            (y1 - y_buf[finite]) / (y1 - y0 + 1e-6) * 255.0, 0, 255
        ).astype(np.uint8)
    image = pixels.reshape(height, width)
    meta = {
        "x_min": x_min,
        "x_max": x_max,
        "z_min": z_min,
        "z_max": z_max,
        "width": width,
        "height": height,
    }
    return image, meta


def _prepare_face_images(depth: np.ndarray) -> list[np.ndarray]:
    import cv2

    filled = depth.copy()
    kernel = np.ones((5, 5), np.uint8)
    for _ in range(4):
        dilated = cv2.dilate(filled, kernel)
        filled = np.where(filled == 0, dilated, filled)

    variants: list[np.ndarray] = []
    for gray in (depth, filled, cv2.equalizeHist(filled)):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        blur = cv2.GaussianBlur(clahe, (5, 5), 0)
        variants.append(cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB))
        variants.append(cv2.cvtColor(255 - blur, cv2.COLOR_GRAY2RGB))
    return variants


def _pixel_to_xz(px: float, py: float, meta: dict) -> np.ndarray:
    x = meta["x_min"] + (px / max(meta["width"] - 1, 1)) * (meta["x_max"] - meta["x_min"])
    z = meta["z_max"] - (py / max(meta["height"] - 1, 1)) * (meta["z_max"] - meta["z_min"])
    return np.array([x, z], dtype=float)


def _lm_xy(landmarks, index: int) -> tuple[float, float] | None:
    if index < 0 or index >= len(landmarks):
        return None
    item = landmarks[index]
    return float(item.x), float(item.y)


def _detect_landmarks(rgb: np.ndarray):
    h, w = rgb.shape[:2]
    try:
        from mediapipe.solutions.face_mesh import FaceMesh

        with FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.2,
        ) as mesh:
            result = mesh.process(rgb)
            if result.multi_face_landmarks:
                return result.multi_face_landmarks[0].landmark, w, h
    except ImportError:
        pass

    global _FACE_LANDMARKER
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    with _FACE_LANDMARKER_LOCK:
        if _FACE_LANDMARKER is None:
            if not _FACE_LANDMARKER_MODEL.exists():
                raise FileNotFoundError(f"MediaPipe 모델이 없습니다: {_FACE_LANDMARKER_MODEL}")
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(_FACE_LANDMARKER_MODEL)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.2,
                min_face_presence_confidence=0.2,
            )
            _FACE_LANDMARKER = vision.FaceLandmarker.create_from_options(options)

    rgb = np.ascontiguousarray(rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _FACE_LANDMARKER.detect(mp_image)
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0], w, h


def _map_xz_to_3d(xz: np.ndarray, tree: cKDTree, pts: np.ndarray) -> np.ndarray:
    """X-Z에서 가까운 점들 중 가장 앞면(Y 최소)을 고른다."""
    _, idxs = tree.query(xz, k=min(16, len(pts)))
    neighbors = pts[np.atleast_1d(idxs)]
    return neighbors[np.argmin(neighbors[:, 1])]


def _lr_tilt_deg(left_pt: np.ndarray, right_pt: np.ndarray) -> float:
    """좌→우 선분의 수평 대비 기울기. 양수면 우측이 높다."""
    return float(np.degrees(np.arctan2(right_pt[2] - left_pt[2], right_pt[0] - left_pt[0])))


def _sagittal_tilt_deg(p_from: np.ndarray, p_to: np.ndarray) -> float:
    """시상면(Y-Z)에서 수평 대비 기울기."""
    return float(
        np.degrees(np.arctan2(p_to[2] - p_from[2], abs(p_to[1] - p_from[1])))
    )


def _tilt_side_label(angle_deg: float) -> str:
    if angle_deg > 0.05:
        return "R↑"
    if angle_deg < -0.05:
        return "L↑"
    return "Level"


def _subsample(pts: np.ndarray, max_pts: int = 120000) -> np.ndarray:
    if len(pts) <= max_pts:
        return pts
    return pts[:: max(1, len(pts) // max_pts)]


def _annotate_mid(ax, x1, y1, x2, y2, text: str, color: str) -> None:
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    ax.text(
        mid_x,
        mid_y,
        text,
        ha="center",
        va="center",
        fontsize=9,
        color=color,
        weight="bold",
        zorder=7,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
    )


def _draw_horizontal_ref(ax, vertex: np.ndarray, target: np.ndarray, arc_color: str) -> None:
    """시상면 꼭짓점에서 수평 점선과 축 사이 각도 호를 그린다. (Y-Z)"""
    y0, z0 = float(vertex[1]), float(vertex[2])
    y1 = float(target[1])
    z1 = float(target[2])
    ax.plot(
        [y0, y1],
        [z0, z0],
        color="gray",
        linestyle="--",
        alpha=0.6,
        linewidth=1.5,
        zorder=2,
    )
    dy = y1 - y0
    dz = z1 - z0
    if abs(dy) < 1e-6 and abs(dz) < 1e-6:
        return
    theta_h = 180.0 if dy < 0 else 0.0
    theta_a = float(np.degrees(np.arctan2(dz, dy))) % 360.0
    t1, t2 = (theta_h, theta_a) if theta_h <= theta_a else (theta_a, theta_h)
    if (t2 - t1) > 180.0:
        t1, t2 = t2, t1 + 360.0
    span = abs((float(np.degrees(np.arctan2(dz, abs(dy))))))
    radius = 34.0 if span >= 25.0 else 26.0
    ax.add_patch(
        Arc(
            (y0, z0),
            width=radius * 2,
            height=radius * 2,
            angle=0.0,
            theta1=t1,
            theta2=t2,
            color=arc_color,
            linewidth=1.6,
            zorder=6,
        )
    )


def analyze_stl(filepath: str) -> Path | None:
    """MediaPipe Face Mesh + KDTree로 얼굴을 찾고 한의학 임상 리포트를 생성한다."""
    print(f"[analyze] 분석 시작: {filepath}")
    result_path: Path | None = None
    try:
        with _analyze_lock:
            pts = read_stl_binary(filepath)
            if len(pts) == 0:
                print(f"[analyze] 실패: 유효한 3D 데이터가 없습니다. ({filepath})")
                return

            mins, maxs = pts.min(0), pts.max(0)
            mid_x = float((mins[0] + maxs[0]) / 2.0)
            mid_y = float((mins[1] + maxs[1]) / 2.0)
            mid_z = float((mins[2] + maxs[2]) / 2.0)
            height = float(maxs[2] - mins[2])
            z_cut = float(np.percentile(pts[:, 2], 25))
            face_pts = pts[(pts[:, 1] <= mid_y) & (pts[:, 2] >= z_cut)]
            if len(face_pts) < 500:
                face_pts = pts[pts[:, 1] <= mid_y]
            if len(face_pts) < 500:
                face_pts = pts

            depth, meta = _project_front_depth(face_pts)
            tree = cKDTree(pts[:, [0, 2]])

            mapped: dict[str, np.ndarray] = {}
            for rgb in _prepare_face_images(depth):
                detected = _detect_landmarks(rgb)
                if detected is None:
                    continue
                landmarks, width, height_px = detected

                def px(index: int) -> np.ndarray | None:
                    xy = _lm_xy(landmarks, index)
                    if xy is None:
                        return None
                    return _pixel_to_xz(
                        xy[0] * (width - 1), xy[1] * (height_px - 1), meta
                    )

                nose_xz = px(LM_NOSE)
                if nose_xz is None:
                    continue
                for name, index in FACE_LMS.items():
                    xz = px(index)
                    if xz is not None:
                        mapped[name] = _map_xz_to_3d(xz, tree, pts)

                traguses = []
                for index in (LM_TRAGUS_L, LM_TRAGUS_R):
                    xz = px(index)
                    if xz is not None:
                        traguses.append(_map_xz_to_3d(xz, tree, pts))
                if traguses:
                    mapped["tragus"] = max(traguses, key=lambda p: float(p[1]))
                break

            if "nose" not in mapped:
                print("[analyze] 실패: MediaPipe가 얼굴 랜드마크를 찾지 못했습니다.")
                return

            nose_pt = mapped["nose"]
            glabella_pt = mapped.get("glabella", nose_pt + np.array([0.0, 0.0, 25.0]))
            menton_pt = mapped.get("menton", nose_pt + np.array([0.0, 10.0, -50.0]))
            tragus_pt = mapped.get("tragus", nose_pt + np.array([-70.0, 90.0, 0.0]))
            nasion_pt = mapped.get("nasion", glabella_pt + np.array([0.0, 2.0, -8.0]))

            nose_x = float(nose_pt[0])
            occiput_zone = pts[
                (pts[:, 2] > mid_z) & (np.abs(pts[:, 0] - nose_x) <= 20.0)
            ]
            if len(occiput_zone) == 0:
                occiput_zone = pts[pts[:, 2] > mid_z]
            occiput_pt = occiput_zone[np.argmax(occiput_zone[:, 1])]

            z_bottom_20 = float(mins[2] + height * 0.2)
            c7_zone = pts[
                (pts[:, 2] < z_bottom_20) & (np.abs(pts[:, 0] - nose_x) <= 20.0)
            ]
            if len(c7_zone) == 0:
                c7_zone = pts[pts[:, 2] < z_bottom_20]
            c7_pt = (
                c7_zone[np.argmax(c7_zone[:, 1])]
                if len(c7_zone) > 0
                else occiput_pt - np.array([0.0, 0.0, 100.0])
            )

            z_shoulder = float(mins[2] + height * 0.15)
            shoulder_band = pts[pts[:, 2] < z_shoulder]
            left_sh = shoulder_band[shoulder_band[:, 0] < mid_x]
            right_sh = shoulder_band[shoulder_band[:, 0] > mid_x]
            acromion_l = (
                left_sh[np.argmax(left_sh[:, 2])] if len(left_sh) else None
            )
            acromion_r = (
                right_sh[np.argmax(right_sh[:, 2])] if len(right_sh) else None
            )

            cva_angle = float(
                np.degrees(
                    np.arctan2(tragus_pt[2] - c7_pt[2], abs(tragus_pt[1] - c7_pt[1]))
                )
            )
            cranial_tilt = _sagittal_tilt_deg(tragus_pt, nasion_pt)
            menton_dev_mm = float(glabella_pt[0] - menton_pt[0])
            menton_dir = (
                "Left"
                if menton_dev_mm > 0
                else "Right"
                if menton_dev_mm < 0
                else "Center"
            )

            gonion_tilt = None
            if "gonion_l" in mapped and "gonion_r" in mapped:
                gonion_tilt = _lr_tilt_deg(mapped["gonion_l"], mapped["gonion_r"])
            pupillary_tilt = None
            if "eye_l" in mapped and "eye_r" in mapped:
                pupillary_tilt = _lr_tilt_deg(mapped["eye_l"], mapped["eye_r"])
            alar_tilt = None
            if "alar_l" in mapped and "alar_r" in mapped:
                alar_tilt = _lr_tilt_deg(mapped["alar_l"], mapped["alar_r"])
            commissure_tilt = None
            if "mouth_l" in mapped and "mouth_r" in mapped:
                commissure_tilt = _lr_tilt_deg(mapped["mouth_l"], mapped["mouth_r"])
            acromion_tilt = None
            if acromion_l is not None and acromion_r is not None:
                acromion_tilt = _lr_tilt_deg(acromion_l, acromion_r)

            v_line = c7_pt - occiput_pt
            v_len = float(np.linalg.norm(v_line))
            z_lo = min(float(occiput_pt[2]), float(c7_pt[2]))
            z_hi = max(float(occiput_pt[2]), float(c7_pt[2]))
            y_back = min(float(occiput_pt[1]), float(c7_pt[1])) - 25.0
            nape = pts[
                (np.abs(pts[:, 0] - nose_x) <= 20.0)
                & (pts[:, 2] >= z_lo)
                & (pts[:, 2] <= z_hi)
                & (pts[:, 1] >= y_back)
            ]
            cervical_depth = 0.0
            deepest_pt = None
            depth_foot = occiput_pt.copy()
            if len(nape) > 0 and v_len > 0:
                bins = np.arange(z_lo, z_hi + 2.0, 2.0)
                outline = []
                for i in range(len(bins) - 1):
                    sl = nape[(nape[:, 2] >= bins[i]) & (nape[:, 2] < bins[i + 1])]
                    if len(sl):
                        outline.append(sl[np.argmax(sl[:, 1])])
                if outline:
                    outline_pts = np.vstack(outline)
                    dz = float(c7_pt[2] - occiput_pt[2])
                    if abs(dz) > 1e-6:
                        t = (outline_pts[:, 2] - occiput_pt[2]) / dz
                        y_on_line = occiput_pt[1] + t * (c7_pt[1] - occiput_pt[1])
                        inward = outline_pts[outline_pts[:, 1] <= y_on_line + 2.0]
                        if len(inward):
                            outline_pts = inward
                    line_yz = np.array(
                        [0.0, c7_pt[1] - occiput_pt[1], c7_pt[2] - occiput_pt[2]]
                    )
                    line_yz_len = float(np.linalg.norm(line_yz))
                    if line_yz_len > 1e-6:
                        offsets = np.column_stack(
                            [
                                np.zeros(len(outline_pts)),
                                outline_pts[:, 1] - occiput_pt[1],
                                outline_pts[:, 2] - occiput_pt[2],
                            ]
                        )
                        dist = (
                            np.linalg.norm(np.cross(offsets, line_yz), axis=1)
                            / line_yz_len
                        )
                        idx = int(np.argmax(dist))
                        deepest_pt = outline_pts[idx]
                        cervical_depth = float(dist[idx])
                        depth_foot = _point_to_line_foot(deepest_pt, occiput_pt, v_line)

            def _fmt_tilt(angle: float | None) -> str:
                if angle is None:
                    return "N/A"
                return f"{abs(angle):.2f}° ({_tilt_side_label(angle)})"

            print(
                f"[analyze] CVA={cva_angle:.1f}°, Cranial Tilt={abs(cranial_tilt):.2f}°, "
                f"Menton={abs(menton_dev_mm):.1f}mm ({menton_dir}), "
                f"Cervical Depth={cervical_depth:.1f}mm"
            )
            print(
                "[analyze] tilts "
                f"gonion={_fmt_tilt(gonion_tilt)}, "
                f"pupillary={_fmt_tilt(pupillary_tilt)}, "
                f"alar={_fmt_tilt(alar_tilt)}, "
                f"commissure={_fmt_tilt(commissure_tilt)}, "
                f"acromion={_fmt_tilt(acromion_tilt)}"
            )

            cloud = _subsample(pts)
            fig, (ax1, ax2, ax3) = plt.subplots(
                1,
                3,
                figsize=(26, 10),
                gridspec_kw={"width_ratios": [1.15, 1.15, 0.85]},
            )
            annot_bbox = dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2)

            ax1.scatter(
                cloud[:, 1],
                cloud[:, 2],
                s=0.2,
                color="#4a4a4a",
                alpha=0.1,
                rasterized=True,
                linewidths=0,
            )
            ax1.plot(
                nasion_pt[1],
                nasion_pt[2],
                "s",
                color=C_CRANIAL,
                markersize=9,
                zorder=5,
                label="Nasion",
            )
            ax1.plot(
                tragus_pt[1],
                tragus_pt[2],
                "o",
                color=C_CVA,
                markersize=10,
                zorder=5,
                label="Tragus",
            )
            ax1.plot(
                occiput_pt[1],
                occiput_pt[2],
                "o",
                color=C_DEPTH,
                markersize=10,
                zorder=5,
                label="Occiput",
            )
            ax1.plot(
                c7_pt[1],
                c7_pt[2],
                "o",
                color=C_CVA,
                markersize=10,
                zorder=5,
                label="C7",
            )
            _draw_horizontal_ref(ax1, c7_pt, tragus_pt, C_CVA)
            ax1.plot(
                [c7_pt[1], tragus_pt[1]],
                [c7_pt[2], tragus_pt[2]],
                color=C_CVA,
                linewidth=3.5,
                zorder=4,
                label="CVA",
            )
            _annotate_mid(
                ax1,
                c7_pt[1],
                c7_pt[2],
                tragus_pt[1],
                tragus_pt[2],
                f"{cva_angle:.1f}°",
                C_CVA,
            )
            _draw_horizontal_ref(ax1, tragus_pt, nasion_pt, C_CRANIAL)
            ax1.plot(
                [tragus_pt[1], nasion_pt[1]],
                [tragus_pt[2], nasion_pt[2]],
                color=C_CRANIAL,
                linewidth=3.5,
                zorder=4,
                label="Cranial Tilt",
            )
            _annotate_mid(
                ax1,
                tragus_pt[1],
                tragus_pt[2],
                nasion_pt[1],
                nasion_pt[2],
                f"{abs(cranial_tilt):.2f}°",
                C_CRANIAL,
            )
            ax1.plot(
                [occiput_pt[1], c7_pt[1]],
                [occiput_pt[2], c7_pt[2]],
                color="gray",
                linestyle="--",
                linewidth=3,
                zorder=3,
                label="Occiput-C7",
            )
            if deepest_pt is not None:
                ax1.plot(
                    deepest_pt[1],
                    deepest_pt[2],
                    "x",
                    color=C_DEPTH,
                    markersize=12,
                    markeredgewidth=3,
                    zorder=5,
                    label="Cervical Depth",
                )
                ax1.plot(
                    [deepest_pt[1], depth_foot[1]],
                    [deepest_pt[2], depth_foot[2]],
                    color=C_DEPTH,
                    linewidth=3,
                    zorder=4,
                )
                _annotate_mid(
                    ax1,
                    deepest_pt[1],
                    deepest_pt[2],
                    depth_foot[1],
                    depth_foot[2],
                    f"{cervical_depth:.1f} mm",
                    C_DEPTH,
                )
            ax1.invert_xaxis()
            ax1.set_aspect("equal", adjustable="box")
            ax1.set_title("I. Sagittal View", fontsize=16, fontweight="bold")
            ax1.set_xlabel("Front / Back (Y, mm)")
            ax1.set_ylabel("Up / Down (Z, mm)")
            ax1.legend(loc="upper left", fontsize=8, framealpha=0.92)

            ax2.scatter(
                cloud[:, 0],
                cloud[:, 2],
                s=0.2,
                color="#4a4a4a",
                alpha=0.1,
                rasterized=True,
                linewidths=0,
            )
            ax2.plot(
                [glabella_pt[0], menton_pt[0]],
                [glabella_pt[2], menton_pt[2]],
                color=C_MIDLINE,
                linewidth=3.5,
                zorder=4,
                label="Facial Midline",
            )
            ax2.axvline(
                x=float(glabella_pt[0]),
                color="red",
                linestyle="--",
                alpha=0.6,
                linewidth=1.5,
                zorder=3,
                label="Absolute Vertical",
            )
            _dx_mid = float(menton_pt[0] - glabella_pt[0])
            _dz_mid = float(menton_pt[2] - glabella_pt[2])
            _facial_tilt_vis = (
                float(np.degrees(np.arctan2(_dx_mid, -_dz_mid)))
                if _dz_mid != 0
                else 0.0
            )
            _theta_v = 270.0
            _theta_m = float(np.degrees(np.arctan2(_dz_mid, _dx_mid))) % 360.0
            _t1, _t2 = (
                (_theta_v, _theta_m) if _theta_v <= _theta_m else (_theta_m, _theta_v)
            )
            if (_t2 - _t1) > 180.0:
                _t1, _t2 = _t2, _t1 + 360.0
            ax2.add_patch(
                Arc(
                    (float(glabella_pt[0]), float(glabella_pt[2])),
                    width=36.0,
                    height=36.0,
                    angle=0.0,
                    theta1=_t1,
                    theta2=_t2,
                    color=C_MIDLINE,
                    linewidth=1.6,
                    zorder=6,
                )
            )
            ax2.text(
                float(glabella_pt[0]) + (12.0 if _dx_mid >= 0 else -12.0),
                float(glabella_pt[2]) - 8.0,
                f"{abs(_facial_tilt_vis):.2f}°",
                color=C_MIDLINE,
                weight="bold",
                fontsize=9,
                ha="left" if _dx_mid >= 0 else "right",
                va="top",
                zorder=7,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
            )
            ax2.plot(
                glabella_pt[0],
                glabella_pt[2],
                "s",
                color=C_MIDLINE,
                markersize=8,
                zorder=5,
            )
            ax2.plot(
                menton_pt[0],
                menton_pt[2],
                "s",
                color=C_MENTON,
                markersize=8,
                zorder=5,
            )
            ax2.text(
                menton_pt[0],
                menton_pt[2],
                f"  {abs(menton_dev_mm):.1f} mm",
                color=C_MENTON,
                weight="bold",
                fontsize=9,
                ha="left",
                va="bottom",
                zorder=7,
                bbox=annot_bbox,
            )
            if "eye_l" in mapped and "eye_r" in mapped:
                ax2.plot(
                    [mapped["eye_l"][0], mapped["eye_r"][0]],
                    [mapped["eye_l"][2], mapped["eye_r"][2]],
                    color=C_PUPIL,
                    linewidth=3.5,
                    zorder=4,
                    label="Pupillary",
                )
                if pupillary_tilt is not None:
                    _annotate_mid(
                        ax2,
                        mapped["eye_l"][0],
                        mapped["eye_l"][2],
                        mapped["eye_r"][0],
                        mapped["eye_r"][2],
                        f"{abs(pupillary_tilt):.2f}°",
                        C_PUPIL,
                    )
            if "alar_l" in mapped and "alar_r" in mapped:
                ax2.plot(
                    [mapped["alar_l"][0], mapped["alar_r"][0]],
                    [mapped["alar_l"][2], mapped["alar_r"][2]],
                    color=C_ALAR,
                    linewidth=3.5,
                    zorder=4,
                    label="Alar",
                )
                if alar_tilt is not None:
                    _annotate_mid(
                        ax2,
                        mapped["alar_l"][0],
                        mapped["alar_l"][2],
                        mapped["alar_r"][0],
                        mapped["alar_r"][2],
                        f"{abs(alar_tilt):.2f}°",
                        C_ALAR,
                    )
            if "mouth_l" in mapped and "mouth_r" in mapped:
                ax2.plot(
                    [mapped["mouth_l"][0], mapped["mouth_r"][0]],
                    [mapped["mouth_l"][2], mapped["mouth_r"][2]],
                    color=C_COMMISSURE,
                    linewidth=3.5,
                    zorder=4,
                    label="Commissure",
                )
                if commissure_tilt is not None:
                    _annotate_mid(
                        ax2,
                        mapped["mouth_l"][0],
                        mapped["mouth_l"][2],
                        mapped["mouth_r"][0],
                        mapped["mouth_r"][2],
                        f"{abs(commissure_tilt):.2f}°",
                        C_COMMISSURE,
                    )
            if "gonion_l" in mapped and "gonion_r" in mapped:
                ax2.plot(
                    [mapped["gonion_l"][0], mapped["gonion_r"][0]],
                    [mapped["gonion_l"][2], mapped["gonion_r"][2]],
                    color=C_GONION,
                    linewidth=3.5,
                    zorder=4,
                    label="Gonion",
                )
                if gonion_tilt is not None:
                    _annotate_mid(
                        ax2,
                        mapped["gonion_l"][0],
                        mapped["gonion_l"][2],
                        mapped["gonion_r"][0],
                        mapped["gonion_r"][2],
                        f"{abs(gonion_tilt):.2f}°",
                        C_GONION,
                    )
            if acromion_l is not None and acromion_r is not None:
                ax2.plot(
                    [acromion_l[0], acromion_r[0]],
                    [acromion_l[2], acromion_r[2]],
                    color=C_ACROMION,
                    linewidth=3.5,
                    zorder=4,
                    label="Acromion",
                )
                ax2.plot(
                    acromion_l[0],
                    acromion_l[2],
                    "o",
                    color=C_ACROMION,
                    markersize=8,
                    zorder=5,
                )
                ax2.plot(
                    acromion_r[0],
                    acromion_r[2],
                    "o",
                    color=C_ACROMION,
                    markersize=8,
                    zorder=5,
                )
                if acromion_tilt is not None:
                    _annotate_mid(
                        ax2,
                        acromion_l[0],
                        acromion_l[2],
                        acromion_r[0],
                        acromion_r[2],
                        f"{abs(acromion_tilt):.2f}°",
                        C_ACROMION,
                    )
            ax2.set_aspect("equal", adjustable="box")
            ax2.set_title("II. Coronal View", fontsize=16, fontweight="bold")
            ax2.set_xlabel("Left / Right (X, mm)")
            ax2.set_ylabel("Up / Down (Z, mm)")
            ax2.legend(loc="upper right", fontsize=8, framealpha=0.92)

            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis("off")
            ax3.set_title("III. Clinical Summary", fontsize=16, fontweight="bold")
            ax3.set_facecolor("#fafafa")

            sections: list[tuple[str, list[tuple[str, str, str, bool]]]] = [
                (
                    "TMJ & Mandible",
                    [
                        (
                            "Menton Deviation",
                            f"{abs(menton_dev_mm):.1f} mm ({menton_dir})",
                            C_MENTON,
                            abs(menton_dev_mm) > DEV_WARN_MM,
                        ),
                        (
                            "Gonion Level Tilt",
                            _fmt_tilt(gonion_tilt),
                            C_GONION,
                            False,
                        ),
                    ],
                ),
                (
                    "Cranio-Cervical",
                    [
                        (
                            "CVA",
                            f"{cva_angle:.1f}°",
                            C_CVA,
                            cva_angle < CVA_WARN_DEG,
                        ),
                        (
                            "Cranial Tilt",
                            f"{abs(cranial_tilt):.2f}°",
                            C_CRANIAL,
                            False,
                        ),
                        (
                            "Cervical Depth",
                            f"{cervical_depth:.1f} mm",
                            C_DEPTH,
                            False,
                        ),
                    ],
                ),
                (
                    "Cross-Horizontal",
                    [
                        (
                            "Pupillary Tilt",
                            _fmt_tilt(pupillary_tilt),
                            C_PUPIL,
                            pupillary_tilt is not None
                            and abs(pupillary_tilt) > TILT_WARN_DEG,
                        ),
                        (
                            "Alar Tilt",
                            _fmt_tilt(alar_tilt),
                            C_ALAR,
                            alar_tilt is not None and abs(alar_tilt) > TILT_WARN_DEG,
                        ),
                        (
                            "Commissure Tilt",
                            _fmt_tilt(commissure_tilt),
                            C_COMMISSURE,
                            commissure_tilt is not None
                            and abs(commissure_tilt) > TILT_WARN_DEG,
                        ),
                        (
                            "Acromion Tilt",
                            _fmt_tilt(acromion_tilt),
                            C_ACROMION,
                            False,
                        ),
                    ],
                ),
            ]

            y = 0.93
            ax3.text(
                0.5,
                y,
                "CerviCa Diagnostic Report",
                ha="center",
                va="top",
                fontsize=13,
                fontweight="bold",
                color="black",
                transform=ax3.transAxes,
            )
            y -= 0.07
            ax3.plot(
                [0.06, 0.94],
                [y + 0.02, y + 0.02],
                color="#bdbdbd",
                linewidth=1,
                transform=ax3.transAxes,
                clip_on=False,
            )
            for title, rows in sections:
                ax3.text(
                    0.06,
                    y,
                    title.upper(),
                    ha="left",
                    va="top",
                    fontsize=11,
                    fontweight="bold",
                    color="black",
                    transform=ax3.transAxes,
                )
                y -= 0.055
                for label, value, metric_color, warn in rows:
                    ax3.text(
                        0.08,
                        y,
                        label,
                        ha="left",
                        va="top",
                        fontsize=10,
                        color="black",
                        transform=ax3.transAxes,
                    )
                    ax3.text(
                        0.86 if warn else 0.94,
                        y,
                        value,
                        ha="right",
                        va="top",
                        fontsize=10,
                        fontweight="bold",
                        color=metric_color,
                        transform=ax3.transAxes,
                    )
                    if warn:
                        ax3.text(
                            0.94,
                            y,
                            "[!]",
                            ha="right",
                            va="top",
                            fontsize=10,
                            fontweight="bold",
                            color="red",
                            transform=ax3.transAxes,
                        )
                    y -= 0.048
                y -= 0.025

            ax3.text(
                0.06,
                0.04,
                "[!]  CVA < 50°  ·  Menton Dev > 3 mm  ·  Tilt > 2°",
                ha="left",
                va="bottom",
                fontsize=8,
                color="#757575",
                transform=ax3.transAxes,
            )

            fig.tight_layout()
            src = Path(filepath)
            output_path = src.with_name(f"{src.stem}_clinical_report_AI.png")
            try:
                fig.savefig(output_path, dpi=160, bbox_inches="tight")
            except OSError:
                output_path = src.with_name("clinical_report_AI.png")
                fig.savefig(output_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            print(f"[analyze] 완료: 분석 이미지 저장됨 -> {output_path}")
            result_path = output_path
    except Exception as e:
        print(f"[analyze] 에러 발생: {e}")
        traceback.print_exc()
        return None
    return result_path


def wait_until_file_ready(filepath: str) -> bool:
    previous_size = -1
    stable_count = 0
    deadline = time.monotonic() + READY_TIMEOUT_SEC

    while time.monotonic() < deadline:
        try:
            current_size = os.path.getsize(filepath)
        except OSError:
            time.sleep(STABLE_INTERVAL_SEC)
            continue

        if current_size > 0 and current_size == previous_size:
            stable_count += 1
            if stable_count >= STABLE_CHECKS and _can_open_for_read(filepath):
                return True
        else:
            stable_count = 0
            previous_size = current_size

        time.sleep(STABLE_INTERVAL_SEC)

    print(f"[watch] timeout waiting for file to finish writing: {filepath}")
    return False


def _can_open_for_read(filepath: str) -> bool:
    try:
        with open(filepath, "rb"):
            return True
    except OSError:
        return False


def _process_stl(filepath: str) -> None:
    try:
        if not wait_until_file_ready(filepath):
            return
        analyze_stl(os.path.abspath(filepath))
    finally:
        with _lock:
            _in_progress.discard(filepath)


def _is_inside_watch_dir(filepath: str) -> bool:
    try:
        Path(filepath).resolve().relative_to(WATCH_DIR.resolve())
        return True
    except ValueError:
        return False


def _schedule_if_new_stl(path: str) -> None:
    if not path.lower().endswith(".stl"):
        return
    filepath = os.path.abspath(path)
    if not _is_inside_watch_dir(filepath):
        return

    with _lock:
        if filepath in _in_progress:
            return
        _in_progress.add(filepath)

    thread = threading.Thread(
        target=_process_stl,
        args=(filepath,),
        name=f"stl-ready-{Path(filepath).name}",
        daemon=True,
    )
    thread.start()


class StlCreatedHandler(PatternMatchingEventHandler):
    def __init__(self) -> None:
        super().__init__(
            patterns=["*.stl", "*.STL"],
            ignore_directories=True,
            case_sensitive=False,
        )

    def on_created(self, event: FileSystemEvent) -> None:
        _schedule_if_new_stl(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        _schedule_if_new_stl(event.dest_path)


def process_existing_stls() -> None:
    for path in sorted(WATCH_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".stl":
            continue
        result = path.with_name(f"{path.stem}_clinical_report_AI.png")
        if result.exists():
            print(f"[watch] skip (already analyzed): {path.name}")
            continue
        print(f"[watch] queue existing file: {path.name}")
        _schedule_if_new_stl(str(path))


def main() -> None:
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    observer = Observer()
    observer.schedule(StlCreatedHandler(), str(WATCH_DIR), recursive=False)
    observer.start()
    print(f"[watch] monitoring: {WATCH_DIR}")
    print("[watch] existing .stl files will be analyzed now.")
    print("[watch] drop new .stl files here. Ctrl+C to stop.")
    process_existing_stls()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[watch] stopping...")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
