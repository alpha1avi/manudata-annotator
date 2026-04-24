"""ManuData Annotator — Trajectory Extraction (Stage 5).

Extracts 6-DoF hand trajectories, grasp signals, and kinematic
derivatives from per-frame annotations using pure NumPy / SciPy.
No ML models — runs in milliseconds per video.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from models.hand_pose import WRIST, THUMB_TIP, INDEX_TIP, FINGERTIP_IDS
from pipeline.local_vision_pipeline import FrameAnnotation

logger = logging.getLogger(__name__)

# MediaPipe landmark indices for orientation vectors
INDEX_MCP = 5
PINKY_MCP = 17
MIDDLE_MCP = 9

# Maximum gap (in frames) to interpolate across
_MAX_INTERP_GAP = 5


# ── dataclasses ───────────────────────────────────────────────────────


@dataclass
class Trajectory:
    """Full kinematic trajectory for one hand."""

    hand: str                                  # "left" | "right"
    timestamps: List[float]
    positions_2d: np.ndarray                   # (T, 2) pixel wrist coords
    positions_3d: Optional[np.ndarray]         # (T, 3) if depth available
    orientations: Optional[np.ndarray]         # (T, 3) roll, pitch, yaw
    trajectory_6dof: Optional[np.ndarray]      # (T, 6) [x, y, z, r, p, y]
    velocity: np.ndarray                       # (T-1, D)
    acceleration: np.ndarray                   # (T-2, D)
    grasp_signal: np.ndarray                   # (T,) 0=open, 1=closed
    tool_tip_trajectory: Optional[np.ndarray]  # (T, 2 or 3)
    path_length: float
    max_velocity: float
    coordinate_frame: str                      # "pixel" | "camera_relative"


@dataclass
class GraspEvent:
    """Single grasp open/close transition."""

    timestamp: float
    event_type: str                # "open" | "close"
    duration_to_next: Optional[float]


# ── extractor ─────────────────────────────────────────────────────────


class TrajectoryExtractor:
    """Extract hand trajectories from frame annotations."""

    def __init__(self, smoothing_window: int = 5) -> None:
        self.smoothing_window = smoothing_window

    # ── main entry point ──────────────────────────────────────────────

    def extract(
        self, frame_annotations: List[FrameAnnotation]
    ) -> Dict:
        """Extract trajectories for each detected hand.

        Returns:
            ``{"left": Trajectory|None, "right": Trajectory|None,
              "grasp_events": List[GraspEvent], "bimanual": bool}``
        """
        if not frame_annotations:
            return {"left": None, "right": None, "grasp_events": [], "bimanual": False}

        timestamps = [a.timestamp for a in frame_annotations]
        dt = self._median_dt(timestamps)

        results: Dict[str, Optional[Trajectory]] = {}
        all_grasp_events: List[GraspEvent] = []

        for hand_label in ("left", "right"):
            # Check if this hand appears in any frame
            count = sum(
                1 for a in frame_annotations
                for h in a.hand_pose.hands
                if h.handedness == hand_label
            )
            if count < 2:
                results[hand_label] = None
                continue

            traj = self._build_trajectory(frame_annotations, timestamps, dt, hand_label)
            results[hand_label] = traj

            if traj is not None:
                events = self.detect_grasp_events(traj.grasp_signal, timestamps)
                all_grasp_events.extend(events)

        bimanual = results["left"] is not None and results["right"] is not None

        logger.info(
            "Trajectory extraction: left=%s, right=%s, bimanual=%s, grasp_events=%d",
            "yes" if results["left"] else "no",
            "yes" if results["right"] else "no",
            bimanual,
            len(all_grasp_events),
        )

        return {
            "left": results["left"],
            "right": results["right"],
            "grasp_events": all_grasp_events,
            "bimanual": bimanual,
        }

    # ── component extractors ──────────────────────────────────────────

    def extract_wrist_trajectory(
        self, annotations: List[FrameAnnotation], hand: str = "right"
    ) -> np.ndarray:
        """Get wrist pixel position across frames.

        Gaps < 5 frames are linearly interpolated; longer gaps are NaN.

        Returns:
            ``(T, 2)`` array of pixel positions.
        """
        T = len(annotations)
        pos = np.full((T, 2), np.nan, dtype=np.float64)

        for i, ann in enumerate(annotations):
            for h in ann.hand_pose.hands:
                if h.handedness == hand:
                    wrist = h.keypoints_pixel[WRIST]
                    pos[i] = [wrist[0], wrist[1]]
                    break

        pos = self._interpolate_gaps(pos, max_gap=_MAX_INTERP_GAP)
        return pos

    def extract_3d_trajectory(
        self, annotations: List[FrameAnnotation], hand: str = "right"
    ) -> Optional[np.ndarray]:
        """Combine wrist (x, y) with depth → (x, y, z).

        Returns:
            ``(T, 3)`` array or ``None`` if no depth data.
        """
        has_depth = any(a.depth is not None for a in annotations)
        if not has_depth:
            return None

        T = len(annotations)
        pos3d = np.full((T, 3), np.nan, dtype=np.float64)

        for i, ann in enumerate(annotations):
            for h in ann.hand_pose.hands:
                if h.handedness == hand:
                    wrist = h.keypoints_pixel[WRIST]
                    x, y = wrist[0], wrist[1]
                    z = 0.0
                    if ann.depth is not None:
                        z = ann.depth.depth_at_point(x, y)
                    pos3d[i] = [x, y, z]
                    break

        pos3d = self._interpolate_gaps(pos3d, max_gap=_MAX_INTERP_GAP)
        return pos3d

    def extract_orientation(
        self, annotations: List[FrameAnnotation], hand: str = "right"
    ) -> Optional[np.ndarray]:
        """Compute hand orientation (roll, pitch, yaw) from keypoints.

        - Palm normal = (MCP_index - wrist) x (MCP_pinky - wrist)
        - Finger direction = wrist → middle_MCP
        - Convert to RPY via ``atan2``.

        Returns:
            ``(T, 3)`` array or ``None`` if insufficient data.
        """
        T = len(annotations)
        orient = np.full((T, 3), np.nan, dtype=np.float64)
        any_valid = False

        for i, ann in enumerate(annotations):
            for h in ann.hand_pose.hands:
                if h.handedness == hand and len(h.keypoints_pixel) >= 21:
                    wrist = np.array(h.keypoints_pixel[WRIST], dtype=np.float64)
                    idx_mcp = np.array(h.keypoints_pixel[INDEX_MCP], dtype=np.float64)
                    pnk_mcp = np.array(h.keypoints_pixel[PINKY_MCP], dtype=np.float64)
                    mid_mcp = np.array(h.keypoints_pixel[MIDDLE_MCP], dtype=np.float64)

                    v1 = idx_mcp - wrist   # wrist → index MCP
                    v2 = pnk_mcp - wrist   # wrist → pinky MCP

                    # Palm normal (2D cross product gives scalar z-component)
                    cross_z = v1[0] * v2[1] - v1[1] * v2[0]

                    # Finger direction
                    fdir = mid_mcp - wrist
                    fdir_len = np.linalg.norm(fdir) + 1e-8

                    yaw = math.atan2(fdir[1], fdir[0])
                    pitch = math.atan2(cross_z, fdir_len)
                    # Roll approximation from v1-v2 angle
                    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
                    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
                    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                    roll = math.acos(dot)

                    orient[i] = [roll, pitch, yaw]
                    any_valid = True
                    break

        if not any_valid:
            return None

        orient = self._interpolate_gaps(orient, max_gap=_MAX_INTERP_GAP)
        return orient

    def compute_grasp_signal(
        self, annotations: List[FrameAnnotation], hand: str = "right"
    ) -> np.ndarray:
        """Track thumb-to-index distance normalised to 0 (open) – 1 (closed).

        A sigmoid is applied so transitions are gradual.

        Returns:
            ``(T,)`` float array.
        """
        T = len(annotations)
        raw_dists = np.full(T, np.nan, dtype=np.float64)

        for i, ann in enumerate(annotations):
            for h in ann.hand_pose.hands:
                if h.handedness == hand and len(h.keypoints_pixel) >= 21:
                    thumb = np.array(h.keypoints_pixel[THUMB_TIP], dtype=np.float64)
                    index = np.array(h.keypoints_pixel[INDEX_TIP], dtype=np.float64)
                    raw_dists[i] = np.linalg.norm(thumb - index)
                    break

        # Interpolate small gaps
        raw_dists = self._interpolate_gaps_1d(raw_dists, max_gap=_MAX_INTERP_GAP)

        # Normalise: max distance → 0 (open), min → 1 (closed)
        valid = raw_dists[~np.isnan(raw_dists)]
        if len(valid) < 2:
            return np.zeros(T, dtype=np.float64)

        d_min = np.percentile(valid, 5)
        d_max = np.percentile(valid, 95)
        rng = d_max - d_min
        if rng < 1e-6:
            return np.zeros(T, dtype=np.float64)

        normalised = (d_max - raw_dists) / rng
        normalised = np.clip(normalised, 0.0, 1.0)

        # Replace NaN with 0 (open) assumption
        normalised = np.nan_to_num(normalised, nan=0.0)

        # Sigmoid smoothing: push values toward 0 or 1
        k = 6.0  # steepness
        smoothed = 1.0 / (1.0 + np.exp(-k * (normalised - 0.5)))

        return smoothed

    def detect_grasp_events(
        self, grasp_signal: np.ndarray, timestamps: List[float]
    ) -> List[GraspEvent]:
        """Find grasp open/close transitions at the 0.5 crossing.

        Returns:
            List of :class:`GraspEvent` in temporal order.
        """
        events: List[GraspEvent] = []
        T = len(grasp_signal)
        if T < 2:
            return events

        for i in range(T - 1):
            prev_val = grasp_signal[i]
            curr_val = grasp_signal[i + 1]

            if prev_val < 0.5 <= curr_val:
                events.append(GraspEvent(
                    timestamp=timestamps[i + 1],
                    event_type="close",
                    duration_to_next=None,
                ))
            elif prev_val >= 0.5 > curr_val:
                events.append(GraspEvent(
                    timestamp=timestamps[i + 1],
                    event_type="open",
                    duration_to_next=None,
                ))

        # Fill duration_to_next
        for j in range(len(events) - 1):
            events[j].duration_to_next = round(
                events[j + 1].timestamp - events[j].timestamp, 4
            )

        return events

    def smooth_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Savitzky-Golay smoothing with NaN interpolation.

        Args:
            trajectory: ``(T, D)`` or ``(T,)`` array (may contain NaN).

        Returns:
            Smoothed array of the same shape.
        """
        traj = trajectory.copy()
        win = self.smoothing_window
        if win % 2 == 0:
            win += 1  # must be odd
        if len(traj) < win:
            return traj

        if traj.ndim == 1:
            traj = self._interpolate_gaps_1d(traj, max_gap=len(traj))
            traj = np.nan_to_num(traj, nan=0.0)
            return savgol_filter(traj, win, polyorder=2)

        traj = self._interpolate_gaps(traj, max_gap=len(traj))
        traj = np.nan_to_num(traj, nan=0.0)
        for d in range(traj.shape[1]):
            traj[:, d] = savgol_filter(traj[:, d], win, polyorder=2)
        return traj

    def compute_derivatives(
        self, trajectory: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute velocity and acceleration from a position trajectory.

        Returns:
            ``(velocity, acceleration)`` — shapes ``(T-1, D)`` and ``(T-2, D)``.
        """
        if dt <= 0:
            dt = 1.0

        velocity = np.diff(trajectory, axis=0) / dt
        velocity = self.smooth_trajectory(velocity)

        acceleration = np.diff(velocity, axis=0) / dt
        acceleration = self.smooth_trajectory(acceleration)

        return velocity, acceleration

    # ── private: full trajectory builder ──────────────────────────────

    def _build_trajectory(
        self,
        annotations: List[FrameAnnotation],
        timestamps: List[float],
        dt: float,
        hand: str,
    ) -> Optional[Trajectory]:
        """Build a complete :class:`Trajectory` for one hand."""
        pos_2d = self.extract_wrist_trajectory(annotations, hand)

        # Check we have enough valid data
        valid_count = int(np.sum(~np.isnan(pos_2d[:, 0])))
        if valid_count < 2:
            return None

        pos_2d_smooth = self.smooth_trajectory(pos_2d)

        pos_3d = self.extract_3d_trajectory(annotations, hand)
        if pos_3d is not None:
            pos_3d = self.smooth_trajectory(pos_3d)

        orient = self.extract_orientation(annotations, hand)
        if orient is not None:
            orient = self.smooth_trajectory(orient)

        # 6-DoF
        traj_6dof: Optional[np.ndarray] = None
        if pos_3d is not None and orient is not None:
            traj_6dof = np.hstack([pos_3d, orient])  # (T, 6)

        # Derivatives on the best available position data
        primary = pos_3d if pos_3d is not None else pos_2d_smooth
        velocity, acceleration = self.compute_derivatives(primary, dt)

        # Grasp
        grasp_sig = self.compute_grasp_signal(annotations, hand)

        # Tool-tip trajectory (index fingertip as proxy)
        tool_tip = self._extract_tool_tip(annotations, hand, pos_3d is not None)

        # Path length & max velocity
        valid_mask = ~np.isnan(primary[:, 0])
        if valid_mask.sum() >= 2:
            diffs = np.diff(primary[valid_mask], axis=0)
            step_lengths = np.linalg.norm(diffs, axis=1)
            path_length = float(np.nansum(step_lengths))
        else:
            path_length = 0.0

        vel_mags = np.linalg.norm(velocity, axis=1)
        max_vel = float(np.nanmax(vel_mags)) if len(vel_mags) > 0 else 0.0

        coord_frame = "camera_relative" if pos_3d is not None else "pixel"

        return Trajectory(
            hand=hand,
            timestamps=timestamps,
            positions_2d=pos_2d_smooth,
            positions_3d=pos_3d,
            orientations=orient,
            trajectory_6dof=traj_6dof,
            velocity=velocity,
            acceleration=acceleration,
            grasp_signal=grasp_sig,
            tool_tip_trajectory=tool_tip,
            path_length=round(path_length, 2),
            max_velocity=round(max_vel, 2),
            coordinate_frame=coord_frame,
        )

    # ── helpers ───────────────────────────────────────────────────────

    def _extract_tool_tip(
        self,
        annotations: List[FrameAnnotation],
        hand: str,
        has_depth: bool,
    ) -> Optional[np.ndarray]:
        """Extract index-fingertip trajectory as a tool-tip proxy."""
        T = len(annotations)
        dim = 3 if has_depth else 2
        tip = np.full((T, dim), np.nan, dtype=np.float64)

        for i, ann in enumerate(annotations):
            for h in ann.hand_pose.hands:
                if h.handedness == hand and len(h.keypoints_pixel) >= 21:
                    idx_tip = h.keypoints_pixel[INDEX_TIP]
                    x, y = idx_tip[0], idx_tip[1]
                    if has_depth and ann.depth is not None:
                        z = ann.depth.depth_at_point(x, y)
                        tip[i] = [x, y, z]
                    else:
                        tip[i, :2] = [x, y]
                    break

        valid = int(np.sum(~np.isnan(tip[:, 0])))
        if valid < 2:
            return None

        tip = self._interpolate_gaps(tip, max_gap=_MAX_INTERP_GAP)
        return self.smooth_trajectory(tip)

    @staticmethod
    def _median_dt(timestamps: List[float]) -> float:
        """Compute median time-step between consecutive timestamps."""
        if len(timestamps) < 2:
            return 0.5  # default
        dts = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        return float(np.median(dts))

    @staticmethod
    def _interpolate_gaps(arr: np.ndarray, max_gap: int) -> np.ndarray:
        """Linearly interpolate NaN gaps <= *max_gap* frames (2-D)."""
        result = arr.copy()
        T, D = result.shape

        for d in range(D):
            col = result[:, d]
            nan_mask = np.isnan(col)
            if not nan_mask.any():
                continue

            # Find gap starts and lengths
            changes = np.diff(nan_mask.astype(int))
            gap_starts = np.where(changes == 1)[0] + 1
            gap_ends = np.where(changes == -1)[0] + 1

            # Handle edge cases
            if nan_mask[0]:
                gap_starts = np.concatenate([[0], gap_starts])
            if nan_mask[-1]:
                gap_ends = np.concatenate([gap_ends, [T]])

            for gs, ge in zip(gap_starts, gap_ends):
                gap_len = ge - gs
                if gap_len > max_gap:
                    continue  # leave as NaN
                # Interpolate
                left = col[gs - 1] if gs > 0 else np.nan
                right = col[ge] if ge < T else np.nan
                if np.isnan(left) and np.isnan(right):
                    continue
                if np.isnan(left):
                    left = right
                if np.isnan(right):
                    right = left
                result[gs:ge, d] = np.linspace(left, right, gap_len)

        return result

    @staticmethod
    def _interpolate_gaps_1d(arr: np.ndarray, max_gap: int) -> np.ndarray:
        """1-D version of gap interpolation."""
        result = arr.copy()
        T = len(result)
        nan_mask = np.isnan(result)
        if not nan_mask.any():
            return result

        changes = np.diff(nan_mask.astype(int))
        gap_starts = np.where(changes == 1)[0] + 1
        gap_ends = np.where(changes == -1)[0] + 1

        if nan_mask[0]:
            gap_starts = np.concatenate([[0], gap_starts])
        if nan_mask[-1]:
            gap_ends = np.concatenate([gap_ends, [T]])

        for gs, ge in zip(gap_starts, gap_ends):
            gap_len = ge - gs
            if gap_len > max_gap:
                continue
            left = result[gs - 1] if gs > 0 else np.nan
            right = result[ge] if ge < T else np.nan
            if np.isnan(left) and np.isnan(right):
                continue
            if np.isnan(left):
                left = right
            if np.isnan(right):
                right = left
            result[gs:ge] = np.linspace(left, right, gap_len)

        return result


# ── standalone test with synthetic data ───────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.hand_pose import HandDetection, HandPoseResult
    from config import AnnotatorConfig
    from utils.logging_config import setup_logging

    setup_logging(verbose=True)

    # Create synthetic annotations (30 frames, ~15 seconds at 2 fps)
    T = 30
    timestamps = [i * 0.5 for i in range(T)]

    annotations: List[FrameAnnotation] = []
    for i in range(T):
        # Simulate a right hand moving across the frame
        x = 300 + int(100 * math.sin(i * 0.3))
        y = 400 + int(50 * math.cos(i * 0.2))

        # Simulate thumb-index distance (grasping cycle)
        thumb_x = x + int(30 * math.cos(i * 0.5))
        thumb_y = y - 20
        index_x = x + int(30 * math.cos(i * 0.5 + 1.5))
        index_y = y - 40

        # Build 21 keypoints (only wrist, thumb tip, index tip matter for test)
        kp = [(x, y)] + [(x, y)] * 3 + [(thumb_x, thumb_y)]  # 0-4
        kp += [(x + 20, y - 30)] + [(x, y)] * 2 + [(index_x, index_y)]  # 5-8
        kp += [(x + 5, y - 35)] + [(x, y)] * 2 + [(x + 5, y - 50)]  # 9-12
        kp += [(x - 10, y - 25)] + [(x, y)] * 2 + [(x - 10, y - 45)]  # 13-16
        kp += [(x - 25, y - 20)] + [(x, y)] * 2 + [(x - 25, y - 40)]  # 17-20

        hand = HandDetection(
            handedness="right",
            confidence=0.9,
            keypoints_2d=[(px / 640, py / 480) for px, py in kp],
            keypoints_pixel=kp,
            bbox=(x - 50, y - 60, x + 50, y + 20),
        )

        ann = FrameAnnotation(
            timestamp=timestamps[i],
            frame_path=f"frame_{i:04d}.jpg",
            hand_pose=HandPoseResult(hands=[hand], num_hands=1, hand_visibility="right_only"),
            objects=[],
            depth=None,
            interactions=[],
            activity_score=0.5,
            hand_visibility="right_only",
            processing_time_ms=5.0,
        )
        annotations.append(ann)

    extractor = TrajectoryExtractor(smoothing_window=5)
    result = extractor.extract(annotations)

    print(f"\nBimanual: {result['bimanual']}")
    print(f"Grasp events: {len(result['grasp_events'])}")

    for hand_label in ("left", "right"):
        traj = result[hand_label]
        if traj is None:
            print(f"\n{hand_label}: not detected")
            continue
        print(f"\n{hand_label} hand trajectory:")
        print(f"  Frames:       {len(traj.timestamps)}")
        print(f"  Path length:  {traj.path_length:.1f} px")
        print(f"  Max velocity: {traj.max_velocity:.1f} px/s")
        print(f"  Coord frame:  {traj.coordinate_frame}")
        print(f"  Velocity shape:     {traj.velocity.shape}")
        print(f"  Acceleration shape: {traj.acceleration.shape}")
        print(f"  Grasp signal range: [{traj.grasp_signal.min():.2f}, {traj.grasp_signal.max():.2f}]")
        if traj.orientations is not None:
            print(f"  Orientations shape: {traj.orientations.shape}")

    for ev in result["grasp_events"]:
        print(f"  Grasp event: {ev.event_type} at t={ev.timestamp:.2f}s (next in {ev.duration_to_next}s)")
