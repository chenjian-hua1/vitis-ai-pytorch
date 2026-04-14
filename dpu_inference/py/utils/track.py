# =============================================================================
# botsort_core.py  —  Standalone BoT-SORT tracker (no extra dependencies)
# =============================================================================
#
# MIT License
#
# Original work:
#   BoT-SORT: Robust Associations Multi-Pedestrian Tracking
#   Copyright (c) 2022  Nir Aharon, Roy Orfaig, Ben-Zion Bobrovsky
#   https://github.com/NirAharon/BoT-SORT
#   arXiv: https://arxiv.org/abs/2206.14651
#
# Derivative works this file is based on:
#   ByteTrack  — Copyright (c) 2022 Zhang et al.
#                https://github.com/ifzhang/ByteTrack  (MIT)
#   StrongSORT — Copyright (c) 2022 Du et al.
#                https://github.com/dyhBUPT/StrongSORT  (MIT)
#
# This standalone re-implementation:
#   Copyright (c) 2024  (your name / organisation)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# =============================================================================
# Notes
# -----
# * No dependency on the original BoT-SORT repo; requires only numpy, scipy,
#   and opencv-python.
# * ReID model is NOT included; external embeddings can be passed via `feats`.
# * GMC ships three pure-Python backends: sparseOptFlow / ecc / orb.
# * Entry point: BoTSORT.update(boxes, scores, classes, img, feats)
#
# Quick install:
#   pip install numpy scipy opencv-python
# =============================================================================

from __future__ import annotations

import cv2
import numpy as np
from collections import deque
from enum import IntEnum
from scipy.optimize import linear_sum_assignment

__all__ = ["BoTSORT", "BoTSORTConfig", "Track"]
__version__ = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

class BoTSORTConfig:
    """Central configuration object for all tracker hyper-parameters."""

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float  = 0.1,
        new_track_thresh: float  = 0.6,
        track_buffer: int        = 30,
        match_thresh: float      = 0.8,
        proximity_thresh: float  = 0.5,
        appearance_thresh: float = 0.25,
        frame_rate: int          = 30,
        cmc_method: str          = "sparseOptFlow",  # "sparseOptFlow"|"ecc"|"orb"|"none"
        fuse_score: bool         = True,
    ):
        self.track_high_thresh  = track_high_thresh
        self.track_low_thresh   = track_low_thresh
        self.new_track_thresh   = new_track_thresh
        self.track_buffer       = track_buffer
        self.match_thresh       = match_thresh
        self.proximity_thresh   = proximity_thresh
        self.appearance_thresh  = appearance_thresh
        self.frame_rate         = frame_rate
        self.cmc_method         = cmc_method
        self.fuse_score         = fuse_score


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Kalman Filter  (8-dim state: cx, cy, w, h, vx, vy, vw, vh)
# ─────────────────────────────────────────────────────────────────────────────

class KalmanFilter:
    """
    Standard Kalman Filter for bounding-box tracking.

    State vector  : [cx, cy, w, h, vx, vy, vw, vh]
    Measurement   : [cx, cy, w, h]
    """

    def __init__(self):
        ndim = 4
        dt   = 1.0

        # Transition matrix F
        self._F = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = dt

        # Measurement matrix H
        self._H = np.eye(ndim, 2 * ndim)

        # Process noise weights
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    # ── public ──────────────────────────────────────────────────────────────

    def initiate(self, measurement: np.ndarray):
        """
        Parameters
        ----------
        measurement : [cx, cy, w, h]

        Returns
        -------
        mean, covariance
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean     = np.concatenate([mean_pos, mean_vel])

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        Q   = np.diag(np.square(std))
        mean       = self._F @ mean
        covariance = self._F @ covariance @ self._F.T + Q
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        R           = np.diag(np.square(std))
        S           = self._H @ covariance @ self._H.T + R
        K           = covariance @ self._H.T @ np.linalg.inv(S)
        innovation  = measurement - self._H @ mean
        mean        = mean + K @ innovation
        covariance  = covariance - K @ S @ K.T
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        R              = np.diag(np.square(std))
        proj_mean      = self._H @ mean
        proj_cov       = self._H @ covariance @ self._H.T + R
        return proj_mean, proj_cov

    @staticmethod
    def multi_predict(stracks: list, kalman_filter: "KalmanFilter"):
        if not stracks:
            return
        multi_mean = np.asarray([t.mean.copy() for t in stracks])
        multi_cov  = np.asarray([t.covariance  for t in stracks])

        # zero out velocity for non-tracked (lost) states
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][4] = 0
                multi_mean[i][5] = 0
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0

        # batch predict
        F   = kalman_filter._F
        std_wp = kalman_filter._std_weight_position
        std_wv = kalman_filter._std_weight_velocity

        for i in range(len(stracks)):
            m = multi_mean[i]
            std = [
                std_wp * m[2], std_wp * m[3], std_wp * m[2], std_wp * m[3],
                std_wv * m[2], std_wv * m[3], std_wv * m[2], std_wv * m[3],
            ]
            Q = np.diag(np.square(std))
            multi_mean[i]  = F @ m
            multi_cov[i]   = F @ multi_cov[i] @ F.T + Q

        for st, mean, cov in zip(stracks, multi_mean, multi_cov):
            st.mean       = mean
            st.covariance = cov


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Track state & base Track class
# ─────────────────────────────────────────────────────────────────────────────

class TrackState(IntEnum):
    New      = 0
    Tracked  = 1
    Lost     = 2
    Removed  = 3


class Track:
    """Single tracked object, managing its full lifecycle and Kalman state."""

    _count = 0  # global ID counter

    @classmethod
    def reset_id(cls):
        cls._count = 0

    @classmethod
    def _next_id(cls):
        cls._count += 1
        return cls._count

    def __init__(self, tlbr: np.ndarray, score: float, cls: int,
                 feat: np.ndarray | None = None):
        # raw detection box stored as-is; Kalman state uses cx,cy,w,h
        self._tlbr       = tlbr.copy()
        self.score       = score
        self.cls         = int(cls)
        self.track_id    = -1           # assigned in activate()
        self.state       = TrackState.New
        self.frame_id    = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self.is_activated = False

        # Kalman state
        self.mean       = None
        self.covariance = None

        # ReID feature smoothing (exponential moving average)
        self.smooth_feat   = None
        self._feat_history = deque(maxlen=50)
        self.alpha         = 0.9        # EMA decay coefficient
        if feat is not None:
            self._update_feat(feat)

    # ── bbox helpers ────────────────────────────────────────────────────────

    @property
    def tlwh(self) -> np.ndarray:
        t = self._tlbr
        return np.array([t[0], t[1], t[2] - t[0], t[3] - t[1]])

    @property
    def tlbr(self) -> np.ndarray:
        if self.mean is None:
            return self._tlbr.copy()
        cx, cy, w, h = self.mean[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    @property
    def xywh(self) -> np.ndarray:
        t = self.tlbr
        return np.array([(t[0] + t[2]) / 2, (t[1] + t[3]) / 2,
                         t[2] - t[0], t[3] - t[1]])

    @staticmethod
    def tlbr_to_xywh(tlbr: np.ndarray) -> np.ndarray:
        return np.array([(tlbr[0] + tlbr[2]) / 2, (tlbr[1] + tlbr[3]) / 2,
                         tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]])

    # ── feature ─────────────────────────────────────────────────────────────

    def _update_feat(self, feat: np.ndarray):
        feat = feat / (np.linalg.norm(feat) + 1e-12)
        self._feat_history.append(feat)
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            self.smooth_feat /= (np.linalg.norm(self.smooth_feat) + 1e-12)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        self.track_id     = Track._next_id()
        self.state        = TrackState.Tracked
        self.frame_id     = frame_id
        self.start_frame  = frame_id
        self.tracklet_len = 1
        self.is_activated = True

        meas               = Track.tlbr_to_xywh(self._tlbr)
        self.mean, self.covariance = kalman_filter.initiate(meas)

    def re_activate(self, new_track: "Track", kalman_filter: KalmanFilter,
                    frame_id: int, new_id: bool = False):
        meas = Track.tlbr_to_xywh(new_track._tlbr)
        self.mean, self.covariance = kalman_filter.update(
            self.mean, self.covariance, meas)

        if new_track.smooth_feat is not None:
            self._update_feat(new_track.smooth_feat)

        self.score        = new_track.score
        self.cls          = new_track.cls
        self.state        = TrackState.Tracked
        self.is_activated = True
        self.frame_id     = frame_id
        self.tracklet_len = 0
        if new_id:
            self.track_id = Track._next_id()

    def update(self, new_track: "Track", kalman_filter: KalmanFilter, frame_id: int):
        meas = Track.tlbr_to_xywh(new_track._tlbr)
        self.mean, self.covariance = kalman_filter.update(
            self.mean, self.covariance, meas)

        if new_track.smooth_feat is not None:
            self._update_feat(new_track.smooth_feat)

        self.score        = new_track.score
        self.cls          = new_track.cls
        self.state        = TrackState.Tracked
        self.is_activated = True
        self.frame_id     = frame_id
        self.tracklet_len += 1

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def predict(self, kalman_filter: KalmanFilter):
        mean = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean[4] = mean[5] = mean[6] = mean[7] = 0
        self.mean, self.covariance = kalman_filter.predict(mean, self.covariance)

    @property
    def end_frame(self) -> int:
        return self.frame_id

    def __repr__(self):
        return (f"Track(id={self.track_id}, state={self.state.name}, "
                f"cls={self.cls}, score={self.score:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Matching utilities
# ─────────────────────────────────────────────────────────────────────────────

def _box_iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between two sets of [x1,y1,x2,y2] boxes. a:(M,4) b:(N,4)"""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w  = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h  = np.clip(inter_y2 - inter_y1, 0, None)
    inter    = inter_w * inter_h

    union    = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-9)


def iou_distance(tracks: list[Track], detections: list[Track]) -> np.ndarray:
    """Cost matrix = 1 - IoU."""
    if not tracks or not detections:
        return np.empty((len(tracks), len(detections)))
    a = np.stack([t.tlbr for t in tracks])
    b = np.stack([d.tlbr for d in detections])
    return 1.0 - _box_iou_batch(a, b)


def embedding_distance(tracks: list[Track], detections: list[Track]) -> np.ndarray:
    """Cosine distance matrix (1 - cosine similarity)."""
    M, N = len(tracks), len(detections)
    dist = np.ones((M, N))
    t_feats = np.stack([t.smooth_feat for t in tracks])     # (M, D)
    d_feats = np.stack([d.smooth_feat for d in detections]) # (N, D)
    # cosine similarity
    sim  = t_feats @ d_feats.T
    dist = 1.0 - sim
    return dist


def fuse_score(cost_matrix: np.ndarray, detections: list[Track]) -> np.ndarray:
    """Fuse detection confidence into the IoU cost matrix (BoT-SORT paper Eq. 2)."""
    if cost_matrix.size == 0:
        return cost_matrix
    scores = np.array([d.score for d in detections])
    iou_sim = 1 - cost_matrix
    det_sim = iou_sim * scores[np.newaxis, :]
    return 1 - det_sim


def linear_assignment(cost_matrix: np.ndarray,
                       thresh: float) -> tuple[list, list, list]:
    """
    Hungarian algorithm assignment; pairs with cost > thresh are rejected.

    Returns
    -------
    matches         : list of (row, col)
    unmatched_rows  : list of row indices
    unmatched_cols  : list of col indices
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    rows, cols = linear_sum_assignment(cost_matrix)
    matches, u_rows, u_cols = [], [], []

    matched_mask_r = np.zeros(cost_matrix.shape[0], dtype=bool)
    matched_mask_c = np.zeros(cost_matrix.shape[1], dtype=bool)

    for r, c in zip(rows, cols):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            matched_mask_r[r] = True
            matched_mask_c[c] = True

    u_rows = np.where(~matched_mask_r)[0].tolist()
    u_cols = np.where(~matched_mask_c)[0].tolist()
    return matches, u_rows, u_cols


# ─────────────────────────────────────────────────────────────────────────────
# 5.  GMC — Global Motion Compensation (Python implementations)
# ─────────────────────────────────────────────────────────────────────────────

class GMC:
    """
    Camera motion compensation (GMC).

    Methods
    -------
    sparseOptFlow  : Lucas-Kanade sparse optical flow (fastest, recommended)
    ecc            : Enhanced Correlation Coefficient
    orb            : ORB feature matching
    none           : disabled (static camera)
    """

    def __init__(self, method: str = "sparseOptFlow", downscale: float = 2.0):
        self.method    = method
        self.downscale = max(1.0, downscale)
        self._prev_frame: np.ndarray | None = None
        self._prev_kps: list | None         = None
        self._prev_desc: np.ndarray | None  = None

        if method == "orb":
            self._detector = cv2.ORB_create(nfeatures=1000)
            self._matcher  = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif method == "ecc":
            self._warp_mode = cv2.MOTION_EUCLIDEAN
            self._criteria  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                1000, 1e-5)

    def reset_params(self):
        self._prev_frame = None
        self._prev_kps   = None
        self._prev_desc  = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        frame : BGR image (H, W, 3)

        Returns
        -------
        H : (2, 3) affine warp matrix  (identity if cannot estimate)
        """
        identity = np.eye(2, 3, dtype=np.float32)

        if frame is None or frame.size == 0:
            return identity

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = gray.shape
        dw   = int(w / self.downscale)
        dh   = int(h / self.downscale)
        gray = cv2.resize(gray, (dw, dh))

        if self.method == "sparseOptFlow":
            warp = self._apply_sparse_optflow(gray)
        elif self.method == "ecc":
            warp = self._apply_ecc(gray)
        elif self.method == "orb":
            warp = self._apply_orb(gray)
        else:
            warp = identity

        self._prev_frame = gray.copy()
        return warp

    def _apply_sparse_optflow(self, gray: np.ndarray) -> np.ndarray:
        identity = np.eye(2, 3, dtype=np.float32)
        if self._prev_frame is None:
            return identity

        kps = cv2.goodFeaturesToTrack(
            self._prev_frame, maxCorners=200, qualityLevel=0.01,
            minDistance=7, blockSize=7)
        if kps is None or len(kps) < 4:
            return identity

        kps_next, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_frame, gray, kps, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        good_prev = kps[status[:, 0] == 1]
        good_next = kps_next[status[:, 0] == 1]
        if len(good_prev) < 4:
            return identity

        H, _ = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC)
        return H if H is not None else identity

    def _apply_ecc(self, gray: np.ndarray) -> np.ndarray:
        identity = np.eye(2, 3, dtype=np.float32)
        if self._prev_frame is None:
            return identity
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(
                self._prev_frame, gray, warp, self._warp_mode,
                self._criteria, None, 1)
        except cv2.error:
            pass
        return warp

    def _apply_orb(self, gray: np.ndarray) -> np.ndarray:
        identity = np.eye(2, 3, dtype=np.float32)
        kps, desc = self._detector.detectAndCompute(gray, None)
        if desc is None or len(kps) < 4:
            self._prev_kps   = kps
            self._prev_desc  = desc
            return identity

        if self._prev_desc is None:
            self._prev_kps   = kps
            self._prev_desc  = desc
            return identity

        matches = self._matcher.knnMatch(self._prev_desc, desc, k=2)
        good    = [m for m, n in matches if m.distance < 0.9 * n.distance]
        if len(good) < 4:
            self._prev_kps   = kps
            self._prev_desc  = desc
            return identity

        src_pts = np.float32([self._prev_kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _    = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

        self._prev_kps   = kps
        self._prev_desc  = desc
        return H if H is not None else identity

    @staticmethod
    def apply_to_tracks(tracks: list[Track], H: np.ndarray):
        """Apply affine warp matrix H to the Kalman state of all tracks."""
        if H is None or np.allclose(H, np.eye(2, 3)):
            return
        for t in tracks:
            if t.mean is None:
                continue
            cx, cy = t.mean[0], t.mean[1]
            pt     = np.array([[cx, cy]], dtype=np.float32)
            pt_new = cv2.transform(pt.reshape(1, 1, 2), H).reshape(2)
            t.mean[0] = pt_new[0]
            t.mean[1] = pt_new[1]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main Tracker
# ─────────────────────────────────────────────────────────────────────────────

class BoTSORT:
    """
    BoT-SORT multi-object tracker (no built-in ReID model; accepts external embeddings).

    Usage
    -----
    tracker = BoTSORT(BoTSORTConfig(), frame_rate=30)

    for frame in video:
        boxes   = ...  # np.ndarray (N, 4) [x1,y1,x2,y2]
        scores  = ...  # np.ndarray (N,)
        classes = ...  # np.ndarray (N,) int
        feats   = None # np.ndarray (N, D) or None

        tracks = tracker.update(boxes, scores, classes, frame, feats)
        for t in tracks:
            print(t.track_id, t.tlbr, t.score, t.cls)
    """

    def __init__(self, cfg: BoTSORTConfig | None = None, frame_rate: int = 30):
        self.cfg        = cfg or BoTSORTConfig(frame_rate=frame_rate)
        self.frame_id   = 0

        # track lists
        self.tracked_stracks:  list[Track] = []
        self.lost_stracks:     list[Track] = []
        self.removed_stracks:  list[Track] = []

        # max frames a lost track stays alive before removal
        self.max_time_lost = int(frame_rate / 30.0 * self.cfg.track_buffer)

        self.kalman_filter = KalmanFilter()
        self.gmc           = GMC(method=self.cfg.cmc_method)

        Track.reset_id()

    # ── main entry ───────────────────────────────────────────────────────────

    def update(
        self,
        boxes:   np.ndarray,
        scores:  np.ndarray,
        classes: np.ndarray,
        img:     np.ndarray | None = None,
        feats:   np.ndarray | None = None,
    ) -> list[Track]:
        """
        Parameters
        ----------
        boxes   : (N, 4) float32  [x1, y1, x2, y2]
        scores  : (N,)   float32  confidence
        classes : (N,)   int      class id
        img     : BGR frame (for GMC); optional
        feats   : (N, D) float32  ReID embeddings; optional

        Returns
        -------
        List of active Track objects.
        """
        self.frame_id += 1
        cfg = self.cfg

        # ── 0. Build detection Track objects ────────────────────────────────
        high_mask = scores >= cfg.track_high_thresh
        low_mask  = (scores >= cfg.track_low_thresh) & ~high_mask

        def _make_dets(mask):
            dets = []
            for i in np.where(mask)[0]:
                f = feats[i] if feats is not None else None
                dets.append(Track(boxes[i], float(scores[i]), int(classes[i]), f))
            return dets

        detections_high = _make_dets(high_mask)
        detections_low  = _make_dets(low_mask)

        # ── 1. Kalman predict ────────────────────────────────────────────────
        strack_pool = _join_lists(self.tracked_stracks, self.lost_stracks)
        KalmanFilter.multi_predict(strack_pool, self.kalman_filter)

        # ── 2. GMC — camera motion compensation ─────────────────────────────
        if img is not None:
            H = self.gmc.apply(img)
            GMC.apply_to_tracks(strack_pool, H)

        # ── 3. First association  (high-score detections) ────────────────────
        dists = iou_distance(strack_pool, detections_high)
        if cfg.fuse_score:
            dists = fuse_score(dists, detections_high)

        # Fuse appearance distance when ReID embeddings are available.
        # Only pairs whose IoU exceeds proximity_thresh are eligible;
        # pairs with cosine distance above appearance_thresh are also blocked.
        if feats is not None and any(d.smooth_feat is not None for d in detections_high):
            emb_dists = embedding_distance(strack_pool, detections_high)
            iou_sim   = 1 - iou_distance(strack_pool, detections_high)
            emb_dists[iou_sim < cfg.proximity_thresh]    = 1.0  # too far apart — block ReID
            emb_dists[emb_dists > cfg.appearance_thresh] = 1.0  # too dissimilar — block ReID
            dists = np.minimum(dists, emb_dists)

        matches, u_track, u_det_high = linear_assignment(dists, thresh=cfg.match_thresh)

        activated_stracks    = []
        refind_stracks       = []

        for itrk, idet in matches:
            track = strack_pool[itrk]
            det   = detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.kalman_filter, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.kalman_filter, self.frame_id)
                refind_stracks.append(track)

        # ── 4. Second association (low-score detections) ──────────────────────
        r_tracked = [strack_pool[i] for i in u_track
                     if strack_pool[i].state == TrackState.Tracked]
        dists2    = iou_distance(r_tracked, detections_low)
        matches2, u_track2, _ = linear_assignment(dists2, thresh=0.5)

        for itrk, idet in matches2:
            track = r_tracked[itrk]
            det   = detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.kalman_filter, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.kalman_filter, self.frame_id)
                refind_stracks.append(track)

        # mark lost
        lost_stracks = []
        for i in u_track2:
            track = r_tracked[i]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # ── 5. Init new tracks (unmatched high-score dets) ──────────────────
        for i in u_det_high:
            det = detections_high[i]
            if det.score >= cfg.new_track_thresh:
                det.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(det)

        # ── 6. Remove long-lost tracks ───────────────────────────────────────
        removed_stracks = []
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # ── 7. Update tracker state ──────────────────────────────────────────
        self.tracked_stracks = [t for t in self.tracked_stracks
                                 if t.state == TrackState.Tracked]
        self.tracked_stracks = _join_lists(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = _join_lists(self.tracked_stracks, refind_stracks)

        self.lost_stracks    = _sub_lists(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks   += lost_stracks
        self.lost_stracks    = _sub_lists(self.lost_stracks, self.removed_stracks)

        self.removed_stracks += removed_stracks
        # cap removed list size to avoid unbounded growth
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-500:]

        # ── 8. Return only confirmed active tracks ───────────────────────────
        return [t for t in self.tracked_stracks if t.is_activated]

    def reset(self):
        """Reset the tracker; call this when starting a new video."""
        self.tracked_stracks  = []
        self.lost_stracks     = []
        self.removed_stracks  = []
        self.frame_id         = 0
        self.gmc.reset_params()
        Track.reset_id()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _join_lists(a: list[Track], b: list[Track]) -> list[Track]:
    exists = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in exists]


def _sub_lists(a: list[Track], b: list[Track]) -> list[Track]:
    remove_ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in remove_ids]


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Example usage  (python botsort_core.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("BoT-SORT standalone core — quick smoke test")
    print("=" * 60)

    cfg     = BoTSORTConfig(cmc_method="none")  # disable GMC when no real video frames
    tracker = BoTSORT(cfg, frame_rate=30)

    # simulate 5 frames with 3 detections each
    for fid in range(1, 6):
        boxes = np.array([
            [100 + fid,  50 + fid, 200 + fid, 150 + fid],
            [300 + fid, 200 + fid, 420 + fid, 320 + fid],
            [500 + fid,  10 + fid, 600 + fid, 110 + fid],
        ], dtype=np.float32)
        scores  = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        classes = np.array([0, 0, 1],        dtype=np.int32)

        tracks = tracker.update(boxes, scores, classes)
        print(f"\nFrame {fid:02d}  — active tracks: {len(tracks)}")
        for t in tracks:
            print(f"  {t}")