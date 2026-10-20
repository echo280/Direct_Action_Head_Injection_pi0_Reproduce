"""Metric, unaugmented camera geometry. No learned model dependencies."""
from dataclasses import dataclass
import numpy as np


def lift_pixel(uv, depth_m, intrinsics, camera_to_base):
    depth = np.asarray(depth_m, dtype=np.float64).squeeze()
    k = np.asarray(intrinsics, dtype=np.float64)
    transform = np.asarray(camera_to_base, dtype=np.float64)
    point = np.asarray(uv, dtype=np.float64)
    if depth.ndim != 2 or k.shape != (3, 3) or transform.shape != (4, 4) or point.shape != (2,):
        raise ValueError("Expected depth[H,W], K[3,3], T_base_cam[4,4], uv[2]")
    if not np.isfinite(point).all() or not np.isfinite(k).all() or not np.isfinite(transform).all():
        raise ValueError("Non-finite camera geometry")
    if not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("Extrinsic must be homogeneous")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError("Extrinsic rotation is not in SO(3)")
    # Explicit nearest-pixel lookup, not Python's ties-to-even rounding.
    u, v = np.floor(point + 0.5).astype(int)
    if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
        raise ValueError(f"Pixel outside depth image: {(u, v)}")
    z = depth[v, u]
    if not np.isfinite(z) or z <= 0:
        raise ValueError("Invalid metric depth at target pixel")
    camera_point = z * np.linalg.solve(k, np.array([u, v, 1.0]))
    return (rotation @ camera_point + transform[:3, 3]).astype(np.float32)


def project_point(point_base, intrinsics, camera_to_base):
    transform = np.asarray(camera_to_base, dtype=np.float64)
    camera = transform[:3, :3].T @ (np.asarray(point_base) - transform[:3, 3])
    if camera[2] <= 0:
        raise ValueError("Target is behind the camera")
    pixel = np.asarray(intrinsics) @ camera
    return pixel[:2] / pixel[2]


def innermost_pixel(mask):
    from scipy.ndimage import distance_transform_edt
    mask = np.asarray(mask, dtype=bool).squeeze()
    if mask.ndim != 2 or not mask.any():
        raise ValueError("Target mask is empty or not 2D; no centroid fallback")
    # Padding makes an image-edge boundary explicit.
    distance = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    v, u = np.unravel_index(np.argmax(distance), mask.shape)
    return np.array([u, v], dtype=np.int32)


@dataclass
class StageSwitch:
    initial_object_z: float
    threshold_m: float = 0.01
    placing: bool = False

    def update(self, object_z):
        if not np.isfinite(object_z):
            raise ValueError("Object height must be finite")
        # Latched switch is our implementation choice; the paper does not specify recovery.
        self.placing = self.placing or float(object_z) > self.initial_object_z + self.threshold_m
        return int(self.placing)
