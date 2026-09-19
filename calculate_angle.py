import numpy as np
from typing import Sequence


def calculate(
    A: Sequence[float],
    B: Sequence[float],
    C: Sequence[float],
) -> float:
    """
    Calculate the angle at joint B given three points A, B and C.

    Points may be 2D [x, y] or 3D [x, y, z].
    """

    a = np.asarray(A, dtype=float)
    b = np.asarray(B, dtype=float)
    c = np.asarray(C, dtype=float)

    vec_ba = a - b
    vec_bc = c - b

    magnitude_ba = np.linalg.norm(vec_ba)
    magnitude_bc = np.linalg.norm(vec_bc)

    # Prevent division by zero if two landmarks overlap.
    if magnitude_ba == 0 or magnitude_bc == 0:
        raise ValueError("Cannot calculate angle: zero-length vector.")

    cosine_angle = np.dot(vec_ba, vec_bc) / (
        magnitude_ba * magnitude_bc
    )

    # Protect against floating-point precision errors.
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)

    return float(np.degrees(angle))

def calculate_ama(data: Sequence[int]) -> float:
    return sum(data) / 3

def calculate_ema(data: Sequence[int], N=3) -> Sequence[float]:
    alpha = 2 / (N + 1)
    radians = np.radians(data)
    npa = np.asarray(radians)
    ema_x = round(np.cos(npa[0]), 2)
    ema_y = round(np.sin(npa[0]), 2)

    angles = [np.degrees(npa[0])]
    for v in npa[1:]:
        x = np.cos(v)
        y = np.sin(v)
        ema_x = (x * alpha) + (ema_x * (1 - alpha))
        ema_y = (y * alpha) + (ema_y * (1 - alpha))
        angle = np.atan2(ema_y, ema_x)
        angles.append(np.degrees(angle))

    return [round(r, 2) for r in angles]
