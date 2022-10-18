import numpy as np
from matplotlib import cm


def visualize_depth(nparray, maxdepth=10.0):
    """
    Takes metric scale np.array and returns a colormapped np.array with type np.uint8.
    """
    normalized_depth = 1.0 - np.clip(nparray, 0.0, maxdepth) / maxdepth
    return (cm.inferno(normalized_depth) * 255).astype(np.uint8)
