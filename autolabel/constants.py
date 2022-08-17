import numpy as np
from matplotlib import cm

colors = (cm.tab10(np.linspace(0, 1, 10)) * 255.0)[:, :3].astype(np.uint8)
COLORS = np.concatenate([colors, colors, colors, colors], axis=0)
