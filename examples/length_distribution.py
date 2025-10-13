from sphedron import Icosphere
from sphedron.helpers import compute_edges_angles
import matplotlib.pyplot as plt
import numpy as np


mesh = Icosphere(refine_factor=16, refine_by_angle=False)
distances = compute_edges_angles(mesh.nodes, mesh.edges)
print("mean:", np.mean(distances))
print("std:", np.std(distances))
print("max:", np.max(distances))
print("min:", np.min(distances))
# print(plt.hist(distances, bins=40))
plt.show()
