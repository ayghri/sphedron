from sphedron.cube import CubeSphere
from sphedron.utils.plots import plot_vertices

import matplotlib.pyplot as plt

# mesh = StratifiedIcospheres([1, 2])
# mesh = Icosphere(2)
# mesh = CubeSphere(depth=3, use_length=False)
mesh = CubeSphere(depth=3)

plot_vertices(mesh)
