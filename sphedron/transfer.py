"""
Author: Ayoub Ghriss, dev@ayghri.com
Date: 2024

License: Non-Commercial Use Only

Permission is granted to use, copy, modify, and distribute this software
for non-commercial purposes only, with attribution to the original author.
Commercial use requires explicit permission.

This software is provided "as is", without warranty of any kind.
"""

from typing import Literal, Callable
from numpy.typing import NDArray
import numpy as np

from .mesh import Mesh
from .utils import query_nearest


class MeshTransfer:
    """A class to facilitate the transfer of values between meshes.

    This class is designed to handle the transfer of data from a sender mesh
    to a receiver mesh, taking into account the specified number of neighbors
    for interpolation.

    Args:
        sender_mesh: The mesh from which values will be sent.
        receiver_mesh: The mesh to which values will be received.
        n_neighbors: The number of neighboring points to consider for
            value transfer. Defaults to 5.
    """

    def __init__(
        self,
        sender_mesh: Mesh,
        receiver_mesh: Mesh,
        n_neighbors: int = 5,
    ) -> None:
        self._sender_mesh = sender_mesh
        self._receiver_mesh = receiver_mesh
        self._n_neighbors = n_neighbors
        self._nearest_senders = None
        self._nearest_distances = None

    def transfer(
        self,
        sent_values: NDArray,
        aggregation: Literal["mean", "sum", "max", "min"] = "mean",
        recompute: bool = False,
    ) -> NDArray:
        """
        Transfers values from the source mesh to the target mesh using nearest
        neighbor interpolation.

        Args:
            sent_values: The values to be transferred from the sender mesh,
                shaped (sender_mesh.num_nodes, d).
            aggregation: Method of aggregation for the transferred values.
                Can be "mean", "sum", "max", or "min". Defaults to "mean".
            recompute: Flag indicating whether to recompute the nearest
                neighbors. Defaults to False.

        Returns:
            The transferred values for the target mesh, of shape
            (receiver_mesh.num_nodes, d).

        Raises:
            ValueError: if sent_values.shape is not (sender_mesh.num_nodes, ?)
            AttributeError: if the aggregation argument is not supported.
        """
        if sent_values.shape[0] != self._sender_mesh.num_nodes:
            raise ValueError(
                "sent_values and sender mesh do correspond"
                "to the same number of nodes."
            )
        self.compute_neighbors(recompute)
        nearest_vals = sent_values[self._nearest_senders]
        try:
            aggregated_values = getattr(np, aggregation)(nearest_vals, axis=1)
            return aggregated_values
        except AttributeError as exc:
            raise AttributeError(
                f"aggregation {aggregation} is invalid, "
                "allowed : ['mean', 'sum', 'max', 'min']"
            ) from exc

    def weighted_transfer(
        self,
        sent_values: NDArray,
        weight_func: Callable | None = None,
        recompute: bool = False,
    ) -> NDArray:
        """
        Transfers values from the source mesh to the target mesh using nearest
        neighbor interpolation.

        Args:
            sent_values (NDArray): The values to be transferred from the
                sender mesh, shaped (sender_mesh.num_nodes, d).
            weight_func (Callable | None, optional): A function to compute
                weights for the transfer, should take (num_nodes, num_neighbors)
                and return weights of the same shape. If None, uniform weights
                are used. Defaults to None.
            recompute (bool, optional): Flag indicating whether to recompute
                the nearest neighbors. Defaults to False.

        Returns:
            NDArray: The transferred values for the target mesh, of shape
            (receiver_mesh.num_nodes, d).
        """
        self.compute_neighbors(recompute)

        nearest_vals = sent_values[self._nearest_senders]
        if weight_func is None:

            def weighing_func(x):
                return np.ones_like(x)

        else:
            weighing_func = weight_func

        weights = weighing_func(self._nearest_distances)
        weights = weights / weights.sum(axis=1, keepdims=True)

        return (nearest_vals * weights).sum(axis=1)

    def compute_neighbors(self, recompute: bool):
        """Get neighbors for sender to receiver mesh"""
        if self._nearest_senders is None or recompute:
            self._nearest_senders = query_nearest(
                self._sender_mesh.nodes,
                self._receiver_mesh.nodes,
                n_neighbors=self._n_neighbors,
            )
            self._nearest_distances = np.linalg.norm(
                self._receiver_mesh.nodes[:, None, :]
                - self._sender_mesh.nodes[self._nearest_senders],
                axis=-1,
            )
