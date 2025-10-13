Construction of the icosphere
=============================

At level 1, the icosphere has 12 nodes, 20 (triangular) faces, and 30
edges. The mesh is refined recursively by quadrupling the number of
faces at each iteration. This yields a number of nodes at depth
:math:`\nu` equal to:

.. math::


   |V| = 12 + 10\times4^{\nu-1} - 10

We can achieve much more precise control by refining the base icosphere
so that the edges of the refined mesh are equal to :math:`1/n` of the
base icosphere. This approach results in a number of nodes that is equal
to:

:math:``|V| = 12 + 10*(\nu^2-1)``

The nodes at level 1 are circular permutations of the base nodes.

.. math::


   \begin{align}
     v_{\alpha,\epsilon} = \frac{1}{\sqrt{1+\phi^2}}[\alpha,\ \epsilon \phi,\ 0],\ \text{where } \ \alpha, \epsilon \in\{-1, +1\}, \text{and } \phi=\frac{1+\sqrt{5}}{2}
   \end{align}

The coordinates of these 4 base nodes are then permuted circularly (3
permutations), which yields the 12 nodes at depth 1.

\_\ **Note** : Instead of using indices :math:`0,1,...,6` as in the
GraphCast paper, the corresponding indices for the same meshes following
our approach would be :math:`M_1, M_2, M_4, M_8, M_{16}, M*{32}, M*{64}`
where :math:`|M_{64}| = 40962`\ \_

Subdivision of a mesh
---------------------

Assume we have a set :math:`V` of :math:`m` nodes and set :math:`F` of
faces such that :math:`F \subset \{(i,j,k)| 1\leq i< j< k \leq m\}`.

-  The number of nodes and faces are related: if the number of nodes is
   :math:`m= 12 + 10*(\nu^2-1)` for depth :math:`\nu`, then
   :math:`|F|=20 * \nu^2`

For each :math:`(i,j,k)\in F` we define the edges
:math:`(i,j), (i,k), (j,k)` and note :math:`E` the set of unique edges.

Number of edges
---------------

Starting from the base mesh with 20 faces and 30 edges, refining it to
depth :math:`\nu` means that every edge is split into :math:`\nu` edges,
and each face is refined into :math:`\nu^2` faces.

To avoid counting the edges twice, we count the internal new edges on
each face, which yields:

-  Each new node on an edge adds 1 cross-floor internal edge:
   :math:`2*(\nu-1)`
-  At depth :math:`\nu`, we obtain :math:`(\nu-1)\nu/2` internal floor
   edges
-  Each internal node adds 2 edges to the next floor:
   :math:`(\nu-2)(\nu-1)`

| Thus, for each face, the refinement adds
  :math:`\nu-1 + (\nu-1)\nu/2 + (\nu-2)(\nu-1)`
| = :math:`(\nu-1)(3\nu-4)/2 + 2(\nu-1)`.

In total, we have: :math:`30 * \nu + 20 * [(\nu-1)(3\nu-4)/2 + \nu-1]`.

For depth 64 we get a total of
:math:`30 * \nu + 20 * [(\nu-1)(3\nu-4)/2 + \nu-1]` edges.

.. code:: python

   num_edges = lambda x: 30*x + 20*((x-1)*(3*x-4)//2 + 2*(x -1))

Angle between 2 nodes
---------------------

Since the mesh base are triangles, the longitude/latitude gap is not
global measure of the resolution. The length of edges of the base mesh
is :math:`\frac{2}{\sqrt{1+\phi^2}}\approx1.052`, which is equivalent to
an angle between the nodes of
:math:`\omega =2 \arcsin(\frac{1}{\sqrt{1+\phi^2}})\approx 63.44\degree`.

A depth 64 mesh is equivalent to a 1.5-degree angle between each
connected nodes. A 32 depth is around 2.3
