# Construction of the icosphere

At level 1, the icosphere has 12 vertices that form 20 (triangular) faces. In the
GraphCast paper, the mesh is refined recursively by quadrupling the number of
faces at each iteration. This yields a number of vertices equal to:

$$
|V| = 12 + 10\times4^{\nu-1} - 10
$$

We can use a mush more fine-grained control by taking the base icosphere and
refining it so that the edges of the refined mesh = to $1/n$ of the base
icosphere. This approach yields a number of vertices that is equal to:

$$
|V| = 12 + 10*(\nu^2-1)
$$

Instead of the notation in the GraphCast paper that uses indices $0,1,...,6$,
the corresponding indices for the same meshes in this approach would be
$M_1, M_2, M_4, M_8, M_{16}, M_{32}, M_{64}$ where $|M_{64}| = 40962$

The vertices at level 1 are circular permutations of the base vertices.

$$
\begin{align}
v_{\alpha,\epsilon} = \frac{1}{\sqrt{1+\phi^2}}[\alpha,\ \epsilon \phi,\ 0],\
\text{where } \ \alpha, \epsilon \in\{-1, +1\},
\ \text{and } \phi=\frac{1+\sqrt{5}}{2}
\end{align}
$$

The coordinates of 4 base vertices are then permuted circularly (3 permutations),
which makes the depth 1 mesh contain 12 vertices.

## Size of subdivisions

Assume we have a set $V$ of $m$ vertices and set $F$ of faces such that
$F \subset \{(i,j,k)| 1\leq i< j< k \leq m\}$.

-   The number of vertices and faces are related: if the number of vertices is:
    $m= 12 + 10*(\nu^2-1)$, depth $\nu$, $|F|=20 * \nu^2$

For each $(i,j,k)\in F$ we define the edges $(i,j), (i,k), (j,k)$ and note $E$
the set of unique edges.

## Angle between 2 vertices

Since the mesh base are triangles, the longitude/latitude gap is not global
measure of the resolution. The length of edges of the base mesh is
$\frac{2}{\sqrt{1+\phi^2}}\approx1.052$, which is equivalent to an angle
between the vertices of
$\omega =2 \arcsin(\frac{1}{\sqrt{1+\phi^2}})\approx 63.44\degree$.

## Refinement of the base mesh


A depth 64 mesh is equivalent to a 1.5-degree angle between each connected
vertices. A 32 depth is around 2.3

