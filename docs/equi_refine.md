# Construction of the icosphere: arc length

At level 1, the icosphere has 12 vertices, 20 (triangular) faces, and 30 edges.
In the GraphCast paper, the mesh is refined recursively by quadrupling the
number of faces at each iteration. This yields a number of vertices equal to:

$$
|V| = 12 + 10\times4^{\nu-1} - 10
$$

We can achieve much more precise control by refining the base icosphere so that
the edges of the refined mesh are equal to $1/n$ of the base icosphere. This
approach results in a number of vertices that is equal to:

$$
|V| = 12 + 10*(\nu^2-1)
$$

Instead of using indices $0,1,...,6$ as noted in the GraphCast paper, the
corresponding indices for the same meshes in this approach would be
$M_1, M_2, M_4, M_8, M_{16}, M_{32}, M_{64}$ where $|M_{64}| = 40962$

The vertices at level 1 are circular permutations of the base vertices.

$$
\begin{align}
v_{\alpha,\epsilon} = \frac{1}{\sqrt{1+\phi^2}}[\alpha,\ \epsilon \phi,\ 0],\ \text{where } \ \alpha, \epsilon \in\{-1, +1\},
\ \text{and } \phi=\frac{1+\sqrt{5}}{2}
\end{align}
$$

The coordinates of 4 base vertices are then permuted circularly (3
permutations), which yields the 12 vertices at depth 1.

## Subdivision of a mesh

Assume we have a set $V$ of $m$ vertices and set $F$ of faces such that
$F \subset \{(i,j,k)| 1\leq i< j< k \leq m\}$.

-   The number of vertices and faces are related: if the number of vertices is
    $m= 12 + 10*(\nu^2-1)$ for depth $\nu$, then $|F|=20 * \nu^2$

For each $(i,j,k)\in F$ we define the edges $(i,j), (i,k), (j,k)$ and note $E$
the set of unique edges.

## Number of edges

Starting from the base mesh with 20 faces and 30 edges, refining it to depth
$\nu$ means that every edge is split into $\nu$ edges, and each face is refined
into $\nu^2$ faces.

To avoid counting the edges twice, we count the internal new edges on each face,
which yields:

-   Each new vertex on an edge adds 1 cross-floor internal edge: $2*(\nu-1)$
-   At depth $\nu$, we obtain $(\nu-1)\nu/2$ internal floor edges
-   Each internal vertex adds 2 edges to the next floor: $(\nu-2)(\nu-1)$

Thus, for each face, the refinement adds
$\nu-1 + (\nu-1)\nu/2 + (\nu-2)(\nu-1)$  
= $(\nu-1)(3\nu-4)/2 + 2(\nu-1)$.

In total, we have: $30 * \nu + 20 * [(\nu-1)(3\nu-4)/2 + \nu-1]$.

For depth 64 we get a total of $30 * \nu + 20 * [(\nu-1)(3\nu-4)/2 + \nu-1]$
edges.

```python
num_edges = lambda x: 30*x + 20*((x-1)*(3*x-4)//2 + 2*(x -1))
```

## Angle between 2 vertices

Since the mesh base are triangles, the angle is not a global measure of the
resolution. The length of edges of the base mesh is $1$, which translates to
~$\theta =2 \sin(\frac{\sqrt{2}}{2})\approx 64$ degrees. A depth 64 mesh is
equivalent to a 1.5-degree angle between each connected vertices. A 32 depth is
around 2.3
