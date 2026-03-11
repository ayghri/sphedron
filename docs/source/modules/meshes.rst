Mesh Classes
============

Base Classes
------------

.. automodule:: sphedron.mesh.base
   :no-members:

.. autoclass:: sphedron.mesh.base.Mesh
   :no-members:

   .. rubric:: Methods

   .. autosummary::

      ~Mesh.from_base
      ~Mesh.from_graph
      ~Mesh.refine
      ~Mesh.base
      ~Mesh.reset
      ~Mesh.mask_nodes
      ~Mesh.faces2triangles
      ~Mesh.triangle2face_index
      ~Mesh.face2triangle_index
      ~Mesh.build_trimesh
      ~Mesh.query_edges_from_faces
      ~Mesh.query_edges_from_radius
      ~Mesh.query_edges_from_neighbors

   .. automethod:: from_base
   .. automethod:: from_graph
   .. automethod:: refine
   .. automethod:: base
   .. automethod:: reset
   .. automethod:: mask_nodes
   .. automethod:: faces2triangles
   .. automethod:: triangle2face_index
   .. automethod:: face2triangle_index
   .. automethod:: build_trimesh
   .. automethod:: query_edges_from_faces
   .. automethod:: query_edges_from_radius
   .. automethod:: query_edges_from_neighbors

.. autoclass:: sphedron.mesh.base.NodesOnlyMesh
   :no-members:
   :show-inheritance:

.. autoclass:: sphedron.mesh.base.TriangularMesh
   :no-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~TriangularMesh.refine
      ~TriangularMesh.faces2triangles
      ~TriangularMesh.triangle2face_index
      ~TriangularMesh.face2triangle_index

   .. automethod:: refine
   .. automethod:: faces2triangles
   .. automethod:: triangle2face_index
   .. automethod:: face2triangle_index

.. autoclass:: sphedron.mesh.base.RectangularMesh
   :no-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~RectangularMesh.refine
      ~RectangularMesh.faces2triangles
      ~RectangularMesh.triangle2face_index
      ~RectangularMesh.face2triangle_index

   .. automethod:: refine
   .. automethod:: faces2triangles
   .. automethod:: triangle2face_index
   .. automethod:: face2triangle_index


Refinable Meshes
----------------

.. automodule:: sphedron.mesh.refinables
   :no-members:

.. autoclass:: sphedron.mesh.refinables.Icosphere
   :no-members:
   :show-inheritance:

   .. automethod:: base

.. autoclass:: sphedron.mesh.refinables.Octasphere
   :no-members:
   :show-inheritance:

   .. automethod:: base

.. autoclass:: sphedron.mesh.refinables.Cubesphere
   :no-members:
   :show-inheritance:

   .. automethod:: base

.. autoclass:: sphedron.mesh.refinables.UniformMesh
   :no-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~UniformMesh.reshape

   .. automethod:: reshape


Nested Meshes
-------------

.. automodule:: sphedron.mesh.nested
   :no-members:

.. autoclass:: sphedron.mesh.nested.NestedMeshes
   :no-members:

   .. rubric:: Methods

   .. autosummary::

      ~NestedMeshes.__getitem__
      ~NestedMeshes.__len__
      ~NestedMeshes.reset
      ~NestedMeshes.mask_nodes
      ~NestedMeshes.build_trimesh
      ~NestedMeshes.query_edges_from_faces
      ~NestedMeshes.query_edges_from_radius
      ~NestedMeshes.query_edges_from_neighbors

   .. automethod:: __getitem__
   .. automethod:: __len__
   .. automethod:: reset
   .. automethod:: mask_nodes
   .. automethod:: build_trimesh
   .. automethod:: query_edges_from_faces
   .. automethod:: query_edges_from_radius
   .. automethod:: query_edges_from_neighbors

.. autoclass:: sphedron.mesh.nested.NestedIcospheres
   :no-members:
   :show-inheritance:

.. autoclass:: sphedron.mesh.nested.NestedOctaspheres
   :no-members:
   :show-inheritance:

.. autoclass:: sphedron.mesh.nested.NestedCubespheres
   :no-members:
   :show-inheritance:
