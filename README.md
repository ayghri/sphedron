# Sphedron: Polyhedral meshes on the sphere

A python package for creating refinable polyhedral meshes on the sphere. The refinement is either rectangle or triangle based.

This was developed as a component of Graph Neural Networks (GNN) for Geospatial ML projects, and it's designed with that in mind.

The package implements:

- Icosphere
- Octasphere
- Cubesphere
- Uniform Latitude/Longitude

It is also straightforward to extend it to include other triangular/rectangular meshes.

You can find code examples in the [examples/](./examples) folder.

To understand the inner workings of a mesh and how it is refined, refer to the
[Icosphere tutorial](./docs/source/tutorials/icosphere.rst).

## Requirements
- python >= 3.10
- numpy
- scipy (nearest neighbor query)
- trimesh (radius query)

Optional dependencies:
- cartopy (plots)
- shapely (land mask feature)

## Install

```bash
pip install .
```

## Using notebooks in ReadTheDocs

This project's Sphinx docs can render Jupyter notebooks via `nbsphinx`.

- Notebook sources live under `docs/source/notebooks/` (copied from `examples/` as needed).
- The docs build defaults to **not executing** notebooks (it renders the saved inputs/outputs).
- To execute notebooks locally during a Sphinx build:
	- `SPHEDRON_DOCS_EXECUTE_NOTEBOOKS=auto`
