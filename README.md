# Sphedron: Polyhedral meshes on the sphere

A python package for creating refinable polyhedral meshes on the sphere. The refinement is either rectangle or triangle based.

The was developed as a component of Graph Neural Networks (GNN) for Geospatial ML projects, and it's designed with that in mind.

The package implements:

- Icosphere 
- Octasphere
- Cubesphere
- Uniform Latitude/Longitude 

It is also straighforward to extend it to include other triangular/rectangular meshes.

You can find code examples here:

If you're looking to implement your own rectangle/triangle based mesh:

For the full documentation:

## Requirements
- python >= 3.10
- numpy
- scipy (nearest neighbor query)
- trimesh (radius query)

Optional dependencies:
- cartopy (plots)
- shapely (land mask feature)

## Install


To understand the inner workings of a mesh and how its refined, refer to
[Icosphere Mesh Documentation](./docs/icosphere.md)
