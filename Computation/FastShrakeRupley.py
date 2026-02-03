"""Accelerated calculation of solvent accessible surface areas for Bio.PDB entities.

Uses the "rolling ball" algorithm developed by Shrake & Rupley algorithm,
which uses a sphere (of equal radius to a solvent molecule) to probe the
surface of the molecule.

Made with Claude Sonnet
Optimizations:
- scipy.spatial.KDTree for faster spatial queries
- numba JIT compilation for core computational loops  
- Vectorized operations and reduced memory allocations
- Efficient set operations using numpy arrays

Reference:
    Shrake, A; Rupley, JA. (1973). J Mol Biol
    "Environment and exposure to solvent of protein atoms. Lysozyme and insulin".
"""

import collections
import math
from collections.abc import MutableMapping

import numpy as np
import numba
from scipy.spatial import KDTree

__all__ = ["ShrakeRupleyAccelerated"]

_ENTITY_HIERARCHY = {
    "A": 0,
    "R": 1,
    "C": 2,
    "M": 3,
    "S": 4,
}

# vdW radii taken from:
# https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
ATOMIC_RADII: MutableMapping[str, float] = collections.defaultdict(lambda: 2.0)
ATOMIC_RADII.update(
    {
        "H": 1.200,
        "HE": 1.400,
        "C": 1.700,
        "N": 1.550,
        "O": 1.520,
        "F": 1.470,
        "NA": 2.270,
        "MG": 1.730,
        "P": 1.800,
        "S": 1.800,
        "CL": 1.750,
        "K": 2.750,
        "CA": 2.310,
        "NI": 1.630,
        "CU": 1.400,
        "ZN": 1.390,
        "SE": 1.900,
        "BR": 1.850,
        "CD": 1.580,
        "I": 1.980,
        "HG": 1.550,
    }
)


@numba.jit(nopython=True, cache=True)
def _compute_sphere_numba(n_points):
    """Numba-compiled sphere point generation using golden spiral."""
    dl = np.pi * (3 - 5**0.5)
    dz = 2.0 / n_points
    
    longitude = 0.0
    z = 1.0 - dz / 2.0
    
    coords = np.zeros((n_points, 3), dtype=np.float64)
    
    for k in range(n_points):
        r = (1.0 - z * z) ** 0.5
        coords[k, 0] = math.cos(longitude) * r
        coords[k, 1] = math.sin(longitude) * r
        coords[k, 2] = z
        z -= dz
        longitude += dl
    
    return coords


@numba.jit(nopython=True, cache=True)
def _compute_distances_squared(points, center):
    """Compute squared distances from points to center."""
    n_points = points.shape[0]
    distances_sq = np.zeros(n_points, dtype=np.float64)
    
    for i in range(n_points):
        dx = points[i, 0] - center[0]
        dy = points[i, 1] - center[1] 
        dz = points[i, 2] - center[2]
        distances_sq[i] = dx*dx + dy*dy + dz*dz
    
    return distances_sq


@numba.jit(nopython=True, cache=True)
def _find_accessible_points(sphere_points, coords, radii, atom_idx, 
                           neighbor_indices, n_points):
    """Core SASA computation with numba compilation."""
    n_neighbors = len(neighbor_indices)
    accessible = np.ones(n_points, dtype=numba.boolean)
    
    atom_center = coords[atom_idx]
    atom_radius = radii[atom_idx]
    
    # Transform sphere points to atom center
    scaled_sphere = np.zeros((n_points, 3), dtype=np.float64)
    for i in range(n_points):
        scaled_sphere[i, 0] = sphere_points[i, 0] * atom_radius + atom_center[0]
        scaled_sphere[i, 1] = sphere_points[i, 1] * atom_radius + atom_center[1]
        scaled_sphere[i, 2] = sphere_points[i, 2] * atom_radius + atom_center[2]
    
    # Check each neighbor for occlusion
    for n_idx in range(n_neighbors):
        neighbor = neighbor_indices[n_idx]
        
        if neighbor == atom_idx:
            continue
            
        neighbor_center = coords[neighbor]
        neighbor_radius = radii[neighbor]
        neighbor_radius_sq = neighbor_radius * neighbor_radius
        
        # Find points within neighbor's radius
        distances_sq = _compute_distances_squared(scaled_sphere, neighbor_center)
        
        for pt_idx in range(n_points):
            if accessible[pt_idx] and distances_sq[pt_idx] <= neighbor_radius_sq:
                accessible[pt_idx] = False
    
    return np.sum(accessible)


class ShrakeRupleyAccelerated:
    """Accelerated SASA calculator using scipy KDTree and numba compilation."""

    def __init__(self, probe_radius=1.40, n_points=100, radii_dict=None):
        """Initialize the accelerated SASA calculator.

        :param probe_radius: radius of the probe in Å. Default is 1.40
        :param n_points: resolution of the surface of each atom. Default is 100
        :param radii_dict: user-provided dictionary of atomic radii
        """
        if probe_radius <= 0.0:
            raise ValueError(
                f"Probe radius must be a positive number: {probe_radius} <= 0"
            )

        self.probe_radius = float(probe_radius)

        if n_points < 1:
            raise ValueError(
                f"Number of sphere points must be larger than 1: {n_points}"
            )
        self.n_points = n_points

        # Update radii with user provided values
        self.radii_dict = ATOMIC_RADII.copy()
        if radii_dict is not None:
            self.radii_dict.update(radii_dict)

        # Pre-compute reference sphere using numba
        self._sphere = _compute_sphere_numba(self.n_points)

    def compute(self, entity, level="A"):
        """Calculate surface accessibility surface area for an entity.
        
        Accelerated version using scipy KDTree and numba compilation.
        """
        # Input validation (same as original)
        is_valid = hasattr(entity, "level") and entity.level in {"R", "C", "M", "S"}
        if not is_valid:
            raise ValueError(
                f"Invalid entity type '{type(entity)}'. "
                "Must be Residue, Chain, Model, or Structure"
            )

        if level not in _ENTITY_HIERARCHY:
            raise ValueError(f"Invalid level '{level}'. Must be A, R, C, M, or S.")
        elif _ENTITY_HIERARCHY[level] > _ENTITY_HIERARCHY[entity.level]:
            raise ValueError(
                f"Level '{level}' must be equal or smaller than input entity: {entity.level}"
            )

        # Get atoms and coordinates
        atoms = list(entity.get_atoms())
        n_atoms = len(atoms)
        if not n_atoms:
            raise ValueError("Entity has no child atoms.")

        # Prepare coordinate and radii arrays
        coords = np.array([a.coord for a in atoms], dtype=np.float64)
        radii = np.array([self.radii_dict[a.element] for a in atoms], dtype=np.float64)
        radii += self.probe_radius

        # Build scipy KDTree for efficient neighbor queries
        max_radius = np.max(radii)
        tree = KDTree(coords)
        
        # Pre-allocate result array
        asa_counts = np.zeros(n_atoms, dtype=np.int64)
        
        # Process each atom
        for i in range(n_atoms):
            # Find neighbors within interaction distance
            search_radius = radii[i] + max_radius
            neighbor_indices = tree.query_ball_point(coords[i], search_radius)
            neighbor_indices = np.array(neighbor_indices, dtype=np.int64)
            
            # Use numba-compiled function for core computation
            accessible_points = _find_accessible_points(
                self._sphere, coords, radii, i, neighbor_indices, self.n_points
            )
            
            asa_counts[i] = accessible_points

        # Convert accessible point count to surface area in Ų
        surface_area_factor = 4.0 * np.pi / self.n_points
        asa_values = asa_counts * (radii * radii * surface_area_factor)

        # Set atom SASA values
        for i, atom in enumerate(atoms):
            atom.sasa = asa_values[i]

        # Aggregate values per entity level if necessary
        if level != "A":
            entities = set(atoms)
            target = _ENTITY_HIERARCHY[level]
            for _ in range(target):
                entities = {e.parent for e in entities}

            atomdict = {a.full_id: idx for idx, a in enumerate(atoms)}
            for e in entities:
                e_atoms = [atomdict[a.full_id] for a in e.get_atoms()]
                e.sasa = asa_values[e_atoms].sum()


# Compatibility alias
#ShrakeRupley = ShrakeRupleyAccelerated
