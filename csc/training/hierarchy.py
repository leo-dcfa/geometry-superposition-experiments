"""Tree-structured features — the independent variable of CSC-2.

CSC used i.i.d. exchangeable features: every feature independent, none more
general than another, no containment relations. Hyperbolic geometry had nothing
to match, and measurably bought nothing.

A hierarchy supplies the three properties hyperbolic space is shaped for:

1. **Exponential growth with depth.** A tree of branching factor b has b^D
   nodes at depth D; hyperbolic volume grows as e^(√|K|·R). Set radius ∝ depth
   and the growth rates coincide — which is why trees embed in H² at
   arbitrarily low distortion (Sarkar) while Euclidean space needs many
   dimensions for the same fidelity.
2. **Distance via a common ancestor.** Tree distance is
   depth(u) + depth(v) − 2·depth(lca(u, v)). Hyperbolic geodesics between two
   rim points bend toward the origin, so the path routes through the region
   where the common ancestor sits. Flat geodesics are straight and have no
   ancestor to route through.
3. **Radius means generality.** General concepts near the origin, specific ones
   near the rim. This is the coordinate CSC's features gave no meaning to, and
   which the models correctly ignored — leaving 12× the volume unused.

Generative model: sample a root-to-leaf path; every node on the path activates
together, with magnitude decaying by ``level_decay`` per level, then each
activation is dropped independently with probability ``sparsity``. So a parent
co-occurs with its children far more often than with unrelated nodes, and the
feature correlation matrix carries the tree.

``depth=0`` degenerates to a flat set of independent features and reproduces
CSC's setting — the anchor H2-a requires, so the same generator covers both.
"""

from __future__ import annotations

import torch
from torch import Tensor


class FeatureTree:
    """A complete b-ary tree of ``depth`` levels; features are its nodes."""

    def __init__(self, depth: int, branching: int = 2) -> None:
        if depth < 0 or branching < 1:
            raise ValueError("depth must be >= 0 and branching >= 1")
        self.depth = depth
        self.branching = branching
        self.parent: list[int] = [-1]
        self.level: list[int] = [0]
        frontier = [0]
        for lvl in range(1, depth + 1):
            nxt = []
            for node in frontier:
                for _ in range(branching):
                    self.parent.append(node)
                    self.level.append(lvl)
                    nxt.append(len(self.parent) - 1)
            frontier = nxt
        self.leaves = frontier if depth > 0 else [0]

    @property
    def n_features(self) -> int:
        return len(self.parent)

    def path_to_root(self, node: int) -> list[int]:
        path = []
        while node != -1:
            path.append(node)
            node = self.parent[node]
        return path

    def tree_distance_matrix(self) -> Tensor:
        """d(u,v) = depth(u) + depth(v) − 2·depth(lca(u,v)) — the target metric.

        This is what an embedding is judged against (P2-1 distortion), and what
        hyperbolic space can reproduce at low distortion while flat space
        cannot without many dimensions.
        """
        n = self.n_features
        ancestors = [set(self.path_to_root(i)) for i in range(n)]
        d = torch.zeros(n, n)
        for u in range(n):
            for v in range(u + 1, n):
                common = ancestors[u] & ancestors[v]
                lca_level = max(self.level[c] for c in common)
                dist = self.level[u] + self.level[v] - 2 * lca_level
                d[u, v] = d[v, u] = float(dist)
        return d

    def level_tensor(self) -> Tensor:
        return torch.tensor(self.level, dtype=torch.float32)

    def path_matrix(self, level_decay: float) -> Tensor:
        """``(n_leaves, n_features)`` — row L is the activation pattern of leaf L.

        Precomputed once and gathered per batch. The obvious implementation
        walks each sampled leaf's path in Python every step, which is
        O(batch x depth) interpreter operations per training step — ~15M per
        run at batch 256 and 10k steps, and by far the dominant cost. One
        gather replaces all of it.
        """
        key = ("path_matrix", level_decay)
        cached = getattr(self, "_cache", {}).get(key)
        if cached is not None:
            return cached
        m = torch.zeros(len(self.leaves), self.n_features)
        for row, leaf in enumerate(self.leaves):
            for node in self.path_to_root(leaf):
                m[row, node] = level_decay ** self.level[node]
        if not hasattr(self, "_cache"):
            self._cache = {}
        self._cache[key] = m
        return m


def sample_hierarchical_batch(
    batch_size: int,
    tree: FeatureTree,
    sparsity: float,
    level_decay: float = 0.7,
    generator: torch.Generator | None = None,
    device=None,
) -> Tensor:
    """``(batch_size, tree.n_features)`` activations carrying the tree structure.

    One root-to-leaf path activates per sample, magnitudes decaying with depth,
    then independent dropout at rate ``sparsity`` so that overall density
    matches the flat setting and the two are comparable.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity is P(feature dropped) and must lie in [0, 1)")
    kw = {"generator": generator, "device": device}
    n = tree.n_features
    paths = tree.path_matrix(level_decay).to(device)
    leaf_idx = torch.randint(len(tree.leaves), (batch_size,), **kw)
    x = paths[leaf_idx]

    values = torch.rand(batch_size, n, **kw)
    keep = torch.rand(batch_size, n, **kw) >= sparsity
    return x * values * keep


def hierarchy_importance(tree: FeatureTree, decay: float = 0.9, device=None) -> Tensor:
    """Importance by tree level, not by index.

    A flat geometric spectrum over an arbitrary index ordering would make
    importance a property of the *labelling*. Tying it to level keeps it a
    property of the structure, so that "important" and "general" coincide the
    way they do in real hierarchies.
    """
    levels = tree.level_tensor().to(device)
    imp = decay**levels
    return imp / imp.mean()
