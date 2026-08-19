# Box-derived E10 U-shape fixture

This fixture replaces each frozen E3 box with a U having the same initial outer envelope and
center before the recorded versioned guidance adjustments. Its opening faces the episode start.

The ten U-object dimensions, centers, yaws, component geometry, and matched constraint
files are finalized as version 1. Do not edit the v1 guidance or generated fixture in place; create
a new versioned fixture for any later geometry change.

The Rerun policy cloud is the saved obstacle-free
dataset reset; `inspection/proposed_u_surface` is a deterministic sample of the proposed geometry.
The geometry-only review artifact itself does not embed the later MPLib paths, live PhysX results,
whole-arm clearance results, or any ITPS/reranking comparison outcome. Object finalization is
separate from per-episode benchmark validity; consult the non-convex experiment plan for the
predeclared validity strata.
