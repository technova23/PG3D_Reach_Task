# Research brief

## One-sentence thesis

Executable geometric programs can provide an explicit test-time constraint interface for 3D diffusion policies, enabling new combinations of constraints without retraining while preserving learned manipulation priors.

## Updated project thesis

The core project now includes a kinematic point-cloud world model:

> Programmatic constraints over imagined 3D point-cloud rollouts for steering 3D diffusion robot policies.

The first world model is not learned. It is a geometry/kinematics compositor that removes current robot points from the observed point cloud, renders/samples the robot mesh at future joint states implied by a candidate action chunk, and produces imagined future point-cloud observations and end-effector trajectories.

## Scientific ladder

1. Constrained reach: prove policy + world model + constraint reranking.
2. Reach with projection/keep-out zones: prove path-level constraints and multi-chunk imagination.
3. Pick-and-place: add grasp/contact/object interaction.
4. Pick-and-place with carried-object constraints: add object proxy attachment and no-overflight.
5. Place-into-container: add relational/narrow-placement constraints.
6. Constraint composition and LLM-generated constraints: move toward paper-scale experiments.

## P0 MVP definition

A DP3-style point-cloud diffusion policy trained on nominal ManiSkill reach demonstrations is evaluated with an unseen keep-out region. At inference time, pg3d samples candidate action chunks, uses the robot-geometry point-cloud world model to imagine future end-effector/robot motion, evaluates a handwritten `avoid_region` constraint, and reranks candidates in receding-horizon mode. Success means improved combined reach-and-constraint success over base DP3 and simple rejection/filtering, with clear visualizations of imagined rollouts.

## What not to implement yet

- xArm or any real-robot execution path.
- Multi-task or language-conditioned policies.
- RISE/MinkowskiEngine.
- CodeDiffuser-style baseline.
- Full mesh/SDF collision checking.
- Learned dynamics/world model.
- LLM-generated constraints before handwritten constraints work.
