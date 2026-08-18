# E10 P1 U-shape visualization fixture

This smoke fixture places a three-box U-shaped obstacle into output episode 0 / dataset episode 305
from the frozen 75 cm candidate-midpath suite. It is for geometry, camera, point-cloud, and artifact
inspection before freezing an E10 benchmark population.

- Envelope: `0.28 x 0.30 x 0.60 m`.
- Opening: local `-Y`, oriented toward the episode start.
- Closed back: lies beyond the stored candidate-midpath anchor toward the goal.
- Initial sampled whole-robot clearance: `0.03229 m`.
- Serialized target: whole robot.

The U is represented by two side-wall `BoxRegion`s and one back-wall `BoxRegion`. The simulator
creates the same three collidable actors. Do not treat this single visualization episode as a
performance result or definitive benchmark fixture.
