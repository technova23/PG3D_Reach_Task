# E10 P1 U-shape visualization in selected episode 004

This fixture moves the first three-box U-shaped obstacle to output episode 004 / dataset episode
1010 from the frozen 75 cm candidate-midpath suite. It keeps the original U dimensions while using
the episode-004 candidate-midpath anchor.

- Envelope: `0.28 x 0.30 x 0.60 m`.
- Opening: local `-Y`, oriented toward the episode start.
- Closed back: lies beyond the candidate-midpath anchor toward the goal.
- Initial sampled whole-robot clearance: `0.07909 m`.
- Serialized target: whole robot.

Because this is evaluated as a one-entry smoke, its generated artifacts use local filename
`episode_000`; `fixture.json` records that the source selection is episode 004. This is not a
definitive benchmark fixture or performance result.
