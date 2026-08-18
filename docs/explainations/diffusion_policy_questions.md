# Diffusion Policy Questions

This file collects focused questions that arise while reading
[`diffusion_policy.md`](diffusion_policy.md). Answers refer to the active pg3d implementation and
the Panda reach checkpoint at `/scratch2/skills/train_final_Arya/step_00065000.pt`.

## 1. If the checkpoint was trained with DDIM, and DDIM is deterministic by default, how do we obtain multiple trajectories for reranking?

### Short answer

DDIM is deterministic **given the starting noise and observation**. It does not map every possible
starting noise tensor to the same trajectory.

For reranking, the code creates a batch of repeated observations but initializes every batch item
with an independently drawn Gaussian trajectory. The DDIM transitions are deterministic, but each
candidate follows those transitions from a different starting point. Consequently, the final
trajectories can differ.

### Step-by-step code path

Suppose reranking requests 32 candidates.

1. `sample_action_chunks(..., k=32)` repeats the same observation window 32 times
   ([`eval_constrained_reach.py:162-182`](../../scripts/eval_constrained_reach.py#L162-L182)).
2. The repeated batch is passed once to `policy.predict_action`
   ([`eval_constrained_reach.py:227-230`](../../scripts/eval_constrained_reach.py#L227-L230)).
3. `conditional_sample` creates a Gaussian tensor with the complete batch shape
   ([`policy.py:211-224`](../../pg3d/policies/dp3/policy.py#L211-L224)). For this checkpoint, its
   shape is:

   ```text
   [32 candidates, 16 trajectory steps, 7 action dimensions]
   ```

4. Every `[16, 7]` batch entry receives its own random values. Call these starting trajectories
   \(x_T^{(1)},\ldots,x_T^{(32)}\).
5. The same observation and deterministic DDIM update are applied to all 32 entries in parallel
   ([`policy.py:225-233`](../../pg3d/policies/dp3/policy.py#L225-L233)).

Mathematically, with the observation \(o\) fixed, deterministic DDIM defines a function:

\[
x_0=F_\theta(x_T,o).
\]

Deterministic means:

\[
F_\theta(x_T,o)=F_\theta(x_T,o)
\]

when the same \(x_T\), observation, model, and numerical environment are reused. It does **not**
mean:

\[
F_\theta(x_T^{(1)},o)=F_\theta(x_T^{(2)},o)
\]

for different initial noise tensors.

Therefore, the 32 candidates are:

```text
same observation + noise sample 1  -> DDIM -> trajectory 1
same observation + noise sample 2  -> DDIM -> trajectory 2
...
same observation + noise sample 32 -> DDIM -> trajectory 32
```

The world model then evaluates these completed trajectories and reranking selects among them. The
world model does not create the diversity; the 32 independent initial noise tensors do.

### What would produce identical trajectories?

If one `[16, 7]` Gaussian tensor were drawn and copied identically into all 32 batch entries, then
deterministic DDIM would produce identical candidates, assuming identical observations and ordinary
deterministic execution.

Likewise, rerunning the entire evaluation with the same global PyTorch seed and the same sequence
of random-number-consuming operations should reproduce the same candidate batch. The evaluator
seeds the global CPU and CUDA PyTorch generators in
[`eval_constrained_reach.py:3382-3385`](../../scripts/eval_constrained_reach.py#L3382-L3385).

### Current implementation detail

`DP3ChunkPolicyAdapter.sample_action_chunks` accepts a NumPy `rng`, but the ordinary base/reranking
path does not currently turn that RNG into a `torch.Generator`. `_predict_actions` calls
`policy.predict_action(batch)` without a generator
([`eval_constrained_reach.py:227-230`](../../scripts/eval_constrained_reach.py#L227-L230)). Thus,
ordinary DDIM candidate diversity currently comes from PyTorch's global random-number generator.

This does not prevent generating 32 different trajectories, but it means candidate randomness is
tied to global PyTorch RNG state and call order rather than being fully isolated behind the
adapter's `rng` argument.

## 2. Can we use a DDPM scheduler during inference if a DDIM scheduler was used while training the checkpoint?

### Short answer

Yes. The checkpoint's denoiser can be used with DDPM during inference because the model was trained
to predict noise under a forward noise schedule, and DDIM and DDPM share that forward process when
configured with the same beta schedule. DDIM versus DDPM primarily determines how the learned noise
predictions are converted into reverse-diffusion transitions at inference time.

This repository already supports both:

- `set_scheduler("ddpm")` replaces the policy's ordinary inference scheduler
  ([`policy.py:174-209`](../../pg3d/policies/dp3/policy.py#L174-L209)).
- The current ITPS path constructs an isolated DDIM scheduler without altering the base policy's
  scheduler
  ([`policy.py:350-360`](../../pg3d/policies/dp3/policy.py#L350-L360)).

The second point is deliberately separate from DDPM compatibility: this branch used DDPM for ITPS
historically, but commit `642c515` switched current ITPS inference to DDIM. Old DDPM-ITPS artifacts
must not be compared as if they came from the current sampler.

### Why this is valid

Saying that the checkpoint was “trained with DDIM” is slightly misleading. The training code does
instantiate a `DDIMScheduler`, but during training it uses the scheduler for the forward operation:

\[
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
\]

The call is `scheduler.add_noise(...)` at
[`policy.py:425-433`](../../pg3d/policies/dp3/policy.py#L425-L433). The network is then trained to
recover \(\epsilon\), selected as the target at
[`policy.py:442-448`](../../pg3d/policies/dp3/policy.py#L442-L448).

The model is therefore learning:

\[
\epsilon_\theta(x_t,t,o)\approx\epsilon,
\]

not learning a hard-coded DDIM transition function.

For matching values of `num_train_timesteps`, `beta_start`, `beta_end`, `beta_schedule`, and
`prediction_type`, DDIM and DDPM expose the model to the same noisy marginal \(q(x_t\mid x_0)\).
They use the predicted noise differently during inference:

- DDIM with `eta=0` uses a deterministic reverse transition once \(x_T\) is fixed.
- DDPM samples from a reverse transition and normally injects posterior noise at successive
  denoising levels.

### How the repository preserves compatibility

The base scheduler configuration is created in
[`policy.py:143-152`](../../pg3d/policies/dp3/policy.py#L143-L152). When switching to DDPM,
`set_scheduler` copies the training timestep count, beta endpoints, beta schedule, clipping setting,
and prediction type into the new scheduler
([`policy.py:185-194`](../../pg3d/policies/dp3/policy.py#L185-L194)).

The current ITPS path separately copies the same configuration into an isolated DDIM scheduler
([`policy.py:350-362`](../../pg3d/policies/dp3/policy.py#L350-L362)). It passes the trained U-Net's
epsilon prediction, after adding the guidance gradient, to that scheduler
([`policy.py:271-298`](../../pg3d/policies/dp3/policy.py#L271-L298)). Neither the optional ordinary
DDPM switch nor current DDIM-ITPS requires retraining or weight conversion.

### What changes when DDPM is used?

The learned network and its weights remain unchanged. The reverse sampler changes:

```text
DDIM:
initial random x_T
    -> deterministic reverse transitions
    -> final trajectory

DDPM:
initial random x_T
    -> stochastic reverse transition + new scheduler noise
    -> stochastic reverse transition + new scheduler noise
    -> final trajectory
```

Consequently:

- DDPM can produce different outputs even when starting from the same \(x_T\), if its reverse-step
  random noise differs.
- DDPM usually consumes more random numbers than deterministic DDIM.
- Sampling quality, diversity, and runtime can change, so the scheduler choice should be evaluated
  empirically even though it is mathematically compatible.
- The inference timestep count can be lower than the 350 training levels. This checkpoint uses 100
  selected inference levels.

### Why current ITPS uses an isolated DDIM scheduler

Current ITPS uses deterministic DDIM for the reverse transition. Its stochasticity comes from the
initial Gaussian trajectory and the fresh Gaussian noise used to re-noise the predicted clean
sample back to the same diffusion level during intermediate annealed-MCMC inner steps
([`policy.py:262-318`](../../pg3d/policies/dp3/policy.py#L262-L318)). Only the final inner step
advances to the next diffusion level.

The local scheduler is important operationally: evaluating ITPS does not mutate the scheduler used
by later base or reranking calls. The supplied generator controls both the initial trajectory and
the explicit inner-loop re-noising.

### Reproducibility caveat for ordinary `set_scheduler("ddpm")`

`conditional_sample` accepts a generator and uses it for the initial Gaussian sample, but its
ordinary scheduler call does not pass that generator into `scheduler.step`
([`policy.py:219-233`](../../pg3d/policies/dp3/policy.py#L219-L233)). This is sufficient for DDIM's
default deterministic transition. If the same function is switched to DDPM, reverse-step noise
comes from global PyTorch RNG state rather than the supplied generator.

Current ITPS does not rely on stochastic reverse-step noise: its DDIM reverse transition is
deterministic for a fixed sample. It explicitly uses the supplied generator for its initial sample
and intermediate forward re-noising at
[`policy.py:262-318`](../../pg3d/policies/dp3/policy.py#L262-L318).

## Summary

1. Reranking gets multiple DDIM trajectories by starting deterministic DDIM from multiple
   independently sampled Gaussian trajectories.
2. A denoiser trained through the shared forward noise process can be sampled with either DDIM or
   DDPM, provided their noise schedule and prediction configuration are compatible.
3. DDIM is deterministic conditional on its starting noise; DDPM also introduces randomness during
   reverse transitions.
4. The current ITPS path deliberately uses an isolated DDIM scheduler, with stochasticity from its
   initial trajectory and intermediate forward re-noising; historical ITPS runs used DDPM.

## 3. What does `conditional_sample` do, and what does `trajectory[condition_mask] = condition_data[condition_mask]` mean?

### Purpose of `conditional_sample`

`conditional_sample` is the complete ordinary reverse-diffusion loop
([`policy.py:211-235`](../../pg3d/policies/dp3/policy.py#L211-L235)). It accepts:

- `condition_data`: a tensor with the shape of the object being generated, containing values that
  should be fixed wherever the mask is true;
- `condition_mask`: a Boolean tensor of the same shape, identifying which values are known;
- `global_cond`: observation features supplied separately to the U-Net;
- `generator`: the random-number generator used to create the initial Gaussian trajectory.

It performs five operations:

1. Create a completely random tensor called `trajectory`.
2. Configure the descending DDIM inference timesteps.
3. At every timestep, restore any known values selected by `condition_mask`.
4. Predict noise and use the scheduler to move to the next lower noise level.
5. Restore the known values one final time and return the denoised tensor.

In simplified form:

```python
trajectory = random_gaussian_with_same_shape_as(condition_data)

for timestep in scheduler.timesteps:
    clamp_known_values(trajectory)
    predicted_noise = model(trajectory, timestep, global_cond)
    trajectory = scheduler.step(predicted_noise, timestep, trajectory).prev_sample

clamp_known_values(trajectory)
return trajectory
```

### Meaning of the masked assignment

The expression:

```python
trajectory[condition_mask] = condition_data[condition_mask]
```

is Boolean-indexed assignment. For every tensor element where `condition_mask` is `True`, it copies
the corresponding value from `condition_data` into `trajectory`. Elements where the mask is
`False` are left unchanged.

It is equivalent in meaning to:

```python
trajectory = torch.where(condition_mask, condition_data, trajectory)
```

For a small example:

```text
trajectory      = [ 0.7, -0.4,  1.2,  0.3 ]
condition_data  = [ 9.0,  8.0,  7.0,  6.0 ]
condition_mask  = [False, True, False, True]
```

After the assignment:

```text
trajectory      = [ 0.7,  8.0,  1.2,  6.0 ]
```

Only positions 1 and 3 were replaced.

### Why restore known values at every denoising step?

Diffusion starts from a random tensor and the scheduler updates the entire tensor after every model
call. If part of that tensor represents information already known—such as observation features—it
must not be freely generated as if it were unknown.

The masked assignment implements diffusion inpainting or clamping:

```text
known fields:    force them back to their supplied values
unknown fields:  allow reverse diffusion to generate them
```

It runs before every U-Net call so the model always sees the correct known values. It runs once
after the loop because the last scheduler step can modify all tensor entries, including entries
that are supposed to remain fixed.

The mask is produced by `LowdimMaskGenerator`
([`modules.py:499-559`](../../pg3d/policies/dp3/modules.py#L499-L559)). That generic component can
mark observation dimensions in the first `n_obs_steps` as known while leaving action dimensions
unknown.

### Generic non-global-conditioning case

When `obs_as_global_cond=False`, the tensor being diffused contains both action dimensions and
encoded observation dimensions:

```text
trajectory shape = [batch, horizon, action_dim + observation_feature_dim]
```

`_build_conditioning` writes encoded observations into the observation dimensions for the first
`obs_steps` and marks those entries true
([`policy.py:490-500`](../../pg3d/policies/dp3/policy.py#L490-L500)). Conceptually, the mask looks
like:

```text
                       action fields     observation fields
first observed step    False ... False   True ... True
second observed step   False ... False   True ... True
later step             False ... False   False ... False
```

The actions and future observation fields remain generated, while the supplied past observations
are clamped.

### What happens for the current checkpoint?

The 65,000-step checkpoint has `obs_as_global_cond=True`. Its observations do not occupy fields in
the diffused trajectory. They are encoded separately and passed to the U-Net through
`global_cond`.

For this path, `_build_conditioning` creates:

```python
cond_data = torch.zeros([batch, 16, 7])
cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
```

See [`policy.py:473-489`](../../pg3d/policies/dp3/policy.py#L473-L489).

Therefore:

```text
condition_mask contains no True values
condition_data[condition_mask] is empty
trajectory[condition_mask] is empty
```

and:

```python
trajectory[condition_mask] = condition_data[condition_mask]
```

changes nothing for this checkpoint. The generated tensor contains only the normalized 16-by-7
action trajectory, and all of it is denoised. Observation conditioning happens through
cross-attention using `global_cond` instead.

The clamping line remains in `conditional_sample` because the same sampler supports both globally
conditioned and inpainting-style policy configurations.

## 4. Are the two conditioning types inpainting and cross-attention?

The proposed interpretation is mostly correct, with two adjustments.

### Adjustment 1: global conditioning is broader than cross-attention

The main configuration switch in this policy is `obs_as_global_cond`:

- `obs_as_global_cond=True`: diffuse only the action trajectory and provide observations through a
  separate conditioning input.
- `obs_as_global_cond=False`: concatenate encoded observation features with actions in the tensor
  being diffused, then clamp the known observation portion with a mask.

Cross-attention is one way to use the separate global-conditioning input. When
`obs_as_global_cond=True`, the policy computes `global_cond_dim` differently depending on
`condition_type`
([`policy.py:123-132`](../../pg3d/policies/dp3/policy.py#L123-L132)):

- `condition_type="cross_attention"` preserves the observations as separate tokens;
- `condition_type="film"` flattens the observation features into one conditioning vector.

Therefore, the current checkpoint uses:

```text
global conditioning
    implemented specifically with cross-attention
```

but “global conditioning” and “cross-attention” are not interchangeable terms in the generic
implementation.

### Adjustment 2: the inpainting configuration also predicts future observation features

With `obs_as_global_cond=False`, the diffused tensor contains:

```text
[actions | encoded observation features]
```

for every step in the horizon. `_build_conditioning` fills and masks the observation-feature fields
for the first `obs_steps`
([`policy.py:490-500`](../../pg3d/policies/dp3/policy.py#L490-L500)). The mask leaves all action
fields unknown. With the current `action_visible=False` setting
([`policy.py:154-160`](../../pg3d/policies/dp3/policy.py#L154-L160)), no past action values are
clamped.

Consequently, reverse diffusion generates:

- all action fields; and
- observation-feature fields after the known observation window.

It does not generate the masked observation features for the first `obs_steps`; those are restored
from `condition_data` before every model call.

The more precise statement is therefore:

> If actions and encoded observations share the diffused tensor, the known observation-feature
> prefix is masked and clamped. The model generates all unknown actions and any unmasked future
> observation-feature slots.

The values in the diffused tensor are encoded observation features, not raw 1024-point clouds.

### Current checkpoint

The current checkpoint has:

```text
obs_as_global_cond = True
condition_type     = cross_attention
```

Thus:

1. The diffused tensor contains only normalized actions with shape `[B, 16, 7]`.
2. Recent observations are encoded into three conditioning tokens.
3. The action U-Net attends to those tokens through cross-attention.
4. `cond_mask` is entirely false.
5. `trajectory[condition_mask] = condition_data[condition_mask]` is a no-op.

So the corrected summary is:

> This implementation supports separate observation conditioning and inpainting-style
> conditioning. Our checkpoint diffuses only actions and separately conditions the denoiser on
> observation tokens through cross-attention. In the alternative configuration, encoded
> observations are included in the diffused tensor; the known observation prefix is clamped, while
> unknown actions and future observation features are generated.

## 5. Is the goal point cloud encoded like the arm and environment, and is there additional action conditioning?

### The goal uses reserved points, but a separate encoder branch

Each policy observation contains one point-cloud array with 1024 XYZ points. It is more precise to
say that 192 **points**, rather than 192 point clouds, are allocated to the goal.

For this checkpoint, the array is divided as follows:

```text
point_cloud [1024, 3]

first 832 points   scene branch: robot arm + table/environment scene points
last 192 points    goal-marker branch: synthetic points centered on the goal
```

The 192 goal points are deterministic synthetic marker points. They are created around the target
position with radius 0.055 m and overwrite the final 192 slots
([`goal_markers.py:65-116`](../../pg3d/policies/dp3/goal_markers.py#L65-L116)). The dataset loader
performs this insertion for every step at
[`reach_dataset.py:171-182`](../../pg3d/policies/dp3/reach_dataset.py#L171-L182).

The goal points are therefore part of the `[1024, 3]` point-cloud tensor, but they are **not**
encoded in the same way as the other 832 points.

In `DP3Encoder.forward`:

```python
scene_points = points[:, :-goal_marker_points]
marker_points = points[:, -goal_marker_points:]

scene_feature = PointNet(scene_points)
marker_feature = marker_mlp(flatten(marker_points))
```

See [`modules.py:457-474`](../../pg3d/policies/dp3/modules.py#L457-L474).

The two branches differ:

- The first 832 points use the PointNet-style scene encoder: a shared per-point MLP followed by max
  pooling. Their order is discarded
  ([`modules.py:331-375`](../../pg3d/policies/dp3/modules.py#L331-L375)).
- The final 192 goal-marker points are flattened in their fixed order and passed through a separate
  goal-marker MLP
  ([`modules.py:423-430`](../../pg3d/policies/dp3/modules.py#L423-L430)).

This separation prevents the relatively small goal marker from disappearing in PointNet's global
max pooling and preserves its structured location signal.

The policy does not receive the dataset's `robot_mask` or segmentation labels. Consequently, within
the first 832 scene slots, arm and environment points are processed by the same PointNet branch.
Their XYZ geometry is visible, but their semantic identity is not separately supplied to the base
policy.

### Additional explicit goal conditioning

Yes. The goal-marker points are not the only goal signal used by this checkpoint.

The checkpoint has `use_goal_encoder=True`. The policy computes the 3D relative displacement:

\[
g_{\mathrm{rel}}=g_{\mathrm{world}}-p_{\mathrm{EEF}},
\]

where `goal_xyz` is the target position and `ee_position` is the current TCP position
([`policy.py:503-508`](../../pg3d/policies/dp3/policy.py#L503-L508)). This vector is passed through
a separate goal MLP producing a 64D feature
([`modules.py:441-447`](../../pg3d/policies/dp3/modules.py#L441-L447)).

Thus, the goal is represented twice:

```text
192 ordered XYZ goal-marker points -> marker MLP -> 32D feature
3D goal-relative displacement      -> goal MLP   -> 64D feature
```

### Other observation conditioning

For each observation step, the complete conditioning feature is:

```text
832 scene points -> PointNet       -> 128D
192 goal points  -> marker MLP     ->  32D
9D robot state   -> state MLP      ->  64D
3D goal relative -> goal MLP       ->  64D
                                      ----
                                      288D
```

The feature widths are assembled in
[`modules.py:434-455`](../../pg3d/policies/dp3/modules.py#L434-L455). Because this checkpoint uses
three observation steps, it produces three 288D tokens:

```text
global_cond shape = [batch, 3, 288]
```

Those tokens condition the action U-Net through cross-attention. The U-Net also receives a
sinusoidal embedding of the current diffusion timestep, because it must know the current noise
level
([`modules.py:260-291`](../../pg3d/policies/dp3/modules.py#L260-L291)).

Therefore, action prediction is conditioned on:

1. scene geometry, including arm and environment XYZ points;
2. ordered synthetic goal-marker points;
3. the 9D robot state;
4. the explicit goal-relative EEF displacement;
5. three recent observation steps; and
6. the current diffusion timestep.

The first five describe the robot task state. The diffusion timestep describes the denoising state
and is not an environment observation.
