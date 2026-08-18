# How the Base Diffusion Policy Works

This document explains the ordinary, unguided diffusion policy implemented in this repository.
It follows the actual code path used to train and evaluate the Panda reach checkpoint, rather than
describing diffusion policies only in the abstract.

ITPS will be added to this document later. Until then, everything below describes the base policy
without ITPS, geometric guidance, rejection, reranking, or world-model costs.

Focused follow-up questions are collected in
[`diffusion_policy_questions.md`](diffusion_policy_questions.md).

## Reference scope

The code references below were verified on branch `itps-stochastic-sampling-baseline` after cleanup
commit `9382279`. Line numbers should be updated if the referenced source files change.

The canonical implementation is under `pg3d/policies/dp3/`:

- [`policy.py`](../../pg3d/policies/dp3/policy.py#L49-L168) constructs the policy and scheduler.
- [`modules.py`](../../pg3d/policies/dp3/modules.py#L35-L309) implements the timestep embedding,
  conditional one-dimensional U-Net, and cross-attention blocks.
- [`modules.py`](../../pg3d/policies/dp3/modules.py#L331-L496) also implements the PointNet-style
  observation encoder.
- [`normalizer.py`](../../pg3d/policies/dp3/normalizer.py#L11-L179) implements normalization and
  unnormalization.
- [`reach_dataset.py`](../../pg3d/policies/dp3/reach_dataset.py#L24-L193) constructs training
  sequences.
- [`train_dp3_reach.py`](../../scripts/train_dp3_reach.py#L94-L142) runs optimization and EMA
  updates.
- [`checkpoint.py`](../../pg3d/policies/dp3/checkpoint.py#L42-L93) saves and reloads the policy.

## 1. What the policy models

The policy models a distribution over a sequence of robot actions conditioned on recent
observations:

\[
p_\theta(a_{0:H-1}\mid o_{-S+1:0}).
\]

Here:

- \(H\) is the predicted action horizon;
- \(S\) is the number of observation steps;
- each observation contains a point cloud and robot state, plus goal information for this
  checkpoint;
- each action contains seven Panda arm-joint targets.

The network does not directly output the final action trajectory in one pass. It is trained to
predict noise added to an action trajectory. During inference, repeated noise predictions are used
by a diffusion scheduler to turn a random trajectory into a structured action trajectory.

`SimpleDP3` is the concrete policy class
([`policy.py:49-168`](../../pg3d/policies/dp3/policy.py#L49-L168)). Its two main learned components
are created here:

1. `DP3Encoder` converts each observation into a conditioning feature
   ([`policy.py:109-122`](../../pg3d/policies/dp3/policy.py#L109-L122)).
2. `ConditionalUnet1D` predicts the diffusion target for the complete action sequence
   ([`policy.py:123-142`](../../pg3d/policies/dp3/policy.py#L123-L142)).

## 2. Configuration of the 65,000-step checkpoint

The generic class and training script have defaults, but checkpoints store the exact constructor
arguments used for that run. `load_reach_policy_from_checkpoint` reconstructs `SimpleDP3` from
`checkpoint["policy_kwargs"]`
([`checkpoint.py:75-92`](../../pg3d/policies/dp3/checkpoint.py#L75-L92)).

The checkpoint `/scratch2/skills/train_final_Arya/step_00065000.pt` stores:

| Property | Value |
|---|---:|
| Training optimizer step | 65,000 |
| Action horizon \(H\) | 16 |
| Observation steps \(S\) | 3 |
| Actions executed per prediction | 8 |
| Action dimension | 7 |
| Point-cloud shape per observation | 1024 x 3 |
| Robot-state dimension | 9 |
| Diffusion training timesteps | 350 |
| Diffusion inference steps | 100 |
| Prediction type | `epsilon` |
| Conditioning | `cross_attention` |
| U-Net channel widths | 512, 1024, 2048 |
| Goal-marker points | 192 |
| Goal-marker radius | 0.055 m |
| Explicit goal-relative encoder | enabled |
| EMA weights | present |

This distinction matters. For example, the current CLI default is 350 inference steps
([`train_dp3_reach.py:352-354`](../../scripts/train_dp3_reach.py#L352-L354)), but this checkpoint
uses the stored value of 100.

## 3. Constructing a training example

### 3.1 Sequence extraction

`ReachSequenceDataset` opens the Zarr dataset and builds fixed-length sequence indices without
crossing episode boundaries
([`reach_dataset.py:68-104`](../../pg3d/policies/dp3/reach_dataset.py#L68-L104)). Near an episode
boundary, missing values are padded by repeating the first or last available value
([`reach_dataset.py:377-398`](../../pg3d/policies/dp3/reach_dataset.py#L377-L398)).

For this checkpoint, one item contains approximately:

```text
batch["obs"]["point_cloud"]  [16, 1024, 3]
batch["obs"]["agent_pos"]    [16, 9]
batch["obs"]["goal_xyz"]     [16, 3]
batch["obs"]["ee_position"]  [16, 3]
batch["action"]               [16, 7]
```

The exact conversion to tensors is in
[`reach_dataset.py:171-193`](../../pg3d/policies/dp3/reach_dataset.py#L171-L193). Although a
16-step sequence is loaded for every observation field, the globally conditioned policy later uses
only the first `n_obs_steps=3` observations.

### 3.2 Goal markers

The final 192 point-cloud slots are overwritten with a deterministic pattern centered on the goal
([`goal_markers.py:81-116`](../../pg3d/policies/dp3/goal_markers.py#L81-L116)). They are ordered
tokens, not ordinary randomly sampled scene points.

The dataset performs this insertion in
[`reach_dataset.py:174-182`](../../pg3d/policies/dp3/reach_dataset.py#L174-L182). The encoder knows
that the trailing 192 points are special and sends them to a separate MLP instead of the
permutation-invariant PointNet branch.

### 3.3 Normalization

The training dataset fits affine normalization statistics in
[`reach_dataset.py:128-162`](../../pg3d/policies/dp3/reach_dataset.py#L128-L162):

- observations are standardized;
- actions use min-max normalization by default, mapping their observed range to `[-1, 1]`.

For one field, normalization is:

\[
x_{\mathrm{norm}} = x\,s+b,
\]

and unnormalization is:

\[
x = \frac{x_{\mathrm{norm}}-b}{s}.
\]

These operations are implemented at
[`normalizer.py:46-54`](../../pg3d/policies/dp3/normalizer.py#L46-L54). Mapping actions to
`[-1, 1]` is important because the diffusion scheduler is configured with `clip_sample=True`.

## 4. Encoding the observation

At training time, `compute_loss` first normalizes the observations and actions
([`policy.py:387-399`](../../pg3d/policies/dp3/policy.py#L387-L399)). When the explicit goal encoder
is enabled, it also computes:

\[
g_{\mathrm{rel}} = g_{\mathrm{world}} - p_{\mathrm{EEF}},
\]

using [`policy.py:503-508`](../../pg3d/policies/dp3/policy.py#L503-L508).

Because this checkpoint uses global conditioning, the first three observations are flattened from
`[B, 3, ...]` to `[B*3, ...]`, encoded independently, and then reshaped back into three observation
tokens
([`policy.py:407-416`](../../pg3d/policies/dp3/policy.py#L407-L416)).

For each observation step, `DP3Encoder` combines four branches:

1. The first 832 scene points go through a PointNet-style encoder.
2. The final 192 goal-marker points are flattened and passed through a marker MLP.
3. The 9D robot state goes through a state MLP.
4. The 3D goal-relative vector goes through a goal MLP.

The branch construction is in
[`modules.py:378-455`](../../pg3d/policies/dp3/modules.py#L378-L455), and the forward pass is in
[`modules.py:457-492`](../../pg3d/policies/dp3/modules.py#L457-L492).

The PointNet branch applies an MLP independently to every point and then max-pools over points
([`modules.py:331-375`](../../pg3d/policies/dp3/modules.py#L331-L375)). For this checkpoint, the
concatenated observation feature has 288 dimensions:

```text
PointNet scene feature       128
robot-state feature           64
goal-marker feature           32
goal-relative feature         64
                            -----
total                         288
```

The resulting global condition has shape `[B, 3, 288]`.

## 5. The diffusion model

### 5.1 Forward diffusion

Let \(x_0\) denote a clean, normalized 16-by-7 action trajectory. The scheduler defines a noise
variance \(\beta_t\), with:

\[
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{i=0}^{t}\alpha_i.
\]

The forward diffusion equation is:

\[
x_t=\sqrt{\bar\alpha_t}\,x_0+
\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

The policy constructs a `DDIMScheduler` with 350 training timesteps, a scaled-linear beta schedule
from `0.0001` to `0.05`, clipping enabled, and epsilon prediction
([`policy.py:143-152`](../../pg3d/policies/dp3/policy.py#L143-L152)). DDIM and DDPM use the same
marginal forward-noising equation; their important difference here is the inference transition.

### 5.2 What one training step does

`compute_loss` implements one diffusion behavior-cloning loss
([`policy.py:387-453`](../../pg3d/policies/dp3/policy.py#L387-L453)):

1. Normalize the clean action sequence to obtain \(x_0\).
2. Encode the observation into conditioning tokens.
3. Draw independent Gaussian noise \(\epsilon\) with the same shape as \(x_0\).
4. Draw one random timestep for each batch item, uniformly from `0` through `349`.
5. Use `scheduler.add_noise` to construct \(x_t\).
6. Give \(x_t\), timestep \(t\), and the observation tokens to the U-Net.
7. Train the U-Net output to match the exact noise \(\epsilon\) that was added.

The sampling of noise and timesteps occurs at
[`policy.py:425-433`](../../pg3d/policies/dp3/policy.py#L425-L433). The denoiser call is at
[`policy.py:436-440`](../../pg3d/policies/dp3/policy.py#L436-L440).

For epsilon prediction, the target is selected explicitly here:

```python
if pred_type == "epsilon":
    target = noise
```

See [`policy.py:442-448`](../../pg3d/policies/dp3/policy.py#L442-L448).

The implementation uses elementwise Huber loss with `delta=1`, rather than plain mean-squared
error, and then averages over the batch, horizon, and action dimensions
([`policy.py:450-453`](../../pg3d/policies/dp3/policy.py#L450-L453)):

\[
\mathcal L(\theta)=
\mathbb E_{x_0,t,\epsilon}
\left[
\operatorname{Huber}\left(
\epsilon_\theta(x_t,t,o),\epsilon
\right)
\right].
\]

With this checkpoint's globally conditioned action-only trajectory, the conditioning mask is all
false. Therefore, all 16-by-7 action values are noised and included in the loss. The mask machinery
also supports another configuration in which observation features are concatenated into the
trajectory, but that is not the configuration used here
([`policy.py:417-435`](../../pg3d/policies/dp3/policy.py#L417-L435)).

### 5.3 What the U-Net predicts

`ConditionalUnet1D.forward` receives:

```text
sample       x_t             [B, 16, 7]
timestep     t               [B] or scalar
global_cond  observation     [B, 3, 288]
```

It rearranges the action trajectory to `[B, 7, 16]` so one-dimensional convolutions operate over
the 16-step temporal axis
([`modules.py:260-274`](../../pg3d/policies/dp3/modules.py#L260-L274)). The scalar timestep is
converted to a sinusoidal embedding and processed by an MLP
([`modules.py:35-48`](../../pg3d/policies/dp3/modules.py#L35-L48) and
[`modules.py:195-203`](../../pg3d/policies/dp3/modules.py#L195-L203)).

For cross-attention, the timestep embedding is attached to each of the three observation tokens
([`modules.py:275-291`](../../pg3d/policies/dp3/modules.py#L275-L291)). Conditional residual blocks
let action-time features attend to those observation tokens
([`modules.py:99-120`](../../pg3d/policies/dp3/modules.py#L99-L120) and
[`modules.py:123-177`](../../pg3d/policies/dp3/modules.py#L123-L177)).

The U-Net downsamples and upsamples along the action horizon, using skip connections, and returns a
tensor with the original shape `[B, 16, 7]`
([`modules.py:293-309`](../../pg3d/policies/dp3/modules.py#L293-L309)). Every output element is a
prediction of the noise added to the corresponding normalized action element.

## 6. Updating the model

The trainer constructs `SimpleDP3`, installs the dataset normalizer, and creates an AdamW optimizer
in [`train_dp3_reach.py:94-109`](../../scripts/train_dp3_reach.py#L94-L109).

For every optimizer step it:

1. gets a batch;
2. computes the diffusion loss;
3. backpropagates;
4. clips gradients;
5. updates the trainable model;
6. updates the learning-rate scheduler;
7. updates the EMA copy.

This loop is at
[`train_dp3_reach.py:124-142`](../../scripts/train_dp3_reach.py#L124-L142). The EMA update is a
weighted average:

\[
\theta_{\mathrm{EMA}}
\leftarrow
d\,\theta_{\mathrm{EMA}}+(1-d)\,\theta,
\]

implemented at [`modules.py:562-607`](../../pg3d/policies/dp3/modules.py#L562-L607).

The checkpoint stores raw weights, EMA weights, normalization statistics, optimizer state,
learning-rate scheduler state, policy constructor arguments, and the optimizer-step number
([`checkpoint.py:42-72`](../../pg3d/policies/dp3/checkpoint.py#L42-L72)). Evaluation normally loads
the EMA weights.

## 7. Ordinary inference

Training teaches the network to answer this question:

> Given a noisy action trajectory \(x_t\), noise level \(t\), and observation \(o\), what noise
> \(\epsilon\) was probably added?

Inference repeatedly asks that question to construct a clean trajectory.

### 7.1 Build the live observation window

At the start of an episode, the first observation is repeated three times. Later, the oldest
observation is dropped whenever a new one arrives
([`rollout_dp3_reach_policy.py:417-437`](../../scripts/rollout_dp3_reach_policy.py#L417-L437)).

The window is converted to batched tensors, with goal markers inserted using the same convention
as training
([`rollout_dp3_reach_policy.py:440-470`](../../scripts/rollout_dp3_reach_policy.py#L440-L470)).

`predict_action` then calls `_build_conditioning`
([`policy.py:362-377`](../../pg3d/policies/dp3/policy.py#L362-L377)). This method applies the same
normalization and observation encoder used during training
([`policy.py:455-501`](../../pg3d/policies/dp3/policy.py#L455-L501)).

For this globally conditioned checkpoint:

```text
cond_data   zeros with shape [B, 16, 7]
cond_mask   false with shape [B, 16, 7]
global_cond encoded observations with shape [B, 3, 288]
```

The zero `cond_data` is only a shape/device template in this configuration. Observations are
provided through `global_cond`, not inserted into the action trajectory.

### 7.2 Start from Gaussian noise

`conditional_sample` begins with:

\[
x_T\sim\mathcal N(0,I),
\]

implemented by `torch.randn` at
[`policy.py:211-224`](../../pg3d/policies/dp3/policy.py#L211-L224). This initial tensor already has
the final trajectory shape `[B, 16, 7]`, but initially contains only noise.

The checkpoint requests 100 inference steps. `scheduler.set_timesteps(100)` selects a descending
sequence of 100 noise levels from the 350-level training schedule
([`policy.py:225-226`](../../pg3d/policies/dp3/policy.py#L225-L226)).

### 7.3 One reverse-diffusion step

At the current level \(t\), the U-Net predicts:

\[
\hat\epsilon_\theta=\epsilon_\theta(x_t,t,o).
\]

From that prediction, the scheduler can estimate the clean trajectory:

\[
\hat x_0=
\frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon_\theta}
{\sqrt{\bar\alpha_t}}.
\]

This is why reverse diffusion should not be understood as simply executing
`x_t - predicted_noise`. The scheduler combines the sample and predicted noise using coefficients
from the noise schedule, applies its configured clipping behavior, and computes the sample for the
next selected noise level.

The actual base-policy loop is only these operations:

```python
for timestep in self.noise_scheduler.timesteps:
    trajectory[condition_mask] = condition_data[condition_mask]
    model_output = self.model(
        sample=trajectory,
        timestep=timestep,
        global_cond=global_cond,
    )
    trajectory = self.noise_scheduler.step(
        model_output,
        timestep,
        trajectory,
    ).prev_sample
```

See [`policy.py:225-235`](../../pg3d/policies/dp3/policy.py#L225-L235).

The default base scheduler is DDIM. With its default `eta=0`, each scheduler transition is
deterministic once the initial Gaussian trajectory is fixed. Changing the initial random sample can
still produce a different final action trajectory.

This predict-and-step operation is repeated 100 times:

```text
Gaussian x_T
  -> predict noise -> scheduler step -> x_t1
  -> predict noise -> scheduler step -> x_t2
  -> ...
  -> predict noise -> scheduler step -> x_0 approximation
```

There is no obstacle energy, gradient, world-model call, candidate scoring, rejection, or
reranking in this loop.

### 7.4 Unnormalize and select executable actions

After denoising, `predict_action` extracts the seven action dimensions and unnormalizes the complete
16-step trajectory
([`policy.py:378-379`](../../pg3d/policies/dp3/policy.py#L378-L379)).

It returns:

```text
action_pred  all 16 predicted actions
action       the 8-action execution slice
```

The slice starts at `n_obs_steps - 1`. For this checkpoint:

```text
start = 3 - 1 = 2
end   = 2 + 8 = 10
action = action_pred[:, 2:10]
```

The exact slicing is at
[`policy.py:380-385`](../../pg3d/policies/dp3/policy.py#L380-L385). This offset follows the diffusion
policy sequence convention: the predicted trajectory is aligned with the padded observation/action
window, and execution begins at the action aligned with the most recent observation.

Each 7D policy action is an absolute Panda arm-joint target for this dataset. The rollout adapter
adds the gripper command to construct the simulator's 8D action and clips it to the simulator action
bounds
([`rollout_dp3_reach_policy.py:496-528`](../../scripts/rollout_dp3_reach_policy.py#L496-L528)).

## 8. What `--methods base` means

The constrained evaluator's `base` branch requests exactly one action chunk:

```python
if method == "base":
    chunk = adapter.sample_action_chunks(obs_window, k=1, rng=rng)[0]
```

See
[`eval_constrained_reach.py:1241-1249`](../../scripts/eval_constrained_reach.py#L1241-L1249).
The adapter invokes `policy.predict_action` under `torch.inference_mode()`
([`eval_constrained_reach.py:227-230`](../../scripts/eval_constrained_reach.py#L227-L230)).

Therefore, `base` means:

1. encode the current observation window;
2. draw one Gaussian action trajectory;
3. run ordinary DDIM reverse diffusion;
4. execute the returned eight-action slice;
5. observe again and replan.

The evaluator may calculate obstacle-clearance metrics after execution, but those constraints do
not influence base-policy sampling or action selection.

## 9. End-to-end summary

### Training

```text
demonstrated joint trajectory x_0
        +
recent point clouds, robot states, and goal
        |
        v
normalize and encode observations
        |
sample timestep t and Gaussian noise epsilon
        |
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon
        |
conditional U-Net predicts epsilon_hat
        |
Huber(epsilon_hat, epsilon)
        |
optimizer update + EMA update
```

### Inference

```text
recent live observations
        |
normalize and encode into three conditioning tokens
        |
initialize x_T as a random [16, 7] trajectory
        |
100 iterations of:
    epsilon_hat = U-Net(x_t, t, observation)
    x_previous = DDIM_scheduler.step(epsilon_hat, t, x_t)
        |
unnormalize the complete [16, 7] trajectory
        |
return action_pred[:, 0:16]
execute action_pred[:, 2:10]
```

## 10. ITPS extension

Reserved for the next part of this explanation. It will begin from the ordinary inference loop in
Section 7 and identify exactly which operations ITPS changes, which operations remain unchanged,
and how the same trained noise-prediction model is reused without retraining.
