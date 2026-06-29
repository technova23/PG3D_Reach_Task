# Dependency mirroring and submodule runbook

Use private mirrors, not public GitHub forks, for dependency repos we expect to modify. A private mirror lets us experiment without exposing private patches, notes, or in-progress research code.

## Repos to mirror

Required now:

- DP3 / 3D-Diffusion-Policy -> private mirror at
  `git@github.com:krrish94/dp3-for-pg3d.git`, submodule at `external/dp3`, branch `main`.
  This submodule is reference material while the narrow runtime policy core is ported into
  `pg3d/policies/dp3`.

Optional only if we patch them:

- ManiSkill/SAPIEN -> private mirror or pinned dependency.

Do not mirror every dependency by default. Mirror only when we need to patch or pin fragile code.

## Create a private mirror on GitHub

1. Create an empty private repo, e.g. `krrish94/dp3-for-pg3d`.
2. Mirror-push the upstream repo:

```bash
mkdir -p ~/src/mirrors
cd ~/src/mirrors

git clone --bare https://github.com/YanjieZe/3D-Diffusion-Policy.git dp3-for-pg3d.git
cd dp3-for-pg3d.git
git push --mirror git@github.com:krrish94/dp3-for-pg3d.git
cd ..
rm -rf dp3-for-pg3d.git
```

3. Clone the private mirror and add upstream remote:

```bash
git clone git@github.com:krrish94/dp3-for-pg3d.git
cd dp3-for-pg3d
git remote add upstream https://github.com/YanjieZe/3D-Diffusion-Policy.git
git remote -v
```

4. Use `main` as the pg3d-owned private branch:

```bash
git branch --show-current
# If needed:
git branch -m master main
git push -u origin main
```

## Add private mirror as pg3d submodule

From the root of `pg3d`:

```bash
mkdir -p external
git submodule add -b main git@github.com:krrish94/dp3-for-pg3d.git external/dp3
git commit -m "Add private DP3 mirror as submodule"
```

Runtime pg3d code should not import from `external/dp3`. Import `pg3d.policies.dp3` instead.

Clone/update on another workstation:

```bash
git clone --recurse-submodules git@github.com:YOUR_ORG/pg3d.git
cd pg3d
git submodule update --init --recursive
```

## Update private DP3 mirror from upstream

Inside `external/dp3`:

```bash
git fetch upstream
git checkout main
git merge upstream/master
# resolve conflicts, test, then:
git push origin main
```

Then update submodule pointer in main repo:

```bash
cd ../..
git status
git add external/dp3
git commit -m "Update DP3 submodule pointer"
```

## Rules for Codex

- Do not edit `external/dp3` unless explicitly asked.
- If editing `external/dp3`, summarize changes both in the DP3 mirror and in pg3d work logs.
- Never vendor a full dependency repo into `pg3d` outside `external/`. Port only narrow,
  reviewed runtime slices such as `pg3d/policies/dp3`.
- Keep `.gitmodules` URLs SSH-based for private submodules.
- Record submodule commit hashes in experiment logs.
