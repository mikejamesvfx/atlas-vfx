# ADR-021 Amendment 4 (draft, for Mike): Atlas Comp may route its own jobs, with preflight and a result check

Draft written in atlas-vfx on 2026-10-31 for the owner to place in atlas-nexus
(`docs/decisions/ADR-021-comfy-farm-across-the-estate.md`). Nothing here takes effect until he does.

## Why

Rule 9 says "No scheduler", and a job router needs its own decision and evidence. The evidence is B10 and B11: a
dispatch to a box without the node pack or the model "succeeds with no image".

Atlas Comp now needs to send renders from a queue tab to render, hub or Comfy Cloud, so ws keeps working and
presets can run overnight.

## Proposed rule (narrow)

Atlas Comp (atlas-vfx `atlas_vfx/comp`) may submit its own recipe jobs to a farm ComfyUI or to Comfy Cloud, and
nothing else may, under five conditions:

1. **Preflight, every time, before anything is sent.** The target must report:
   - every `class_type` in the job's graph (`/object_info`, or Cloud's node catalogue);
   - every model file the graph names (the loaders' options, or the Cloud account's models);
   - the job's inputs (uploaded, or on a path the target can read).

   One miss and the job is refused with the reason. No silent fallback to another box.
2. **Result check.** A job is done only when the expected outputs exist: the frame count the recipe promises,
   readable as images. Otherwise it is failed, and the layer does not switch to it.
3. **One job per target at a time**, in the order queued. No load balancing or splitting across boxes. The artist
   picks the target, or "first free".
4. **Fail closed on auth.** Farm calls use the existing farm bearer token (an environment variable, named in `docs/MACHINES.local.md` only). Cloud
   calls use the owner's Comfy account. No token or LAN address in any tracked file.
5. **Cloud is opt-in per job.** The job states that reference frames leave the estate, and shows the estimated
   credits for the artist to confirm.

## What it does not change

- Remote-model proxies (rule 6).
- The shared model store (rule 5).
- Listening rules (rule 8).
- Any other tool's ability to submit jobs. Still none.

## Evidence to collect before "accepted"

- **Preflight refusal:** a job sent to a target missing one node is refused before upload. Test, and once on hub.
- **Farm run:** a relight preview on render (3090) returns 17 frames that match a ws run with the same seed and the
  cached float32 latent.
- **Cloud run:** an Alpha-Gen matte on Comfy Cloud, with its credits estimated beforehand and its frames checked
  after.
