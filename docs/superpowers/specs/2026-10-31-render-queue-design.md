# Latent pre-render, guide-from-latent node, and the render queue

Design, 2026-10-31. Proposed by Claude. Mike: "we just need a pre render system on remote via api or home server
tab". Targets chosen: home farm, Comfy Cloud, other machines later. Pre-render, in Mike's words: "an initial latent
encode at float 32 for any input footage". Also "Utilize comfy cloud options", and "Do it but add developing that
ic-Lora as a priority" (the guide-from-latent node, chosen over training a LoRA).

## Intent

Atlas Comp renders on ws today. Two things are wanted:
- Renders sent elsewhere: the home farm (render, hub), Comfy Cloud, more machines later, from a queue tab, so ws
  keeps working and presets can be queued overnight.
- Inputs encoded once at float32 and reused: the same latent on every machine, and a small file to send instead of
  frames.

## Findings this design rests on (checked 2026-10-31)

- **LTX guide nodes encode internally.** `LTXAddVideoICLoRAGuide` (ComfyUI-LTXVideo `iclora.py`) VAE-encodes the
  image, calls `nodes_lt.LTXVAddGuide.append_keyframe`, then `iclora_attention.append_guide_attention_entry`.
  - No node takes a pre-encoded latent as the guide.
- **Downscale factor.** The Relight and Alpha-Gen LoRAs both carry `reference_downscale_factor` 1, so no dilation
  step applies.
- **Comfy Cloud has:**
  - the LTX-2.5 distilled int8 base, Gemma 4 int8, the LTX-2.5 video and audio VAEs;
  - the Alpha-Gen, clean-plate and restore IC-LoRAs;
  - `LTXAddVideoICLoRAGuide`, `SaveLatent` and `LoadLatent`.
- **Comfy Cloud lacks:**
  - the LTX-2.3 Relight IC-LoRA: not in the public catalog, but importable into the account;
  - atlas-vfx's nodes.
- **ADR-021 rule 9 (no scheduler):** a job router needs its own decision, because a dispatch to a box without the
  pack or model "succeeds with no image". A draft amendment is in `docs/dev/ADR-021-amendment-4-draft.md`.
- **Encode cost.** Encoding a 97-frame 1280x704 reference is about 10-20 s of a 4-7 min relight. The win from a
  cached latent is consistency and transfer size (about 6 MB against about 150 MB of frames), not speed. Measure it.

## Goals

1. **Priority: guide-from-latent.**
   - **Atlas VFX Latent Pre-render** encodes a reference clip once with the VAE in float32 and writes a `.latent`
     (safetensors) plus a sidecar: hash of the frames, VAE file, size, frame count.
   - **Atlas VFX Guide From Latent** loads it and adds it as the IC-LoRA guide exactly as `LTXAddVideoICLoRAGuide`
     would (frame 0, downscale 1, the same keyframe and attention entries).
   - **Proof:** with the same seed, a relight run from the cached latent matches the image-guided run.
2. **The render queue tab** in Atlas Comp.
   - **Targets:** ws, render, hub, Comfy Cloud, plus any ComfyUI added by address.
   - **Three stages per job:** pre (local: Prep, the ball, reference frames, the cached latent), generate (any
     target: standard and LTX nodes only, plus the guide-from-latent node where installed) and post (local: Return,
     Relight Transfer Files, a new version).
   - **Preflight before sending:**
     - nodes, from the target's `/object_info` or Cloud's node search;
     - models, from loader options or the Cloud account's models;
     - inputs;
     - on Cloud, estimated credits, which the artist confirms.
   - **After the run:** the expected frame count must come back, or the job fails.
   - **Running:** one job per target, in order, with cancel and live status.
3. **Comfy Cloud target.**
   - Reference frames uploaded with its upload API, the graph submitted, the outputs fetched.
   - Credits estimated first and confirmed by the artist.
   - The Relight LoRA imported into Mike's account, his action to approve.

## Non-goals

- Training our own IC-LoRA. Scoped later: a relight LoRA that obeys the sun direction without the mirror trick.
- A general farm scheduler beyond Atlas Comp jobs.
- Running atlas-vfx nodes on Comfy Cloud.

## Anti-goals

- A job that "succeeds with no image".
- Plates leaving the machine without the artist knowing: a Cloud job says so and shows its cost first.
- Tokens or LAN addresses in tracked files. Targets live in an untracked local config: the farm token
  stays an environment variable, and addresses go in `docs/MACHINES.local.md` or a `*.local.json` file.

## Constraints

- The licence register comes first for anything new.
- No new dependencies.
- Widgets and HTTP input are untrusted. Latent files are safetensors, read with `safe_load`, never pickle.
- atlas-vfx imports ComfyUI-LTXVideo's `nodes_lt` / `iclora_attention` only inside the guide node, lazily. The pack
  is already required by every LTX graph.
- One decision, Mike's: the ADR-021 amendment.

## Order

1. Guide-from-latent (two nodes, tests, live equivalence on V141).
2. The render queue with ws and farm targets and preflight.
3. The Comfy Cloud target.
