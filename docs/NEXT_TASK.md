# Next task — diagnose the depth scale before fixing the 3D panel

This answers the A/B/C question about the empty 3D panel. **Do not pick a
rendering fix yet.** One number in the smoke-test report needs checking
first, because it may be a data bug rather than a display problem.

## Why

The report says WiLoR estimates each hand's absolute depth at **9–15 m**,
with the two hands **0.84 m apart**.

Both are physically impossible for this footage. These are cap-mounted
egocentric cameras: hands are at arm's length, **0.4–0.7 m** from the
lens, and a person working a shoe holds them **0.2–0.4 m** apart. Being
out by a factor of ~20 in depth is not weak-perspective jitter, it is a
scale error — almost certainly focal length.

This matters beyond the panel. If the translation is wrong, `kp3d` in the
delivered `.npz` is wrong, and `README_FOR_CUSTOMER.md` currently promises
"camera-space, metres". Shipping that to a technical evaluator who checks
it would be exactly the kind of quiet inaccuracy this tool exists to
prevent. Rendering wrist-relative would hide the symptom and leave the
delivered data wrong.

## Step 1 — separate "translation is wrong" from "everything is wrong"

MANO's *relative* joint positions are metric and trustworthy; only the
global translation is in question. So measure the hand against itself, on
a frame where both hands are tracked:

```
wrist -> middle-finger MCP   (joint 0 -> joint 9)    expect ~0.09 m
thumb tip -> pinky tip       (joint 4 -> joint 20)   expect ~0.18-0.22 m
```

- Intra-hand distances correct, translation absurd → the bug is isolated
  to `pred_cam_t_full` / focal length. **Fixable.**
- Intra-hand distances also wrong (e.g. a 2 m wide hand) → the whole scale
  is off, and the `.npz` units claim must change.

Print the actual numbers.

## Step 2 — check the focal length

Report what WiLoR-mini returns for focal length (`scaled_focal_length` or
equivalent), and confirm that the value used in the crop-to-full-image
camera translation is **the same one**, scaled to *this* video's
resolution rather than the model's nominal crop size.

A mismatch here scales depth linearly and is the most likely culprit.

Print the value, the video's width/height, and the conversion actually
applied.

## Step 3 — then, and only then, fix the panel

**If the translation is fixable:** prefer a correct camera-space panel.
Widen `view_half_extent_m` to fit two hands at their true separation
(0.2–0.4 m apart plus hand size suggests roughly 0.35–0.45 m half-extent)
and keep the fixed-scale, no-autoscale rule. A truthful panel beats a
flattering one.

**If it is genuinely not fixable,** take option A (per-hand wrist-relative
rendering), with these conditions — all three are required, not optional:

1. Change the 3D panel caption to include **WRIST-RELATIVE**. As it
   stands the caption implies a shared metric space, and after this change
   that is no longer what the panel shows.
2. Do **not** draw both hands wrist-relative at a common origin as though
   that were their true relative position. Either offset them by a fixed
   display constant (and say so in the caption), or draw two sub-panels.
   Inventing a plausible-looking inter-hand geometry is worse than showing
   none.
3. Amend `README_FOR_CUSTOMER.md`: state that the 3D panel is displayed
   wrist-relative for legibility, that absolute depth is not reliable at
   this stage, and exactly what the `.npz` contains.

## Step 4 — README correction, regardless of the above

This is independent of the 3D panel and needs doing now.

With the WiLoR-mini backend, `pose_recovery_pct` is structurally 100%: the
regressor never reports failure, so every detected hand gets a pose. The
"hand visible but pose lost" case cannot be measured with this backend.

Update `README_FOR_CUSTOMER.md` so that:

- `pose_recovery_pct` is **no longer the headline quality metric**.
- The document states plainly that this backend cannot distinguish
  "tracked" from "hand visible but pose not recovered", and that a 100%
  figure therefore reflects the model's behaviour, not measured accuracy.
- The `hand_visible` / `valid` section says the two flags are equal for
  data produced by this backend, and points at the `.npz` field
  `pose_recovery_measurable=False` as the machine-readable marker.

A 100% recovery rate reaching a technical evaluator unqualified is the
single worst outcome available here. It would be read as a claim, checked,
and found hollow.

## Step 5 — throughput, before any full batch

Inference measured 5.5 fps, described as per-frame YOLO+pose with no
batching. That is ~2.3 h per 45k-frame video single-worker.

Try batching the pose regressor across detections (and across frames if
the pipeline allows) and report the new fps. Several-fold gains are
plausible and worth having before committing to ten full-length videos.

Do not start the full batch. Report back with the numbers from steps 1, 2
and 5, and what you changed for 3 and 4.

## Note for later — a real limitation in this tool

`--smoke-test N` bounds the *render* window but not the analysis pass, so
a smoke test on a 25-minute source still runs inference over the whole
file. That is why trimming the clip was necessary. Worth fixing, but not
now.
