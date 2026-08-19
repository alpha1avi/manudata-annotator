# ManuData — Hand Pose QC Renders

This note accompanies the QC videos and `qc_report.csv`. It describes what
produced the keypoints, the conventions they follow, and how to read the
visibility flags that ship with every frame.

---

## What produced the keypoints

| | |
|---|---|
| Hand detection | YOLO hand detector bundled with WiLoR-mini |
| Pose regression | **WiLoR-mini** (ViT-based 3D hand pose), 21 joints per hand |
| Joint convention | MANO / OpenPose 21-joint hand ordering |
| Max hands per frame | 2 (one left, one right) |

Two properties of this backend shape how the numbers below must be read;
both are stated where they matter and summarised here so nothing is buried:

- It exposes **no per-hand pose confidence** and never reports a pose
  failure, so "hand visible but pose not recovered" cannot be measured —
  see *The visibility flags*.
- It carries **no camera calibration**, so absolute depth rests on an
  assumed field of view — see *Camera intrinsics*.

The exact checkpoint identifier and package version used for a given
delivery are recorded in the `meta` block of every `.npz` keypoint file
(`model`, `model_version`) and in the `.done.json` sidecar next to each
rendered MP4. Read them from the delivery rather than from this document,
which describes the pipeline rather than any one run.

## Coordinate convention and units

**3D keypoints** (`kp3d`, shape `(frames, 2, 21, 3)`)

- **Units: metres.**
- Camera-space, not world-space: the origin is the camera's optical
  centre for that frame.
- Axes follow the standard vision convention — **+X right, +Y down,
  +Z forward** (away from the camera, into the scene).
- Hand index `0` is the **left** hand, index `1` is the **right**.
- Joint `0` is the wrist; joints `1–4` thumb, `5–8` index, `9–12` middle,
  `13–16` ring, `17–20` pinky, each running proximal to distal.

Because the frame is camera-space, both hands in a frame are in one
consistent metric space and inter-hand distances are meaningful. Across
frames the origin moves with the camera; if you need a static world
frame, compose with the camera trajectory from the SLAM stage.

**2D keypoints** (`kp2d`, shape `(frames, 2, 21, 2)`) are the same joints
projected into **source-video pixel coordinates**, origin at the top-left
of the frame, in the video's native resolution.

## Camera intrinsics

These are cap-mounted cameras on a factory floor and we do **not** have
per-unit calibrated intrinsics for them. Rather than imply a precision we
do not have, we state the assumption plainly:

- WiLoR resolves the inherent monocular depth ambiguity with a fixed
  nominal focal length. Left as-is that places the hands at ~12 m, which is
  physically wrong for arm's-length footage, so we rescale depth to a
  plausible **assumed horizontal field of view** (default 65°, giving
  hands at roughly arm's length). Depth is linear in focal length and the
  2D projection is invariant to it, so this corrects the depth **scale**
  without moving a single 2D keypoint.
- The assumed FOV, the focal length and principal point used, and the flag
  `absolute_depth_calibrated=False` are written into every `.npz`
  (`assumed_hfov_deg`, `focal_length_px`, `principal_point_px`).

The consequence: **relative** 3D geometry — finger articulation, grasp
aperture, and the in-frame hand-to-hand separation — is well conditioned,
while the **absolute** depth of a hand from the camera is an assumption,
not a measurement, and the front-to-back offset between the two hands
inherits that same uncertainty. If you intend to use absolute metric
depth, calibrate against a known object in the scene, or tell us the
camera's true field of view and we will re-run with it.

## The visibility flags — please filter on these

Every frame carries two independent boolean flags per hand. They mean
different things and are deliberately not merged:

| Flag | Meaning |
|---|---|
| `hand_visible[t, h]` | The hand detector found hand `h` in frame `t`. A hand is in frame and not fully occluded. **A property of the scene.** |
| `valid[t, h]` | The pose regressor recovered a 21-joint pose for that detection. **A property of our tracker.** |

Which gives three states:

1. **`hand_visible=1, valid=1`** — tracked. `kp3d`/`kp2d` are populated.
2. **`hand_visible=1, valid=0`** — the hand was there and we did not
   recover its pose. Keypoints are `NaN`. **This is our failure**, and
   it is the only thing that should count against tracking quality.
3. **`hand_visible=0`** — no hand present, or fully occluded. Keypoints
   are `NaN`. **This is ground truth about the footage**, not an error.

Missing keypoints are always `NaN`, never zero or a held-over value, so
`np.isnan` is a safe test and no interpolated value can be mistaken for a
measurement.

> **Important for this delivery.** The WiLoR-mini backend used here has no
> per-hand pose confidence and never reports a pose failure: every detected
> hand is given a pose. State 2 above (`hand_visible=1, valid=0`) therefore
> **cannot occur** with this backend, and `hand_visible` is set **equal to**
> `valid` for every hand. The machine-readable marker is
> `pose_recovery_measurable=False` in the `.npz` `meta` block. The practical
> effect: we can measure *whether a hand was detected*, but not *whether we
> failed to pose a detected hand* — so "tracked" and "detected" are the same
> event in this data, and the recovery figure below is not a measured
> accuracy. This is a limitation of the current backend, called out so the
> number is not mistaken for one it cannot be.

### Occlusion is an expected property of this data

These are real assembly and finishing operations on an operating factory
line, captured egocentrically. Hands go behind the workpiece, under
tooling, out of the cap camera's field of view, and behind the operator's
own body — continuously and by the nature of the work. A recording of
this work with no occluded frames would indicate a staged capture, not a
better one.

That is why we report the two cases separately. Collapsing them into a
single "missing data rate" would describe the factory and the tracker with
one number that measures neither.

### Pose-recovery rate — and why it is not the headline for this delivery

> **Pose-recovery rate = recovered poses ÷ hand-slots where a hand was
> visible.**

This is `pose_recovery_pct` in `qc_report.csv`, and it is designed to be
the measure of our tracker: computed strictly over slots where a hand was
detected, so occlusion neither flatters nor penalises it.

**With the WiLoR-mini backend it is structurally 100%.** As noted above,
the regressor poses every detected hand, so recovered always equals
visible. A 100% here reflects the model's behaviour — it always returns a
pose — **not** a measured accuracy, and it should not be read as one. It
stays in the CSV and the header for continuity and because it becomes a
real measurement the moment a backend with a pose-failure signal is used;
until then, treat `pose_recovery_measurable=False` as the caveat that
travels with it. What *is* honestly measured here is **detection
coverage** — `occluded_or_absent_pct`, the share of hand-slots with no
hand in view — reported alongside it as a property of the footage. Both
are per-video in the CSV, with portfolio-wide figures at the end of the
ranked summary.

## Reading the QC videos

**Left panel** — source frame with the 21 keypoints projected and drawn as
a skeleton. Each finger has its own colour; the left hand uses a warm
palette and the right a cool one, labelled `L` and `R` at the wrist.

**Right panel** — the same keypoints in 3D on a fixed ground grid with an
axis triad, orbiting at 15°/s so depth is readable. The view scale and
centre are **fixed for the whole video** — the skeleton's apparent size is
therefore comparable across the clip and between clips.

**Header** — site, task, source resolution and frame rate, the current
frame index, and a status tag when the current frame has no live pose.

**Timeline strip** (under the footage) — the whole video at a glance, one
row per hand: bright where tracked, dimmed where ghosted, amber where the
hand was visible but unrecovered, dark where no hand was in view. The
white line is the playhead.

### What the dimmed skeletons mean

- **Dimmed / ghosted skeleton + `pose unrecovered`** — the hand is still
  detected but we lost its pose, so the last recovered pose is held,
  dimmed, for up to half a second. It is shown dimmed precisely so it
  cannot be mistaken for a live measurement, and it is **not** written to
  the delivered keypoints — those stay `NaN`.
- **`no hand in view`** — the detector reports no hand. The skeleton
  dissolves within a fraction of a second and nothing is drawn. We do not
  hold a stale pose here, because that would assert a hand the data says
  is not present.

Ghosting is a display convenience so the render does not strobe. **No
interpolated or held value appears anywhere in the delivered data.**

## Delivered files

| File | Contents |
|---|---|
| `<video>_qc.mp4` | The QC render described above |
| `<video>.npz` | Keypoints: `kp3d`, `kp2d`, `conf`, `det_conf`, `hand_visible`, `valid`, plus a `meta` block |
| `qc_report.csv` | One row per video — see the column list below |
| `manudata_qc_reel.mp4` | The top recommended clips, concatenated |

`qc_report.csv` columns: `filename`, `site`, `task`, `duration_s`,
`total_frames`, `frames_both_hands`, `frames_one_hand`,
`frames_zero_hands`, `frames_visible_no_pose`, `occluded_or_absent_pct`,
`pose_recovery_pct`, `longest_gap_s`, `mean_confidence`,
`recommended_clip_start_s`, `recommended_clip_end_s`.

`frames_both_hands` / `frames_one_hand` / `frames_zero_hands` count frames
by how many hands have a **recovered** pose. `longest_gap_s` is the
longest continuous stretch with no pose for either hand, from any cause.
A blank `pose_recovery_pct` means no hand was visible anywhere in that
video, so the rate is undefined rather than zero.

## Questions we expect

**Why is `pose_recovery_pct` exactly 100%?** Because the current backend
(WiLoR-mini) poses every hand it detects and has no way to report a pose
failure, so the recovered-over-visible ratio is 100% by construction, not
by measured perfection. See *Pose-recovery rate* above and the
`pose_recovery_measurable=False` flag in each `.npz`. A backend that scores
or rejects individual poses would turn this back into a real quality
number; the cases such a number would expose — motion blur, partial
occlusion detectable but not articulable, extreme grasps against the
workpiece — are still present in the footage and visible in the renders,
they are simply not counted against the tracker today.

**Can we get the frames you dropped?** Nothing is dropped. Every frame of
every source video has a row in the keypoint arrays, flagged as described
above. You can reconstruct any subset by filtering on the flags.

**Can we get a world-frame version?** Yes — that requires the camera
trajectory from the SLAM stage, which we can deliver alongside. The QC
renders can include a camera-trajectory panel showing it.
