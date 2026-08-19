# ManuData — Hand Pose QC Renders

This note accompanies the QC videos and `qc_report.csv`. It describes what
produced the keypoints, the conventions they follow, and how to read the
visibility flags that ship with every frame.

---

## What produced the keypoints

| | |
|---|---|
| Hand detection | YOLO hand detector bundled with WiLoR |
| Pose regression | **WiLoR** (ViT-based 3D hand pose), 21 joints per hand |
| Joint convention | MANO / OpenPose 21-joint hand ordering |
| Max hands per frame | 2 (one left, one right) |

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

- The pose model is applied with its **nominal training focal length**,
  scaled to the source frame's long edge, with the principal point taken
  as the image centre.
- The focal length and principal point actually used are written into
  every `.npz` as `focal_length_px` and `principal_point_px`.

The consequence: **relative** 3D geometry — finger articulation, grasp
aperture, hand-to-hand distance — is well conditioned, while **absolute**
depth carries the scale error of the assumed focal length. If you intend
to use absolute metric depth, calibrate against a known object in the
scene, or tell us and we will run a calibration pass on the capture rig.

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

### The figure that measures our tracking

> **Pose-recovery rate = recovered poses ÷ hand-slots where a hand was
> visible.**

This is `pose_recovery_pct` in `qc_report.csv`, it is burned into the
header band of every rendered frame, and it is what the video ranking
sorts on. It is computed strictly over frames where the detector found a
hand, so occlusion neither flatters nor penalises it.

`occluded_or_absent_pct` is reported alongside it as a property of the
footage. Both are per-video in the CSV, and the portfolio-wide figures
are printed at the end of the ranked summary.

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

**Why is `pose_recovery_pct` not 100%?** Motion blur at 60 fps under
factory lighting, partial occlusion where enough of the hand is visible to
detect but not to articulate, and extreme grasp poses against the
workpiece. These are the cases we are actively working on, and they are
visible in the renders rather than filtered out of them.

**Can we get the frames you dropped?** Nothing is dropped. Every frame of
every source video has a row in the keypoint arrays, flagged as described
above. You can reconstruct any subset by filtering on the flags.

**Can we get a world-frame version?** Yes — that requires the camera
trajectory from the SLAM stage, which we can deliver alongside. The QC
renders can include a camera-trajectory panel showing it.
