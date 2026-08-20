"""Tests for the properties this tool's credibility rests on.

Weighted toward the two failure modes that would quietly ruin a
customer-facing render: panels drifting out of sync, and the two
missing-pose cases being conflated into one number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qc.config import RenderConfig, VideoMeta
from qc.io.sizing import MIN_BITRATE_KBPS, plan_quality
from qc.manifest import Label, ManifestError, prettify_site, resolve
from qc.pose.gapfill import (
    ABSENT,
    FADING,
    HELD,
    TAG_NO_HAND,
    TAG_UNRECOVERED,
    TRACKED,
    UNRECOVERED,
    build_render_plan,
)
from qc.pose.schema import N_HANDS, PoseTrack, PoseTrackError
from qc.report.analyze import analyze, best_window, longest_run

FPS = 30.0


def make_track(n_frames: int = 120) -> PoseTrack:
    """A track with every hand visible and recovered."""
    track = PoseTrack.empty(n_frames)
    track.hand_visible[:] = True
    track.valid[:] = True
    track.det_conf[:] = 0.9
    track.conf[:] = 0.8
    # Distinct, finite poses so a frame can be identified by its values.
    idx = np.arange(n_frames, dtype=np.float32)
    track.kp3d[:] = idx[:, None, None, None] * 0.001
    track.kp2d[:] = idx[:, None, None, None]
    return track


def make_meta(n_frames: int = 120, tmp_path=None) -> VideoMeta:
    return VideoMeta(
        path=(tmp_path / "clip.mp4") if tmp_path else __import__("pathlib").Path("clip.mp4"),
        width=1280, height=720, fps=FPS, n_frames=n_frames,
    )


# ── schema invariants ─────────────────────────────────────────────────


def test_valid_without_visible_is_rejected():
    """A recovered pose implies a detection; the inverse is corrupt data."""
    track = make_track(10)
    track.hand_visible[3, 0] = False
    with pytest.raises(PoseTrackError, match="valid but not hand_visible"):
        track.assert_consistent()


def test_frame_count_mismatch_raises():
    track = make_track(100)
    with pytest.raises(PoseTrackError, match="Frame-count mismatch"):
        track.check_alignment(101, "clip.mp4")


def test_save_load_round_trip(tmp_path):
    track = make_track(20)
    track.hand_visible[5, 1] = False
    track.valid[5, 1] = False
    track.meta["model"] = "WiLoR"

    path = tmp_path / "t.npz"
    track.save(path)
    loaded = PoseTrack.load(path)

    assert loaded.n_frames == 20
    assert loaded.meta["model"] == "WiLoR"
    np.testing.assert_array_equal(loaded.valid, track.valid)
    np.testing.assert_array_equal(loaded.hand_visible, track.hand_visible)
    np.testing.assert_allclose(loaded.kp3d, track.kp3d)


def test_recovery_rate_is_over_visible_slots_only():
    """Occlusion must not count against the tracker."""
    track = PoseTrack.empty(100)
    # Only a quarter of slots have a hand in view at all...
    track.hand_visible[:25] = True
    # ...and we recover a pose on all but one of them.
    track.valid[:25] = True
    track.valid[24] = False

    assert track.n_visible_slots == 50
    assert track.n_recovered_slots == 48
    assert track.pose_recovery_rate == pytest.approx(48 / 50)


def test_recovery_rate_undefined_when_nothing_visible():
    """Zero opportunities is not zero percent."""
    track = PoseTrack.empty(10)
    assert np.isnan(track.pose_recovery_rate)


# ── visibility state machine ──────────────────────────────────────────


def test_visible_but_unrecovered_holds_then_gives_up():
    """Case (b): hold a ghost briefly, then say so rather than lie."""
    cfg = RenderConfig(hold_limit_s=0.5)
    track = make_track(90)
    track.valid[10:, 0] = False  # hand still visible throughout

    plan = build_render_plan(track, FPS, cfg)

    assert plan.status[9, 0] == TRACKED
    # Within the hold limit: ghosted copy of the last good pose.
    assert plan.status[12, 0] == HELD
    assert plan.src_index[12, 0] == 9
    assert plan.alpha[12, 0] < 1.0
    # Well past it: nothing drawn, and the header says why.
    assert plan.status[80, 0] == UNRECOVERED
    assert plan.src_index[80, 0] == -1


def test_absent_hand_is_not_held_as_a_ghost():
    """Case (a): never assert a hand the detector says is not there."""
    cfg = RenderConfig(hold_limit_s=0.5)
    track = make_track(90)
    track.valid[10:, :] = False
    track.hand_visible[10:, :] = False

    plan = build_render_plan(track, FPS, cfg)

    # A brief dissolve is allowed so the skeleton does not snap off...
    assert plan.status[11, 0] == FADING
    # ...but it is far shorter than the hold used when a hand IS visible,
    # and nothing survives to the hold limit.
    hold_limit_frame = 10 + int(cfg.hold_limit_s * FPS)
    assert plan.status[hold_limit_frame, 0] == ABSENT
    assert plan.src_index[hold_limit_frame, 0] == -1


def test_header_tag_distinguishes_the_two_cases():
    cfg = RenderConfig()
    track = make_track(60)

    track.valid[30:, 0] = False          # visible, unrecovered
    track.valid[30:, 1] = False
    track.hand_visible[30:, 1] = False   # absent
    plan = build_render_plan(track, FPS, cfg)
    assert plan.header_tag(55) == TAG_UNRECOVERED

    both_gone = make_track(60)
    both_gone.valid[30:, :] = False
    both_gone.hand_visible[30:, :] = False
    plan2 = build_render_plan(both_gone, FPS, cfg)
    assert plan2.header_tag(55) == TAG_NO_HAND

    assert plan.header_tag(10) == ""  # tracked frames stay clean


def test_no_pose_before_the_first_detection_is_never_invented():
    cfg = RenderConfig()
    track = make_track(60)
    track.valid[:20, :] = False
    plan = build_render_plan(track, FPS, cfg)
    assert (plan.src_index[:20] == -1).all()


# ── frame-exact synchronisation ───────────────────────────────────────


def test_plan_indexes_are_absolute_and_never_reordered():
    """The render plan must map frame t to a pose at t, not at t-offset."""
    cfg = RenderConfig()
    track = make_track(200)
    plan = build_render_plan(track, FPS, cfg)
    expected = np.arange(200)
    np.testing.assert_array_equal(plan.src_index[:, 0], expected)
    np.testing.assert_array_equal(plan.src_index[:, 1], expected)


def test_composer_rejects_out_of_range_frames(tmp_path):
    from qc.render.compositor import FrameComposer

    cfg = RenderConfig()
    track = make_track(50)
    meta = make_meta(50, tmp_path)
    plan = build_render_plan(track, FPS, cfg)
    composer = FrameComposer(meta, track, plan, cfg, "Site", "Task")

    frame = np.zeros((720, 1280, 3), np.uint8)
    with pytest.raises(IndexError):
        composer.compose(50, frame)


def test_composer_output_is_canvas_sized(tmp_path):
    from qc.render.compositor import FrameComposer

    cfg = RenderConfig()
    track = make_track(30)
    meta = make_meta(30, tmp_path)
    plan = build_render_plan(track, FPS, cfg)
    composer = FrameComposer(meta, track, plan, cfg, "Site", "Task")

    out = composer.compose(5, np.zeros((720, 1280, 3), np.uint8))
    assert out.shape == (cfg.canvas_h, cfg.canvas_w, 3)
    assert out.dtype == np.uint8


# ── analysis ──────────────────────────────────────────────────────────


def test_longest_run():
    assert longest_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)) == 3
    assert longest_run(np.zeros(5, bool)) == 0
    assert longest_run(np.ones(4, bool)) == 4


def test_best_window_prefers_the_two_handed_stretch():
    n = int(60 * FPS)
    track = PoseTrack.empty(n)
    track.hand_visible[:] = True
    track.valid[:] = True
    # Wreck the first half so the window must land in the second.
    track.valid[: n // 2, 1] = False

    window = best_window(track, FPS)
    assert window is not None
    assert window.start_s >= 25.0
    assert 20.0 <= window.duration_s <= 30.5
    assert window.both_hands_pct > 95.0


def test_short_video_still_yields_a_window():
    """A 14s clip is usable; returning nothing would drop it from the reel."""
    n = int(14 * FPS)
    track = PoseTrack.empty(n)
    track.hand_visible[:] = True
    track.valid[:] = True
    window = best_window(track, FPS)
    assert window is not None
    assert window.start_frame == 0
    assert window.end_frame == n


def test_analyze_separates_occlusion_from_tracking_failure(tmp_path):
    n = 100
    track = PoseTrack.empty(n)
    track.hand_visible[:] = True
    track.valid[:] = True
    # 20 slots occluded — case (a).
    track.hand_visible[:10, :] = False
    track.valid[:10, :] = False
    # 10 slots visible but unrecovered — case (b).
    track.valid[10:20, 0] = False
    track.conf[track.valid] = 0.75

    stats = analyze(track, make_meta(n, tmp_path), "Alpine VL", "wire harness")

    assert stats.frames_visible_no_pose == 10
    assert stats.occluded_or_absent_pct == pytest.approx(10.0)
    assert stats.pose_recovery_pct == pytest.approx(100 * 170 / 180)
    assert stats.frames_zero_hands == 10
    assert stats.frames_one_hand == 10
    assert stats.frames_both_hands == 80


# ── sizing ────────────────────────────────────────────────────────────


def test_bitrate_cap_scales_with_duration():
    short = plan_quality(30.0, 50.0)
    long = plan_quality(120.0, 50.0)
    assert short.max_bitrate_kbps > long.max_bitrate_kbps
    assert long.bufsize_kbps == long.max_bitrate_kbps * 2


def test_no_cap_without_a_target():
    assert plan_quality(30.0, None).max_bitrate_kbps is None


def test_bitrate_clamped_at_the_quality_floor():
    quality = plan_quality(3600.0, 1.0)
    assert quality.max_bitrate_kbps == MIN_BITRATE_KBPS


# ── manifest ──────────────────────────────────────────────────────────


def test_blank_task_is_rejected():
    labels = {"a.mp4": Label(site="Alpine VL", task="")}
    with pytest.raises(ManifestError, match="blank"):
        resolve(__import__("pathlib").Path("a.mp4"), labels)


def test_unknown_video_is_rejected():
    with pytest.raises(ManifestError, match="no row"):
        resolve(__import__("pathlib").Path("missing.mp4"), {})


def test_filename_convention_yields_both_labels():
    """Country_City_Site_Task_NNN_NNN fills the manifest with no typing."""
    from qc.manifest import parse_filename

    label = parse_filename("India_Faridabad_AlpineFootwear01_ShoeAssembly_003_039")
    assert label is not None
    assert label.site == "Alpine Footwear 01"
    assert label.task == "shoe assembly"
    assert "Faridabad" in label.notes and "India" in label.notes


@pytest.mark.parametrize("stem", [
    "Cam01_2026_0526_143513_014",   # raw camera name: 'Cam01' is not alphabetic
    "2026_0509_154309_010",         # all numeric
    "random_name",                  # too few tokens
    "India_Faridabad_Site_Task",    # no trailing clip numbers
])
def test_off_convention_names_are_not_guessed(stem):
    """Better a blank task than a confidently wrong one on a customer frame."""
    from qc.manifest import parse_filename

    assert parse_filename(stem) is None


def test_split_camel():
    from qc.manifest import split_camel

    assert split_camel("AlpineFootwear01") == "Alpine Footwear 01"
    assert split_camel("ShoeAssembly") == "Shoe Assembly"
    assert split_camel("PCBSolder") == "PCB Solder"


def test_prettify_site_uppercases_initialisms():
    assert prettify_site("tangerine shoes vl") == "Tangerine Shoes VL"
    assert prettify_site("alpine Vl") == "Alpine VL"


# ── config fingerprint drives --resume ────────────────────────────────


def test_resume_rejects_a_truncated_output(tmp_path):
    """+faststart means a truncated MP4 still advertises its full length.

    So the size check has to catch it, and it has to run before the
    frame-count check that the truncated file would happily pass.
    """
    from qc.io.video_writer import write_sidecar
    from qc.runner import output_is_complete

    cfg = RenderConfig()
    out = tmp_path / "clip_qc.mp4"
    out.write_bytes(b"x" * 5000)
    write_sidecar(out, {
        "source": "clip.mp4",
        "config_fingerprint": cfg.fingerprint(),
        "frames": 300,
        "size_bytes": 5000,
    })

    # Same bytes as recorded: the size gate passes (a real file would then
    # go on to the decode checks).
    out.write_bytes(b"x" * 4000)  # now truncated
    assert output_is_complete(out, cfg, 300, "clip.mp4") is False


def test_resume_rejects_missing_sidecar_and_wrong_settings(tmp_path):
    from qc.io.video_writer import write_sidecar
    from qc.runner import output_is_complete

    cfg = RenderConfig()
    out = tmp_path / "clip_qc.mp4"
    out.write_bytes(b"x" * 100)

    # No sidecar at all — the write never finished.
    assert output_is_complete(out, cfg, 300, "clip.mp4") is False

    write_sidecar(out, {
        "source": "clip.mp4",
        "config_fingerprint": RenderConfig(max_size_mb=5.0).fingerprint(),
        "frames": 300,
        "size_bytes": 100,
    })
    assert output_is_complete(out, cfg, 300, "clip.mp4") is False

    write_sidecar(out, {
        "source": "a_different_video.mp4",
        "config_fingerprint": cfg.fingerprint(),
        "frames": 300,
        "size_bytes": 100,
    })
    assert output_is_complete(out, cfg, 300, "clip.mp4") is False


def test_shard_splits_without_overlap_or_loss():
    """Several pods must cover the batch exactly once between them."""
    from qc.parallel import shard

    items = [f"v{i}.mp4" for i in range(10)]
    parts = [shard(items, i, 3) for i in (1, 2, 3)]

    assert sum(len(p) for p in parts) == len(items)
    assert sorted(x for p in parts for x in p) == sorted(items)
    for a in range(3):
        for b in range(a + 1, 3):
            assert not set(parts[a]) & set(parts[b])


def test_shard_is_round_robin_not_contiguous():
    """Round-robin keeps long and short videos spread across pods."""
    from qc.parallel import shard

    items = list(range(9))
    assert shard(items, 1, 3) == [0, 3, 6]
    assert shard(items, 2, 3) == [1, 4, 7]
    assert shard(items, 3, 3) == [2, 5, 8]


def test_shard_rejects_bad_ranges():
    from qc.parallel import shard

    with pytest.raises(ValueError):
        shard([1, 2, 3], 0, 3)
    with pytest.raises(ValueError):
        shard([1, 2, 3], 4, 3)


@pytest.mark.parametrize("text,expected", [("1/3", (1, 3)), ("3/3", (3, 3))])
def test_parse_shard_accepts_valid(text, expected):
    from qc.cli import _parse_shard

    assert _parse_shard(text) == expected


@pytest.mark.parametrize("text", ["0/3", "4/3", "abc", "1/0", "1"])
def test_parse_shard_rejects_invalid(text):
    import argparse

    from qc.cli import _parse_shard

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_shard(text)


def test_doctor_flags_unlabelled_manifest(tmp_path):
    from qc.doctor import FAIL, PASS, check_manifest

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "filename,site,task,notes\na.mp4,Alpine VL,wire harness,\nb.mp4,Alpine VL,,\n",
        encoding="utf-8",
    )
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]

    assert check_manifest(manifest, videos).status == FAIL
    assert check_manifest(manifest, [videos[0]]).status == PASS


def test_doctor_flags_missing_manifest_row(tmp_path):
    from qc.doctor import FAIL, check_manifest

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("filename,site,task\na.mp4,Alpine VL,wire harness\n",
                        encoding="utf-8")
    check = check_manifest(manifest, [tmp_path / "a.mp4", tmp_path / "unknown.mp4"])
    assert check.status == FAIL
    assert "unknown.mp4" in check.detail


def test_doctor_report_exits_nonzero_only_on_failures():
    from qc.doctor import FAIL, PASS, WARN, Check, format_report

    clean = [Check("a", PASS), Check("b", WARN, "slower")]
    assert not any(c.failed for c in clean)
    assert "Ready, with 1 warning" in format_report(clean)

    broken = clean + [Check("c", FAIL, "no CUDA")]
    assert any(c.failed for c in broken)
    assert "1 blocking problem" in format_report(broken)


def test_fingerprint_changes_with_render_affecting_settings():
    base = RenderConfig()
    assert base.fingerprint() == RenderConfig().fingerprint()
    assert base.fingerprint() != RenderConfig(with_slam=True).fingerprint()
    assert base.fingerprint() != RenderConfig(max_size_mb=25.0).fingerprint()


def test_inference_progress_time_formatting():
    """Inference reports minutes and hours, not raw seconds."""
    from qc.pose.wilor_mini_backend import _human

    assert _human(45) == "45s"
    assert _human(600) == "10m00s"
    assert _human(2580) == "43m00s"
    assert _human(7860) == "2h11m"
    assert _human(float("inf")) == "unknown"


# ── mid-inference checkpointing ───────────────────────────────────────
#
# The property under test is not "resume works" but "a resumed track is
# indistinguishable from an uninterrupted one". A checkpoint that silently
# mixes frames from two different parameter sets would produce a track that
# passes every shape and consistency check and is wrong in half its frames.


def _filled_track(n_frames: int, upto: int) -> PoseTrack:
    """A track with frames [0, upto) populated and the rest missing."""
    track = PoseTrack.empty(n_frames)
    for i in range(upto):
        track.kp3d[i, 0] = float(i)
        track.kp2d[i, 0] = float(i)
        track.conf[i, 0] = 0.9
        track.det_conf[i, 0] = 0.9
        track.hand_visible[i, 0] = True
        track.valid[i, 0] = True
    return track


def test_checkpoint_round_trips_partial_work(tmp_path):
    from qc.pose.checkpoint import InferenceCheckpoint

    fp = {"backend": "WiLoR-mini", "assumed_hfov_deg": 65.0}
    ckpt = InferenceCheckpoint(tmp_path / "v.npz.partial", fp, every=100)
    ckpt.save(_filled_track(500, 300), done=300)

    restored, start = InferenceCheckpoint(tmp_path / "v.npz.partial", fp).resume(500)
    assert start == 300
    assert restored is not None
    assert restored.valid[:300, 0].all()
    assert not restored.valid[300:, 0].any()
    assert restored.kp3d[299, 0, 0, 0] == 299.0


def test_checkpoint_saved_track_is_not_polluted_by_bookkeeping(tmp_path):
    """The in-memory track must come out of save() exactly as it went in."""
    from qc.pose.checkpoint import InferenceCheckpoint

    track = _filled_track(50, 10)
    track.meta["model"] = "WiLoR-mini"
    ckpt = InferenceCheckpoint(tmp_path / "v.npz.partial", {"a": 1})
    ckpt.save(track, done=10)

    assert track.meta == {"model": "WiLoR-mini"}


def test_checkpoint_refuses_a_partial_from_different_settings(tmp_path):
    """Mixing depth scales would corrupt half a track plausibly. Refuse."""
    from qc.pose.checkpoint import InferenceCheckpoint

    path = tmp_path / "v.npz.partial"
    InferenceCheckpoint(path, {"assumed_hfov_deg": 65.0}).save(
        _filled_track(500, 300), done=300
    )

    restored, start = InferenceCheckpoint(path, {"assumed_hfov_deg": 90.0}).resume(500)
    assert (restored, start) == (None, 0)
    assert not path.exists(), "an incompatible partial must be removed, not left to rot"


def test_checkpoint_refuses_a_partial_of_the_wrong_length(tmp_path):
    from qc.pose.checkpoint import InferenceCheckpoint

    fp = {"a": 1}
    path = tmp_path / "v.npz.partial"
    InferenceCheckpoint(path, fp).save(_filled_track(500, 300), done=300)

    restored, start = InferenceCheckpoint(path, fp).resume(400)
    assert (restored, start) == (None, 0)


def test_checkpoint_survives_a_corrupt_partial(tmp_path):
    """A bad cache file costs recomputation, never a failed run."""
    from qc.pose.checkpoint import InferenceCheckpoint

    path = tmp_path / "v.npz.partial"
    path.write_bytes(b"not an npz")

    restored, start = InferenceCheckpoint(path, {"a": 1}).resume(500)
    assert (restored, start) == (None, 0)
    assert not path.exists()


def test_checkpoint_missing_partial_starts_from_zero(tmp_path):
    from qc.pose.checkpoint import InferenceCheckpoint

    restored, start = InferenceCheckpoint(tmp_path / "absent.partial", {}).resume(500)
    assert (restored, start) == (None, 0)


def test_checkpoint_writes_only_on_the_interval(tmp_path):
    from qc.pose.checkpoint import InferenceCheckpoint

    path = tmp_path / "v.npz.partial"
    ckpt = InferenceCheckpoint(path, {"a": 1}, every=100)
    track = _filled_track(500, 500)

    for done in range(1, 100):
        ckpt.maybe_save(track, done)
    assert not path.exists()

    ckpt.maybe_save(track, 100)
    assert path.exists()


def test_checkpoint_clear_removes_the_partial(tmp_path):
    from qc.pose.checkpoint import InferenceCheckpoint

    path = tmp_path / "v.npz.partial"
    ckpt = InferenceCheckpoint(path, {"a": 1})
    ckpt.save(_filled_track(10, 5), done=5)
    assert path.exists()

    ckpt.clear()
    assert not path.exists()
    ckpt.clear()  # idempotent — a second run must not trip over a gone file


def test_resumed_progress_reports_the_rate_it_actually_achieved():
    """A resumed pass must not claim throughput from the run that died."""
    from pathlib import Path

    from qc.config import VideoMeta as VM
    from qc.pose.wilor_mini_backend import _InferenceProgress

    meta = VM(path=Path("v.mp4"), width=1920, height=1080, fps=30.0,
              n_frames=1000, n_frames_exact=True)
    progress = _InferenceProgress(meta, start=900)
    # 100 fresh frames in 10 seconds is 10 fps, not 90.
    assert progress._rate(1000, 10.0) == pytest.approx(10.0)


# ── progress reporting ────────────────────────────────────────────────


def test_periodic_progress_rate_excludes_resumed_work():
    from qc.progress import PeriodicProgress

    p = PeriodicProgress("v.mp4", total=1000, start=900)
    assert p.rate(1000, 10.0) == pytest.approx(10.0)

    fresh = PeriodicProgress("v.mp4", total=1000)
    assert fresh.rate(1000, 10.0) == pytest.approx(100.0)


def test_periodic_progress_respects_its_interval(caplog):
    from qc.progress import PeriodicProgress

    p = PeriodicProgress("v.mp4", total=100, interval_s=3600)
    with caplog.at_level("INFO", logger="qc.progress"):
        for i in range(1, 51):
            p.update(i)
    assert caplog.records == []


def test_periodic_progress_is_silent_when_disabled(caplog):
    from qc.progress import PeriodicProgress

    p = PeriodicProgress("v.mp4", total=100, interval_s=0.0, enabled=False)
    with caplog.at_level("INFO", logger="qc.progress"):
        p.update(50)
        p.close(50)
    assert caplog.records == []


def test_periodic_progress_reports_when_due(caplog):
    from qc.progress import PeriodicProgress

    p = PeriodicProgress("v.mp4", total=100, interval_s=0.0)
    with caplog.at_level("INFO", logger="qc.progress"):
        p.update(50)
    assert "50/100" in caplog.text and "50.0%" in caplog.text


def test_human_duration_formats_for_operators():
    from qc.progress import human_duration

    assert human_duration(45) == "45s"
    assert human_duration(600) == "10m00s"
    assert human_duration(7860) == "2h11m"
    assert human_duration(float("inf")) == "unknown"
    assert human_duration(float("nan")) == "unknown"


def test_pool_never_spawns_more_workers_than_jobs():
    """Three processes for one job runs silent and buys nothing."""
    from qc.parallel import JobResult, execute

    seen = []

    def fake(job):
        seen.append(job)
        return JobResult(video=Path(str(job)), value=job)

    # A single job must take the in-process path, not a spawned pool.
    results = execute(fake, ["only-one"], workers=3, label="render")
    assert len(results) == 1
    assert seen == ["only-one"], "the job should have run in this process"
