import pytest

from src import recording_overlay as overlay_module
from src.recording_overlay import (
    ERROR,
    ERROR_HIDE_MS,
    HIDDEN,
    PROCESSING,
    RECORDING,
    SUCCESS,
    SUCCESS_HIDE_MS,
    SUCCESS_MESSAGE,
    RecordingOverlay,
)


class FakeMaster:
    """Record Tk ``after`` jobs so tests can fire or inspect them."""

    def __init__(self):
        self.jobs = {}
        self.cancelled = []
        self._next_id = 0

    def after(self, delay_ms, callback):
        self._next_id += 1
        job_id = f"after#{self._next_id}"
        self.jobs[job_id] = (delay_ms, callback)
        return job_id

    def after_cancel(self, job_id):
        self.cancelled.append(job_id)
        self.jobs.pop(job_id, None)

    def delays(self):
        return sorted(delay for delay, _callback in self.jobs.values())

    def fire(self, delay_ms):
        [job_id] = [
            job_id for job_id, (delay, _cb) in self.jobs.items() if delay == delay_ms
        ]
        _delay, callback = self.jobs.pop(job_id)
        callback()


class FakeView:
    def __init__(self, master, fail_on_show=False):
        self.master = master
        self.fail_on_show = fail_on_show
        self.shown = []
        self.hide_calls = 0
        self.destroyed = False

    def show(self, state, message, elapsed):
        if self.fail_on_show:
            raise RuntimeError("canvas broke")
        self.shown.append((state, message, elapsed))

    def hide(self):
        self.hide_calls += 1

    def destroy(self):
        self.destroyed = True


def make_overlay(**view_options):
    master = FakeMaster()
    views = []

    def factory(parent):
        view = FakeView(parent, **view_options)
        views.append(view)
        return view

    clock_values = iter(float(tick) for tick in range(1000))
    overlay = RecordingOverlay(
        master, view_factory=factory, clock=lambda: next(clock_values)
    )
    return overlay, master, views


def test_recording_shows_bars_and_keeps_animating():
    overlay, master, views = make_overlay()

    assert overlay.request(RECORDING, session=1) is True
    [view] = views
    assert view.shown[0][:2] == (RECORDING, "")
    assert master.delays() == [overlay_module.ANIMATION_FRAME_MS]

    master.fire(overlay_module.ANIMATION_FRAME_MS)

    assert len(view.shown) == 2
    assert view.shown[1][0] == RECORDING
    assert view.shown[1][2] > view.shown[0][2]
    assert master.delays() == [overlay_module.ANIMATION_FRAME_MS]


def test_processing_replaces_recording_animation():
    overlay, master, views = make_overlay()
    overlay.request(RECORDING, session=1)
    [recording_job] = master.jobs

    overlay.request(PROCESSING, session=1)

    assert recording_job in master.cancelled
    assert views[0].shown[-1][:2] == (PROCESSING, "")
    assert master.delays() == [overlay_module.ANIMATION_FRAME_MS]


@pytest.mark.parametrize(
    ("state", "message", "shown_message", "hide_delay"),
    [
        (SUCCESS, "", SUCCESS_MESSAGE, SUCCESS_HIDE_MS),
        (ERROR, "Transcription failed", "Transcription failed", ERROR_HIDE_MS),
    ],
)
def test_result_states_stop_animation_and_auto_hide(
    state, message, shown_message, hide_delay
):
    overlay, master, views = make_overlay()
    overlay.request(PROCESSING, session=1)

    overlay.request(state, session=1, message=message)

    assert views[0].shown[-1][:2] == (state, shown_message)
    assert master.delays() == [hide_delay]

    master.fire(hide_delay)

    assert overlay.state == HIDDEN
    assert views[0].hide_calls == 1
    assert master.jobs == {}


def test_late_events_cannot_hide_or_replace_newer_session():
    overlay, master, views = make_overlay()
    overlay.request(RECORDING, session=1)
    overlay.request(RECORDING, session=2)

    assert overlay.request(PROCESSING, session=1) is False
    assert overlay.request(ERROR, session=1, message="old") is False

    assert overlay.state == RECORDING
    assert views[0].shown[-1][0] == RECORDING
    assert master.delays() == [overlay_module.ANIMATION_FRAME_MS]


def test_session_stages_cannot_move_backwards():
    overlay, _master, views = make_overlay()
    overlay.request(SUCCESS, session=3)

    assert overlay.request(PROCESSING, session=3) is False
    assert overlay.request(RECORDING, session=3) is False
    assert views[0].shown[-1][0] == SUCCESS


def test_new_recording_cancels_pending_result_hide():
    overlay, master, _views = make_overlay()
    overlay.request(ERROR, session=1, message="No speech detected")
    [hide_job] = master.jobs

    overlay.request(RECORDING, session=2)

    assert hide_job in master.cancelled
    assert overlay.state == RECORDING
    assert master.delays() == [overlay_module.ANIMATION_FRAME_MS]


def test_hidden_request_hides_without_creating_window_first():
    overlay, master, views = make_overlay()

    assert overlay.request(HIDDEN) is True
    assert views == []

    overlay.request(RECORDING, session=1)
    overlay.request(HIDDEN)

    assert views[0].hide_calls == 1
    assert master.jobs == {}


def test_unknown_state_is_ignored_without_disabling_overlay():
    overlay, _master, views = make_overlay()

    assert overlay.request("listening", session=1) is False
    assert overlay.request(RECORDING, session=1) is True
    assert len(views) == 1


def test_window_creation_failure_logs_once_and_disables_overlay(capsys):
    master = FakeMaster()
    attempts = []

    def failing_factory(parent):
        attempts.append(parent)
        raise RuntimeError("secret text")

    overlay = RecordingOverlay(master, view_factory=failing_factory)

    assert overlay.request(RECORDING, session=1) is False
    assert overlay.request(PROCESSING, session=1) is False

    assert len(attempts) == 1
    err = capsys.readouterr().err
    assert err.count("Recording overlay disabled after a UI error") == 1
    assert "RuntimeError" in err
    assert "secret text" not in err


def test_animation_failure_disables_and_destroys_window(capsys):
    overlay, master, views = make_overlay()
    overlay.request(RECORDING, session=1)
    views[0].fail_on_show = True

    master.fire(overlay_module.ANIMATION_FRAME_MS)

    assert views[0].destroyed is True
    assert master.jobs == {}
    assert overlay.request(PROCESSING, session=1) is False
    assert capsys.readouterr().err.count("Recording overlay disabled") == 1


def test_close_cancels_timers_and_destroys_window():
    overlay, master, views = make_overlay()
    overlay.request(RECORDING, session=1)

    overlay.close()

    assert views[0].destroyed is True
    assert master.jobs == {}
    assert overlay.state == HIDDEN

    overlay.request(RECORDING, session=2)
    assert len(views) == 2


def test_overlay_size_widens_only_for_text_states():
    recording = overlay_module.overlay_size(RECORDING, 0, 1.0)
    processing = overlay_module.overlay_size(PROCESSING, 0, 1.0)
    success = overlay_module.overlay_size(SUCCESS, 120, 1.0)

    assert recording == processing == (64, 36)
    assert success == (2 * 14 + 16 + 8 + 120, 36)
    assert overlay_module.overlay_size(RECORDING, 0, 1.5) == (96, 54)


@pytest.mark.parametrize(
    ("work_area", "size", "scale", "position"),
    [
        ((0, 0, 1920, 1040), (64, 36), 1.0, (928, 988)),
        ((0, 0, 2560, 1400), (96, 54), 1.5, (1232, 1322)),
        ((-1920, 0, 0, 1040), (64, 36), 1.0, (-992, 988)),
    ],
)
def test_overlay_position_centers_above_work_area_bottom(
    work_area, size, scale, position
):
    assert overlay_module.overlay_position(work_area, size, scale) == position
