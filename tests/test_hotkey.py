import threading
from types import SimpleNamespace

import keyboard
import pytest

from src import hotkey as hotkey_module
from src.hotkey import HotkeyManager, HotkeyState


@pytest.fixture
def hotkey_events(monkeypatch):
    """Exercise the real keyboard dispatcher without installing a Windows hook."""
    codes = {"ctrl": 29, "shift": 42, "space": 57, "enter": 28, "a": 30}
    monkeypatch.setattr(keyboard._os_keyboard, "init", lambda: None)
    listener = keyboard._KeyboardListener()
    listener.init()
    monkeypatch.setattr(listener, "start_if_necessary", lambda: None)
    monkeypatch.setattr(keyboard, "_listener", listener)
    monkeypatch.setattr(keyboard, "_pressed_events", {})
    monkeypatch.setattr(keyboard, "_logically_pressed_keys", {})
    monkeypatch.setattr(keyboard, "_hooks", {})
    monkeypatch.setattr(keyboard, "_hotkeys", {})
    monkeypatch.setattr(
        keyboard,
        "key_to_scan_codes",
        lambda key: (codes[key] if isinstance(key, str) else key,),
    )
    monkeypatch.setattr(keyboard, "is_modifier", lambda code: code in {29, 42})
    monkeypatch.setattr(keyboard, "press", lambda key: None)
    monkeypatch.setattr(keyboard, "release", lambda key: None)
    monkeypatch.setattr(
        hotkey_module, "get_config", lambda: SimpleNamespace(hotkey="ctrl+shift+space")
    )
    manager = HotkeyManager()
    states = []
    assert manager.register(on_state_change=states.append)

    def dispatch(name, event_type):
        return listener.direct_callback(
            keyboard.KeyboardEvent(event_type, codes[name], name=name)
        )

    yield manager, states, dispatch, listener
    manager.unregister()


@pytest.mark.parametrize(
    "release_order",
    [
        ("ctrl", "shift", "space"),
        ("shift", "space", "ctrl"),
        ("space", "ctrl", "shift"),
    ],
)
def test_hotkey_consumes_space_release_in_any_order(hotkey_events, release_order):
    manager, states, dispatch, _ = hotkey_events
    for expected_state in (HotkeyState.RECORDING, HotkeyState.PROCESSING):
        for key in ("ctrl", "shift", "space"):
            assert not dispatch(key, keyboard.KEY_DOWN)
        for key in release_order:
            assert not dispatch(key, keyboard.KEY_UP)
        assert manager.get_state() == expected_state
    assert states == [HotkeyState.RECORDING, HotkeyState.PROCESSING]
    manager.set_idle()
    assert dispatch("space", keyboard.KEY_DOWN)
    assert dispatch("space", keyboard.KEY_UP)


def test_hotkey_consumes_repeats_after_modifiers_are_released(hotkey_events):
    manager, states, dispatch, _ = hotkey_events
    for key in ("ctrl", "shift", "space"):
        dispatch(key, keyboard.KEY_DOWN)
    for _ in range(3):
        assert not dispatch("space", keyboard.KEY_DOWN)
    dispatch("ctrl", keyboard.KEY_UP)
    dispatch("shift", keyboard.KEY_UP)
    assert not dispatch("space", keyboard.KEY_DOWN)
    assert not dispatch("space", keyboard.KEY_UP)
    assert states == [HotkeyState.RECORDING]
    assert manager.get_state() == HotkeyState.RECORDING


def test_unmatched_chord_and_typing_pass_through(hotkey_events):
    _, states, dispatch, _ = hotkey_events
    dispatch("ctrl", keyboard.KEY_DOWN)
    assert dispatch("space", keyboard.KEY_DOWN)
    assert dispatch("space", keyboard.KEY_UP)
    dispatch("ctrl", keyboard.KEY_UP)
    assert dispatch("a", keyboard.KEY_DOWN)
    assert dispatch("a", keyboard.KEY_UP)
    assert states == []


@pytest.mark.parametrize("repeat_count", [0, 3])
@pytest.mark.parametrize(
    "release_order", [("ctrl", "shift", "space"), ("space", "ctrl", "shift")]
)
def test_space_pressed_before_modifiers_keeps_its_release(
    hotkey_events, repeat_count, release_order
):
    manager, states, dispatch, _ = hotkey_events
    assert dispatch("space", keyboard.KEY_DOWN)
    dispatch("ctrl", keyboard.KEY_DOWN)
    dispatch("shift", keyboard.KEY_DOWN)
    for _ in range(repeat_count):
        assert dispatch("space", keyboard.KEY_DOWN)
    for key in release_order:
        accepted = dispatch(key, keyboard.KEY_UP)
        if key == "space":
            assert accepted
    assert manager.get_state() == HotkeyState.IDLE
    assert states == []
    for key in ("ctrl", "shift", "space"):
        dispatch(key, keyboard.KEY_DOWN)
    assert manager.get_state() == HotkeyState.RECORDING


def test_only_final_fresh_key_is_consumed_in_multi_key_chord(hotkey_events):
    manager, states, dispatch, _ = hotkey_events
    manager.unregister()
    manager.config.hotkey = "ctrl+space+a"
    assert manager.register(on_state_change=states.append)
    dispatch("ctrl", keyboard.KEY_DOWN)
    assert dispatch("space", keyboard.KEY_DOWN)
    assert not dispatch("a", keyboard.KEY_DOWN)
    dispatch("ctrl", keyboard.KEY_UP)
    assert dispatch("space", keyboard.KEY_UP)
    assert not dispatch("a", keyboard.KEY_UP)
    assert states == [HotkeyState.RECORDING]


def test_unregister_removes_only_owned_hooks(hotkey_events):
    manager, _, dispatch, listener = hotkey_events

    def unrelated(event):
        return True

    remove_unrelated = keyboard.hook_key("space", unrelated, suppress=True)
    manager.unregister()
    assert listener.blocking_keys[57] == [unrelated]
    assert not any(listener.blocking_hotkeys.values())
    assert dispatch("space", keyboard.KEY_DOWN)
    assert dispatch("space", keyboard.KEY_UP)
    remove_unrelated()


def test_failed_registration_removes_partial_hooks(hotkey_events, monkeypatch):
    manager, _, _, listener = hotkey_events
    manager.unregister()

    def fail(*args, **kwargs):
        raise RuntimeError("hook unavailable")

    monkeypatch.setattr(keyboard, "add_hotkey", fail)
    assert not manager.register()
    assert manager.get_last_registration_error() == "hook unavailable"
    assert not manager.is_registered()
    assert not any(listener.blocking_keys.values())


def test_unregister_removes_every_activation_key_hook(hotkey_events):
    manager, _, _, listener = hotkey_events
    manager.unregister()
    manager.config.hotkey = "ctrl+space+a"
    assert manager.register()
    assert listener.blocking_keys[30]
    assert listener.blocking_keys[57]
    manager.unregister()
    assert not any(listener.blocking_keys.values())


def test_sequence_hotkey_retains_library_release_handling(hotkey_events):
    manager, states, dispatch, _ = hotkey_events
    manager.unregister()
    manager.config.hotkey = "a, enter"
    assert manager.register(on_state_change=states.append)
    for _ in range(2):
        for key in ("a", "enter"):
            assert not dispatch(key, keyboard.KEY_DOWN)
            assert not dispatch(key, keyboard.KEY_UP)
    assert states == [HotkeyState.RECORDING, HotkeyState.PROCESSING]


def test_second_press_cancels_request_before_start_callback_initializes(hotkey_events):
    manager, _, dispatch, _ = hotkey_events
    manager.unregister()
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    requests = []
    outcomes = []

    def start(request):
        requests.append(request)
        entered.set()
        assert release.wait(2)
        outcomes.append(request.commit_capture())
        finished.set()

    manager.config.hotkey = "ctrl+shift+space"
    assert manager.register(on_start=start, on_stop=lambda request, committed: None)
    try:
        for key in ("ctrl", "shift", "space"):
            dispatch(key, keyboard.KEY_DOWN)
        assert entered.wait(2)
        for key in ("ctrl", "shift", "space"):
            dispatch(key, keyboard.KEY_UP)
        for key in ("ctrl", "shift", "space"):
            dispatch(key, keyboard.KEY_DOWN)
        assert requests[0].cancelled.is_set()
        release.set()
        assert finished.wait(2)
    finally:
        release.set()
    assert outcomes == [False]
