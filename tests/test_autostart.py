import pytest

from src import autostart as autostart_module


def test_launch_command_uses_pythonw_module(tmp_path, monkeypatch):
    python_path = tmp_path / "venv" / "Scripts" / "python.exe"
    monkeypatch.setattr(autostart_module.sys, "executable", str(python_path))
    monkeypatch.setattr(autostart_module.sys, "frozen", False, raising=False)

    assert autostart_module._get_launch_command() == (
        f'"{python_path.with_name("pythonw.exe")}" -m src'
    )


def test_launch_command_uses_packaged_executable(monkeypatch):
    monkeypatch.setattr(
        autostart_module.sys,
        "executable",
        r"C:\Program Files\murmur\murmur.exe",
    )
    monkeypatch.setattr(autostart_module.sys, "frozen", True, raising=False)

    assert autostart_module._get_launch_command() == (
        '"C:\\Program Files\\murmur\\murmur.exe"'
    )


def test_set_autostart_noops_when_winreg_is_unavailable(monkeypatch):
    monkeypatch.setattr(autostart_module, "winreg", None)

    autostart_module.set_autostart(True)

    assert autostart_module.is_autostart_enabled() is False


class FakeRegistry:
    HKEY_CURRENT_USER = object()
    KEY_SET_VALUE = object()
    KEY_READ = object()
    REG_SZ = 1

    def __init__(self):
        self.calls = {}
        self.errors = {}
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def _call(self, operation, args):
        self.calls[operation] = args
        if operation in self.errors:
            raise self.errors[operation]

    def OpenKey(self, *args):  # noqa: N802 - match the winreg API
        self._call("open", args)
        return self

    def SetValueEx(self, *args):  # noqa: N802 - match the winreg API
        self._call("set", args)

    def DeleteValue(self, *args):  # noqa: N802 - match the winreg API
        self._call("delete", args)

    def QueryValueEx(self, *args):  # noqa: N802 - match the winreg API
        self._call("query", args)
        return ("command", self.REG_SZ)


@pytest.fixture
def registry(monkeypatch):
    fake_winreg = FakeRegistry()
    monkeypatch.setattr(autostart_module, "winreg", fake_winreg)
    monkeypatch.setattr(autostart_module, "_get_launch_command", lambda: "command")
    return fake_winreg


def test_set_autostart_uses_pythonw_and_registry(registry):
    autostart_module.set_autostart(True)

    assert registry.calls["open"] == (
        registry.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows\CurrentVersion\Run",
        0,
        registry.KEY_SET_VALUE,
    )
    assert registry.calls["set"] == (
        registry,
        "murmur",
        0,
        registry.REG_SZ,
        "command",
    )
    assert registry.closed
    assert "delete" not in registry.calls


def test_disable_autostart_removes_value_and_closes_handle(registry):
    autostart_module.set_autostart(False)

    assert registry.calls["delete"] == (registry, "murmur")
    assert registry.closed


@pytest.mark.parametrize("operation", ["open", "delete"])
def test_disable_autostart_is_idempotent_for_missing_entries(registry, operation):
    registry.errors[operation] = FileNotFoundError("missing")

    autostart_module.set_autostart(False)

    assert registry.closed is (operation == "delete")


@pytest.mark.parametrize(
    ("enabled", "operation"),
    [(True, "open"), (False, "open"), (True, "set"), (False, "delete")],
)
def test_set_autostart_propagates_failure_and_closes_handle(
    registry, enabled, operation
):
    error = PermissionError("access denied")
    registry.errors[operation] = error

    with pytest.raises(PermissionError) as caught:
        autostart_module.set_autostart(enabled)

    assert caught.value is error
    assert registry.closed is (operation != "open")


def test_is_autostart_enabled_closes_handle(registry):
    assert autostart_module.is_autostart_enabled() is True
    assert registry.calls["query"] == (registry, "murmur")
    assert registry.closed


@pytest.mark.parametrize("operation", ["open", "query"])
def test_is_autostart_enabled_returns_false_for_missing_entries(registry, operation):
    registry.errors[operation] = FileNotFoundError("missing")

    assert autostart_module.is_autostart_enabled() is False
    assert registry.closed is (operation == "query")


@pytest.mark.parametrize("operation", ["open", "query"])
def test_is_autostart_enabled_propagates_failure_and_closes_handle(registry, operation):
    error = PermissionError("access denied")
    registry.errors[operation] = error

    with pytest.raises(PermissionError) as caught:
        autostart_module.is_autostart_enabled()

    assert caught.value is error
    assert registry.closed is (operation == "query")
