import contextlib
import importlib.util
from pathlib import Path

import pytest

from src import windows_identity

pytest.importorskip("winreg", reason="toast icon registration is Windows-only")

MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "tools" / "register_toast_icon.py"
)
SPEC = importlib.util.spec_from_file_location("register_toast_icon", MODULE_PATH)
register_toast_icon = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(register_toast_icon)


@pytest.fixture
def fake_registry(monkeypatch):
    keys = {}
    winreg = register_toast_icon.winreg

    @contextlib.contextmanager
    def create_key(root, sub_key):
        yield keys.setdefault((root, sub_key), {})

    def set_value(key, name, _reserved, _value_type, value):
        key[name] = value

    def delete_key(root, sub_key):
        if (root, sub_key) not in keys:
            raise FileNotFoundError(sub_key)
        del keys[(root, sub_key)]

    monkeypatch.setattr(winreg, "CreateKey", create_key)
    monkeypatch.setattr(winreg, "SetValueEx", set_value)
    monkeypatch.setattr(winreg, "DeleteKey", delete_key)
    return keys


def test_app_user_model_id_matches_runtime_identity():
    assert (
        register_toast_icon.APP_USER_MODEL_ID
        == windows_identity._WINDOWS_APP_USER_MODEL_ID
    )


def _make_repo(tmp_path, icon_bytes=b"icon"):
    repo_root = tmp_path / "repo"
    (repo_root / "app").mkdir(parents=True, exist_ok=True)
    (repo_root / "app" / "toast-icon.png").write_bytes(icon_bytes)
    return repo_root


def test_register_copies_icon_and_writes_identity(tmp_path, fake_registry):
    repo_root = _make_repo(tmp_path)
    app_data_dir = tmp_path / "app-data"

    icon_path = register_toast_icon.register(repo_root, app_data_dir)

    assert icon_path.parent == app_data_dir
    assert icon_path.name.startswith("toast-icon-")
    assert icon_path.read_bytes() == b"icon"
    key = (
        register_toast_icon.winreg.HKEY_CURRENT_USER,
        register_toast_icon.REGISTRY_KEY,
    )
    assert fake_registry[key] == {"DisplayName": "murmur", "IconUri": str(icon_path)}


def test_register_requires_app_icon(tmp_path, fake_registry):
    with pytest.raises(FileNotFoundError):
        register_toast_icon.register(tmp_path, tmp_path / "app-data")

    assert fake_registry == {}


def test_register_renames_changed_icon_and_removes_old_copy(tmp_path, fake_registry):
    app_data_dir = tmp_path / "app-data"
    first_path = register_toast_icon.register(_make_repo(tmp_path), app_data_dir)

    second_path = register_toast_icon.register(
        _make_repo(tmp_path, b"new icon"), app_data_dir
    )

    assert second_path != first_path
    assert list(app_data_dir.iterdir()) == [second_path]


def test_remove_deletes_identity_and_icon(tmp_path, fake_registry):
    repo_root = _make_repo(tmp_path)
    app_data_dir = tmp_path / "app-data"
    icon_path = register_toast_icon.register(repo_root, app_data_dir)

    register_toast_icon.remove(app_data_dir)
    register_toast_icon.remove(app_data_dir)

    assert fake_registry == {}
    assert not icon_path.exists()
