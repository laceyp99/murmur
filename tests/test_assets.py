from src import assets


def test_logo_path_uses_project_root_in_source_mode(tmp_path, monkeypatch):
    logo_path = tmp_path / "murmur tray logo.png"
    logo_path.write_bytes(b"logo")
    monkeypatch.setattr(assets, "PROJECT_ROOT", tmp_path)
    monkeypatch.delattr(assets.sys, "_MEIPASS", raising=False)

    assert assets.get_logo_path() == logo_path


def test_logo_path_uses_bundle_root_when_frozen(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    bundle_root = tmp_path / "bundle"
    source_root.mkdir()
    bundle_root.mkdir()
    logo_path = bundle_root / "murmur tray logo.png"
    logo_path.write_bytes(b"logo")
    monkeypatch.setattr(assets, "PROJECT_ROOT", source_root)
    monkeypatch.setattr(assets.sys, "_MEIPASS", str(bundle_root), raising=False)

    assert assets.get_logo_path() == logo_path


def test_app_icon_uses_canonical_app_data_dir(tmp_path, monkeypatch):
    logo_path = tmp_path / "logo.png"
    icon_dir = tmp_path / "app-data"
    icon_path = icon_dir / assets.APP_ICON_FILENAME
    logo_path.write_bytes(b"logo")
    icon_dir.mkdir()
    icon_path.write_bytes(b"icon")
    icon_path.touch()

    monkeypatch.setattr(assets, "get_logo_path", lambda: logo_path)
    monkeypatch.setattr(assets, "get_app_data_dir", lambda: icon_dir)

    assert assets.get_app_icon_path() == icon_path
