from src import assets


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
