from pathlib import Path

from PyInstaller.building import build_main
from PyInstaller.utils.hooks import collect_data_files


_find_binary_dependencies = build_main.find_binary_dependencies


def find_binary_dependencies(binaries, import_packages, symlink_suppression_patterns):
    """Initialize Torch's DLL path once instead of importing every subpackage."""
    if any(
        package == "torch" or package.startswith("torch.")
        for package in import_packages
    ):
        import_packages = [
            "torch",
            *(
                package
                for package in import_packages
                if package != "torch" and not package.startswith("torch.")
            ),
        ]
    return _find_binary_dependencies(
        binaries, import_packages, symlink_suppression_patterns
    )


build_main.find_binary_dependencies = find_binary_dependencies

repo_root = Path(SPECPATH).parent
generated_root = repo_root / "build" / "windows"
icon_path = generated_root / "murmur.ico"
version_path = generated_root / "version_info.txt"

datas = collect_data_files("customtkinter")
datas += collect_data_files("whisper")
datas += [
    (str(repo_root / "murmur tray logo.png"), "."),
    (str(repo_root / "murmur logo.png"), "."),
]

hidden_imports = [
    "pystray._win32",
    "winrt.windows.foundation",
    "winrt.windows.foundation.collections",
    "winrt.windows.media.control",
]

excluded_modules = [
    "_pytest",
    "functorch",
    "pytest",
    "torch._dynamo",
    "torch._inductor",
    "torch._lazy",
    "torch._numpy",
    "torch.onnx",
    "torch.utils.hipify",
    "torch.utils.tensorboard",
]

analysis = Analysis(
    [str(repo_root / "run.py")],
    pathex=[str(repo_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[str(repo_root / "packaging" / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="murmur",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path),
    version=str(version_path),
)

coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Murmur",
)
