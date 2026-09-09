"""Bundle Torch for Murmur without collecting unrelated training toolchains."""

from PyInstaller.utils.hooks import (
    PY_DYLIB_PATTERNS,
    collect_data_files,
    collect_dynamic_libs,
)

module_collection_mode = "pyz+py"
warn_on_missing_hiddenimports = False

datas = collect_data_files(
    "torch",
    excludes=[
        "**/*.h",
        "**/*.hpp",
        "**/*.cuh",
        "**/*.lib",
        "**/*.cpp",
        "**/*.pyi",
        "**/*.cmake",
    ],
)
binaries = collect_dynamic_libs(
    "torch",
    search_patterns=[*PY_DYLIB_PATTERNS, "*.so.*"],
)
hiddenimports = [
    "torch._C",
    "torch.autograd",
    "torch.backends.cuda",
    "torch.backends.cudnn",
    "torch.cuda",
    "torch.distributions",
    "torch.distributions.categorical",
    "torch.nn",
    "torch.nn.functional",
]
