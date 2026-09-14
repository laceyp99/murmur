"""Collect metadata for the maintained webrtcvad-wheels distribution."""

from PyInstaller.utils.hooks import copy_metadata

datas = copy_metadata("webrtcvad-wheels")
