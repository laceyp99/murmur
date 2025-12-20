"""
System tray management for Murmur.
"""

import threading
import os
from PIL import Image
import pystray
from pystray import MenuItem as item

from .config import get_config
from .settings_gui import show_settings

class TrayManager:
    """
    Manages the system tray icon and menu for Murmur.
    """
    
    def __init__(self, on_exit_callback=None):
        self.config = get_config()
        self.on_exit_callback = on_exit_callback
        self.icon = None
        self._status_text = "Ready"
        
    def _create_image(self):
        """Load and prepare the tray icon image."""
        # Try to find the logo file
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "murmur tray logo.png"),
            os.path.join(os.path.dirname(__file__), "..", "murmur.png"),
        ]
        
        logo_path = None
        for path in possible_paths:
            if os.path.exists(path):
                logo_path = path
                break
        
        if logo_path:
            try:
                image = Image.open(logo_path)
                # Resize for tray (usually 16x16 or 32x32, but Pillow handles scaling)
                image = image.resize((64, 64), Image.Resampling.LANCZOS)
                return image
            except Exception as e:
                print(f"Error loading tray icon: {e}")
        
        # Fallback: Create a simple colored square if no image found
        image = Image.new('RGB', (64, 64), color=(73, 109, 137))
        return image

    def set_status(self, status: str):
        """Update the status text in the tray menu."""
        self._status_text = status
        if self.icon:
            self.icon.update_menu()

    def _on_settings(self):
        """Open the settings window."""
        # Run in a separate thread to avoid blocking the tray
        threading.Thread(target=show_settings, daemon=True).start()

    def _on_exit(self, icon, item):
        """Handle exit from tray menu."""
        if self.on_exit_callback:
            self.on_exit_callback()
        icon.stop()

    def run(self):
        """Start the system tray icon loop."""
        menu = pystray.Menu(
            item(lambda text: f"Status: {self._status_text}", lambda: None, enabled=False),
            pystray.Menu.SEPARATOR,
            item("Settings", self._on_settings),
            item("Exit", self._on_exit)
        )
        
        self.icon = pystray.Icon(
            "murmur",
            self._create_image(),
            "murmur",
            menu
        )
        
        self.icon.run()

    def stop(self):
        """Stop the tray icon."""
        if self.icon:
            self.icon.stop()
