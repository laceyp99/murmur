"""
Clipboard module for Murmur.
Handles copying transcribed text to the system clipboard.
"""

import pyperclip
from typing import Optional


class ClipboardManager:
    """
    Manages clipboard operations for copying transcribed text.
    """
    
    def __init__(self):
        self._last_copied: Optional[str] = None
    
    def copy(self, text: str) -> bool:
        """
        Copy text to the system clipboard.
        
        Args:
            text: The text to copy to clipboard.
        
        Returns:
            True if successful, False otherwise.
        """
        if not text:
            return False
        
        try:
            pyperclip.copy(text)
            self._last_copied = text
            return True
        except Exception as e:
            print(f"Failed to copy to clipboard: {e}")
            return False
    
    def get_last_copied(self) -> Optional[str]:
        """Get the last text that was copied."""
        return self._last_copied
    
    def get_current(self) -> Optional[str]:
        """Get the current clipboard content."""
        try:
            return pyperclip.paste()
        except Exception:
            return None
    
    def clear(self) -> bool:
        """Clear the clipboard."""
        try:
            pyperclip.copy('')
            return True
        except Exception:
            return False


# Global clipboard manager instance
_clipboard_manager: Optional[ClipboardManager] = None


def get_clipboard_manager() -> ClipboardManager:
    """Get the global clipboard manager instance."""
    global _clipboard_manager
    if _clipboard_manager is None:
        _clipboard_manager = ClipboardManager()
    return _clipboard_manager


def copy_to_clipboard(text: str) -> bool:
    """
    Convenience function to copy text to clipboard.
    
    Args:
        text: The text to copy.
    
    Returns:
        True if successful, False otherwise.
    """
    return get_clipboard_manager().copy(text)
