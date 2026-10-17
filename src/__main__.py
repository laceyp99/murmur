"""
Entry point for running murmur as a module.
Usage: python -m src
"""

from .launch import ensure_output_streams

if __name__ == "__main__":
    ensure_output_streams()

    from .main import main

    main()
