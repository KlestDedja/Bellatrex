def check_and_import_gui_dependencies():
    """Import the NiceGUI backend.

    Returns the ``nicegui`` module.

    Raises :class:`ImportError` when NiceGUI is not installed.
    """
    try:
        import nicegui

        return nicegui
    except ImportError as e:
        raise ImportError(
            "Optional dependency 'nicegui' is not installed. "
            "Install it with: pip install bellatrex"
        ) from e
