def check_and_import_gui_dependencies():
    try:
        import nicegui
        import plotly
    except ImportError as e:
        raise ImportError(
            "Optional dependencies for the GUI are not installed, "
            "namely nicegui>=2.0.0 and plotly>=5.0. "
            "Please install them using: pip install bellatrex[gui]"
        ) from e
    return nicegui, plotly
