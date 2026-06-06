try:
    from app.bellatrex._nicegui import runtime
except ImportError:
    from bellatrex._nicegui import runtime


def test_detect_native_window_support_requires_pythonnet_on_windows(monkeypatch):
    def fake_find_spec(name: str):
        if name == "webview":
            return object()
        if name == "pythonnet":
            return None
        return None

    monkeypatch.setattr(runtime.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(runtime.sys, "platform", "win32")

    assert runtime.detect_native_window_support() is False


def test_detect_native_window_support_accepts_windows_with_pythonnet(monkeypatch):
    def fake_find_spec(name: str):
        if name in {"webview", "pythonnet"}:
            return object()
        return None

    monkeypatch.setattr(runtime.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(runtime.sys, "platform", "win32")

    assert runtime.detect_native_window_support() is True


def test_detect_native_window_support_only_needs_webview_off_windows(monkeypatch):
    def fake_find_spec(name: str):
        if name == "webview":
            return object()
        if name == "pythonnet":
            return None
        return None

    monkeypatch.setattr(runtime.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(runtime.sys, "platform", "linux")

    assert runtime.detect_native_window_support() is True
