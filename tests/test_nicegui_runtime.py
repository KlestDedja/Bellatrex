try:
    from app.bellatrex._nicegui import runtime
except ImportError:
    from bellatrex._nicegui import runtime


def test_run_subprocess_app_times_out_and_kills_process(monkeypatch):
    calls = []

    class FakeProcess:
        def __init__(self, command):
            self.command = command
            self.wait_count = 0

        def wait(self, timeout=None):
            self.wait_count += 1
            calls.append(("wait", timeout))
            if self.wait_count == 1:
                raise runtime.subprocess.TimeoutExpired(self.command, timeout)
            if self.wait_count == 2:
                raise runtime.subprocess.TimeoutExpired(self.command, timeout)
            return 0

        def terminate(self):
            calls.append(("terminate", None))

        def kill(self):
            calls.append(("kill", None))

    monkeypatch.setenv("BELLATREX_GUI_SUBPROCESS_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setattr(runtime.subprocess, "Popen", FakeProcess)

    try:
        runtime.run_subprocess_app("payload.pkl", blocking=True)
    except RuntimeError as exc:
        assert "timed out after 0.1 seconds" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")

    assert calls == [
        ("wait", 0.1),
        ("terminate", None),
        ("wait", 5),
        ("kill", None),
        ("wait", None),
    ]


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
