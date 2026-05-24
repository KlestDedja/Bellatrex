import importlib.util
import os
import pickle
import shutil
import socket
import subprocess
import sys
import tempfile
from uuid import uuid4


def detect_native_window_support() -> bool:
    return importlib.util.find_spec("webview") is not None


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def ensure_nicegui_screen_test_port(port: int) -> None:
    os.environ.setdefault("NICEGUI_SCREEN_TEST_PORT", str(port))


def prepare_session_temp_dir(
    base_temp_files_dir: str,
    colorbar_paths: list[str],
) -> tuple[str, list[str]]:
    """Move one explorer session into its own temp directory.

    This keeps concurrent ``blocking=False`` sessions isolated from each other.
    Bellatrex reuses tree ids across loop iterations, so isolation is the safe
    guarantee here; early deletion is only possible after that specific session
    closes.
    """
    session_temp_dir = os.path.abspath(
        os.path.join(base_temp_files_dir, f"bellatrex_session_{uuid4().hex}")
    )
    os.makedirs(session_temp_dir, exist_ok=True)

    session_colorbar_paths = []
    for colorbar_path in colorbar_paths:
        session_colorbar_path = os.path.abspath(
            os.path.join(session_temp_dir, os.path.basename(colorbar_path))
        )
        if os.path.exists(colorbar_path):
            shutil.move(colorbar_path, session_colorbar_path)
        session_colorbar_paths.append(session_colorbar_path)

    return session_temp_dir, session_colorbar_paths


def prepare_tree_window_temp_dir(
    session_temp_dir: str,
    source_image_path: str,
) -> tuple[str, str, str]:
    """Copy a cached tree image into a child window temp dir."""
    base_temp_files_dir = os.path.dirname(os.path.abspath(session_temp_dir))
    child_temp_dir = os.path.abspath(
        os.path.join(base_temp_files_dir, f"bellatrex_tree_window_{uuid4().hex}")
    )
    os.makedirs(child_temp_dir, exist_ok=True)

    image_name = os.path.basename(source_image_path)
    child_image_path = os.path.abspath(os.path.join(child_temp_dir, image_name))
    shutil.copy2(source_image_path, child_image_path)
    return child_temp_dir, image_name, child_image_path


def write_subprocess_payload(payload: dict, temp_files_dir: str) -> str:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".bellatrex-gui.pkl",
        delete=False,
        dir=temp_files_dir,
    ) as handle:
        pickle.dump(payload, handle)
        return handle.name


def build_main_window_payload(
    plots,
    colorbar_paths,
    render_context: dict | None,
    native: bool,
    port: int,
    temp_files_dir: str,
) -> str:
    return write_subprocess_payload(
        {
            "kind": "main_window",
            "plots": plots,
            "colorbar_paths": colorbar_paths,
            "render_context": render_context,
            "native": native,
            "port": port,
            "temp_files_dir": temp_files_dir,
        },
        temp_files_dir,
    )


def build_tree_window_payload(
    image_name: str,
    title: str,
    subtitle: str | None,
    native: bool,
    port: int,
    temp_files_dir: str,
) -> str:
    return write_subprocess_payload(
        {
            "kind": "tree_window",
            "image_name": image_name,
            "title": title,
            "subtitle": subtitle,
            "native": native,
            "port": port,
            "temp_files_dir": temp_files_dir,
        },
        temp_files_dir,
    )


def run_subprocess_app(payload_path: str, blocking: bool) -> None:
    command = [sys.executable, "-m", "bellatrex.nicegui_plots_code", "--payload", payload_path]
    process = subprocess.Popen(command)

    if blocking:
        exit_code = process.wait()
        if exit_code != 0:
            raise RuntimeError(f"NiceGUI window process exited with code {exit_code}.")


def cleanup_temp_artifacts(
    temp_files_dir: str,
    colorbar_paths: list[str] | None = None,
    tree_image_paths: list[str] | None = None,
    payload_path: str | None = None,
) -> None:
    paths_to_remove = set(colorbar_paths or [])
    paths_to_remove.update(tree_image_paths or [])
    if payload_path:
        paths_to_remove.add(payload_path)

    if os.path.isdir(temp_files_dir):
        for entry in os.listdir(temp_files_dir):
            if (
                entry.startswith("temp_colourbar")
                and entry.endswith(".png")
                or entry.startswith("tree_")
                and entry.endswith(".png")
                or entry.endswith(".bellatrex-gui.pkl")
            ):
                paths_to_remove.add(os.path.join(temp_files_dir, entry))

    for path in paths_to_remove:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    try:
        if os.path.isdir(temp_files_dir) and not os.listdir(temp_files_dir):
            os.rmdir(temp_files_dir)
    except OSError:
        pass


def cleanup_payload_file(payload_path: str | None) -> None:
    if not payload_path:
        return
    try:
        os.remove(payload_path)
    except FileNotFoundError:
        pass


def serve_payload_file(payload_path: str) -> None:
    from .windows import _run_nicegui_app, _run_tree_window_app

    with open(payload_path, "rb") as handle:
        payload = pickle.load(handle)

    kind = payload.get("kind", "main_window")
    if kind == "tree_window":
        _run_tree_window_app(
            payload["image_name"],
            payload["title"],
            payload.get("subtitle"),
            native=payload["native"],
            port=payload["port"],
            temp_files_dir=payload["temp_files_dir"],
            payload_path=payload_path,
        )
        return

    _run_nicegui_app(
        payload["plots"],
        payload["colorbar_paths"],
        payload.get("render_context"),
        native=payload["native"],
        port=payload["port"],
        temp_files_dir=payload["temp_files_dir"],
        payload_path=payload_path,
    )
