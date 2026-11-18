import os
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from browden.dependencies.selenium import NoSuchWindowException
from browden.web_navigator.interface import TabNotFoundError
import browden.web_navigator.selenium_chrome.backend as backend
from browden.web_navigator.selenium_chrome.backend import (
    DEVTOOLS_WAIT_SECONDS,
    SINGLETON_FILES,
    ChromeLaunchError,
    SeleniumChromeBackend,
    _chrome_args,
    _clear_stale_singletons,
    _find_chrome_binary,
    _launch_chrome,
    _lexists,
    _pid_from_singleton_lock_target,
    _pid_is_alive,
    _profile_holder_pid,
    _singleton_lock_pid,
    _wellknown_chrome_paths,
)


# POSIX SingletonLock is a symlink to a non-existent ``host-pid`` path. Unit
# tests that plant one skip on Windows, where that lock is a regular file and
# creating symlinks often needs a privilege the CI runner does not have.
_posix_lock = pytest.mark.skipif(
    os.name == "nt", reason="POSIX SingletonLock is a symlink"
)

# The backend deals in raw Selenium handles — no namespace. profile_dir is
# required, defaulted here so tests that don't care can omit it.
def _backend(**kwargs):
    kwargs.setdefault("profile_dir", "/tmp/bg-unit-profile")
    return SeleniumChromeBackend(**kwargs)


def _make_fake_driver(handles=("h1",), dead=False):
    drv = MagicMock(name="driver")
    if dead:
        type(drv).window_handles = PropertyMock(
            side_effect=Exception("invalid session id")
        )
    else:
        drv.window_handles = list(handles)
    return drv


def _patch_launch(mock_launch, port=9222):
    """Make _launch_chrome return a fresh fake (proc, port) pair per call."""
    mock_launch.side_effect = lambda profile: (MagicMock(name="chrome_proc"), port)


def _debugger_address(mock_webdriver):
    """The debuggerAddress experimental option Selenium was attached with."""
    opts = mock_webdriver.Chrome.call_args.kwargs["options"]
    return opts.experimental_options["debuggerAddress"]


@patch("browden.web_navigator.selenium_chrome.backend._launch_chrome")
@patch("browden.web_navigator.selenium_chrome.backend.webdriver")
def test_drv_lazy_init(mock_webdriver, mock_launch):
    _patch_launch(mock_launch)
    fake = _make_fake_driver()
    mock_webdriver.Chrome.return_value = fake

    backend = _backend()
    assert backend._driver is None

    drv = backend._drv()
    assert drv is fake
    assert mock_webdriver.Chrome.call_count == 1

    drv2 = backend._drv()
    assert drv2 is fake
    assert mock_webdriver.Chrome.call_count == 1  # cached, not recreated
    assert mock_launch.call_count == 1


@patch("browden.web_navigator.selenium_chrome.backend._launch_chrome")
@patch("browden.web_navigator.selenium_chrome.backend.webdriver")
def test_drv_launches_with_provided_profile_dir(mock_webdriver, mock_launch, tmp_path):
    _patch_launch(mock_launch, port=7000)
    mock_webdriver.Chrome.return_value = _make_fake_driver()
    profile = tmp_path / "custom-profile"
    backend = _backend(profile_dir=str(profile))

    assert backend.get_profile_dir() == profile

    backend._drv()

    mock_launch.assert_called_once_with(profile)
    # Selenium attaches to the launched Chrome rather than spawning its own.
    assert _debugger_address(mock_webdriver) == "127.0.0.1:7000"


@patch("browden.web_navigator.selenium_chrome.backend._launch_chrome")
@patch("browden.web_navigator.selenium_chrome.backend.webdriver")
def test_drv_recreates_after_dead_session(mock_webdriver, mock_launch):
    _patch_launch(mock_launch)
    dead = _make_fake_driver(dead=True)
    alive = _make_fake_driver()
    mock_webdriver.Chrome.return_value = alive

    backend = _backend()
    backend._driver = dead  # simulate a cached, dead driver
    dead_proc = MagicMock(name="dead_chrome")
    dead_proc.poll.return_value = 1  # process actually exited — safe to relaunch
    backend._chrome_proc = dead_proc

    drv = backend._drv()

    assert drv is alive
    dead.quit.assert_called_once()
    dead_proc.terminate.assert_not_called()  # never kill; it already exited
    assert mock_webdriver.Chrome.call_count == 1  # one new driver after the dead one
    assert mock_launch.call_count == 1


@patch("browden.web_navigator.selenium_chrome.backend._launch_chrome")
@patch("browden.web_navigator.selenium_chrome.backend.webdriver")
def test_drv_does_not_relaunch_or_kill_when_chrome_is_still_running(mock_webdriver, mock_launch):
    """A wedged driver with a live Chrome must not spawn a second instance.

    That is the cookie-wipe path: SIGTERM/SIGKILL the starter, clear
    SingletonSocket, launch again on the same profile.
    """
    dead = _make_fake_driver(dead=True)
    backend = _backend()
    backend._driver = dead
    live_proc = MagicMock(name="live_chrome")
    live_proc.poll.return_value = None
    live_proc.pid = 4242
    backend._chrome_proc = live_proc

    with pytest.raises(ChromeLaunchError, match="PID 4242") as exc:
        backend._drv()

    assert "Not launching another Chrome" in str(exc.value)
    live_proc.terminate.assert_not_called()
    live_proc.kill.assert_not_called()
    mock_launch.assert_not_called()
    mock_webdriver.Chrome.assert_not_called()
    dead.quit.assert_called_once()  # detached the dead driver only
    assert backend._chrome_proc is live_proc  # still tracked


def test_devtools_wait_is_five_minutes():
    assert DEVTOOLS_WAIT_SECONDS == 300


def test_clear_stale_singletons_removes_files(tmp_path):
    for name in SINGLETON_FILES:
        (tmp_path / name).write_text("stale")
    (tmp_path / "Default").mkdir()  # unrelated content must survive

    _clear_stale_singletons(tmp_path)

    for name in SINGLETON_FILES:
        assert not (tmp_path / name).exists()
    assert (tmp_path / "Default").is_dir()


def test_clear_stale_singletons_noop_when_absent(tmp_path):
    _clear_stale_singletons(tmp_path)  # must not raise on empty dir


@_posix_lock
def test_clear_stale_singletons_removes_dangling_lock_for_dead_pid(tmp_path):
    """POSIX SingletonLock is a symlink to a non-existent ``host-pid`` path.

    ``Path.exists()`` follows it and returns False, which is how the old
    clearer left the lock in place while still unlinking SingletonSocket —
    the exact sequence that lets a second Chrome SIGKILL the first.
    A lock that names a dead PID is safe to remove.
    """
    lock = tmp_path / "SingletonLock"
    sock = tmp_path / "SingletonSocket"
    lock.symlink_to("testhost-999999")
    sock.symlink_to("/tmp/does-not-exist-browden-socket")
    assert not lock.exists()  # dangling: exists() is the trap
    assert lock.is_symlink()
    assert not _pid_is_alive(999999)

    _clear_stale_singletons(tmp_path)

    assert not _lexists(lock)
    assert not _lexists(sock)


@_posix_lock
def test_clear_stale_singletons_refuses_when_lock_pid_is_alive(tmp_path):
    lock = tmp_path / "SingletonLock"
    sock = tmp_path / "SingletonSocket"
    lock.symlink_to(f"testhost-{os.getpid()}")
    sock.write_text("socket")

    with pytest.raises(ChromeLaunchError, match=f"PID {os.getpid()}") as exc:
        _clear_stale_singletons(tmp_path)

    assert "already holds this profile" in str(exc.value)
    assert lock.is_symlink()  # untouched
    assert sock.exists()  # SingletonSocket must not be unlinked either


@_posix_lock
def test_clear_stale_singletons_refuses_unparseable_posix_lock(tmp_path):
    lock = tmp_path / "SingletonLock"
    lock.symlink_to("no-pid-here")  # no trailing -<digits>
    with pytest.raises(ChromeLaunchError, match="could not be verified"):
        _clear_stale_singletons(tmp_path)
    assert lock.is_symlink()


def test_pid_from_singleton_lock_target_splits_on_last_hyphen():
    assert _pid_from_singleton_lock_target("example.com-9156") == 9156
    assert _pid_from_singleton_lock_target("my-host-name-42") == 42
    assert _pid_from_singleton_lock_target("no-pid") is None
    assert _pid_from_singleton_lock_target("host-0") is None
    assert _pid_from_singleton_lock_target("") is None


@_posix_lock
def test_singleton_lock_pid_and_holder(tmp_path):
    assert _singleton_lock_pid(tmp_path) is None
    assert _profile_holder_pid(tmp_path) is None
    lock = tmp_path / "SingletonLock"
    lock.symlink_to(f"testhost-{os.getpid()}")
    assert _singleton_lock_pid(tmp_path) == os.getpid()
    assert _profile_holder_pid(tmp_path) == os.getpid()
    lock.unlink()
    lock.symlink_to("testhost-999999")
    assert _singleton_lock_pid(tmp_path) == 999999
    assert _profile_holder_pid(tmp_path) is None  # dead PID is not a holder
    lock.unlink()
    lock.write_text("not-a-symlink")
    assert _singleton_lock_pid(tmp_path) is None  # Windows-style regular file


def test_singleton_lock_pid_ignores_regular_file(tmp_path):
    """Windows (and leftover regular files) have no PID we can parse."""
    (tmp_path / "SingletonLock").write_text("not-a-symlink")
    assert _singleton_lock_pid(tmp_path) is None
    assert _profile_holder_pid(tmp_path) is None


def test_chrome_args_have_no_automation_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("BROWDEN_CHROME_BINARY", "/usr/bin/google-chrome")
    monkeypatch.delenv("BROWDEN_HEADLESS", raising=False)
    args = _chrome_args(tmp_path / "profile", 9222)

    assert args[0] == "/usr/bin/google-chrome"
    assert f"--user-data-dir={tmp_path / 'profile'}" in args
    assert "--remote-debugging-port=9222" in args
    # None of chromedriver's automation tells should appear — that's the point.
    joined = " ".join(args)
    assert "enable-automation" not in joined
    assert "AutomationControlled" not in joined
    assert "--test-type" not in joined
    # Headed by default: no headless switch unless asked for.
    assert not any(a.startswith("--headless") for a in args)
    # Software WebGL fallback is always on: a GPU-less host must still expose a
    # WebGL context, or anti-bot sensors (Akamai) flag the session — issue #29.
    assert "--enable-unsafe-swiftshader" in args


def test_chrome_args_enable_swiftshader_when_headless(tmp_path, monkeypatch):
    monkeypatch.setenv("BROWDEN_CHROME_BINARY", "/usr/bin/google-chrome")
    monkeypatch.setenv("BROWDEN_HEADLESS", "1")
    args = _chrome_args(tmp_path / "profile", 9222)
    # The WebGL fallback matters headless too (CI, containers): keep it on.
    assert "--enable-unsafe-swiftshader" in args


def test_chrome_args_headless_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("BROWDEN_CHROME_BINARY", "/usr/bin/google-chrome")
    monkeypatch.setenv("BROWDEN_HEADLESS", "1")
    monkeypatch.delenv("BROWDEN_NO_SANDBOX", raising=False)
    args = _chrome_args(tmp_path / "profile", 9222)
    assert "--headless=new" in args
    # Headless does NOT imply --no-sandbox: the sandbox stays on unless opted out.
    assert "--no-sandbox" not in args


def test_chrome_args_no_sandbox_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("BROWDEN_CHROME_BINARY", "/usr/bin/google-chrome")
    monkeypatch.delenv("BROWDEN_NO_SANDBOX", raising=False)
    # Off by default, even headless — the sandbox is kept.
    monkeypatch.setenv("BROWDEN_HEADLESS", "1")
    assert "--no-sandbox" not in _chrome_args(tmp_path / "profile", 9222)
    # Explicit opt-in adds it (the escape hatch for sandbox-less hosts).
    monkeypatch.setenv("BROWDEN_NO_SANDBOX", "1")
    assert "--no-sandbox" in _chrome_args(tmp_path / "profile", 9222)


def test_find_chrome_binary_honours_env(monkeypatch):
    monkeypatch.setenv("BROWDEN_CHROME_BINARY", "/opt/chrome/chrome")
    assert _find_chrome_binary() == "/opt/chrome/chrome"


def test_find_chrome_binary_searches_path(monkeypatch):
    monkeypatch.delenv("BROWDEN_CHROME_BINARY", raising=False)
    monkeypatch.delenv("CHROME_BIN", raising=False)

    def fake_which(name):
        return "/usr/bin/google-chrome" if name == "google-chrome" else None

    monkeypatch.setattr(
        "browden.web_navigator.selenium_chrome.backend.shutil.which", fake_which
    )
    assert _find_chrome_binary() == "/usr/bin/google-chrome"


def test_find_chrome_binary_raises_when_missing(monkeypatch):
    monkeypatch.delenv("BROWDEN_CHROME_BINARY", raising=False)
    monkeypatch.delenv("CHROME_BIN", raising=False)
    monkeypatch.setattr(
        "browden.web_navigator.selenium_chrome.backend.shutil.which",
        lambda name: None,
    )
    # No well-known install either — must raise, not silently return nothing.
    monkeypatch.setattr(backend, "_wellknown_chrome_paths", list)
    with pytest.raises(RuntimeError):
        _find_chrome_binary()


def test_find_chrome_binary_uses_wellknown_when_not_on_path(monkeypatch, tmp_path):
    # macOS/Windows: Chrome isn't on PATH but sits at a canonical install path.
    monkeypatch.delenv("BROWDEN_CHROME_BINARY", raising=False)
    monkeypatch.delenv("CHROME_BIN", raising=False)
    monkeypatch.setattr(backend.shutil, "which", lambda name: None)
    chrome = tmp_path / "Google Chrome"
    chrome.write_text("#!/bin/sh\n")
    monkeypatch.setattr(
        backend, "_wellknown_chrome_paths",
        lambda: [str(tmp_path / "does-not-exist"), str(chrome)],
    )
    assert _find_chrome_binary() == str(chrome)


def test_wellknown_chrome_paths_macos():
    paths = _wellknown_chrome_paths("darwin", "posix")
    assert any(p.endswith("Contents/MacOS/Google Chrome") for p in paths)


def test_wellknown_chrome_paths_windows():
    paths = _wellknown_chrome_paths("win32", "nt")
    assert paths and all(p.endswith("chrome.exe") for p in paths)


def test_wellknown_chrome_paths_linux_is_empty():
    assert _wellknown_chrome_paths("linux", "posix") == []


@patch("browden.web_navigator.selenium_chrome.backend._wait_for_devtools")
@patch("browden.web_navigator.selenium_chrome.backend.subprocess")
@patch("browden.web_navigator.selenium_chrome.backend._free_port", return_value=4321)
@patch("browden.web_navigator.selenium_chrome.backend._chrome_args", return_value=["chrome"])
@patch("browden.web_navigator.selenium_chrome.backend._clear_stale_singletons")
def test_launch_chrome_clears_singletons_before_spawning(
    mock_clear, mock_args, mock_port, mock_subprocess, mock_wait, tmp_path
):
    proc = MagicMock(name="proc")
    mock_subprocess.Popen.return_value = proc

    manager = MagicMock()
    manager.attach_mock(mock_clear, "clear")
    manager.attach_mock(mock_subprocess.Popen, "Popen")

    out_proc, port = _launch_chrome(tmp_path)

    assert (out_proc, port) == (proc, 4321)
    call_names = [c[0] for c in manager.mock_calls]
    assert call_names == ["clear", "Popen"]  # stale lock cleared before spawn
    mock_wait.assert_called_once()
    # Detached from the MCP process group so a server SIGHUP cannot kill Chrome.
    assert mock_subprocess.Popen.call_args.kwargs["start_new_session"] is True


@patch("browden.web_navigator.selenium_chrome.backend._wait_for_devtools")
@patch("browden.web_navigator.selenium_chrome.backend._terminate")
@patch("browden.web_navigator.selenium_chrome.backend.subprocess")
@patch("browden.web_navigator.selenium_chrome.backend._free_port", return_value=4321)
@patch("browden.web_navigator.selenium_chrome.backend._chrome_args", return_value=["chrome"])
@patch("browden.web_navigator.selenium_chrome.backend._clear_stale_singletons")
def test_launch_chrome_does_not_kill_when_devtools_never_comes_up(
    mock_clear, mock_args, mock_port, mock_subprocess, mock_terminate, mock_wait, tmp_path
):
    proc = MagicMock(name="proc")
    proc.poll.return_value = None  # still starting
    proc.pid = 7777
    mock_subprocess.Popen.return_value = proc
    mock_wait.side_effect = RuntimeError("never came up")

    with pytest.raises(ChromeLaunchError, match="PID 7777") as exc:
        _launch_chrome(tmp_path)
    assert "Not killing it" in str(exc.value)
    assert exc.value.proc is proc
    mock_terminate.assert_not_called()  # leave the starter alone


@patch("browden.web_navigator.selenium_chrome.backend._wait_for_devtools")
@patch("browden.web_navigator.selenium_chrome.backend._terminate")
@patch("browden.web_navigator.selenium_chrome.backend.subprocess")
@patch("browden.web_navigator.selenium_chrome.backend._free_port", return_value=4321)
@patch("browden.web_navigator.selenium_chrome.backend._chrome_args", return_value=["chrome"])
@patch("browden.web_navigator.selenium_chrome.backend._clear_stale_singletons")
def test_launch_chrome_does_not_kill_when_chrome_exits_before_devtools(
    mock_clear, mock_args, mock_port, mock_subprocess, mock_terminate, mock_wait, tmp_path
):
    proc = MagicMock(name="proc")
    proc.poll.return_value = 1
    mock_subprocess.Popen.return_value = proc
    mock_wait.side_effect = RuntimeError("Chrome exited (code 1)")

    with pytest.raises(RuntimeError, match="Chrome exited") as exc:
        _launch_chrome(tmp_path)
    assert type(exc.value) is RuntimeError  # already dead: not a "leave it running" error
    mock_terminate.assert_not_called()


@patch("browden.web_navigator.selenium_chrome.backend.subprocess")
@patch("browden.web_navigator.selenium_chrome.backend._free_port")
@patch("browden.web_navigator.selenium_chrome.backend._chrome_args")
@_posix_lock
def test_launch_chrome_refuses_when_profile_is_held(
    mock_args, mock_port, mock_subprocess, tmp_path
):
    (tmp_path / "SingletonLock").symlink_to(f"testhost-{os.getpid()}")
    with pytest.raises(ChromeLaunchError, match=f"PID {os.getpid()}"):
        _launch_chrome(tmp_path)
    mock_subprocess.Popen.assert_not_called()
    mock_port.assert_not_called()


@patch("browden.web_navigator.selenium_chrome.backend._launch_chrome")
@patch("browden.web_navigator.selenium_chrome.backend.webdriver")
def test_drv_swallows_quit_error_on_dead_driver(mock_webdriver, mock_launch):
    _patch_launch(mock_launch)
    dead = _make_fake_driver(dead=True)
    dead.quit.side_effect = Exception("already gone")
    alive = _make_fake_driver()
    mock_webdriver.Chrome.return_value = alive

    backend = _backend()
    backend._driver = dead

    drv = backend._drv()
    assert drv is alive


def _backend_with_driver(drv):
    backend = _backend()
    backend._driver = drv
    return backend


def test_switch_failure_becomes_page_not_found():
    drv = _make_fake_driver(handles=("h1", "h2"))
    drv.switch_to.window.side_effect = NoSuchWindowException("no such window\n  (Session info: ...)")
    backend = _backend_with_driver(drv)

    # page_ids are raw Selenium handles; a handle Selenium rejects must surface
    # as a clean TabNotFoundError, not the driver's stack-trace exception. The
    # tab-targeting methods that still take a handle (select_tab / close_tab) are
    # where that translation lives — get_page_source / reload / screenshot no
    # longer self-focus, so the caller's select_tab is the one gate.
    for call in (lambda: backend.select_tab("dead"),
                 lambda: backend.close_tab("dead")):
        with pytest.raises(TabNotFoundError) as exc:
            call()
        assert "dead" in str(exc.value)
        assert "Session info" not in str(exc.value)  # no driver stack trace leaks through


def test_profile_dir_is_required():
    with pytest.raises(TypeError):
        SeleniumChromeBackend()  # profile_dir is required
    with pytest.raises(ValueError):
        SeleniumChromeBackend(None)  # must be non-empty
    with pytest.raises(ValueError):
        SeleniumChromeBackend("")  # must be non-empty


def test_get_profile_dir_returns_construction_path(tmp_path):
    backend = SeleniumChromeBackend(profile_dir=str(tmp_path / "prof"))
    assert backend.get_profile_dir() == tmp_path / "prof"


@patch("browden.web_navigator.selenium_chrome.backend._launch_chrome")
@patch("browden.web_navigator.selenium_chrome.backend.webdriver")
def test_drv_recovers_when_current_window_is_gone(mock_webdriver, mock_launch):
    # A missing focused window is not a dead browser: refocus a remaining
    # handle rather than tearing the session down and launching another Chrome.
    drv = _make_fake_driver(handles=("h1",))
    type(drv).current_window_handle = PropertyMock(side_effect=NoSuchWindowException("no such window"))

    backend = _backend_with_driver(drv)

    assert backend.list_handles() == ["h1"]
    drv.quit.assert_not_called()
    mock_launch.assert_not_called()
    mock_webdriver.Chrome.assert_not_called()
    drv.switch_to.window.assert_called_with("h1")


def test_is_running_probes_session_only_not_current_window():
    # is_running and _drv share _session_alive, but is_running is the look-only
    # probe: it checks the session (window_handles), NOT the current window (that's
    # _drv's heal-able concern). A live session with a stale current window still
    # reports running — no teardown, no relaunch.
    drv = _make_fake_driver(handles=("h1",))
    type(drv).current_window_handle = PropertyMock(side_effect=NoSuchWindowException("no such window"))
    backend = _backend_with_driver(drv)

    assert backend.is_running() is True
    drv.quit.assert_not_called()  # look-only: never tears the session down


def test_is_running_false_and_tears_down_a_dead_session():
    drv = _make_fake_driver(dead=True)  # window_handles itself raises
    backend = _backend_with_driver(drv)

    assert backend.is_running() is False
    drv.quit.assert_called_once()  # dead driver is detached (but Chrome is not killed)


def test_is_running_false_does_not_kill_a_live_chrome_process():
    drv = _make_fake_driver(dead=True)
    backend = _backend_with_driver(drv)
    live_proc = MagicMock(name="live_chrome")
    live_proc.poll.return_value = None
    live_proc.pid = 99
    backend._chrome_proc = live_proc

    assert backend.is_running() is False
    live_proc.terminate.assert_not_called()
    live_proc.kill.assert_not_called()
    assert backend._chrome_proc is live_proc


@patch("browden.web_navigator.selenium_chrome.backend._launch_chrome")
@patch("browden.web_navigator.selenium_chrome.backend.webdriver")
def test_list_handles_relaunches_a_dead_browser_and_is_running_does_not(
        mock_webdriver, mock_launch):
    """The resurrection mechanism behind the idle reaper.

    ``sweep_idle`` always enumerates with ``list_handles``. That goes through
    ``_drv``, which starts Chrome when the attached session is dead.
    ``is_running`` is the look-only probe ``list_tabs`` uses, and it must not.
    No real Chrome: ``_launch_chrome`` and ``webdriver.Chrome`` are mocked.
    """
    _patch_launch(mock_launch)
    dead = _make_fake_driver(dead=True)
    alive = _make_fake_driver(handles=("h1",))
    mock_webdriver.Chrome.return_value = alive
    backend = _backend_with_driver(dead)

    assert backend.is_running() is False
    mock_launch.assert_not_called()
    mock_webdriver.Chrome.assert_not_called()

    assert backend.list_handles() == ["h1"]
    mock_launch.assert_called_once()
    mock_webdriver.Chrome.assert_called_once()


@patch("browden.web_navigator.selenium_chrome.backend._launch_chrome")
@patch("browden.web_navigator.selenium_chrome.backend.webdriver")
def test_drv_recreates_when_window_handles_fails(mock_webdriver, mock_launch):
    _patch_launch(mock_launch)
    # Driver that fails on window_handles (session dead)
    dead_drv = _make_fake_driver(dead=True)

    # New driver
    alive_drv = _make_fake_driver(handles=("h1",))
    mock_webdriver.Chrome.return_value = alive_drv

    backend = _backend_with_driver(dead_drv)

    drv = backend._drv()
    assert drv is alive_drv
    dead_drv.quit.assert_called_once()
    assert mock_webdriver.Chrome.call_count == 1


def test_close_page_refocuses_a_survivor():
    # Closing the focused tab leaves the driver on a dead handle; close_tab must
    # re-focus a remaining window so the next command doesn't see a "dead" session.
    drv = _make_fake_driver(handles=("h1", "h2"))

    def _close():
        drv.window_handles = ["h1"]  # h2 is gone after drv.close()
    drv.close.side_effect = _close

    backend = _backend_with_driver(drv)
    backend.close_tab("h2")

    drv.close.assert_called_once()
    # Last switch_to.window call targets a surviving handle, not the closed one.
    assert drv.switch_to.window.call_args.args == ("h1",)


def test_close_page_last_tab_is_refused():
    drv = _make_fake_driver(handles=("only",))
    backend = _backend_with_driver(drv)
    with pytest.raises(ValueError):
        backend.close_tab("only")
    drv.close.assert_not_called()


def test_navigate_failure_becomes_page_not_found():
    drv = _make_fake_driver()
    drv.get.side_effect = NoSuchWindowException("no such window")
    backend = _backend_with_driver(drv)
    with pytest.raises(TabNotFoundError):
        backend.navigate("https://example.com")


@patch("browden.web_navigator.selenium_chrome.backend._launch_chrome")
@patch("browden.web_navigator.selenium_chrome.backend.webdriver")
def test_list_tab_ids_returns_handles_without_switching(mock_webdriver, mock_launch):
    _patch_launch(mock_launch)
    drv = _make_fake_driver(handles=("h1", "h2", "h3"))
    mock_webdriver.Chrome.return_value = drv
    backend = _backend()

    assert backend.list_handles() == ["h1", "h2", "h3"]
    drv.switch_to.window.assert_not_called()  # cheap: no per-tab focus changes


@patch("browden.web_navigator.selenium_chrome.backend._launch_chrome")
@patch("browden.web_navigator.selenium_chrome.backend.webdriver")
def test_drv_keeps_proc_and_refuses_second_spawn_after_devtools_timeout(
        mock_webdriver, mock_launch):
    proc = MagicMock(name="starter")
    proc.poll.return_value = None
    proc.pid = 555
    mock_launch.side_effect = ChromeLaunchError("not ready", proc=proc)
    backend = _backend()

    with pytest.raises(ChromeLaunchError, match="not ready"):
        backend._drv()
    assert backend._chrome_proc is proc

    mock_launch.reset_mock()
    with pytest.raises(ChromeLaunchError, match="PID 555") as exc:
        backend._drv()
    assert "Not launching another Chrome" in str(exc.value)
    mock_launch.assert_not_called()
    mock_webdriver.Chrome.assert_not_called()
    proc.terminate.assert_not_called()


@patch("browden.web_navigator.selenium_chrome.backend._launch_chrome")
@patch("browden.web_navigator.selenium_chrome.backend.webdriver")
@_posix_lock
def test_drv_refuses_when_foreign_chrome_holds_the_profile(
        mock_webdriver, mock_launch, tmp_path):
    profile = tmp_path / "prof"
    profile.mkdir()
    (profile / "SingletonLock").symlink_to(f"testhost-{os.getpid()}")
    backend = _backend(profile_dir=str(profile))

    with pytest.raises(ChromeLaunchError, match=f"PID {os.getpid()}"):
        backend._drv()
    mock_launch.assert_not_called()
    mock_webdriver.Chrome.assert_not_called()


@patch("browden.web_navigator.selenium_chrome.backend._launch_chrome")
@patch("browden.web_navigator.selenium_chrome.backend.webdriver")
def test_drv_does_not_kill_chrome_when_selenium_attach_fails(mock_webdriver, mock_launch):
    proc = MagicMock(name="chrome")
    proc.poll.return_value = None
    proc.pid = 12
    mock_launch.return_value = (proc, 9222)
    mock_webdriver.Chrome.side_effect = Exception("chrome not reachable")
    backend = _backend()

    with pytest.raises(ChromeLaunchError, match="could not attach"):
        backend._drv()
    proc.terminate.assert_not_called()
    proc.kill.assert_not_called()
    assert backend._chrome_proc is proc
    assert backend._driver is None
