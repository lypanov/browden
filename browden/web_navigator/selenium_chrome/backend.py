import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from ...common.logger import logger
from ...common.tab import TabInfo
from ...dependencies.selenium import (
    By,
    ChromeOptions,
    InvalidElementStateException,
    Keys,
    NoSuchElementException,
    NoSuchWindowException,
    WebDriverWait,
    webdriver,
)
from ..interface import TabNotFoundError, WebNavigatorBackend
from ..utils.network_utils import get_free_port

# W3C `key` value -> the Selenium `Keys` constant that dispatches it. Covers
# exactly the control keys the press-key gate authorizes (validator.ACTIVATION_KEYS);
# a key that reaches here outside this map is a caller/gate mismatch and raises.
_SELENIUM_KEYS = {
    "Enter": Keys.ENTER,
    "Space": Keys.SPACE,
    "Tab": Keys.TAB,
    "Escape": Keys.ESCAPE,
    "ArrowUp": Keys.ARROW_UP,
    "ArrowDown": Keys.ARROW_DOWN,
    "ArrowLeft": Keys.ARROW_LEFT,
    "ArrowRight": Keys.ARROW_RIGHT,
    "Home": Keys.HOME,
    "End": Keys.END,
    "PageUp": Keys.PAGE_UP,
    "PageDown": Keys.PAGE_DOWN,
}


def _switch(drv, handle: str) -> None:
    """Focus a tab by id, translating Selenium's missing-window error.

    Selenium's NoSuchWindowException stringifies to a multi-line driver stack
    trace; TabNotFoundError carries a clean, actionable message instead.
    """
    try:
        drv.switch_to.window(handle)
    except NoSuchWindowException:
        raise TabNotFoundError(f"tab {handle!r} is not open") from None


def _headless_enabled() -> bool:
    """Whether to launch Chrome headless, controlled by ``BROWDEN_HEADLESS``.

    A real human-facing session wants a visible window, so this defaults to off.
    Set ``BROWDEN_HEADLESS=1`` (or true/yes/on) for environments without a
    display — e2e tests and CI runners.
    """
    return os.environ.get("BROWDEN_HEADLESS", "").strip().lower() in ("1", "true", "yes", "on")


def _no_sandbox_enabled() -> bool:
    """Whether to disable Chrome's setuid/namespace sandbox — opt-in only.

    ``--no-sandbox`` removes a layer of renderer isolation, exactly the layer
    that matters most on the hosts browden is likely to drive at hostile web
    content (CI runners, containers, servers). So we default to keeping the
    sandbox ON and only drop it when the operator explicitly sets
    ``BROWDEN_NO_SANDBOX=1`` (or true/yes/on) — the escape hatch for environments
    where the sandbox cannot start, e.g. an unprivileged container or a runner
    with user namespaces disabled, where Chrome would otherwise refuse to launch.
    """
    return os.environ.get("BROWDEN_NO_SANDBOX", "").strip().lower() in ("1", "true", "yes", "on")


SINGLETON_FILES = ("SingletonLock", "SingletonCookie", "SingletonSocket")
TITLE_WAIT_SECONDS = 3
# Cold Chrome starts can take several seconds; a first launch after a Chrome
# update can take much longer (profile/cookie-DB migration, macOS Keychain
# "Chrome Safe Storage" prompt). Give the DevTools endpoint minutes, not
# seconds — timing out and killing Chrome mid-startup can wipe the cookie
# store. See ChromeLaunchError.
DEVTOOLS_WAIT_SECONDS = 300


class ChromeLaunchError(RuntimeError):
    """Chrome could not be started or attached without risking the profile.

    Raised instead of killing a Chrome that is still starting, clearing
    ``Singleton*`` files a live process may hold, or launching a second Chrome
    on the same ``--user-data-dir``. Those retries are what wipe cookies (the
    profile's SQLite cookie DB is not safe for two processes, and Chromium's
    ProcessSingleton will SIGKILL a peer it cannot talk to on the singleton
    socket). A ``RuntimeError`` subclass so MCP tools that already fold
    session-level RuntimeErrors into an error envelope keep doing so.

    ``proc`` is set when *we* spawned that Chrome (so the backend can remember
    it and refuse a second spawn). ``pid`` is the process that holds the
    profile, whether we spawned it or not.
    """

    def __init__(self, message: str, *, proc: subprocess.Popen | None = None,
                 pid: int | None = None):
        super().__init__(message)
        self.proc = proc
        self.pid = pid if pid is not None else (proc.pid if proc is not None else None)

    @property
    def envelope(self) -> dict:
        return {"error": str(self)}


CHROME_BINARY_NAMES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
)


def _wellknown_chrome_paths(platform: str = sys.platform, os_name: str = os.name) -> list[str]:
    """OS-specific install locations to try when Chrome isn't on PATH.

    On macOS and Windows, Chrome installs to a fixed application directory that
    is *not* on ``PATH`` and whose executable isn't named ``google-chrome`` — so
    ``shutil.which`` never finds it. Check those canonical spots before giving
    up. Linux keeps to PATH (packaged installs land there), so this is empty.

    ``platform``/``os_name`` are injectable so the per-OS branches are testable
    from any host without perturbing ``os.name`` (which flips pathlib's flavour).
    """
    home = Path.home()
    if platform == "darwin":
        return [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            str(home / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    if os_name == "nt":
        bases = [
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local")),
        ]
        return [str(Path(b) / "Google" / "Chrome" / "Application" / "chrome.exe")
                for b in bases if b]
    return []


def _find_chrome_binary() -> str:
    """Locate the Chrome/Chromium executable to launch directly.

    Honours ``BROWDEN_CHROME_BINARY`` (or the common ``CHROME_BIN``) first, then
    the usual binary names on PATH, then each platform's canonical install
    location (macOS ``/Applications``, Windows ``Program Files``) — so a stock
    Chrome install works without setting anything.
    """
    explicit = os.environ.get("BROWDEN_CHROME_BINARY") or os.environ.get("CHROME_BIN")
    if explicit:
        return explicit
    for name in CHROME_BINARY_NAMES:
        found = shutil.which(name)
        if found:
            return found
    for path in _wellknown_chrome_paths():
        if Path(path).exists():
            return path
    raise RuntimeError(
        "Could not find a Chrome/Chromium binary on PATH or in the usual install "
        "locations; set BROWDEN_CHROME_BINARY to its full path."
    )


def _free_port() -> int:
    """Reserve an ephemeral localhost port for Chrome's DevTools endpoint."""
    return get_free_port()


def _chrome_args(profile_dir: Path, port: int) -> list[str]:
    """Build the command line for a human-looking Chrome.

    Deliberately omits every automation switch chromedriver would otherwise
    inject when *it* launches Chrome (``--enable-automation``, the
    ``AutomationControlled`` blink feature, ``--test-type``, …). Because we
    start Chrome ourselves and Selenium only attaches over the DevTools port,
    ``navigator.webdriver`` stays false and there is no "controlled by
    automated test software" infobar — the window passes for an ordinary,
    human-driven browser. Only benign flags a real launcher commonly sets are
    included here.
    """
    args = [
        _find_chrome_binary(),
        f"--user-data-dir={profile_dir}",
        f"--remote-debugging-port={port}",
        # Chrome >= 111 rejects DevTools websocket connections from a foreign
        # origin. Scope allowed origins to the exact loopback origin Selenium
        # connects from (debuggerAddress is 127.0.0.1:<port>) rather than "*":
        # a wildcard lets any web page that reaches this ephemeral loopback
        # port — e.g. a malicious page scanning loopback ports, whose Origin
        # header the browser sets and the page cannot forge — open a DevTools
        # websocket and take full CDP control (read every cookie, run JS in any
        # origin), bypassing every browden gate. Pinning the origin keeps
        # Selenium working while shutting that out. (A hostile *native* local
        # process can spoof any Origin header, so this flag is no defence
        # there — local processes are trusted by design, see SECURITY.md.)
        f"--remote-allow-origins=http://127.0.0.1:{port}",
        "--no-first-run",
        "--no-default-browser-check",
        # Guarantee a WebGL context even on GPU-less hosts (VMs, containers, our
        # Hyper-V deployment box: hyperv_drm exposes no DRM render node, so
        # Chrome's ANGLE/EGL backend can't initialise hardware GL). Chrome >= 121
        # no longer falls back to its bundled SwiftShader for WebGL by default,
        # so getContext('webgl') returns null — a browser advertising *zero*
        # WebGL support is a decisive bot signal for Akamai & friends, which
        # then 403 the site's order-data XHRs (see issue #29). Re-enabling the
        # software fallback yields a real context (ANGLE/SwiftShader renderer)
        # and clears that flag. On a host with a real GPU this is inert — Chrome
        # keeps using hardware GL and only reaches for SwiftShader as a fallback.
        "--enable-unsafe-swiftshader",
    ]
    if _headless_enabled():
        # New headless mode + the flags a headless container commonly needs.
        logger.info("Launching Chrome headless")
        args += [
            "--headless=new",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--window-size=1280,1024",
        ]
    if _no_sandbox_enabled():
        # Opt-in only (BROWDEN_NO_SANDBOX): dropping the sandbox weakens renderer
        # isolation, so we never do it implicitly — not even headless. Set it on
        # hosts where the sandbox can't start (unprivileged containers, runners
        # with user namespaces disabled).
        logger.warning("Launching Chrome with --no-sandbox (renderer sandbox disabled)")
        args.append("--no-sandbox")
    return args


def _wait_for_devtools(port: int, proc: subprocess.Popen, timeout: float = DEVTOOLS_WAIT_SECONDS) -> None:
    """Block until Chrome's DevTools HTTP endpoint answers, or fail loudly.

    Bails early (rather than waiting out the timeout) if the Chrome process
    exits before the port comes up. Does not kill Chrome — the caller decides,
    and a timeout must *not* terminate a process that may still be starting.
    """
    url = f"http://127.0.0.1:{port}/json/version"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Chrome exited (code {proc.returncode}) before its DevTools "
                f"endpoint came up on port {port}."
            )
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.1)
    raise RuntimeError(
        f"Chrome DevTools endpoint not ready on port {port} after {timeout}s "
        f"(PID {proc.pid} is still running)."
    )


def _terminate(proc: subprocess.Popen | None) -> None:
    """Stop a Chrome subprocess, escalating to kill if it lingers.

    Only for an explicit shutdown of a Chrome we launched and already attached
    to — never for a startup timeout. SIGTERM lets Chrome flush the cookie DB;
    SIGKILL is the last resort after that flush has had time to land.
    """
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception as e:
        logger.warning(f"Failed to terminate Chrome process: {e}")


def _pid_is_alive(pid: int) -> bool:
    """True if ``pid`` currently names a process.

    ``os.kill(pid, 0)`` is the POSIX existence probe: ESRCH means gone,
    EPERM means it exists but we can't signal it (still live). Any other
    error is treated as live — a false "dead" here is how a second Chrome
    gets launched on a profile that is still held.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _pid_from_singleton_lock_target(target: str) -> int | None:
    """Parse the PID out of Chromium's ``<hostname>-<pid>`` symlink target.

    Hostname may itself contain hyphens, so this splits on the last ``-``,
    matching ``ParseProcessSingletonLock`` in
    ``chrome/common/process_singleton_lock_posix.cc``.
    """
    _, sep, pid_str = target.rpartition("-")
    if not sep or not pid_str.isdigit():
        return None
    pid = int(pid_str)
    return pid if pid > 0 else None


def _singleton_lock_pid(profile_dir: Path) -> int | None:
    """PID encoded in Chrome's POSIX ``SingletonLock`` symlink, if any.

    The lock is a symlink whose *target is not a real path* (e.g.
    ``example.com-9156``), so ``Path.exists()`` follows it and returns False
    even while the lock is present. Always ``lstat`` / ``readlink``.
    Non-symlink lock files (Windows, or a leftover regular file) have no PID
    we can parse and return None.
    """
    lock = profile_dir / "SingletonLock"
    try:
        if not lock.is_symlink():
            return None
        return _pid_from_singleton_lock_target(os.readlink(lock))
    except OSError:
        return None


def _unparseable_posix_lock(profile_dir: Path) -> bool:
    """True when a POSIX SingletonLock symlink exists but names no PID.

    We cannot prove that lock is stale, so callers must not clear it or
    launch another Chrome on top.
    """
    lock = profile_dir / "SingletonLock"
    try:
        if not lock.is_symlink():
            return False
        return _pid_from_singleton_lock_target(os.readlink(lock)) is None
    except OSError:
        return False


def _profile_holder_pid(profile_dir: Path) -> int | None:
    """Live PID that holds this profile, or None if the lock looks free.

    "Free" means there is no POSIX lock, or it names a PID that has exited.
    An unparseable symlink is *not* free — see ``_unparseable_posix_lock``.
    """
    pid = _singleton_lock_pid(profile_dir)
    if pid is not None and _pid_is_alive(pid):
        return pid
    return None


def _refuse_held_profile(profile_dir: Path) -> None:
    """Raise ``ChromeLaunchError`` if a live (or unproven-dead) lock holds the profile."""
    holder = _profile_holder_pid(profile_dir)
    if holder is not None:
        raise ChromeLaunchError(
            f"Chrome PID {holder} already holds this profile; not launching a "
            f"second instance (that can wipe cookies). Leave that process "
            f"alone, or quit PID {holder} yourself if it is stuck.",
            pid=holder,
        )
    if _unparseable_posix_lock(profile_dir):
        lock = profile_dir / "SingletonLock"
        raise ChromeLaunchError(
            f"A SingletonLock exists at {lock} but its PID could not be "
            f"verified as dead; not clearing it or launching another Chrome "
            f"on this profile.",
            pid=None,
        )


def _lexists(path: Path) -> bool:
    """True if ``path`` exists *or* is a dangling symlink (``exists()`` is False)."""
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return False
    return True


def _launch_chrome(profile_dir: Path) -> tuple[subprocess.Popen, int]:
    """Launch Chrome directly (no chromedriver) with remote debugging on.

    Returns the running process and the DevTools port to attach Selenium to.
    Refuses if this profile is already held. On a DevTools timeout, leaves
    the new Chrome running and raises ``ChromeLaunchError`` carrying that
    process — the caller must remember it so a later call does not spawn
    another Chrome on top of it.
    """
    _refuse_held_profile(profile_dir)
    _clear_stale_singletons(profile_dir)
    port = _free_port()
    args = _chrome_args(profile_dir, port)
    logger.info(f"Launching Chrome directly: {' '.join(args)}")
    # New session so a signal aimed at the MCP server does not also kill
    # Chrome (a SIGHUP mid-write is another way to lose the cookie DB).
    proc = subprocess.Popen(
        args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        _wait_for_devtools(port, proc)
    except Exception as e:
        if proc.poll() is None:
            raise ChromeLaunchError(
                f"Chrome DevTools endpoint not ready on port {port} after "
                f"{DEVTOOLS_WAIT_SECONDS:g}s (PID {proc.pid} is still running). "
                f"Not killing it and not launching another Chrome on this "
                f"profile — a mid-startup kill can destroy cookies. Wait for "
                f"that process to finish, or quit PID {proc.pid} yourself if "
                f"it is stuck.",
                proc=proc,
            ) from e
        raise
    return proc, port


def _wait_for_title(drv, timeout: float = TITLE_WAIT_SECONDS) -> None:
    """Wait briefly for the page title to populate after navigation.

    drv.get() returns when the load event fires, but many tabs set their
    final title via JavaScript after that. Best-effort: don't raise if the
    title never appears.
    """
    try:
        WebDriverWait(drv, timeout).until(lambda d: bool(d.title))
    except Exception:
        pass


def _clear_stale_singletons(profile_dir: Path) -> None:
    """Remove Singleton* files left over from an unclean shutdown.

    Chrome creates these when it starts and removes them on clean exit.
    After a power-off, kill -9, or container restart they point at dead
    PIDs and block the next launch.

    Must not run while a live process may hold the profile: unlinking
    ``SingletonSocket`` while Chrome is still starting is exactly how a
    second Chrome decides the first is hung and SIGKILLs it (see
    ``chrome/browser/process_singleton_posix.cc``). ``Path.exists()`` also
    cannot see the POSIX ``SingletonLock`` symlink (its target is not a
    real path), so this uses ``lstat`` and refuses rather than clearing
    when the lock names a live PID.
    """
    _refuse_held_profile(profile_dir)
    for name in SINGLETON_FILES:
        path = profile_dir / name
        try:
            if _lexists(path):
                logger.info(f"Clearing stale singleton: {path}")
                path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Failed to clear stale singleton {path}: {e}")


class SeleniumChromeBackend(WebNavigatorBackend):
    """Selenium WebDriver implementation of the navigator backend.

    Each instance owns exactly one Chrome session bound to one profile
    directory (``--user-data-dir``). ``profile_dir`` is required — the caller
    picks the path (the MCP server resolves the shared default and hands it in;
    the backend never falls back to a default of its own); distinct profiles
    don't share the per-dir ``SingletonLock``, so two backends on two profiles
    run concurrently without clobbering each other's window focus.

    Tab handles are raw Selenium window handles: the backend has no notion of the
    server's composite ``<profile>-<handle>`` id (see ``WebNavigatorBackend``).
    """

    def __init__(self, profile_dir):
        # profile_dir is required and immutable: it binds this backend to one
        # Chrome --user-data-dir (the caller resolves the shared default path;
        # the backend never falls back to a module default). Exposed read-only
        # via get_profile_dir().
        if not profile_dir:
            raise ValueError("profile_dir is required")
        self._driver = None
        self._chrome_proc = None
        self._profile_dir = Path(profile_dir).expanduser()

    def get_profile_dir(self) -> Path:
        return self._profile_dir

    def shutdown(self) -> None:
        """Public teardown for orderly shutdown (see ``WebNavigatorBackend.shutdown``).

        This is the only path that stops a Chrome we launched. A failed
        liveness probe or a slow start must not reach here — killing Chrome
        mid-startup is how the cookie store gets destroyed.
        """
        self._teardown()

    def _detach_driver(self) -> None:
        """Drop the WebDriver attachment without touching the Chrome process.

        Selenium only *attached* over the DevTools port, so ``driver.quit()``
        detaches without closing the browser. A failed probe is not proof the
        browser (and its profile) is safe to kill.
        """
        if self._driver is not None:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None

    def _teardown(self) -> None:
        """Drop the WebDriver session and stop the Chrome we launched.

        Reserved for explicit ``shutdown``. Selenium only *attached* to Chrome
        over the DevTools port, so ``driver.quit()`` detaches without closing
        the browser — we own the process and must terminate it ourselves on
        the way out, or it (and its SingletonLock) outlives the MCP server.
        """
        self._detach_driver()
        _terminate(self._chrome_proc)
        self._chrome_proc = None

    def _session_alive(self, check_current: bool) -> bool:
        """Whether the attached driver is live; detaches (does not kill Chrome) on failure.

        The single liveness probe both ``is_running`` and ``_drv`` share. Always
        checks ``window_handles`` (the session is up); with ``check_current`` also
        checks ``current_window_handle`` (the focused-window context later commands
        need). A "no such window" here is recovered by focusing a remaining
        handle rather than treating the whole browser as dead. Any other failed
        probe drops the driver attachment only — never the Chrome process — and
        returns False, so callers decide what next: ``is_running`` reports it,
        ``_drv`` launches only if nothing still holds the profile.
        """
        if self._driver is None:
            return False
        try:
            handles = self._driver.window_handles
            if check_current:
                try:
                    _ = self._driver.current_window_handle
                except NoSuchWindowException:
                    if handles:
                        self._driver.switch_to.window(handles[0])
            return True
        except Exception:
            self._detach_driver()
            return False

    def is_running(self) -> bool:
        """True if a live Chrome is currently attached. Probes without launching.

        The observing counterpart to ``_drv()``'s heal-by-relaunch: callers that
        only want to look (list_tabs across profiles) must not spawn a browser
        as a side effect. Only the session-level probe — a valid *current window*
        is ``_drv``'s concern (it can heal one), not this look-only check.
        A failed probe never kills Chrome.
        """
        return self._session_alive(False)

    def _chrome_proc_running(self) -> bool:
        return self._chrome_proc is not None and self._chrome_proc.poll() is None

    def _drv(self):
        # Health-check the attached driver (session AND current-window context).
        # A failed probe has already detached the driver, not killed Chrome.
        if self._session_alive(True):
            return self._driver
        # A Chrome we launched (or one whose SingletonLock is still live) must
        # not be stacked with a second instance — that is the cookie-wipe path.
        if self._chrome_proc_running():
            raise ChromeLaunchError(
                f"Chrome PID {self._chrome_proc.pid} is still running on this "
                f"profile but is not responding over DevTools. Not launching "
                f"another Chrome (that can wipe cookies). Wait, or quit PID "
                f"{self._chrome_proc.pid} yourself if it is stuck.",
                proc=self._chrome_proc,
            )
        _refuse_held_profile(self._profile_dir)
        profile = self._profile_dir
        logger.info(f"Starting new Chrome session (profile={profile})")
        try:
            self._chrome_proc, port = _launch_chrome(profile)
        except ChromeLaunchError as e:
            # Remember a Chrome we spawned but could not attach to, so the
            # next call sees it via _chrome_proc_running and does not spawn
            # another.
            if e.proc is not None:
                self._chrome_proc = e.proc
            raise
        # Attach to the Chrome we just launched instead of letting
        # chromedriver spawn its own (automation-flagged) instance.
        opts = ChromeOptions()
        opts.add_experimental_option("debuggerAddress", f"127.0.0.1:{port}")
        try:
            self._driver = webdriver.Chrome(options=opts)
        except Exception as e:
            self._detach_driver()
            pid = self._chrome_proc.pid if self._chrome_proc is not None else None
            raise ChromeLaunchError(
                f"Chrome PID {pid} is running but Selenium could not attach "
                f"to DevTools on port {port}. Not killing it and not launching "
                f"another Chrome on this profile.",
                proc=self._chrome_proc,
            ) from e
        return self._driver

    def _tabinfo(self, drv, *, selected: bool) -> TabInfo:
        """Snapshot the driver's *currently focused* window as a TabInfo.

        The four TabInfo constructions (list_tabs / new_blank_tab / navigate /
        reload) all read the same fields off the focused window and only differ in
        ``selected`` — so callers switch to the window they mean, then ask here.
        """
        return TabInfo(
            handle=drv.current_window_handle,
            url=drv.current_url,
            title=drv.title,
            selected=selected,
            profile_dir=str(self._profile_dir),
        )

    @staticmethod
    def _resolve_one_visible(drv, css_selector: str):
        """Re-find a policy-validated selector live and return its single actionable element.

        The identical resolve-and-vet block ``click_element`` and
        ``insert_text_element`` both run. Ambiguity is a deny: the policy layer
        validated exactly one element on the snapshot, so anything other than one
        live match — or a match that is hidden or disabled — means the DOM moved
        under us, and the caller must not act on it.
        """
        matches = drv.find_elements(By.CSS_SELECTOR, css_selector)
        if len(matches) == 0:
            raise NoSuchElementException(f"no element matches {css_selector!r}")
        if len(matches) > 1:
            raise ValueError(f"selector {css_selector!r} matched {len(matches)} live elements")
        el = matches[0]
        if not el.is_displayed():
            raise ValueError("target element is not visible")
        if not el.is_enabled():
            raise ValueError("target element is disabled")
        return el

    def list_tabs(self) -> list[TabInfo]:
        drv = self._drv()
        current_handle = drv.current_window_handle
        tabs = []
        for handle in drv.window_handles:
            drv.switch_to.window(handle)
            tabs.append(self._tabinfo(drv, selected=(handle == current_handle)))
        drv.switch_to.window(current_handle)
        return tabs

    def list_handles(self) -> list[str]:
        return list(self._drv().window_handles)

    def new_blank_tab(self) -> TabInfo:
        drv = self._drv()
        drv.switch_to.new_window("tab")
        return self._tabinfo(drv, selected=True)

    def close_tab(self, handle: str) -> None:
        drv = self._drv()
        if len(drv.window_handles) == 1:
            raise ValueError("Cannot close the last tab")
        _switch(drv, handle)
        drv.close()
        # drv.close() leaves the driver focused on the now-dead handle. The next
        # command — or _drv()'s health check, which reads current_window_handle —
        # would then mistake the focused window for gone. Re-focus a survivor so
        # the session stays valid (and so we never treat a missing window as
        # "launch another Chrome").
        remaining = drv.window_handles
        if remaining:
            drv.switch_to.window(remaining[0])

    def select_tab(self, handle: str) -> None:
        _switch(self._drv(), handle)

    def navigate(self, url: str) -> TabInfo:
        drv = self._drv()
        try:
            drv.get(url)
        except NoSuchWindowException:
            raise TabNotFoundError("there is no active tab to navigate") from None
        _wait_for_title(drv)
        return self._tabinfo(drv, selected=True)

    def get_tab_html(self) -> str:
        # Operates on the focused tab; the caller select_tab's first (uniform
        # tab-targeting contract — no method self-focuses from a handle).
        return self._drv().page_source

    def reload(self) -> TabInfo:
        drv = self._drv()
        drv.refresh()
        _wait_for_title(drv)
        return self._tabinfo(drv, selected=True)

    def current_url(self) -> str:
        try:
            return self._drv().current_url
        except NoSuchWindowException:
            raise TabNotFoundError("there is no active tab") from None

    def document_url(self) -> str:
        """URL of the *focused document* (``document.URL``), for policy gating.

        Unlike ``current_url`` — the top-level browsing context's URL, i.e. what the
        address bar shows — this reflects whichever document the driver is currently
        focused on. So once focus is inside an iframe, the read/write gates see the
        frame's own URL rather than the top page's; when focus is at the top the two
        are identical.

        Falls back to ``current_url`` when script can't run in the focused document
        (e.g. a ``chrome://`` internal page disallows ``execute_script``) — those URLs
        are handled by the read policy's special cases anyway.
        """
        drv = self._drv()
        try:
            return drv.execute_script("return document.URL")
        except NoSuchWindowException:
            raise TabNotFoundError("there is no active tab") from None
        except Exception:
            return self.current_url()

    def screenshot(self) -> bytes:
        try:
            return self._drv().get_screenshot_as_png()
        except NoSuchWindowException:
            raise TabNotFoundError("there is no active tab to screenshot") from None

    def click_element(self, css_selector: str) -> dict:
        drv = self._drv()
        try:
            url_before = drv.current_url
            el = self._resolve_one_visible(drv, css_selector)
        except NoSuchWindowException:
            raise TabNotFoundError("there is no active tab to click in") from None
        el.click()
        _wait_for_title(drv)
        return {
            "clicked": True,
            "url_before": url_before,
            "url": drv.current_url,
            "title": drv.title,
        }

    def insert_text_element(self, css_selector: str, value: str) -> dict:
        drv = self._drv()
        try:
            el = self._resolve_one_visible(drv, css_selector)
        except NoSuchWindowException:
            raise TabNotFoundError("there is no active tab to insert text into") from None
        try:
            el.clear()
        except InvalidElementStateException:
            # contenteditable / rich fields don't support clear(); select-all + delete.
            el.send_keys(Keys.CONTROL, "a")
            el.send_keys(Keys.DELETE)
        el.send_keys(value)
        return {
            "inserted": True,
            "value": value,
            "url": drv.current_url,
            "title": drv.title,
        }

    def press_key_element(self, css_selector: str, key: str) -> dict:
        drv = self._drv()
        try:
            url_before = drv.current_url
            el = self._resolve_one_visible(drv, css_selector)
        except NoSuchWindowException:
            raise TabNotFoundError("there is no active tab to press a key in") from None
        selenium_key = _SELENIUM_KEYS.get(key)
        if selenium_key is None:
            # The gate only lets control keys through; anything else is a bug.
            raise ValueError(f"unsupported key {key!r}")
        # Focus by reference (not a coordinate click, so no hit-testing — an
        # overlay can't intercept it), then dispatch the key to the now-active
        # element. send_keys targets `el`, which the explicit focus() guarantees is
        # focused even for non-input controls (a <tr>/<div> a11y widget).
        drv.execute_script("arguments[0].focus();", el)
        el.send_keys(selenium_key)
        _wait_for_title(drv)
        return {
            "pressed": True,
            "key": key,
            "url_before": url_before,
            "url": drv.current_url,
            "title": drv.title,
        }
