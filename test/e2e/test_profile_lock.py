"""End-to-end: a second backend must not launch Chrome on an occupied profile.

The cookie-wipe incident was a second ``--user-data-dir`` start on top of a
Chrome that was still coming up. This covers the held-profile refusal with a
real Chrome: the first backend starts, the second raises ``ChromeLaunchError``,
and the first session is still alive afterwards.
"""
import os

import pytest

from browden.web_navigator.selenium_chrome import ChromeLaunchError


def test_second_backend_refuses_occupied_profile_and_leaves_the_first_alone(
        new_backend, tmp_path):
    profile = tmp_path / "profile"
    first = new_backend(profile)
    tab = first.new_blank_tab()
    assert tab.handle

    lock = profile / "SingletonLock"
    assert lock.is_symlink(), "Chrome should have taken the POSIX profile lock"
    holder = os.readlink(lock)

    second = new_backend(profile)
    with pytest.raises(ChromeLaunchError, match="already holds this profile"):
        second.new_blank_tab()

    # First Chrome was not killed to make room for the second.
    assert first.is_running()
    assert tab.handle in {t.handle for t in first.list_tabs()}
    assert lock.is_symlink()
    assert os.readlink(lock) == holder
    assert second._chrome_proc is None  # never spawned
