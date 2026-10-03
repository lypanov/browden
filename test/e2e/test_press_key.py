"""End-to-end: the press-key primitive against a real (headless) Chrome.

Goes through ``BrowserSessionManager.press_key`` -> ``backend.press_key_element`` —
the same path the ``press_key`` MCP tool takes after gating — so it exercises the
live chain: find one element -> visible/enabled -> JS ``focus()`` + ``send_keys``
of a control key (or, with no selector, ``send_keys`` on the current focus). The
page is a keyboard-operable list of ``<li tabindex>`` rows (the shape a
coordinate ``click`` can't reach): each row's ``keydown`` handler echoes what
happened into ``#picked``, so re-querying that echo (the cache is invalidated
by the press) proves the key actually reached the focused element. A second
inline page of three ``<input>``s proves ``Shift+Tab`` is a real shift-modified
Tab that walks focus backwards. Both pages are ``data:`` documents, so the run
is offline.
"""
import urllib.parse

import pytest

from browden.mcp.session_management.browser_session_manager import BrowserSessionManager

HTML = """<html><body>
  <div id="picked"></div>
  <ul id="list">
    <li id="r1" class="row" tabindex="0">Egg</li>
    <li id="r2" class="row" tabindex="-1">Avocado</li>
    <li id="r3" class="row" tabindex="-1">Quinoa</li>
  </ul>
  <script>
    const picked = document.getElementById('picked');
    document.querySelectorAll('.row').forEach(row => {
      row.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
          picked.textContent = 'picked:' + row.textContent;
        } else if (e.key === 'ArrowDown' && row.nextElementSibling) {
          row.nextElementSibling.focus();
          picked.textContent = 'focus:' + row.nextElementSibling.textContent;
        }
      });
    });
  </script>
</body></html>"""

DATA_URL = "data:text/html," + urllib.parse.quote(HTML)

# Three inputs in document order. A keydown Tab (with or without Shift) is
# recorded, then focusin writes `from->to` so the test can see both that Chrome
# got a shift-modified Tab and that focus walked backwards.
TAB_HTML = """<html><body>
  <div id="picked"></div>
  <input id="a" />
  <input id="b" />
  <input id="c" />
  <script>
    const picked = document.getElementById('picked');
    let lastTab = null;
    document.addEventListener('keydown', e => {
      if (e.key === 'Tab') {
        lastTab = {shift: e.shiftKey, from: e.target.id};
      }
    });
    document.addEventListener('focusin', e => {
      if (lastTab) {
        picked.textContent = (lastTab.shift ? 'shift-tab' : 'tab')
          + ':' + lastTab.from + '->' + e.target.id;
        lastTab = null;
      }
    });
  </script>
</body></html>"""

TAB_URL = "data:text/html," + urllib.parse.quote(TAB_HTML)


@pytest.fixture
def session(new_backend, tmp_path):
    backend = new_backend(tmp_path / "profile")
    return BrowserSessionManager(backend, namespace="e2e", start_reaper=False)


@pytest.mark.asyncio
async def test_enter_activates_focusable_row(session):
    blank = await session.new_blank_tab(max_tabs=10)
    page = await session.navigate(DATA_URL, id=blank["id"])

    # A <li tabindex="0"> is not a <button>/<a>, so click can't reach it — but
    # press-key focuses it and Enter fires its keydown handler.
    res = await session.press_key("#r1", "Enter", id=page["id"])
    assert res["pressed"] is True
    assert res["key"] == "Enter"
    assert res["id"] == page["id"]

    echo = await session.query_selector("#picked", id=page["id"])
    assert echo["found"] is True
    assert echo["element"]["text"] == "picked:Egg"


@pytest.mark.asyncio
async def test_arrow_key_moves_selection(session):
    blank = await session.new_blank_tab(max_tabs=10)
    page = await session.navigate(DATA_URL, id=blank["id"])

    # ArrowDown on the first row moves focus to the next (roving tabindex) — proves
    # a navigation key is delivered to the focused element, not just Enter.
    await session.press_key("#r1", "ArrowDown", id=page["id"])
    echo = await session.query_selector("#picked", id=page["id"])
    assert echo["element"]["text"] == "focus:Avocado"


@pytest.mark.asyncio
async def test_unsupported_key_refused(session):
    blank = await session.new_blank_tab(max_tabs=10)
    page = await session.navigate(DATA_URL, id=blank["id"])

    # The backend only maps control keys; a character key never reaches Chrome here
    # (the MCP gate refuses it earlier too, but the primitive is defence in depth).
    with pytest.raises(ValueError, match="unsupported key"):
        await session.press_key("#r1", "a", id=page["id"])


@pytest.mark.asyncio
async def test_ambiguous_selector_refused(session):
    blank = await session.new_blank_tab(max_tabs=10)
    page = await session.navigate(DATA_URL, id=blank["id"])

    # Three .row elements match — the backend refuses rather than press a key on an
    # arbitrary one (the DOM moved under a snapshot that had validated one match).
    with pytest.raises(ValueError, match="matched 3 live elements"):
        await session.press_key(".row", "Enter", id=page["id"])


@pytest.mark.asyncio
async def test_shift_tab_is_a_real_shift_modified_tab(session):
    blank = await session.new_blank_tab(max_tabs=10)
    page = await session.navigate(TAB_URL, id=blank["id"])

    # Focus #c then Shift+Tab: Chrome must report key=Tab + shiftKey, and focus
    # must land on #b (reverse tab order), not #a or stay on #c.
    res = await session.press_key("#c", "Shift+Tab", id=page["id"])
    assert res["pressed"] is True
    assert res["key"] == "Shift+Tab"

    echo = await session.query_selector("#picked", id=page["id"])
    assert echo["element"]["text"] == "shift-tab:c->b"


@pytest.mark.asyncio
async def test_shift_tab_without_selector_uses_current_focus(session):
    blank = await session.new_blank_tab(max_tabs=10)
    page = await session.navigate(TAB_URL, id=blank["id"])

    # First press focuses #c and walks back to #b. The second omits the
    # selector — retargeting #c would just repeat the first step — so this is
    # the chatgpt.com "Shift+Tab twice" chain.
    await session.press_key("#c", "Shift+Tab", id=page["id"])
    res = await session.press_key(None, "Shift+Tab", id=page["id"])
    assert res["pressed"] is True
    assert res["key"] == "Shift+Tab"

    echo = await session.query_selector("#picked", id=page["id"])
    assert echo["element"]["text"] == "shift-tab:b->a"
