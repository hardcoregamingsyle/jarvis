"""What can be proven about a browser client with no browser in the room.

Nothing here executes JavaScript, so these tests deliberately avoid pretending
to. What they do assert are the properties that, if broken, turn the remote
client into something worse than useless — a page that looks like it works:

* it is one self-contained file, so the phone in the kitchen renders it even
  when the box has no route to the internet;
* nothing from the server can become markup;
* the microphone belongs to the *visitor's* device, and the checks that decide
  whether it is available happen in an order that can actually answer;
* the token screen exists exactly when the server says a token is needed.

The structural checks (well-formed markup, every DOM lookup resolving, the
script element never closing early) are the ones that catch a typo which a
human reviewer reads straight past.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple

import pytest

from jarvis.server.client import (
    CONTENT_TYPE,
    DEFAULT_FEATURES,
    DEFAULT_TITLE,
    ENDPOINTS,
    PAGE,
    index_html,
    render_page,
    render_page_bytes,
)

# The payload shape jarvis.server.app.status_payload() actually emits.
LAN_STATUS = {
    "ok": True,
    "name": "JARVIS",
    "version": "1.1.0",
    "authenticated": False,
    "auth_required": True,
    "tls": False,
    "model": "Qwen/Qwen3.6-27B",
    "llm": "airllm",
    "stt": "faster-whisper",
    "tts": "piper",
    "orchestrator": True,
    "stt_available": True,
}

# Elements that never take a closing tag; anything else must be balanced.
VOID_ELEMENTS = frozenset(
    {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }
)

# Ids that only exist on the token screen, and are therefore legitimately
# absent (and guarded in JavaScript) when no token is required.
GATE_IDS = frozenset({"gate", "gate-error", "gate-form", "gate-go", "gate-token"})

AUTH_FEATURES = {"auth_required": True, "stt": True, "tts": True}


class Structure(HTMLParser):
    """Tracks tag balance, ids and attributes so the page can be inspected."""

    def __init__(self) -> None:
        HTMLParser.__init__(self, convert_charrefs=True)
        self.stack: List[str] = []
        self.faults: List[str] = []
        self.ids: List[str] = []
        self.tags: List[str] = []
        self.linked: List[Tuple[str, str]] = []
        self.text: List[str] = []

    def _record(self, tag: str, attrs) -> None:
        self.tags.append(tag)
        for name, value in attrs:
            if name == "id" and value:
                self.ids.append(value)
            elif name in ("src", "href") and value:
                self.linked.append((tag, value))

    def handle_starttag(self, tag, attrs) -> None:
        self._record(tag, attrs)
        if tag not in VOID_ELEMENTS:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs) -> None:
        self._record(tag, attrs)

    def handle_endtag(self, tag) -> None:
        if tag in VOID_ELEMENTS:
            self.faults.append("closing tag for void element <%s>" % tag)
        elif not self.stack:
            self.faults.append("</%s> with nothing open" % tag)
        elif self.stack[-1] != tag:
            self.faults.append("</%s> closes <%s>" % (tag, self.stack[-1]))
        else:
            self.stack.pop()

    def handle_data(self, data) -> None:
        self.text.append(data)


def parse(page: str) -> Structure:
    parser = Structure()
    parser.feed(page)
    parser.close()
    return parser


def boot_record(page: str) -> Dict:
    """The JSON the server injected for the page to configure itself from."""
    match = re.search(
        r'<script id="jarvis-boot" type="application/json">(.*?)</script>',
        page,
        re.S,
    )
    assert match, "the page carries no boot record, so it cannot know what the server supports"
    return json.loads(match.group(1))


def script_body(page: str) -> str:
    """The client script only, so assertions do not trip over the markup."""
    scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
    assert scripts, "the page has no client script"
    return "\n".join(scripts)


@pytest.fixture
def plain() -> str:
    return render_page()


@pytest.fixture
def guarded() -> str:
    return render_page(title="JARVIS", has_tls=True, features=AUTH_FEATURES)


# --------------------------------------------------------------------------- #
#  Structure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "page_features",
    [None, {"auth_required": True}, dict(AUTH_FEATURES), {"stream": False, "tasks": False}],
    ids=["defaults", "auth", "auth-speech", "minimal"],
)
def test_every_variant_is_well_formed_html(page_features: Optional[dict]) -> None:
    """An unclosed tag renders as a subtly broken layout, not as an error."""
    page = render_page(features=page_features)
    parsed = parse(page)

    assert not parsed.faults, "malformed markup: %s" % parsed.faults
    assert not parsed.stack, "tags left open at end of document: %s" % parsed.stack
    for required in ("html", "head", "body", "title", "style", "script", "textarea"):
        assert required in parsed.tags, "the page has no <%s>" % required


def test_page_declares_a_doctype_and_utf8(plain: str) -> None:
    assert plain.lstrip().lower().startswith("<!doctype html>"), (
        "without a doctype the browser falls into quirks mode and the layout shifts"
    )
    assert '<meta charset="utf-8"' in plain
    assert CONTENT_TYPE == "text/html; charset=utf-8"


def test_ids_are_unique(guarded: str) -> None:
    """Two elements sharing an id means getElementById silently picks one."""
    parsed = parse(guarded)
    duplicates = sorted({i for i in parsed.ids if parsed.ids.count(i) > 1})
    assert not duplicates, "duplicate element ids: %s" % duplicates


def test_every_dom_lookup_resolves_to_a_real_element() -> None:
    """A mistyped id is a null dereference the moment the page loads."""
    page = render_page(features=AUTH_FEATURES)
    ids = set(re.findall(r'id="([A-Za-z0-9_-]+)"', page))
    looked_up = set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', page))
    looked_up |= set(re.findall(r'getElementById\("([A-Za-z0-9_-]+)"\)', page))

    assert len(looked_up) > 10, (
        "found only %d DOM lookups, so this test is not actually checking the "
        "page" % len(looked_up)
    )
    assert not looked_up - ids, "the script looks up ids that do not exist: %s" % sorted(
        looked_up - ids
    )


def test_without_auth_the_only_absent_elements_are_the_token_screen(plain: str) -> None:
    """Everything else must survive the token screen not being rendered."""
    ids = set(re.findall(r'id="([A-Za-z0-9_-]+)"', plain))
    looked_up = set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', plain))
    missing = looked_up - ids
    assert missing <= GATE_IDS, (
        "these lookups have no element and are not guarded gate elements: %s"
        % sorted(missing - GATE_IDS)
    )
    assert "if (gate)" in plain and "if (!gate)" in plain, (
        "the gate elements are absent without auth, so every use of them must be "
        "guarded by a null check"
    )


def test_the_script_element_cannot_be_closed_early(guarded: str) -> None:
    """A stray '</script' inside the JS truncates the whole client."""
    assert guarded.count("<script") == guarded.count("</script"), (
        "unbalanced script tags: the client script is being cut short"
    )
    assert "</script" not in script_body(guarded), (
        "the client script contains a literal closing tag and will be truncated"
    )
    assert "<!--" not in script_body(guarded), (
        "an HTML comment opener inside the script confuses older parsers"
    )


# --------------------------------------------------------------------------- #
#  Self-containment
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scheme", ["http://", "https://", "//cdn.", "//unpkg", "//cdnjs"])
def test_no_external_url_appears_anywhere(guarded: str, scheme: str) -> None:
    """The box may have no internet. A page that needs a CDN is a blank page."""
    assert scheme not in guarded, (
        "the page references %r; it must be entirely self-contained so it "
        "renders with the Linux box offline" % scheme
    )


def test_no_element_fetches_anything(guarded: str) -> None:
    parsed = parse(guarded)
    assert not parsed.linked, (
        "these elements load external resources: %s" % parsed.linked
    )
    assert "@import" not in guarded, "an @import in the CSS is an external fetch"
    assert "url(" not in guarded, "a url() in the CSS is an external fetch"
    assert "integrity=" not in guarded and "crossorigin" not in guarded, (
        "subresource attributes only make sense for third-party assets, which "
        "this page must not have"
    )


def test_css_and_js_are_inlined(guarded: str) -> None:
    assert "prefers-color-scheme" in guarded, "no dark/light handling in the stylesheet"
    body = script_body(guarded)
    assert len(body) > 4000, (
        "the client script is only %d characters, which is too small to contain "
        "the chat, speech and task logic" % len(body)
    )


def test_page_is_pure_ascii(guarded: str) -> None:
    """Keeps the byte count equal to the character count for any transport."""
    non_ascii = sorted({c for c in guarded if ord(c) > 127})
    assert not non_ascii, "non-ASCII characters in the page: %s" % non_ascii


def test_page_size_is_sane(guarded: str) -> None:
    size = len(render_page_bytes(title="JARVIS", has_tls=True, features=AUTH_FEATURES))
    assert size == len(guarded.encode("utf-8"))
    assert 12_000 < size < 120_000, (
        "the page is %d bytes; under ~12k it cannot hold the described features, "
        "over ~120k it is no longer a single hand-written file" % size
    )


# --------------------------------------------------------------------------- #
#  Nothing from the server becomes markup
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sink",
    ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function("],
)
def test_no_markup_injection_sink_is_used(guarded: str, sink: str) -> None:
    """A tool result containing HTML must render as characters, not elements."""
    assert sink not in guarded, (
        "%r appears in the page; server-provided text would become live markup "
        "the first time a reply contains a tag" % sink
    )


def test_server_text_reaches_the_dom_only_through_textcontent(guarded: str) -> None:
    body = script_body(guarded)
    assert body.count(".textContent") >= 5, (
        "expected every rendering path to assign .textContent; found only %d"
        % body.count(".textContent")
    )
    assert "node.textContent = String(text)" in body, (
        "the shared element factory must set text as text"
    )
    assert "createTextNode" in body, "the status line must be built from a text node"


@pytest.mark.parametrize("hostile", ["</script><img src=x onerror=alert(1)>", "<b>bold</b>", "a\"b'c&d"])
def test_a_hostile_title_cannot_inject_markup(hostile: str) -> None:
    """The title comes from config, and config is not trusted to be sane."""
    page = render_page(title=hostile, features={"model": hostile, "backend": hostile})

    parsed = parse(page)
    assert not parsed.faults and not parsed.stack, (
        "a hostile title broke the document structure: %s %s" % (parsed.faults, parsed.stack)
    )
    assert "<img" not in page, "an element was injected through the title"
    assert page.count("<script") == page.count("</script"), (
        "the title terminated the script element"
    )

    record = boot_record(page)
    assert record["title"] == hostile, "the title must survive escaping intact"
    assert record["features"]["model"] == hostile


def test_boot_record_contains_no_raw_angle_brackets() -> None:
    page = render_page(title="</script>", features={"model": "<b>"})
    raw = re.search(
        r'<script id="jarvis-boot" type="application/json">(.*?)</script>', page, re.S
    ).group(1)
    assert "<" not in raw and ">" not in raw, (
        "unescaped angle brackets in the boot JSON can close the script element"
    )


# --------------------------------------------------------------------------- #
#  The token screen
# --------------------------------------------------------------------------- #
def test_token_screen_is_absent_unless_auth_is_required(plain: str) -> None:
    assert '<div id="gate">' not in plain, (
        "a login screen was rendered although the server requires no token"
    )
    assert '<input id="gate-token"' not in plain
    assert "Access token" not in plain


def test_token_screen_is_rendered_when_auth_is_required(guarded: str) -> None:
    assert '<div id="gate">' in guarded, "no token screen although auth is required"
    assert 'id="gate-token"' in guarded and 'type="password"' in guarded, (
        "the token field must not echo the token on screen"
    )
    assert 'id="gate-form"' in guarded
    assert ENDPOINTS["login"] in guarded


def test_the_token_is_never_persisted_in_the_browser(guarded: str) -> None:
    """The session cookie the server sets is the only thing that survives."""
    for store in ("localStorage", "sessionStorage", "indexedDB", "document.cookie ="):
        assert store not in guarded, (
            "%r appears in the page; the access token must live only in the "
            "session cookie the server set" % store
        )
    body = script_body(guarded)
    assert body.count('field.value = ""') >= 2, (
        "the token field must be cleared on both the accepted and the rejected "
        "path, so it is never left sitting in the DOM"
    )
    accepted = body[body.find('if (!res.ok)'):body.find("}).catch(function () {")]
    assert accepted and 'field.value = ""' in accepted, (
        "the token is not cleared after a successful exchange"
    )


def test_the_login_form_cannot_put_the_token_in_a_url(guarded: str) -> None:
    """If the script never wired up the submit handler, the browser submits the
    form itself. A form with no method defaults to GET, which would write the
    access token into the address bar, the history and every proxy log."""
    form = re.search(r'<form id="gate-form"[^>]*>', guarded)
    assert form, "no login form"
    assert 'method="post"' in form.group(0).lower(), (
        "the token form falls back to a GET submit: %s" % form.group(0)
    )
    assert "ev.preventDefault();" in script_body(guarded), (
        "the scripted path must stop the native submit, or the page navigates "
        "away mid-login"
    )


def test_a_refused_request_with_no_token_screen_says_what_does_work(plain: str) -> None:
    """A session can expire after the page was rendered.

    The token screen is only in the markup when the server asked for it, so an
    authenticated visitor whose session lapses has no field to type into. The
    refusal must name the one action that still works rather than leaving an
    instruction that cannot be followed.
    """
    body = script_body(plain)
    lock = re.search(r"function lock\(message\) \{(.*?)\n  \}", body, re.S)
    assert lock, "no lock routine"
    source = lock.group(1)

    head = source[: source.index("if (!gate)")]
    assert "return" not in head, (
        "lock() returns before it decides anything, so a 401 is swallowed and "
        "the visitor is never told the session went away"
    )
    assert "reload" in source.lower(), (
        "with no token screen in this document there is nothing to type into; "
        "the message must say to reload"
    )
    assert "res.status === 401" in body and "lock(" in body, (
        "nothing routes a refused request into lock()"
    )


def test_requests_carry_the_session_cookie_and_stay_same_origin(guarded: str) -> None:
    body = script_body(guarded)
    assert 'options.credentials = "same-origin"' in body, (
        "the shared request helper must attach the session cookie to every call; "
        "without it each request 401s and the page loops back to the login screen"
    )
    assert 'credentials: "same-origin"' in body, (
        "the login POST bypasses the helper, so it must set credentials itself "
        "or the session cookie is never stored"
    )
    assert '"include"' not in body, "credentials: include implies a cross-origin design"
    assert "401" in body and "403" in body, (
        "a rejected session must be noticed and turned back into a login prompt"
    )


# --------------------------------------------------------------------------- #
#  Endpoints and transport
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,path", sorted(ENDPOINTS.items()))
def test_page_calls_the_documented_endpoint(guarded: str, name: str, path: str) -> None:
    assert path in guarded, "the page never references the %s endpoint %s" % (name, path)


def registered_stream_events(page: str) -> set:
    """The event names the client actually binds a listener to.

    Reading the array alone is not enough: a page that declares the names and
    never registers them is exactly as deaf as one that names them wrongly.
    """
    body = script_body(page)
    match = re.search(r"var STREAM_EVENTS = \[(.*?)\];", body, re.S)
    assert match, "the client keeps no list of stream events"
    assert "stream.addEventListener(name" in body, (
        "STREAM_EVENTS is declared but never registered on the EventSource, so "
        "every event the server sends lands on a listener that does not exist"
    )
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def test_streaming_uses_eventsource_with_a_post_fallback(guarded: str) -> None:
    body = script_body(guarded)
    assert "EventSource" in body, "no SSE client, so replies cannot stream"
    assert "new EventSource(API.stream)" in body
    assert 'typeof window.EventSource === "undefined"' in body, (
        "a browser without EventSource must fall through, not throw"
    )
    assert "postJson(API.chat" in body, (
        "the plain POST to /api/chat is the fallback when the stream is absent"
    )


#: Emitted by the server but deliberately not rendered: the handshake, the
#: teardown, this session's own echo, and the tool trace.
UNRENDERED_STREAM_EVENTS = frozenset(
    {"ready", "shutdown", "utterance", "tool_call", "tool_result"}
)


def test_the_client_listens_for_the_names_the_server_actually_puts_on_the_wire(
    guarded: str,
) -> None:
    """The bus channel is 'assistant.chunk'; the SSE layer renames it to 'chunk'.

    Listening for the channel name is silent forever, and silent in the worst
    way: the connection indicator still reads 'live'. Nothing but a comparison
    against the server's own table catches that, so this test imports it.
    """
    from jarvis.server.app import STREAM_CHANNELS

    listened = registered_stream_events(guarded)
    wire = {name for _channel, name in STREAM_CHANNELS}
    assert wire, "the server forwards no bus channels at all"

    missing = sorted((wire - UNRENDERED_STREAM_EVENTS) - listened)
    assert not missing, (
        "the server sends 'event: %s' and the client registers no listener for "
        "any of them, so replies and task updates are delivered nowhere"
        % "', 'event: ".join(missing)
    )


def test_every_listened_event_reaches_a_branch_of_the_dispatcher(guarded: str) -> None:
    """A listener that lands on no branch is the same silence one step later."""
    body = script_body(guarded)
    dispatcher = re.search(r"function dispatch\(name, raw\) \{(.*?)\n  \}", body, re.S)
    assert dispatcher, "no dispatch routine"
    branches = dispatcher.group(1)

    assert "canonical(name)" in branches, (
        "dispatch() compares the raw wire name against branch names, so the "
        "server's short names ('chunk', 'reply', 'task_update') fall through"
    )
    for branch in ('"assistant.chunk"', '"assistant.reply"', 'indexOf("task.")', '"error"'):
        assert branch in branches, "dispatch() has no %s branch" % branch

    rules = re.search(r"function canonical\(name\) \{(.*?)\n  \}", body, re.S)
    assert rules, "no canonicaliser to map a wire name onto a branch"
    for wire, canonical in (("chunk", "assistant.chunk"), ("reply", "assistant.reply")):
        assert '"%s"' % wire in rules.group(1) and '"%s"' % canonical in rules.group(1), (
            "'%s' is never mapped to '%s'" % (wire, canonical)
        )
    assert '"task_"' in rules.group(1), "the task_* family is never mapped to task.*"


def test_a_failed_request_reports_why_rather_than_no_reply(guarded: str) -> None:
    """finishPending() replaces an empty bubble, so the reason must be set as
    the pending text, not written straight onto the node it will overwrite."""
    body = script_body(guarded)
    start = body.find("postJson(API.chat")
    assert start != -1, "no chat POST to fail"
    failure = re.search(r"\}\)\.catch\(function \(err\) \{(.*?)\n    \}\);", body[start:], re.S)
    assert failure, "the chat POST has no failure handler"
    handler = failure.group(1)
    assert "setPending(" in handler, (
        "the failure text must go through setPending or finishPending replaces "
        "it with '(no reply)' and the visitor never learns what broke"
    )
    assert "pending.body.textContent" not in handler, (
        "writing the node directly is exactly the bug: finishPending overwrites it"
    )
    assert "access token" in handler, "a 401 must be distinguished from a crash"


def test_a_reply_nobody_asked_for_is_still_shown(guarded: str) -> None:
    """A reply spoken at the machine itself must appear in the remote session."""
    body = script_body(guarded)
    reply = re.search(r'if \(name === "assistant\.reply"\) \{(.*?)\n      return;', body, re.S)
    assert reply, "the client does not handle assistant.reply"
    assert 'bubble("jarvis", said)' in reply.group(1), (
        "an assistant.reply arriving with no request in flight is dropped, so "
        "anything said at the box is invisible to the remote client"
    )


def test_no_third_party_transport_is_referenced(guarded: str) -> None:
    """The server is stdlib-only; the client must not expect more."""
    for absent in ("WebSocket", "socket.io", "XMLHttpRequest"):
        assert absent not in guarded, (
            "%r appears in the page, but the server is plain http.server with SSE"
            % absent
        )


# --------------------------------------------------------------------------- #
#  Audio belongs to the client device
# --------------------------------------------------------------------------- #
def test_secure_context_is_checked_before_asking_for_a_microphone(guarded: str) -> None:
    """Outside a secure context there is no getUserMedia to call at all.

    Checking in the other order produces a thrown TypeError instead of the
    explanation the visitor needs.
    """
    body = script_body(guarded)
    secure = body.find("isSecureContext")
    capture = body.find("getUserMedia")

    assert secure != -1, "the page never checks isSecureContext"
    assert capture != -1, "the page never captures audio in the browser"
    assert secure < capture, (
        "getUserMedia is referenced at offset %d, before isSecureContext at %d; "
        "the secure-context test must come first" % (capture, secure)
    )

    # Source order alone would still pass if the runtime decision ran the other
    # way round, so check the branch order inside the routine that decides.
    decision = re.search(r"function micProblem\(\) \{(.*?)\n  \}", body, re.S)
    assert decision, "no micProblem routine to order the checks in"
    branches = decision.group(1)
    assert branches.index("secureContext()") < branches.index("hasCapture()"), (
        "micProblem tests for a capture API before testing the secure context, "
        "so an insecure page reports the wrong reason"
    )


def test_the_insecure_context_explanation_names_both_ways_out(guarded: str) -> None:
    body = script_body(guarded)
    assert "not a secure context" in body, "no explanation for a missing microphone"
    assert "HTTPS" in body, "the explanation must name HTTPS as one of the fixes"
    assert "ssh -N -L" in body, "the explanation must show the SSH tunnel that makes it local"
    assert "localhost" in body, "the tunnel is only useful because localhost is trusted"
    assert "--tls-cert" in body, "the explanation must name the TLS flags"


def test_microphone_permission_is_requested_on_the_press_not_on_load(guarded: str) -> None:
    body = script_body(guarded)

    # Referencing getUserMedia to feature-detect it prompts for nothing; only
    # invoking it with constraints does. Exactly one invocation should exist,
    # and it must live inside the press handler's recording routine.
    invocations = [m.start() for m in re.finditer(r"getUserMedia\(\{", body)]
    assert len(invocations) == 1, (
        "expected one getUserMedia invocation, found %d" % len(invocations)
    )
    begin_at = body.find("function beginTalk()")
    assert begin_at != -1, "no beginTalk routine"
    assert begin_at < invocations[0], (
        "capture must be invoked inside the press handler, not at module scope"
    )
    assert "pointerdown" in body and "onPress" in body, "no push-to-talk press handler"
    assert "function start()" in body
    start_at = body.find("function start()")
    assert "getUserMedia" not in body[start_at:], (
        "the boot path asks for the microphone; permission must wait for a gesture"
    )


def test_the_recorder_mime_type_is_feature_detected(guarded: str) -> None:
    """Chromium yields WebM, Safari yields MP4. Guessing loses one of them."""
    body = script_body(guarded)
    candidates = re.findall(r'"(audio/[^"]+)"', body)
    assert len(candidates) > 1, (
        "only %d recorder mime candidate(s) listed: %s" % (len(candidates), candidates)
    )
    families = {c.split(";")[0] for c in candidates}
    assert {"audio/webm", "audio/mp4"} <= families, (
        "missing a container: %s. Without audio/mp4 iOS Safari cannot record, "
        "without audio/webm Chromium cannot." % sorted(families)
    )
    assert "isTypeSupported" in body, "the candidates are listed but never tested"


def test_the_upload_declares_the_container_that_was_actually_produced(guarded: str) -> None:
    body = script_body(guarded)
    assert 'headers: { "Content-Type": type }' in body, (
        "the recording must be posted with its real mime type, or the server "
        "hands the wrong container to the decoder"
    )
    assert "recorder.mimeType" in body, (
        "the negotiated type from the recorder is more reliable than the request"
    )
    assert ENDPOINTS["stt"] in body


def test_transcript_is_placed_in_the_input_and_sent(guarded: str) -> None:
    body = script_body(guarded)
    assert "input.value = text;" in body, "the transcript must appear in the input box"
    assert "send(text, true);" in body, "the transcript is never actually sent"


def test_playback_is_armed_by_the_button_press_for_ios(guarded: str) -> None:
    body = script_body(guarded)
    arm_at = body.find("function armAudio()")
    press_at = body.find("function onPress(")
    assert arm_at != -1, "nothing arms audio playback"
    assert press_at != -1
    assert "armAudio();" in body[press_at:], (
        "iOS Safari refuses audio that no gesture started, so the press handler "
        "must arm playback"
    )
    assert "data:audio/wav;base64," in body, (
        "arming needs a real (silent) source; play() with no src rejects"
    )
    assert ".play()" in body and "started.catch" in body, (
        "a blocked play() rejects a promise, and an unhandled rejection is a "
        "silent failure the visitor cannot diagnose"
    )


def test_the_reply_is_played_from_the_blob_the_server_returned(guarded: str) -> None:
    """Creating an object URL and then not playing it is a silent assistant."""
    body = script_body(guarded)
    play = re.search(r"function play\(blob\) \{(.*?)\n  \}", body, re.S)
    assert play, "no playback routine"
    source = play.group(1)

    assert "URL.createObjectURL(blob)" in source, "the fetched audio is never wrapped"
    assert re.search(r"player\.src = lastObjectUrl;", source), (
        "the object URL is created but never assigned as the player's source, "
        "so /api/tts is fetched and thrown away"
    )
    assert "revokeObjectURL" in source, "each reply would leak an object URL"
    assert "res.blob()" in body, "the synthesised audio is never read as a blob"


def test_audio_is_captured_and_played_on_the_visiting_device(guarded: str) -> None:
    """The whole point: the server's own microphone stays out of it."""
    body = script_body(guarded)
    assert "new window.MediaRecorder" in body, "recording does not happen in the browser"
    assert "URL.createObjectURL" in body, "the reply is not played from a client-side blob"
    assert '<audio id="player"' in guarded, "no client-side audio element"
    for server_side in ("AudioRecorder", "AudioPlayer", "sounddevice", "/api/listen"):
        assert server_side not in guarded, (
            "%r suggests the remote session drives the server's audio devices, "
            "which would mean talking to an empty room" % server_side
        )


# --------------------------------------------------------------------------- #
#  Interface contract
# --------------------------------------------------------------------------- #
def test_enter_sends_and_shift_enter_makes_a_newline(guarded: str) -> None:
    body = script_body(guarded)
    match = re.search(r'if \(ev\.key === "Enter"[^)]*\)', body)
    assert match, "no Enter handling on the composer"
    assert "!ev.shiftKey" in match.group(0), (
        "Shift+Enter must fall through to the textarea as a newline"
    )
    assert "isComposing" in body, (
        "an IME candidate is confirmed with Enter; sending then truncates the word"
    )


def test_status_line_and_task_panel_are_present(guarded: str) -> None:
    body = script_body(guarded)
    for label in ("model ", "backend ", "STT ", "TTS "):
        assert '"%s"' % label in body or "'%s'" % label in body, (
            "the status line does not report %r" % label.strip()
        )
    assert "reconnecting" in body and "live" in body, "no connection state is shown"
    assert ENDPOINTS["tasks"] in body
    assert "function buildTree(" in body, "the task tree is never rendered"
    assert "node.children" in body, "the task panel ignores subagents"
    assert 'id="task-tree"' in guarded


def test_a_flat_task_list_is_nested_by_parent_id(guarded: str) -> None:
    """/api/tasks returns a flat list whose nesting is only implied.

    Each entry carries parent_id and depth; nothing carries children. A
    renderer that reads only .children therefore draws every subagent at the
    top level, and the tree is not a tree.
    """
    body = script_body(guarded)
    assert "function nest(" in body, "nothing reconstructs the implied nesting"
    assert "parent_id" in body, (
        "the client ignores parent_id, so a subagent is drawn beside the task "
        "that spawned it instead of underneath it"
    )
    assert "buildTree(nest(nodes), 0)" in body, (
        "the nesting is computed and then not used for rendering"
    )
    nest = re.search(r"function nest\(nodes\) \{(.*?)\n  \}", body, re.S)
    assert nest and "children: []" in nest.group(1), (
        "the rebuilt nodes must expose children, which is what buildTree reads"
    )
    assert "var total = nodes.length;" in body, (
        "the badge must count every task, not only the roots"
    )


def test_layout_is_mobile_first_with_a_large_touch_target(guarded: str) -> None:
    assert "width=device-width" in guarded, "no viewport meta, so mobile renders zoomed out"
    assert "min-width: 620px" in guarded, "no responsive breakpoint"
    target = re.search(r"#ptt \{(.*?)\}", guarded, re.S)
    assert target, "no push-to-talk button styling"
    sizes = [int(v) for v in re.findall(r"(?:width|height): (\d+)px", target.group(1))]
    assert sizes and min(sizes) >= 44, (
        "the push-to-talk target is %spx; below 44px it is hard to hit on a phone"
        % sizes
    )
    assert "safe-area-inset" in guarded, "the composer will sit under the iPhone home bar"
    assert "touch-action: none" in guarded, "holding the button would scroll the page"


# --------------------------------------------------------------------------- #
#  Rendering API
# --------------------------------------------------------------------------- #
def test_no_placeholder_survives_rendering() -> None:
    for features in (None, {"auth_required": True}, dict(AUTH_FEATURES)):
        page = render_page(title="Box", features=features)
        leftovers = re.findall(r"__JARVIS_[A-Z0-9_]*__", page)
        assert not leftovers, "unfilled template placeholders: %s" % sorted(set(leftovers))


def test_a_placeholder_shaped_title_cannot_expand() -> None:
    """Config is a string from disk; it must not be able to reach the template."""
    page = render_page(title="__JARVIS_BOOT__", features={"model": "__JARVIS_JS__"})
    assert not re.findall(r"__JARVIS_[A-Z0-9_]*__", page), (
        "a title shaped like a placeholder was substituted on"
    )
    assert page.count("<script") == 2, "the page structure changed"


def test_title_is_injected_into_the_tab_and_the_header() -> None:
    page = render_page(title="Study Box")
    assert "<title>Study Box</title>" in page
    assert '<div id="brand">Study Box</div>' in page
    assert boot_record(page)["title"] == "Study Box"


def test_features_reach_the_page_and_defaults_fill_the_gaps() -> None:
    record = boot_record(render_page(features={"stt": True, "model": "Qwen/Qwen3.6-27B"}))
    features = record["features"]

    assert features["stt"] is True
    assert features["model"] == "Qwen/Qwen3.6-27B"
    assert features["tts"] is False, "an unstated feature must default to unavailable"
    assert features["auth_required"] is False
    assert set(DEFAULT_FEATURES) <= set(features), (
        "the page reads keys that were not supplied: %s"
        % sorted(set(DEFAULT_FEATURES) - set(features))
    )


def test_tls_state_is_recorded_for_the_page() -> None:
    assert boot_record(render_page(has_tls=True))["tls"] is True
    assert boot_record(render_page(has_tls=False))["tls"] is False


def test_truthy_feature_values_are_coerced_to_booleans() -> None:
    features = boot_record(render_page(features={"stt": "yes", "tts": 0, "auth_required": 1}))[
        "features"
    ]
    assert features["stt"] is True and features["tts"] is False
    assert features["auth_required"] is True
    assert '<div id="gate">' in render_page(features={"auth_required": 1}), (
        "a truthy auth flag must still produce the token screen"
    )


@pytest.mark.parametrize("junk", [42, "not a mapping", object(), [("a", 1)]])
def test_render_page_never_raises_on_a_bad_features_argument(junk) -> None:
    """A public entry point returns something usable rather than a 500."""
    page = render_page(features=junk)
    assert page.lstrip().lower().startswith("<!doctype html>")
    assert not parse(page).faults


def test_render_page_bytes_is_the_page_in_utf8() -> None:
    assert render_page_bytes(title="Box").decode("utf-8") == render_page(title="Box")


def test_module_constant_matches_a_default_render() -> None:
    assert PAGE == render_page()
    assert DEFAULT_TITLE in PAGE
    assert PAGE.count("<title>") == 1


def test_rendering_is_deterministic() -> None:
    """Two identical renders must be byte-identical, so ETags and caches work."""
    first = render_page(title="Box", has_tls=True, features=dict(AUTH_FEATURES))
    second = render_page(title="Box", has_tls=True, features=dict(AUTH_FEATURES))
    assert first == second


# --------------------------------------------------------------------------- #
#  The seam the server actually calls
# --------------------------------------------------------------------------- #
def test_index_html_takes_the_status_payload_positionally() -> None:
    """The server passes the status dict positionally and by name.

    render_page is keyword-only, so a positional call raises TypeError and the
    server falls back to a default page. That fallback silently loses
    auth_required, which is the one flag that cannot be corrected later: it
    decides whether the token screen is in the markup at all.
    """
    page = index_html(LAN_STATUS)
    assert '<div id="gate">' in page, (
        "a LAN session that needs a token was served a page with no way to enter one"
    )
    assert index_html() == render_page(), "no payload must still yield a usable page"


def test_index_html_maps_the_payload_onto_the_boot_record() -> None:
    features = boot_record(index_html(LAN_STATUS))["features"]
    assert features["model"] == "Qwen/Qwen3.6-27B"
    assert features["backend"] == "airllm", "the llm slot is the backend name"
    assert features["stt"] is True and features["tts"] is True
    assert features["version"] == "1.1.0"
    assert boot_record(index_html(LAN_STATUS))["title"] == "JARVIS"


def test_an_engine_name_of_unavailable_means_unavailable() -> None:
    """The status slots carry engine names, not booleans; 'unavailable' is a name."""
    offline = dict(LAN_STATUS, stt="unavailable", tts="unavailable", stt_available=False)
    features = boot_record(index_html(offline))["features"]
    assert features["stt"] is False, (
        "push-to-talk would be offered with no engine behind it"
    )
    assert features["tts"] is False


def test_the_gate_is_skipped_once_the_session_cookie_is_valid() -> None:
    """Re-showing the token screen to an authenticated visitor is a dead end."""
    assert '<div id="gate">' not in index_html(dict(LAN_STATUS, authenticated=True))
    assert '<div id="gate">' not in index_html(dict(LAN_STATUS, auth_required=False))


def test_tls_and_orchestrator_flags_are_carried_through() -> None:
    secure = boot_record(index_html(dict(LAN_STATUS, tls=True)))
    assert secure["tls"] is True
    without_agent = boot_record(index_html(dict(LAN_STATUS, orchestrator=False)))
    assert without_agent["features"]["tasks"] is False


@pytest.mark.parametrize("junk", [None, {}, "not a mapping", 7, {"model": None}])
def test_index_html_never_raises_on_a_bad_payload(junk) -> None:
    page = index_html(junk)
    assert page.lstrip().lower().startswith("<!doctype html>")
    assert not parse(page).faults


def test_the_client_trusts_live_status_over_the_boot_record(guarded: str) -> None:
    """A page served with stale defaults must still discover the real engines."""
    body = script_body(guarded)
    assert "function present(" in body, "no tolerant availability reader"
    assert "info.stt_available" in body, (
        "the server reports an explicit stt_available boolean and the client "
        "ignores it"
    )
    assert "FEAT.stt = present(" in body and "FEAT.tts = present(" in body, (
        "the live status must update what push-to-talk is allowed to do"
    )
    assert '"unavailable"' in body, (
        "an engine named 'unavailable' must not read as an available engine"
    )


def test_rendering_does_not_mutate_the_caller_or_the_defaults() -> None:
    supplied = {"stt": True}
    before = dict(DEFAULT_FEATURES)
    render_page(features=supplied)
    assert supplied == {"stt": True}, "the caller's mapping was modified"
    assert DEFAULT_FEATURES == before, "the module defaults were modified"
    assert DEFAULT_FEATURES["stt"] is False
