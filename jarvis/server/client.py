"""The browser client: one self-contained page, served to any device.

The owner's rule for remote access is that JARVIS itself never leaves the Linux
box.  A phone, a Mac or a second laptop is a *terminal*, not an installation —
so this module ships the entire user interface as a single HTML string with the
CSS and JavaScript inlined.  No build step, no framework, no CDN: a page that
fetches a script from the internet would break the moment the box is offline,
which is exactly when a local assistant is supposed to still work.

The design point that makes remote access worth having
------------------------------------------------------
JARVIS's own microphone and speakers are wired to the Linux box.  If a session
opened from the kitchen listened on *that* microphone, you would be talking to
an empty room.  So push-to-talk here records with ``getUserMedia`` in the
visitor's browser, uploads the blob to ``/api/stt``, and plays the synthesised
reply through the visitor's speakers.  Server-side audio devices are never
touched by a remote session.

Browsers make that harder than it sounds, and the page says so out loud rather
than presenting a button that silently does nothing:

* ``getUserMedia`` exists only in a **secure context** — TLS, or an origin the
  browser considers local.  Reached over plain HTTP at a LAN address there is
  no microphone at all, and the page explains the two ways out (an SSH tunnel,
  which makes the origin genuinely local, or running the server with TLS).
* iOS Safari refuses to start audio that no user gesture asked for, so playback
  is armed from the push-to-talk press itself.
* ``MediaRecorder`` supports different containers in different browsers —
  Chromium gives WebM, Safari gives MP4 — so the codec is feature-detected and
  the upload carries whatever ``Content-Type`` was actually produced.

Everything the server sends is written to the DOM with ``textContent``.  A tool
result that happens to contain markup is text, and stays text; the string
``innerHTML`` does not appear anywhere in this file on purpose.

Public surface
--------------
:data:`PAGE`
    The page rendered with defaults, for callers that need no customisation.
:func:`render_page`
    The page with the title, TLS state and available features injected.
:data:`ENDPOINTS`
    The paths the page calls, so the server can be checked against them.
"""

from __future__ import annotations

import html
import json
import logging
import re
from typing import Any, Dict, Mapping, Optional

log = logging.getLogger(__name__)

__all__ = [
    "CONTENT_TYPE",
    "DEFAULT_FEATURES",
    "DEFAULT_TITLE",
    "ENDPOINTS",
    "PAGE",
    "index_html",
    "render_page",
    "render_page_bytes",
]

DEFAULT_TITLE = "JARVIS"
CONTENT_TYPE = "text/html; charset=utf-8"

#: Every path the page talks to.  The server module is expected to route all of
#: these; the test-suite asserts the page really references each one.
ENDPOINTS: Dict[str, str] = {
    "chat": "/api/chat",
    "stream": "/api/stream",
    "stt": "/api/stt",
    "tts": "/api/tts",
    "tasks": "/api/tasks",
    "status": "/api/status",
    "login": "/api/login",
    "logout": "/api/logout",
}

#: What the page assumes when the server tells it nothing.  ``auth_required``
#: is the one that changes the markup: without it the token screen is not
#: rendered at all, rather than rendered and hidden.
DEFAULT_FEATURES: Dict[str, Any] = {
    "auth_required": False,
    "chat": True,
    "stream": True,
    "stt": False,
    "tts": False,
    "tasks": True,
    "model": "unknown",
    "backend": "unknown",
    "version": "",
}

_BOOL_FEATURES = ("auth_required", "chat", "stream", "stt", "tts", "tasks")
_TEXT_FEATURES = ("model", "backend", "version")

# Placeholders use a form that cannot occur in CSS or JavaScript, so a stray
# brace in a stylesheet can never be mistaken for a substitution site.
_PLACEHOLDER_RE = re.compile(r"__JARVIS_[A-Z0-9_]*__")


# --------------------------------------------------------------------------- #
#  Stylesheet
# --------------------------------------------------------------------------- #
_CSS = r"""
*, *::before, *::after { box-sizing: border-box; }

:root {
  color-scheme: dark light;
  --bg: #0b0f14;
  --panel: #131b24;
  --sunken: #0e151d;
  --line: #26313f;
  --fg: #e6edf3;
  --dim: #91a1b3;
  --accent: #4aa8ff;
  --accent-fg: #04121f;
  --ok: #4ade80;
  --warn: #ffb454;
  --bad: #ff7676;
  --radius: 14px;
}

@media (prefers-color-scheme: light) {
  :root {
    --bg: #f4f7fa;
    --panel: #ffffff;
    --sunken: #eaeff5;
    --line: #d3dce7;
    --fg: #15202b;
    --dim: #566779;
    --accent: #0b64c8;
    --accent-fg: #ffffff;
  }
}

html, body { height: 100%; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
  -webkit-text-size-adjust: 100%;
}

#app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  max-width: 860px;
  margin: 0 auto;
}

@supports (min-height: 100dvh) {
  #app { min-height: 100dvh; }
}

/* --- header ------------------------------------------------------------- */
#bar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  padding-top: calc(10px + env(safe-area-inset-top, 0px));
  border-bottom: 1px solid var(--line);
  background: var(--panel);
  position: sticky;
  top: 0;
  z-index: 5;
}

#brand { font-weight: 700; letter-spacing: 0.02em; }

#conn-wrap {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-left: auto;
  font-size: 13px;
  color: var(--dim);
}

.dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--dim);
  flex: none;
}

.dot.live { background: var(--ok); }
.dot.down { background: var(--bad); }

#meta {
  margin: 0;
  padding: 6px 14px;
  font-size: 12.5px;
  color: var(--dim);
  border-bottom: 1px solid var(--line);
  background: var(--sunken);
  overflow-wrap: anywhere;
}

/* --- buttons ------------------------------------------------------------ */
button {
  font: inherit;
  color: inherit;
  cursor: pointer;
  border-radius: 10px;
  border: 1px solid var(--line);
  background: var(--sunken);
  padding: 8px 12px;
  min-height: 40px;
}

button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
button[disabled] { opacity: 0.5; cursor: default; }
.ghost { background: transparent; font-size: 13px; min-height: 34px; padding: 5px 10px; }

/* --- conversation ------------------------------------------------------- */
#log {
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  -webkit-overflow-scrolling: touch;
}

.msg { max-width: 90%; }
.msg .who { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--dim); margin-bottom: 3px; }
.msg .body {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  padding: 10px 13px;
  border-radius: var(--radius);
  background: var(--panel);
  border: 1px solid var(--line);
}

.msg.you { align-self: flex-end; text-align: right; }
.msg.you .body { background: var(--accent); color: var(--accent-fg); border-color: transparent; text-align: left; }
.msg.system .body { background: transparent; border-style: dashed; color: var(--dim); }
.msg.thinking .body::after {
  content: "...";
  color: var(--dim);
}
.msg.thinking .body:not(:empty)::after { content: ""; }

/* --- notices ------------------------------------------------------------ */
.panel {
  margin: 0 14px 12px;
  padding: 11px 13px;
  border: 1px solid var(--line);
  border-left: 3px solid var(--warn);
  border-radius: 10px;
  background: var(--panel);
  font-size: 14px;
}

.panel .text { white-space: pre-wrap; overflow-wrap: anywhere; }
.panel .text code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.fine { color: var(--dim); font-size: 12.5px; margin: 6px 0 0; }

/* --- tasks -------------------------------------------------------------- */
#tasks {
  margin: 0 14px 12px;
  padding: 10px 13px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--panel);
}

#tasks h2 { margin: 0 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--dim); }
ul.tree { list-style: none; margin: 0; padding-left: 0; }
ul.tree ul.tree { padding-left: 16px; border-left: 1px solid var(--line); margin-left: 5px; }
.task { display: flex; gap: 8px; align-items: baseline; padding: 2px 0; font-size: 14px; }
.task .goal { overflow-wrap: anywhere; }
.state {
  flex: none;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 1px 7px;
  border-radius: 999px;
  border: 1px solid var(--line);
  color: var(--dim);
}
.state.s-running { color: var(--accent); border-color: var(--accent); }
.state.s-done { color: var(--ok); border-color: var(--ok); }
.state.s-failed, .state.s-cancelled { color: var(--bad); border-color: var(--bad); }

/* --- composer ----------------------------------------------------------- */
#composer {
  position: sticky;
  bottom: 0;
  display: flex;
  gap: 8px;
  align-items: flex-end;
  padding: 10px 14px;
  padding-bottom: calc(10px + env(safe-area-inset-bottom, 0px));
  border-top: 1px solid var(--line);
  background: var(--panel);
}

#input {
  flex: 1 1 auto;
  min-width: 0;
  resize: none;
  font: inherit;
  color: inherit;
  background: var(--sunken);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px 12px;
  max-height: 30vh;
  min-height: 44px;
}

#input:focus { outline: 2px solid var(--accent); outline-offset: -1px; }

#send { min-height: 44px; }

#ptt {
  flex: none;
  width: 60px;
  height: 60px;
  border-radius: 50%;
  font-size: 11px;
  line-height: 1.15;
  padding: 4px;
  background: var(--sunken);
  border: 2px solid var(--line);
  touch-action: none;
  -webkit-user-select: none;
  user-select: none;
  -webkit-tap-highlight-color: transparent;
}

#ptt.armed { border-color: var(--bad); background: var(--bad); color: #17070a; }
#ptt.busy { border-color: var(--accent); }
#ptt.off { opacity: 0.55; }

/* --- token gate --------------------------------------------------------- */
#gate {
  position: fixed;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  background: var(--bg);
  z-index: 20;
}

#gate form {
  width: 100%;
  max-width: 380px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 20px;
}

#gate h1 { margin: 0 0 8px; font-size: 19px; }
#gate p { margin: 0 0 12px; font-size: 14px; color: var(--dim); }
#gate label { display: block; font-size: 13px; margin-bottom: 5px; }

#gate-token {
  width: 100%;
  font: inherit;
  color: inherit;
  background: var(--sunken);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 11px 12px;
  min-height: 44px;
}

#gate button { width: 100%; margin-top: 12px; background: var(--accent); color: var(--accent-fg); border-color: transparent; min-height: 46px; }
#gate-error { color: var(--bad); margin: 10px 0 0; font-size: 14px; }

[hidden] { display: none !important; }

@media (min-width: 620px) {
  #ptt { width: 68px; height: 68px; font-size: 12px; }
}
"""


# --------------------------------------------------------------------------- #
#  Client script
# --------------------------------------------------------------------------- #
# A 45-byte silent WAV.  Playing it inside the push-to-talk gesture is what
# arms the <audio> element on iOS Safari, which otherwise refuses to start the
# spoken reply that arrives a second later, outside any gesture.
_SILENT_WAV = (
    "data:audio/wav;base64,"
    "UklGRiUAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQEAAACA"
)

_JS = r"""
(function () {
  "use strict";

  /* ---------------------------------------------------------------- boot -- */
  var BOOT = (function () {
    var node = document.getElementById("jarvis-boot");
    if (!node) { return {}; }
    try { return JSON.parse(node.textContent || "{}") || {}; }
    catch (err) { return {}; }
  })();

  var FEAT = BOOT.features || {};
  var BRAND = BOOT.title || "JARVIS";

  var API = {
    chat: "/api/chat",
    stream: "/api/stream",
    stt: "/api/stt",
    tts: "/api/tts",
    tasks: "/api/tasks",
    status: "/api/status",
    login: "/api/login",
    logout: "/api/logout"
  };

  var SILENT_WAV = "__JARVIS_SILENT_WAV__";
  var TAP_MS = 400;          /* shorter than this and the button latches on */
  var TASK_POLL_MS = 6000;

  /* ------------------------------------------------------------- helpers -- */
  function $(id) { return document.getElementById(id); }

  function clear(node) {
    while (node && node.firstChild) { node.removeChild(node.firstChild); }
  }

  /* The only way this page puts server data on screen.  textContent means a
     reply or a tool result containing markup renders as the characters the
     model produced, never as elements. */
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  function slug(value) {
    return String(value || "unknown").toLowerCase().replace(/[^a-z0-9]+/g, "-");
  }

  var log = $("log");
  var meta = $("meta");
  var input = $("input");
  var sendBtn = $("send");
  var ptt = $("ptt");
  var player = $("player");
  var notice = $("notice");
  var noticeBody = $("notice-body");
  var micNote = $("micnote");
  var micNoteBody = $("micnote-body");
  var tasksPanel = $("tasks");
  var tasksToggle = $("tasks-toggle");
  var taskTree = $("task-tree");
  var connDot = $("conn-dot");
  var connText = $("conn-text");
  var gate = $("gate");

  function note(message) {
    if (!noticeBody) { return; }
    clear(noticeBody);
    noticeBody.appendChild(el("div", "text", message));
    notice.hidden = false;
  }

  function scrollDown() {
    if (log) { log.scrollTop = log.scrollHeight; }
  }

  /* ---------------------------------------------------------- networking -- */
  /* Same-origin only, and the session cookie the server set is the sole
     credential.  The access token is never held in a variable and never
     written to browser storage, so a shared machine cannot give it away. */
  function api(path, opts) {
    var options = opts || {};
    options.credentials = "same-origin";
    options.cache = "no-store";
    return fetch(path, options).then(function (res) {
      if (res.status === 401 || res.status === 403) {
        lock("The server refused that request. Enter the access token again.");
        var denied = new Error("unauthorised");
        denied.unauthorised = true;
        throw denied;
      }
      return res;
    });
  }

  function apiJson(path, opts) {
    return api(path, opts).then(function (res) {
      return res.text().then(function (body) {
        if (!body) { return {}; }
        try { return JSON.parse(body); }
        catch (err) { return { error: body }; }
      });
    });
  }

  function postJson(path, payload) {
    return apiJson(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
  }

  function quiet() { /* a failed poll is not worth shouting about */ }

  /* ---------------------------------------------------------- token gate -- */
  function lock(message) {
    if (!gate) {
      /* No token screen is in this document: the page was rendered for a
         session that was already valid, and the session has since expired.
         There is nothing here to type a token into, so say the one thing that
         does work rather than leaving a dead end on screen. */
      note((message || "The server refused the request.")
        + "\n\nReload this page to be asked for the access token again.");
      return;
    }
    gate.hidden = false;
    var errorLine = $("gate-error");
    if (message && errorLine) {
      errorLine.textContent = message;
      errorLine.hidden = false;
    }
    var field = $("gate-token");
    if (field) { field.focus(); }
  }

  if (gate) {
    $("gate-form").addEventListener("submit", function (ev) {
      ev.preventDefault();
      var field = $("gate-token");
      var errorLine = $("gate-error");
      var token = field.value;
      errorLine.hidden = true;
      /* Deliberately not routed through api(): a rejected token must show an
         error here, not bounce back into lock(). */
      fetch(API.login, {
        method: "POST",
        credentials: "same-origin",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: token })
      }).then(function (res) {
        if (!res.ok) { throw new Error("rejected"); }
        field.value = "";          /* the token lives only in the cookie now */
        gate.hidden = true;
        start();
      }).catch(function () {
        field.value = "";
        errorLine.textContent = "That token was not accepted.";
        errorLine.hidden = false;
        field.focus();
      });
    });
  }

  /* ------------------------------------------------------------- speech --- */
  /* Secure context first. Browsers expose no capture API at all outside one,
     so asking for a microphone before checking this is the wrong question in
     the wrong order. */
  function secureContext() {
    if (typeof window.isSecureContext === "boolean") { return window.isSecureContext; }
    var host = location.hostname;
    return location.protocol === "https:" || host === "localhost" || host === "127.0.0.1" || host === "::1";
  }

  function hasRecorder() {
    return typeof window.MediaRecorder !== "undefined";
  }

  function hasCapture() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  /* Chromium records WebM/Opus, Safari records MP4/AAC, Firefox records Ogg.
     Whatever wins here is also the Content-Type sent to /api/stt. */
  var MIME_CANDIDATES = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
    "audio/ogg",
    "audio/wav"
  ];

  function pickMime() {
    if (!hasRecorder() || !window.MediaRecorder.isTypeSupported) { return ""; }
    for (var i = 0; i < MIME_CANDIDATES.length; i++) {
      if (window.MediaRecorder.isTypeSupported(MIME_CANDIDATES[i])) { return MIME_CANDIDATES[i]; }
    }
    return "";
  }

  function micProblem() {
    if (!FEAT.stt) { return "nostt"; }
    if (!secureContext()) { return "insecure"; }
    if (!hasCapture()) { return "nocapture"; }
    if (!hasRecorder()) { return "norecorder"; }
    return "";
  }

  function micExplanation(kind) {
    if (kind === "insecure") {
      var port = location.port || "8765";
      var host = location.hostname || "linux-box";
      return "The microphone is unavailable because this page is not a secure context.\n"
        + "Browsers only hand a microphone to a page served over HTTPS, or to one the\n"
        + "browser already considers local. This page came over plain HTTP from " + host + ".\n\n"
        + "Two ways out, both leaving JARVIS entirely on the Linux box:\n\n"
        + "1. Tunnel it, so the page really is local:\n"
        + "     ssh -N -L " + port + ":127.0.0.1:" + port + " you@" + host + "\n"
        + "   then open localhost:" + port + " on this device.\n\n"
        + "2. Or serve with TLS:\n"
        + "     jarvis serve --host 0.0.0.0 --tls-cert cert.pem --tls-key key.pem\n\n"
        + "Typing works either way.";
    }
    if (kind === "nostt") {
      return "Speech-to-text is not loaded on the server, so push-to-talk is off. "
        + "Typing works. Run 'jarvis doctor' on the Linux box to see what is missing.";
    }
    if (kind === "nocapture") {
      return "This browser exposes no microphone capture API, so push-to-talk is unavailable. "
        + "Typing works.";
    }
    if (kind === "norecorder") {
      return "This browser has no MediaRecorder, so audio cannot be captured here. "
        + "On iOS this means Safari 14.3 or newer. Typing works.";
    }
    return "";
  }

  /* --- playback on THIS device, never on the server ----------------------- */
  var audioArmed = false;
  var lastObjectUrl = null;

  function armAudio() {
    if (audioArmed || !player) { return; }
    audioArmed = true;
    try {
      player.src = SILENT_WAV;
      var started = player.play();
      if (started && started.catch) {
        started.catch(function () { audioArmed = false; });
      }
    } catch (err) {
      audioArmed = false;
    }
  }

  function play(blob) {
    if (lastObjectUrl) { URL.revokeObjectURL(lastObjectUrl); }
    lastObjectUrl = URL.createObjectURL(blob);
    player.src = lastObjectUrl;
    var started = player.play();
    if (started && started.catch) {
      started.catch(function () {
        note("This browser blocked playback until you interact with the page. "
          + "Press the talk button once and the reply will be spoken.");
      });
    }
  }

  function speak(text) {
    if (!text || !FEAT.tts) { return; }
    api(API.tts, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    }).then(function (res) {
      if (!res.ok) { throw new Error("tts " + res.status); }
      return res.blob();
    }).then(play).catch(quiet);
  }

  /* --- recording ---------------------------------------------------------- */
  var state = "idle";        /* idle | arming | recording | stopping */
  var latched = false;       /* a tap latched it on; the next tap stops it */
  var pressedAt = 0;
  var recorder = null;
  var micStream = null;
  var chunks = [];

  var PTT_IDLE = "Hold to talk";

  function setPtt(label, cls) {
    if (!ptt) { return; }
    ptt.textContent = label;
    ptt.className = cls || "";
  }

  function closeStream() {
    if (!micStream) { return; }
    var tracks = micStream.getTracks ? micStream.getTracks() : [];
    for (var i = 0; i < tracks.length; i++) {
      try { tracks[i].stop(); } catch (err) { quiet(); }
    }
    micStream = null;
  }

  function beginTalk() {
    if (state !== "idle") { return; }
    state = "arming";
    setPtt("Allow mic", "busy");
    /* Permission is requested here, on the press, never on page load. */
    navigator.mediaDevices.getUserMedia({ audio: true }).then(function (stream) {
      micStream = stream;
      if (state !== "arming") { closeStream(); state = "idle"; setPtt(PTT_IDLE); return; }
      var mime = pickMime();
      try {
        recorder = mime ? new window.MediaRecorder(stream, { mimeType: mime })
                        : new window.MediaRecorder(stream);
      } catch (err) {
        recorder = new window.MediaRecorder(stream);
      }
      chunks = [];
      recorder.ondataavailable = function (ev) {
        if (ev.data && ev.data.size) { chunks.push(ev.data); }
      };
      recorder.onstop = function () {
        var produced = (recorder && recorder.mimeType) || mime || "";
        finishTalk(produced);
      };
      recorder.start();
      state = "recording";
      setPtt(latched ? "Tap to send" : "Release", "armed");
    }).catch(function (err) {
      state = "idle";
      latched = false;
      setPtt(PTT_IDLE);
      var name = (err && err.name) || "error";
      note("The microphone could not be opened (" + name + "). "
        + "Check that permission was granted for this site and that an input device exists.");
    });
  }

  function endTalk() {
    if (state === "arming") { state = "idle"; closeStream(); setPtt(PTT_IDLE); return; }
    if (state !== "recording") { return; }
    state = "stopping";
    setPtt("Sending", "busy");
    try { recorder.stop(); }
    catch (err) { finishTalk(""); }
  }

  function finishTalk(mime) {
    closeStream();
    state = "idle";
    latched = false;
    if (!chunks.length) { setPtt(PTT_IDLE); note("Nothing was recorded."); return; }
    var type = mime || chunks[0].type || "application/octet-stream";
    var blob = new Blob(chunks, { type: type });
    chunks = [];
    setPtt("Hearing", "busy");
    /* The Content-Type is whatever this browser actually produced, so the
       server can hand the right container to the decoder. */
    api(API.stt, { method: "POST", headers: { "Content-Type": type }, body: blob })
      .then(function (res) {
        if (!res.ok) { throw new Error("stt " + res.status); }
        return res.text();
      })
      .then(function (body) {
        var data = {};
        try { data = JSON.parse(body) || {}; } catch (err) { data = { text: body }; }
        setPtt(PTT_IDLE);
        var text = data.text || data.transcript || "";
        if (!text) { note("No speech was recognised in that recording."); return; }
        input.value = text;
        autosize();
        send(text, true);
      })
      .catch(function (err) {
        setPtt(PTT_IDLE);
        if (err && err.unauthorised) { return; }
        note("Transcription failed: " + ((err && err.message) || "unknown error"));
      });
  }

  function onPress(ev) {
    if (ev && ev.cancelable) { ev.preventDefault(); }
    /* iOS Safari only lets audio start from inside a gesture, so the reply
       that arrives seconds later is armed right here. */
    armAudio();
    var problem = micProblem();
    if (problem) { note(micExplanation(problem)); return; }
    if (state === "recording" || state === "arming") { endTalk(); return; }
    pressedAt = Date.now();
    latched = false;
    beginTalk();
  }

  function onRelease(ev) {
    if (ev && ev.cancelable) { ev.preventDefault(); }
    if (state !== "recording" && state !== "arming") { return; }
    if (Date.now() - pressedAt < TAP_MS) {
      /* Too quick to be a hold: treat it as a tap and keep listening. */
      latched = true;
      if (state === "recording") { setPtt("Tap to send", "armed"); }
      return;
    }
    endTalk();
  }

  if (ptt) {
    if (window.PointerEvent) {
      ptt.addEventListener("pointerdown", onPress);
      ptt.addEventListener("pointerup", onRelease);
      ptt.addEventListener("pointercancel", onRelease);
    } else {
      ptt.addEventListener("touchstart", onPress);
      ptt.addEventListener("touchend", onRelease);
      ptt.addEventListener("touchcancel", onRelease);
      ptt.addEventListener("mousedown", onPress);
      ptt.addEventListener("mouseup", onRelease);
    }
    ptt.addEventListener("click", function (ev) { ev.preventDefault(); });
    ptt.addEventListener("keydown", function (ev) {
      if ((ev.key === " " || ev.key === "Enter") && !ev.repeat) { onPress(ev); }
    });
    ptt.addEventListener("keyup", function (ev) {
      if (ev.key === " " || ev.key === "Enter") { onRelease(ev); }
    });
  }

  /* ---------------------------------------------------------- conversation */
  var pending = null;
  var busy = false;

  function bubble(role, text) {
    var wrap = el("div", "msg " + role);
    var who = role === "you" ? "You" : (role === "system" ? "Note" : BRAND);
    wrap.appendChild(el("div", "who", who));
    var body = el("div", "body", text || "");
    wrap.appendChild(body);
    log.appendChild(wrap);
    scrollDown();
    return { wrap: wrap, body: body };
  }

  function setPending(text) {
    if (!pending) { return; }
    pending.text = text || "";
    pending.body.textContent = pending.text;
    scrollDown();
  }

  function appendChunk(text) {
    if (!pending || !text) { return; }
    pending.streamed = true;
    setPending(pending.text + text);
  }

  function finishPending() {
    if (!pending) { return; }
    pending.wrap.classList.remove("thinking");
    if (!pending.text) { pending.body.textContent = "(no reply)"; }
    pending = null;
    busy = false;
    sendBtn.disabled = false;
  }

  function replyOf(data) {
    if (typeof data === "string") { return data; }
    if (!data) { return ""; }
    return data.reply || data.text || data.message || data.content || "";
  }

  function send(text, alreadyInInput) {
    var value = String(text || "").trim();
    if (!value || busy) { return; }
    busy = true;
    sendBtn.disabled = true;
    bubble("you", value);
    input.value = "";
    autosize();
    if (!alreadyInInput) { input.focus(); }

    var slot = bubble("jarvis", "");
    slot.wrap.classList.add("thinking");
    pending = { wrap: slot.wrap, body: slot.body, text: "", streamed: false };

    postJson(API.chat, { text: value }).then(function (data) {
      var reply = replyOf(data);
      if (!pending) {
        /* The stream already closed it out; only speaking is left. */
        if (reply) { speak(reply); }
        return;
      }
      if (!pending.streamed || (reply && reply.length > pending.text.length)) {
        setPending(reply);
      }
      var spoken = pending.text;
      finishPending();
      speak(spoken);
      refreshTasks();
    }).catch(function (err) {
      if (!pending) { busy = false; sendBtn.disabled = false; return; }
      /* Routed through setPending so the text survives finishPending(), which
         replaces an empty bubble with "(no reply)". */
      if (err && err.unauthorised) {
        setPending("(not sent: the server asked for the access token)");
      } else {
        setPending("(failed: " + ((err && err.message) || "no reply") + ")");
      }
      finishPending();
    });
  }

  function autosize() {
    if (!input) { return; }
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, window.innerHeight * 0.3) + "px";
  }

  input.addEventListener("input", autosize);
  input.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && !ev.shiftKey && !ev.altKey && !ev.isComposing) {
      ev.preventDefault();
      send(input.value);
    }
  });
  sendBtn.addEventListener("click", function () { send(input.value); });

  /* ------------------------------------------------------------- stream --- */
  var stream = null;
  var streamErrors = 0;

  /* Two names for every event, on purpose.  The bus channel is
     "assistant.chunk"; the SSE layer in front of it renames that to "chunk" on
     the wire.  Which one arrives is the server's choice, not this page's, and a
     listener bound to the name the server did not pick is silent forever: a
     connection indicator that reads "live" above a page that never updates. */
  var STREAM_EVENTS = [
    "assistant.chunk", "chunk",
    "assistant.reply", "reply",
    "task.created", "task_created",
    "task.update", "task_update",
    "task.done", "task_done",
    "task.failed", "task_failed",
    "error"
  ];

  function canonical(name) {
    if (name === "chunk") { return "assistant.chunk"; }
    if (name === "reply") { return "assistant.reply"; }
    if (String(name).indexOf("task_") === 0) { return "task." + String(name).slice(5); }
    return name;
  }

  function setConn(label, ok) {
    if (!connText) { return; }
    connText.textContent = label;
    connDot.className = "dot " + (ok === true ? "live" : (ok === false ? "down" : ""));
  }

  function textOf(payload) {
    if (typeof payload === "string") { return payload; }
    if (!payload) { return ""; }
    return payload.text || payload.chunk || payload.delta || payload.content
      || payload.reply || payload.message || "";
  }

  function dispatch(name, raw) {
    var payload = raw;
    try { payload = JSON.parse(raw); } catch (err) { payload = { text: raw }; }
    if (payload && typeof payload === "object" && payload.event && payload.data !== undefined) {
      name = payload.event;
      payload = payload.data;
    }
    name = canonical(name);
    if (name === "assistant.chunk") { appendChunk(textOf(payload)); return; }
    if (name === "assistant.reply") {
      var said = textOf(payload);
      if (pending) {
        setPending(said);
        finishPending();
      } else if (said) {
        /* Nothing here asked for this: it was spoken at the machine itself, or
           by another session. Show it rather than dropping it on the floor. */
        bubble("jarvis", said);
      }
      return;
    }
    if (name.indexOf("task.") === 0) { refreshTasks(); return; }
    if (name === "error") { note(textOf(payload) || "The server reported an error."); }
  }

  function connect() {
    if (!FEAT.stream || typeof window.EventSource === "undefined") {
      setConn("no stream", null);
      return;
    }
    try {
      /* EventSource cannot set headers, which is precisely why the server
         hands out a session cookie instead of expecting a bearer token. */
      stream = new EventSource(API.stream);
    } catch (err) {
      setConn("no stream", null);
      return;
    }
    stream.onopen = function () { streamErrors = 0; setConn("live", true); };
    stream.onmessage = function (ev) { dispatch("message", ev.data); };
    stream.onerror = function () {
      streamErrors += 1;
      setConn("reconnecting", false);
      if (streamErrors === 3) { refreshStatus(); }
    };
    for (var i = 0; i < STREAM_EVENTS.length; i++) {
      (function (name) {
        stream.addEventListener(name, function (ev) { dispatch(name, ev.data); });
      })(STREAM_EVENTS[i]);
    }
  }

  /* -------------------------------------------------------------- status -- */
  /* A status slot is a boolean from some builds and an engine name from
     others, and "unavailable" is a name that means the opposite of one. */
  var ABSENT = ["", "none", "null", "unavailable", "unknown", "disabled"];

  function present(value, fallback) {
    if (typeof value === "boolean") { return value; }
    if (value === null || value === undefined) { return !!fallback; }
    var text = String(value).toLowerCase();
    for (var i = 0; i < ABSENT.length; i++) {
      if (text === ABSENT[i]) { return false; }
    }
    return true;
  }

  function describe(value, fallback) {
    if (value === true) { return "ready"; }
    if (value === false) { return "unavailable"; }
    if (value === null || value === undefined || value === "") {
      return fallback ? "ready" : "unavailable";
    }
    return String(value);
  }

  function refreshStatus() {
    return apiJson(API.status).then(function (data) {
      var info = data || {};
      /* The live status wins over the boot record: it is what decides whether
         push-to-talk is offered at all. */
      FEAT.stt = present(
        info.stt_available !== undefined ? info.stt_available : info.stt, FEAT.stt
      );
      FEAT.tts = present(info.tts, FEAT.tts);
      var bits = [
        "model " + (info.model || FEAT.model || "unknown"),
        "backend " + (info.backend || info.llm || FEAT.backend || "unknown"),
        "STT " + describe(info.stt, FEAT.stt),
        "TTS " + describe(info.tts, FEAT.tts)
      ];
      if (info.version || FEAT.version) { bits.push("v" + (info.version || FEAT.version)); }
      clear(meta);
      meta.appendChild(document.createTextNode(bits.join("  |  ")));
      showMicState();
    }).catch(quiet);
  }

  function showMicState() {
    var problem = micProblem();
    if (!problem) {
      setPtt(PTT_IDLE);
      ptt.removeAttribute("aria-disabled");
      micNote.hidden = true;
      return;
    }
    setPtt(problem === "insecure" ? "No mic" : "Mic off", "off");
    ptt.setAttribute("aria-disabled", "true");
    clear(micNoteBody);
    micNoteBody.appendChild(el("div", "text", micExplanation(problem)));
    micNote.hidden = false;
  }

  /* --------------------------------------------------------------- tasks -- */
  function buildTree(nodes, depth) {
    var list = el("ul", "tree");
    for (var i = 0; i < nodes.length; i++) {
      var node = nodes[i] || {};
      var item = el("li");
      var row = el("div", "task");
      row.appendChild(el("span", "state s-" + slug(node.state), node.state || "?"));
      row.appendChild(el("span", "goal", node.goal || node.task_id || "task"));
      item.appendChild(row);
      var kids = node.children;
      if (kids && kids.length && depth < 8) { item.appendChild(buildTree(kids, depth + 1)); }
      list.appendChild(item);
    }
    return list;
  }

  /* The server sends a FLAT list whose nesting is implied by parent_id, so a
     renderer that only reads .children draws every subagent at the top level
     and the tree is not a tree.  Rebuild the nesting when it is only implied,
     and leave an already-nested payload alone. */
  function nest(nodes) {
    var i;
    for (i = 0; i < nodes.length; i++) {
      if (nodes[i] && nodes[i].children) { return nodes; }
    }
    var byId = {};
    var order = [];
    for (i = 0; i < nodes.length; i++) {
      var raw = nodes[i] || {};
      var key = raw.task_id || "#" + i;
      var wrapped = {
        task_id: raw.task_id,
        goal: raw.goal,
        state: raw.state,
        parent_id: raw.parent_id,
        children: []
      };
      byId[key] = wrapped;
      order.push(wrapped);
    }
    var roots = [];
    for (i = 0; i < order.length; i++) {
      var node = order[i];
      var parent = node.parent_id ? byId[node.parent_id] : null;
      if (parent && parent !== node) { parent.children.push(node); }
      else { roots.push(node); }
    }
    /* Every node claiming a parent means the ids form a cycle. Show the flat
       list rather than an empty panel. */
    return roots.length ? roots : order;
  }

  function refreshTasks() {
    if (!FEAT.tasks || !taskTree) { return; }
    apiJson(API.tasks).then(function (data) {
      var nodes = data;
      if (data && !(data instanceof Array)) { nodes = data.tasks || data.tree || []; }
      if (!(nodes instanceof Array)) { nodes = []; }
      var total = nodes.length;
      clear(taskTree);
      if (!total) {
        taskTree.appendChild(el("p", "fine", "No agent tasks running."));
      } else {
        taskTree.appendChild(buildTree(nest(nodes), 0));
      }
      /* The count is every task, not just the roots: a subagent still counts. */
      tasksToggle.textContent = total ? "Tasks (" + total + ")" : "Tasks";
    }).catch(quiet);
  }

  if (tasksToggle) {
    tasksToggle.addEventListener("click", function () {
      var open = tasksPanel.hidden;
      tasksPanel.hidden = !open;
      tasksToggle.setAttribute("aria-expanded", open ? "true" : "false");
      if (open) { refreshTasks(); }
    });
  }

  $("notice-x").addEventListener("click", function () {
    notice.hidden = true;
    clear(noticeBody);
  });

  /* ---------------------------------------------------------------- boot -- */
  var started = false;

  function start() {
    if (started) { return; }
    started = true;
    showMicState();
    refreshStatus();
    refreshTasks();
    connect();
    window.setInterval(refreshTasks, TASK_POLL_MS);
    autosize();
    input.focus();
  }

  /* Any first gesture is a chance to arm playback, so a typed question can be
     answered out loud without a second tap. */
  document.addEventListener("pointerdown", armAudio, true);
  document.addEventListener("keydown", armAudio, true);

  if (gate && !gate.hidden) {
    var field = $("gate-token");
    if (field) { field.focus(); }
  } else {
    start();
  }
})();
"""

_JS = _JS.replace("__JARVIS_SILENT_WAV__", _SILENT_WAV)


# --------------------------------------------------------------------------- #
#  Markup
# --------------------------------------------------------------------------- #
_LOGIN_HTML = """
<div id="gate">
  <!-- method/action matter only if the script never wired up the submit
       handler: a form with neither defaults to GET, which would put the access
       token in the address bar, the browser history and every proxy log. -->
  <form id="gate-form" autocomplete="off" method="post" action="/api/login">
    <h1>Locked</h1>
    <p>This JARVIS is reachable from the network, so it wants the access token
    printed in its console. The token decides who may connect; it does not
    limit what you can do once you are in.</p>
    <label for="gate-token">Access token</label>
    <input id="gate-token" name="token" type="password" autocomplete="off"
           autocapitalize="off" autocorrect="off" spellcheck="false"
           inputmode="text" required />
    <button id="gate-go" type="submit">Unlock</button>
    <p id="gate-error" hidden></p>
    <p class="fine">Exchanged for a session cookie. Never written to browser storage.</p>
  </form>
</div>
"""

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
<meta name="color-scheme" content="dark light" />
<meta name="referrer" content="same-origin" />
<meta name="robots" content="noindex, nofollow" />
<title>__JARVIS_TITLE__</title>
<style>
__JARVIS_CSS__
</style>
</head>
<body>
<script id="jarvis-boot" type="application/json">__JARVIS_BOOT__</script>
<div id="app">
  <header id="bar">
    <div id="brand">__JARVIS_TITLE__</div>
    <div id="conn-wrap">
      <span id="conn-dot" class="dot"></span>
      <span id="conn-text">starting</span>
    </div>
    <button id="tasks-toggle" type="button" class="ghost" aria-expanded="false"
            aria-controls="tasks">Tasks</button>
  </header>
  <p id="meta">connecting</p>
  <div id="micnote" class="panel" hidden>
    <div id="micnote-body"></div>
  </div>
  <div id="notice" class="panel" hidden>
    <div id="notice-body"></div>
    <button id="notice-x" type="button" class="ghost">Dismiss</button>
  </div>
  <section id="tasks" hidden aria-label="Agent tasks">
    <h2>Agent tasks</h2>
    <div id="task-tree"></div>
  </section>
  <main id="log" role="log" aria-live="polite" aria-label="Conversation"></main>
  <footer id="composer">
    <button id="ptt" type="button" aria-label="Push to talk">Hold to talk</button>
    <textarea id="input" rows="1" placeholder="Ask JARVIS. Enter sends, Shift+Enter for a new line."
              aria-label="Message" autocapitalize="sentences" autocorrect="on"></textarea>
    <button id="send" type="button">Send</button>
  </footer>
</div>
__JARVIS_LOGIN__
<audio id="player" preload="none"></audio>
<script>
__JARVIS_JS__
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
#  Rendering
# --------------------------------------------------------------------------- #
def _strip_placeholders(value: Any) -> str:
    """Remove anything shaped like a template slot from caller-supplied text.

    The title and the model name come from configuration, and configuration is
    not trusted to avoid a string this module happens to substitute on.
    """
    return _PLACEHOLDER_RE.sub("", "" if value is None else str(value))


def _normalise_features(features: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Coerce whatever the server passed into the shape the page expects."""
    resolved: Dict[str, Any] = dict(DEFAULT_FEATURES)
    if features:
        try:
            supplied = dict(features)
        except (TypeError, ValueError):
            log.warning(
                "features must be a mapping, got %s; using defaults",
                type(features).__name__,
            )
            supplied = {}
        for key, value in supplied.items():
            resolved[str(key)] = value
    for key in _BOOL_FEATURES:
        resolved[key] = bool(resolved.get(key))
    for key in _TEXT_FEATURES:
        resolved[key] = _strip_placeholders(resolved.get(key, ""))
    return resolved


def _boot_json(payload: Dict[str, Any]) -> str:
    """Serialise the boot record for a ``<script type="application/json">``.

    ``ensure_ascii`` keeps the page byte-for-byte ASCII, and the angle brackets
    are escaped so no value can end the script element early.
    """
    try:
        raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        log.warning("boot payload was not serialisable; falling back to defaults")
        raw = json.dumps(
            {"title": DEFAULT_TITLE, "tls": False, "features": dict(DEFAULT_FEATURES)},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    return raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render_page(
    *,
    title: str = DEFAULT_TITLE,
    has_tls: bool = False,
    features: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return the complete single-page client as one HTML string.

    :param title: shown in the tab and the header; escaped before insertion.
    :param has_tls: whether the server is speaking TLS.  Recorded in the boot
        record so the page can be honest about why the microphone may be
        missing — the browser's own ``isSecureContext`` remains the authority.
    :param features: what the server can actually do, e.g.
        ``{"auth_required": True, "stt": True, "tts": False, "model": "..."}``.
        Unknown keys are passed through to the page; missing ones fall back to
        :data:`DEFAULT_FEATURES`.  A token screen is rendered only when
        ``auth_required`` is true.

    Never raises: an unusable ``features`` argument is logged and ignored.
    """
    safe_title = html.escape(_strip_placeholders(title) or DEFAULT_TITLE, quote=True)
    resolved = _normalise_features(features)
    boot = {
        "title": html.unescape(safe_title),
        "tls": bool(has_tls),
        "features": resolved,
    }

    page = _TEMPLATE
    page = page.replace("__JARVIS_CSS__", _CSS)
    page = page.replace("__JARVIS_JS__", _JS)
    page = page.replace("__JARVIS_BOOT__", _boot_json(boot))
    page = page.replace("__JARVIS_LOGIN__", _LOGIN_HTML if resolved["auth_required"] else "")
    # Last, so a title containing a placeholder-shaped string cannot expand.
    page = page.replace("__JARVIS_TITLE__", safe_title)
    return page


def render_page_bytes(
    *,
    title: str = DEFAULT_TITLE,
    has_tls: bool = False,
    features: Optional[Mapping[str, Any]] = None,
) -> bytes:
    """:func:`render_page` encoded as UTF-8, ready for a ``wfile.write``."""
    return render_page(title=title, has_tls=has_tls, features=features).encode("utf-8")


#: Values an engine slot takes when the engine did not load.
_ABSENT = frozenset({"", "none", "null", "unavailable", "unknown", "disabled"})


def _engine_present(value: Any, default: bool = False) -> bool:
    """Read a status slot that may be a bool, an engine name, or missing."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() not in _ABSENT


def index_html(status: Optional[Mapping[str, Any]] = None) -> str:
    """Render the page from an ``/api/status`` payload.

    This is the hook the server looks for by name, and it takes the status dict
    *positionally* — :func:`render_page` is keyword-only, so calling it with a
    payload raises :class:`TypeError` and the server would quietly fall back to
    a default page.  That fallback is not harmless: ``auth_required`` decides
    whether the token screen exists in the markup at all, so losing it means a
    LAN session never gets asked for the token.

    The gate is rendered only when a token is required *and* this request did
    not already arrive with a valid session cookie.

    Never raises; an unusable payload yields the default page.
    """
    try:
        payload: Dict[str, Any] = dict(status or {})
    except (TypeError, ValueError):
        log.warning("status payload was not a mapping; rendering the default page")
        payload = {}

    needs_token = bool(payload.get("auth_required")) and not bool(payload.get("authenticated"))
    features: Dict[str, Any] = {
        "auth_required": needs_token,
        "chat": True,
        "stream": True,
        "tasks": bool(payload.get("orchestrator", True)),
        "stt": _engine_present(payload.get("stt_available", payload.get("stt"))),
        "tts": _engine_present(payload.get("tts")),
        "model": payload.get("model") or "unknown",
        "backend": payload.get("llm") or payload.get("backend") or "unknown",
        "version": payload.get("version") or "",
    }
    return render_page(
        title=str(payload.get("name") or DEFAULT_TITLE),
        has_tls=bool(payload.get("tls")),
        features=features,
    )


#: The page with default settings — no auth, no speech, unknown model.
PAGE = render_page()
