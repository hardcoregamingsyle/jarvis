"""Configurable access policy for JARVIS — switched off by default.

The rest of the system asks this module three questions before it acts:

* ``check_path(p, write=...)``  — may I read/write this location?
* ``check_command(cmd)``        — may I run this shell command?
* ``check_tool(spec, kwargs)``  — may I invoke this tool with these args?

Each returns a :class:`Decision`, which callers feed through
:meth:`SecurityGate.allows` — the single funnel handling confirmation
callbacks and the unattended policy.

**Out of the box every one of those questions is answered "yes".** The default
mode is ``"open"``: no path is protected, no command is dangerous, nothing ever
prompts.  This is a policy *engine*, not a policy — it enforces only what the
config tells it to, and the shipped config tells it nothing.

Two modes exist for anyone who wants restrictions.  ``"guarded"`` asks for
confirmation before commands matching ``cfg.dangerous_patterns`` or a known
destructive shape, and refuses writes under ``cfg.protected_paths``.
``"readonly"`` refuses mutation outright.  Both lists default to empty, so even
those modes start permissive until configured.

Independently of mode, every decision can be journalled as JSON-lines for
audit.  The journal records; it never refuses.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

from .config import SecurityConfig
from .contracts import ToolSpec, now_ts
from .platform_utils import IS_WINDOWS, data_dir, ensure_dir, is_drive_relative

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Decision
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    """The verdict returned by every :class:`SecurityGate` check.

    ``bool(decision)`` is the fast-path for "may I proceed with no further
    ceremony?" — it is true only when the action is both allowed and does not
    require confirmation.  Callers that can wait for a human should route
    through :meth:`SecurityGate.allows` instead.
    """

    allowed: bool
    reason: str = ""
    requires_confirmation: bool = False
    destructive: bool = False

    def __bool__(self) -> bool:
        return bool(self.allowed) and not bool(self.requires_confirmation)


# --------------------------------------------------------------------------- #
#  Command-shape detectors
# --------------------------------------------------------------------------- #
# Genuinely destructive shell shapes.  A match sets destructive=True *and*
# requires_confirmation=True — the gate never runs these unattended.
_FORK_BOMB = re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:")
_MKFS = re.compile(r"\bmkfs(?:\.[a-zA-Z0-9]+)?\b", re.IGNORECASE)
_FORMAT_DRIVE = re.compile(r"\bformat(?:\.com)?\s+/?[A-Za-z]:\S*", re.IGNORECASE)
_DISKPART = re.compile(r"\bdiskpart\b", re.IGNORECASE)
_DD_TO_DEV = re.compile(r"\bdd\b[^\r\n|;]*\bof=/dev/\S+", re.IGNORECASE)
# curl/wget/Invoke-WebRequest ... | (sh|bash|zsh|pwsh|powershell|iex|python|...)
_PIPE_TO_SHELL = re.compile(
    r"\b(?:curl|wget|fetch|Invoke-WebRequest|iwr)\b[^\r\n|]*\|\s*"
    r"(?:sudo\s+)?(?:sh|bash|zsh|dash|ksh|fish|pwsh|powershell|cmd(?:\.exe)?|"
    r"iex|Invoke-Expression|python|python3|perl|ruby|node)\b",
    re.IGNORECASE,
)

# `rm -rf <root>` and family, including PowerShell / cmd equivalents.
_RM_INVOCATION = re.compile(
    r"(?<![A-Za-z0-9_])(?:sudo\s+)?rm\s+((?:-\S+\s+)+)(\S+)", re.IGNORECASE
)
_DEL_DRIVE_ROOT = re.compile(
    r"\b(?:del|erase)\s+(?:/[a-zA-Z]\s*)*[A-Za-z]:[\\/]?\s*\*?\s*$",
    re.IGNORECASE,
)
_REMOVE_ITEM_ROOT = re.compile(
    r"\bRemove-Item\b[^\r\n|;]*?-Recurse\b[^\r\n|;]*?-Force\b[^\r\n|;]*?"
    r"[\"']?[A-Za-z]:[\\/]?[\"']?\s*(?:\*)?\s*(?:$|[;|])",
    re.IGNORECASE,
)
_REMOVE_ITEM_ROOT_ALT = re.compile(
    r"\bRemove-Item\b[^\r\n|;]*?-Force\b[^\r\n|;]*?-Recurse\b[^\r\n|;]*?"
    r"[\"']?[A-Za-z]:[\\/]?[\"']?\s*(?:\*)?\s*(?:$|[;|])",
    re.IGNORECASE,
)

# Root-shaped target for `rm`: /, /*, C:, C:\, C:/, C:\*
_ROOT_TARGET = re.compile(r"^(?:/\*?|[A-Za-z]:[\\/]?\*?)$")

# Mutating shell tokens.  Not exhaustive — used only to fence off readonly
# mode from obviously write-shaped commands.  Inspection tokens live below.
_MUTATING_TOKENS = frozenset({
    "rm", "rmdir", "rd", "del", "erase", "mv", "move", "cp", "copy",
    "chmod", "chown", "chgrp", "kill", "killall", "taskkill", "pkill",
    "shutdown", "reboot", "poweroff", "halt", "logoff",
    "format", "mkfs", "dd", "diskpart", "fdisk", "parted", "wipefs",
    "apt", "apt-get", "yum", "dnf", "pacman", "brew", "snap", "flatpak",
    "pip", "pip3", "npm", "yarn", "gem", "cargo",
    "systemctl", "service", "sc", "net",
    "remove-item", "move-item", "new-item", "set-content", "add-content",
    "out-file", "copy-item", "rename-item", "clear-content", "start-process",
    "stop-process", "stop-service", "start-service", "restart-service",
    "set-itemproperty", "new-itemproperty", "remove-itemproperty",
    "invoke-expression", "iex", "invoke-webrequest", "invoke-restmethod",
})
_INSPECT_TOKENS = frozenset({
    "ls", "dir", "cat", "type", "more", "less", "head", "tail",
    "ps", "whoami", "echo", "pwd", "hostname", "date", "uname",
    "which", "where", "find", "grep", "select-string",
    "wc", "du", "df", "netstat", "ip", "ifconfig", "printenv", "env",
    "get-childitem", "get-content", "get-process", "get-location",
    "get-service", "get-command", "get-help", "get-item", "test-path",
})

# Redirection operators that write to a file/stream.
_WRITE_OPS = ("|tee", ">", ">>")

# Keys in tool kwargs that name a filesystem path.
_PATH_KEYS = frozenset({
    "path", "paths", "file", "files", "filename", "file_path", "filepath",
    "target", "targets", "directory", "dir", "folder",
    "src", "source", "sources", "input", "input_path", "input_file",
    "dst", "dest", "destination", "output", "output_path", "output_file", "out",
})
# Kwarg keys whose value should be run through :meth:`SecurityGate.check_command`.
_COMMAND_KEYS = frozenset({"command", "cmd", "script", "shell", "script_body"})
# Kwarg keys that always mean "write" for path checks.
_WRITE_KEYS = frozenset({
    "dst", "dest", "destination", "output", "output_path", "output_file", "out",
})


def _strip_quoted(text: str) -> str:
    """Blank out single- and double-quoted spans so operators inside don't leak.

    A quoted ``">"`` in ``echo "1 > 2"`` must not be counted as a real redirect;
    but the outer string must keep its length so span indexes still line up.
    """
    out = list(text)
    i = 0
    n = len(out)
    while i < n:
        ch = out[i]
        if ch in ("'", '"'):
            quote = ch
            out[i] = " "
            j = i + 1
            while j < n and out[j] != quote:
                out[j] = " "
                j += 1
            if j < n:
                out[j] = " "
                i = j + 1
            else:
                # Unterminated quote — leave the rest untouched.
                break
        else:
            i += 1
    return "".join(out)


def _first_token(cmd: str) -> str:
    """Return the first bare token of a stripped command, lowercased."""
    stripped = cmd.lstrip()
    if not stripped:
        return ""
    m = re.match(r"([^\s|&;<>()]+)", stripped)
    return m.group(1).lower() if m else ""


def _split_pipeline(cmd: str) -> list:
    """Split a stripped command on `|`, `&&`, `||`, `;` — returns raw segments."""
    return [seg.strip() for seg in re.split(r"\|\||&&|[|;&]", cmd) if seg.strip()]


def _rm_root_match(cmd: str) -> bool:
    """True if ``cmd`` is a ``rm`` invocation that recursively force-deletes a root."""
    for m in _RM_INVOCATION.finditer(cmd):
        flags = "".join(re.findall(r"-([a-zA-Z]+)", m.group(1)))
        if not flags:
            continue
        low = flags.lower()
        if "r" not in low or "f" not in low:
            continue
        target = m.group(2).strip("\"'")
        if _ROOT_TARGET.match(target):
            return True
    return False


def _find_destructive(cmd: str) -> Optional[str]:
    """Return a human-readable reason if ``cmd`` matches a destructive shape."""
    if _FORK_BOMB.search(cmd):
        return "fork bomb"
    if _MKFS.search(cmd):
        return "disk format (mkfs)"
    if _FORMAT_DRIVE.search(cmd):
        return "disk format (format command)"
    if _DISKPART.search(cmd):
        return "diskpart invocation"
    if _DD_TO_DEV.search(cmd):
        return "dd writing to a device"
    if _PIPE_TO_SHELL.search(cmd):
        return "download piped into a shell interpreter"
    if _rm_root_match(cmd):
        return "recursive force delete of a filesystem root"
    if _DEL_DRIVE_ROOT.search(cmd):
        return "recursive force delete of a Windows drive root"
    if _REMOVE_ITEM_ROOT.search(cmd) or _REMOVE_ITEM_ROOT_ALT.search(cmd):
        return "recursive force delete of a Windows drive root"
    return None


# --------------------------------------------------------------------------- #
#  Path normalisation
# --------------------------------------------------------------------------- #
def _applies_to_host(raw: str) -> bool:
    """True when a protected-path entry belongs to the running OS.

    ``"/etc"`` on Windows would ``abspath`` to ``C:\\etc`` and start protecting
    an unrelated tree; ``"C:\\Windows"`` on Linux would land under the CWD.
    Filter those out before any comparison happens.
    """
    if not isinstance(raw, str):
        return False
    s = raw.strip()
    if not s:
        return False
    windows_style = s.startswith("\\\\") or (
        len(s) >= 2 and s[1] == ":" and s[0].isalpha()
    )
    posix_style = s.startswith("/")
    if IS_WINDOWS:
        return windows_style
    return posix_style


def _normalise_parts(raw: Any) -> Optional[Tuple[str, ...]]:
    """Expand, absolutise, realpath a path and return its components.

    Returns ``None`` when the input cannot be interpreted as a path at all.
    Components are lower-cased on Windows so comparison is case-insensitive
    there; POSIX comparison is always case-sensitive.
    """
    if isinstance(raw, Path):
        raw = str(raw)
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        expanded = os.path.expanduser(text)
        absolute = os.path.abspath(expanded)
        try:
            real = os.path.realpath(absolute)
        except (OSError, ValueError):
            real = absolute
    except (OSError, ValueError, TypeError):
        return None
    parts = Path(real).parts
    if not parts:
        return None
    if IS_WINDOWS:
        return tuple(p.lower() for p in parts)
    return tuple(parts)


def _starts_with(candidate: Tuple[str, ...], prefix: Tuple[str, ...]) -> bool:
    """True when ``candidate`` component tuple begins with ``prefix``."""
    if not prefix or len(prefix) > len(candidate):
        return False
    return candidate[: len(prefix)] == prefix


# --------------------------------------------------------------------------- #
#  Gate
# --------------------------------------------------------------------------- #
ConfirmFn = Callable[[str], bool]


class SecurityGate:
    """Central permission oracle for JARVIS.

    In the default ``"open"`` mode every check short-circuits to "allowed, no
    confirmation" — see :meth:`_open_mode`.  The classification machinery below
    it only runs for the opt-in ``"guarded"`` and ``"readonly"`` modes.

    The gate is stateless with respect to config values — it re-reads
    ``cfg.mode`` and friends on every call so tests can toggle policy without
    re-instantiating.  Audit writes are serialised across threads and never
    raise into the caller: a broken sink disables itself and logs once.
    """

    def __init__(
        self,
        cfg: SecurityConfig,
        audit_path: Optional[Any] = None,
        confirm: Optional[ConfirmFn] = None,
    ) -> None:
        self.cfg = cfg
        self._audit_path = Path(audit_path) if audit_path else None
        self._audit_lock = threading.Lock()
        self._audit_broken = False
        self._confirm: Optional[ConfirmFn] = confirm

    # ------------------------------------------------------------------ #
    #  Callback wiring
    # ------------------------------------------------------------------ #
    def set_confirm_callback(self, fn: Optional[ConfirmFn]) -> None:
        """Install (or clear with ``None``) the interactive confirmation hook."""
        self._confirm = fn

    # ------------------------------------------------------------------ #
    #  Mode
    # ------------------------------------------------------------------ #
    def _mode(self) -> str:
        """The configured mode, lowercased.  Unset means the permissive default."""
        mode = self.cfg.mode
        if not isinstance(mode, str) or not mode.strip():
            return "open"
        return mode.strip().lower()

    @staticmethod
    def _open_mode(what: str) -> Decision:
        """The verdict every check returns in ``"open"`` mode: unconditional yes.

        Open means open.  No protected paths, no destructive-shape carve-outs,
        no confirmation — a mode that secretly refused things would make the
        setting a lie.
        """
        return Decision(
            allowed=True,
            reason=f"{what} allowed (open mode)",
            requires_confirmation=False,
        )

    # ------------------------------------------------------------------ #
    #  Protected-path membership
    # ------------------------------------------------------------------ #
    def _protected_prefixes(self) -> list:
        prefixes = []
        raw_list = self.cfg.protected_paths
        if not raw_list:
            return prefixes
        try:
            entries = list(raw_list)
        except TypeError:
            return prefixes
        for raw in entries:
            if not _applies_to_host(raw):
                continue
            parts = _normalise_parts(raw)
            if parts is not None:
                prefixes.append(parts)
        return prefixes

    def is_protected(self, path: Any) -> bool:
        """True when ``path`` lies inside a configured protected root.

        ``cfg.protected_paths`` is empty by default, so this is ``False`` for
        every input unless someone has opted in.  Odd input (``None``, numbers,
        lists, unterminated garbage) yields ``False`` rather than raising.
        """
        prefixes = self._protected_prefixes()
        if not prefixes:
            return False
        candidate = _normalise_parts(path)
        if candidate is None:
            return False
        for prefix in prefixes:
            if _starts_with(candidate, prefix):
                return True
        return False

    # ------------------------------------------------------------------ #
    #  Path check
    # ------------------------------------------------------------------ #
    def check_path(self, path: Any, write: bool = False) -> Decision:
        """Decide whether the agent may access ``path`` for read or write.

        Always allowed in the default ``"open"`` mode, with the single
        exception of ``cfg.allow_file_write=False`` — an explicitly disabled
        capability is not something a mode gets to re-enable.  That switch
        defaults to ``True``, so nothing is blocked out of the box.
        """
        mode = self._mode()
        if write and not self.cfg.allow_file_write:
            return Decision(allowed=False, reason="file writes are disabled by policy")
        if mode == "open":
            return self._open_mode("path")

        if isinstance(path, str) and is_drive_relative(path):
            return Decision(
                allowed=False,
                reason=(
                    f"{path!r} is a drive-relative path (means 'current directory on "
                    f"that drive'); refuse before it resolves to the wrong place"
                ),
            )
        if not isinstance(path, (str, Path)) or (isinstance(path, str) and not path.strip()):
            return Decision(allowed=False, reason="empty or non-string path")

        protected = self.is_protected(path)

        if write:
            if mode == "readonly":
                return Decision(allowed=False, reason="readonly mode: writes are refused")
            if not self.cfg.allow_file_write:
                return Decision(allowed=False, reason="file writes are disabled by policy")
            if protected:
                return Decision(
                    allowed=False,
                    reason=f"{path!r} is inside a protected system location",
                    destructive=True,
                )
            return Decision(allowed=True, reason="write allowed")

        # Read path.
        if protected and mode == "readonly":
            return Decision(
                allowed=False,
                reason=f"{path!r} is protected and readonly mode denies even reads",
            )
        return Decision(allowed=True, reason="read allowed")

    # ------------------------------------------------------------------ #
    #  Command check
    # ------------------------------------------------------------------ #
    def check_command(self, cmd: Any) -> Decision:
        """Decide whether the agent may run ``cmd`` in a shell.

        Always allowed in the default ``"open"`` mode — including shapes the
        guarded-mode classifier below would flag as destructive.  The one thing
        open mode does not override is ``cfg.allow_shell``: that switch is not
        mode policy, it is the operator saying this capability is off, and a
        mode must not quietly hand back a capability someone disabled.  It
        defaults to ``True``, so nothing is blocked out of the box.
        """
        if not self.cfg.allow_shell:
            return Decision(allowed=False, reason="shell execution is disabled by policy")

        mode = self._mode()
        if mode == "open":
            return self._open_mode("command")

        if not isinstance(cmd, str) or not cmd.strip():
            return Decision(allowed=False, reason="empty or non-string command")

        raw = cmd
        stripped = _strip_quoted(cmd)

        destructive_reason = _find_destructive(stripped)

        # Configured dangerous patterns (matched on the raw string so patterns
        # containing quotes still match).
        dangerous_match: Optional[str] = None
        for pat in self.cfg.dangerous_patterns or ():
            if not isinstance(pat, str) or not pat:
                continue
            if pat.lower() in raw.lower():
                dangerous_match = pat
                break

        # Readonly mode: allow only inspection commands, deny mutations.
        if mode == "readonly":
            if destructive_reason:
                return Decision(
                    allowed=False,
                    reason=f"readonly mode: destructive command ({destructive_reason})",
                    destructive=True,
                )
            if dangerous_match:
                return Decision(
                    allowed=False,
                    reason=f"readonly mode: matches dangerous pattern {dangerous_match!r}",
                )
            if self._is_mutating(stripped):
                return Decision(
                    allowed=False,
                    reason="readonly mode: mutating command refused",
                )
            return Decision(allowed=True, reason="readonly: inspection command allowed")

        # Non-readonly modes.
        if destructive_reason:
            return Decision(
                allowed=True,
                reason=f"destructive: {destructive_reason}",
                requires_confirmation=True,
                destructive=True,
            )
        if dangerous_match:
            return Decision(
                allowed=True,
                reason=f"matches dangerous pattern {dangerous_match!r}",
                requires_confirmation=True,
            )
        if mode == "guarded" and self._is_mutating(stripped):
            # Guarded mutations pass without confirmation — they are not
            # destructive; the gate exists to raise the bar, not to nag.
            return Decision(allowed=True, reason="mutating command allowed (guarded)")
        return Decision(allowed=True, reason="command allowed")

    @staticmethod
    def _is_mutating(stripped: str) -> bool:
        """Rough heuristic: does this command mutate the system?"""
        for op in _WRITE_OPS:
            if op in stripped:
                return True
        for segment in _split_pipeline(stripped):
            token = _first_token(segment)
            if not token:
                continue
            # Strip a leading directory (e.g. "/usr/bin/rm" -> "rm").
            base = token.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            if base in _MUTATING_TOKENS:
                return True
            if base.startswith("./") or base.startswith(".\\"):
                # Running an arbitrary local script is treated as mutating.
                return True
        return False

    # ------------------------------------------------------------------ #
    #  Tool check
    # ------------------------------------------------------------------ #
    def check_tool(self, spec: ToolSpec, kwargs: dict) -> Decision:
        """Decide whether ``spec`` may be invoked with the given ``kwargs``.

        Always allowed in the default ``"open"`` mode, whatever the tool is
        marked and whatever it was handed.  In the opt-in modes a denial from
        any kwarg wins outright, while confirmation and destructive flags
        accumulate across the kwargs and the tool's own ``dangerous`` marker;
        the final decision names the strongest reason.
        """
        mode = self._mode()
        if mode == "open":
            self.audit("tool", f"{getattr(spec, 'name', '?')} (open mode)", True)
            return self._open_mode("tool call")

        reasons: list = []
        requires = False
        destructive = False

        if spec is not None and getattr(spec, "dangerous", False):
            if mode == "readonly":
                return Decision(
                    allowed=False,
                    reason=f"readonly mode: tool {spec.name!r} is marked dangerous",
                )
            reasons.append(f"tool {spec.name!r} is marked dangerous")
            requires = True

        kw = kwargs or {}
        for key, value in kw.items():
            lkey = key.lower() if isinstance(key, str) else ""
            if lkey in _PATH_KEYS and isinstance(value, (str, Path)):
                write = lkey in _WRITE_KEYS or (spec is not None and getattr(spec, "dangerous", False))
                sub = self.check_path(value, write=write)
                if not sub.allowed:
                    return Decision(
                        allowed=False,
                        reason=f"{key}={value!r}: {sub.reason}",
                        destructive=sub.destructive or destructive,
                    )
                if sub.requires_confirmation:
                    requires = True
                    reasons.append(f"{key}={value!r}: {sub.reason}")
                if sub.destructive:
                    destructive = True
            elif lkey in _COMMAND_KEYS and isinstance(value, str):
                sub = self.check_command(value)
                if not sub.allowed:
                    return Decision(
                        allowed=False,
                        reason=f"{key}={value!r}: {sub.reason}",
                        destructive=sub.destructive or destructive,
                    )
                if sub.requires_confirmation:
                    requires = True
                    reasons.append(f"{key}={value!r}: {sub.reason}")
                if sub.destructive:
                    destructive = True

        return Decision(
            allowed=True,
            reason="; ".join(reasons) if reasons else "tool call allowed",
            requires_confirmation=requires,
            destructive=destructive,
        )

    # ------------------------------------------------------------------ #
    #  Allowance funnel + confirmation
    # ------------------------------------------------------------------ #
    def confirm(self, message_or_decision: Union[str, Decision]) -> bool:
        """Ask for confirmation, honouring the configured unattended policy.

        With an interactive callback wired up, the callback's return value is
        authoritative (a raising callback counts as ``False`` — never proceed
        past an error).  With no callback the answer is ``cfg.unattended_policy``:
        ``"allow"`` (the default) proceeds and audits, ``"deny"`` refuses.
        Destructiveness carries no special weight here; a caller that wants
        irreversible actions held back sets ``unattended_policy="deny"``.
        """
        if isinstance(message_or_decision, Decision):
            message = message_or_decision.reason or "confirmation requested"
        else:
            message = str(message_or_decision)

        callback = self._confirm
        if callback is not None:
            try:
                return bool(callback(message))
            except Exception:  # noqa: BLE001 - a raising prompt is a "no".
                log.warning("confirm callback raised; treating as denial")
                return False

        policy = self.cfg.unattended_policy
        policy = policy.strip().lower() if isinstance(policy, str) and policy.strip() else "allow"
        if policy == "deny":
            self.audit("unattended-denied", message, False)
            return False

        self.audit("unattended-approved", message, True)
        return True

    def allows(self, decision: Decision) -> bool:
        """Return True when ``decision`` should proceed, else False."""
        if not decision.allowed:
            return False
        if not decision.requires_confirmation:
            return True
        return self.confirm(decision)

    # ------------------------------------------------------------------ #
    #  Audit sink
    # ------------------------------------------------------------------ #
    def _resolve_audit_path(self) -> Optional[Path]:
        if not self.cfg.audit_log:
            return None
        if self._audit_broken:
            return None
        if self._audit_path is not None:
            return self._audit_path
        # Default: <data_dir>/logs/audit.jsonl.
        return data_dir() / "logs" / "audit.jsonl"

    def audit(self, action: str, detail: str, allowed: bool) -> None:
        """Append a JSON-lines audit record.  Never raises.

        A single write failure disables the sink for the rest of the process
        so a broken disk cannot poison every future check with log spam.
        """
        path = self._resolve_audit_path()
        if path is None:
            return
        record = {
            "ts": now_ts(),
            "action": str(action),
            "detail": str(detail),
            "allowed": bool(allowed),
        }
        try:
            line = json.dumps(record, ensure_ascii=False)
        except (TypeError, ValueError):
            try:
                line = json.dumps(
                    {"ts": now_ts(), "action": str(action),
                     "detail": repr(detail), "allowed": bool(allowed)},
                    ensure_ascii=False,
                )
            except Exception:  # noqa: BLE001
                return
        with self._audit_lock:
            try:
                ensure_dir(path.parent)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError as exc:
                if not self._audit_broken:
                    log.warning("audit log unavailable, disabling: %s", exc)
                    self._audit_broken = True


__all__ = ["Decision", "SecurityGate"]
