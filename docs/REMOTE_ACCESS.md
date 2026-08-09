# Remote access

How to talk to JARVIS from a different device while JARVIS itself never leaves
the machine it is installed on.

The brain, the tools, the memory, the model and the machine control all stay on
the Linux box. The other device — laptop, desktop, phone, tablet, Linux, macOS,
Windows, iOS or Android — runs **nothing but a browser**. There is no client to
install, no agent, no port on the router.

```bash
# on the JARVIS box
jarvis serve

# on the other device
ssh -N -L 8765:127.0.0.1:8765 user@jarvis-box
# then open http://localhost:8765
```

That is the whole answer. The rest of this document explains why that particular
shape, and what the alternatives cost.

---

## 1. Where the microphone is

This is the one thing to understand before anything else, because it determines
every other decision here.

The microphone and speakers wired into JARVIS are attached to **the Linux box**.
If you sit down at a laptop in another room and JARVIS listens on its own
microphone, you are shouting at an empty room and it is hearing that room, not
you.

So the web client does not use the server's audio at all:

| Step | Runs where |
|------|-----------|
| Capture your voice (`getUserMedia` + `MediaRecorder`) | **the browser, on your device** |
| Upload the recorded clip | over the connection |
| Speech to text, the model, tools, memory, machine control | the JARVIS box |
| Synthesise the reply | the JARVIS box |
| **Play the reply** | **the browser, on your device** |

A remote session never opens the server's microphone or speakers. `jarvis voice`
in a terminal on the box still does — that is the local, in-the-room mode, and it
is a different thing (see §5).

The direct consequence: **the browser must be allowed to open your microphone**,
and browsers only allow that on a *secure context*. That single rule is why SSH
forwarding is the recommended route rather than merely the paranoid one. §6.3 is
the troubleshooting entry for it; read it now if you like, it will save you an
afternoon.

---

## 2. Choosing a transport

| Transport | Works from | Needs on the client | Microphone works? | Exposure |
|---|---|---|---|---|
| **SSH port forward** (recommended) | anywhere you can SSH — same LAN, or over the internet if SSH is reachable | an `ssh` client (built into Linux, macOS, Windows 10+) | **Yes** — the page is `localhost` on your device, which browsers trust without TLS | Nothing. JARVIS stays bound to `127.0.0.1`; only SSH is on the network |
| **Tailscale / WireGuard** | anywhere on the internet, no router config | Tailscale app (or a WireGuard client) | Yes with `tailscale cert`/`serve` (real HTTPS); otherwise no — see §4 | Only devices in your tailnet. No public port |
| **Plain LAN bind** (`--host 0.0.0.0`) | same LAN / Wi-Fi only | nothing but a browser | **No**, unless you supply TLS — plain `http://192.168.x.x` is not a secure context | Every device on the network can reach the port; the token is the only thing between them and the machine |
| **Terminal over SSH** (`jarvis chat`) | anywhere you can SSH | an `ssh` client | N/A — text only. `jarvis voice` here uses the *server's* mic (§5) | Nothing beyond SSH |
| **Router port forward** | the public internet | a browser | Only with real TLS | The whole internet can knock. **Don't.** Use Tailscale instead |

If you are not sure: use SSH forwarding. It is the fastest to set up, needs
nothing installed on either side, exposes nothing, and it is the only option that
makes the microphone work with no certificate work at all.

---

## 3. SSH port forwarding — the recommended default

### What it does

`ssh -L` opens a listening socket **on your device** and pipes it through the
encrypted SSH connection to a socket **on the JARVIS box**. JARVIS itself keeps
listening only on `127.0.0.1` and never appears on the network. To the browser on
your laptop, JARVIS is a service running on that laptop.

That last sentence is the whole trick. Browsers grant `getUserMedia` — the
microphone — only in a *secure context*, which means HTTPS **or** an origin the
browser treats as inherently local: `http://localhost`, `http://127.0.0.1`,
`http://[::1]`. Through the tunnel, that is exactly what your browser sees, so
the microphone works over plain HTTP with no certificate, no self-signed warning
and no browser flags. Any other route either needs a real certificate or gets a
dead microphone button.

### On the JARVIS box

```bash
jarvis serve
```

Defaults: binds `127.0.0.1:8765`. Nothing else on the network can reach it, so no
token is required (one is still generated and saved so that the moment you *do*
bind wider, you have one). To use a different port:

```bash
jarvis serve --port 9100
jarvis serve --print-url          # prints the URL and exits; for scripts
```

### On a Linux or macOS client

`ssh` is already installed.

```bash
ssh -N -L 8765:127.0.0.1:8765 user@jarvis-box
```

* `-N` — do not run a command, just hold the tunnel open.
* `-L 8765:127.0.0.1:8765` — *listen on my 8765, forward to 127.0.0.1:8765 as
  resolved on the far side*. The `127.0.0.1` is from the server's point of view,
  which is precisely why JARVIS can stay loopback-bound.

Leave that terminal open, then browse to **http://localhost:8765**.

To run it in the background instead:

```bash
ssh -f -N -L 8765:127.0.0.1:8765 user@jarvis-box    # -f: fork after auth
```

Make it survive sleep and network changes by putting this in `~/.ssh/config` on
the client:

```
Host jarvis-box
    HostName 192.168.1.42
    User you
    LocalForward 8765 127.0.0.1:8765
    ServerAliveInterval 30
    ServerAliveCountMax 3
    ExitOnForwardFailure yes
```

Then the tunnel is just `ssh -N jarvis-box`. `ExitOnForwardFailure yes` matters:
without it, ssh happily connects when the local port is already taken and you
spend ten minutes wondering why the page shows an old session.

For a tunnel that reconnects itself, `autossh -M 0 -f -N jarvis-box`.

### On a Windows client

OpenSSH ships with Windows 10 (1809+) and Windows 11 — `ssh` is on `PATH` in
PowerShell and in `cmd`. The command is identical:

```powershell
ssh -N -L 8765:127.0.0.1:8765 user@jarvis-box
```

If it reports that `ssh` is not recognised, install the optional feature:

```powershell
Get-WindowsCapability -Online -Name OpenSSH.Client*
Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
```

Then open **http://localhost:8765** in Edge, Chrome or Firefox.

`~/.ssh/config` works the same way on Windows; the file lives at
`C:\Users\<you>\.ssh\config`.

**PuTTY** is only an alternative, not a requirement — but if you prefer it:
Session → Host Name `jarvis-box`; Connection → SSH → Tunnels → Source port
`8765`, Destination `127.0.0.1:8765`, select **Local**, click **Add**; back to
Session → Open. `plink -N -L 8765:127.0.0.1:8765 user@jarvis-box` is the
command-line form.

### From a phone

Any SSH client with port forwarding does it — Termius, JuiceSSH, Blink on iOS.
The mobile browser then opens `http://localhost:8765`. Honestly, Tailscale (§4)
is less fiddly on a phone.

---

## 4. Tailscale / WireGuard — from outside the house

SSH forwarding needs SSH to be reachable. If the JARVIS box sits behind a home
router and you are on mobile data, it is not — and forwarding a port on the
router to get there is the one thing this document actively recommends against
(§7). A mesh VPN solves it without opening anything.

### Tailscale

Install on the JARVIS box and on the client (Linux, macOS, Windows, iOS,
Android):

```bash
# Linux box
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Both devices join your private tailnet and get a stable `100.x.y.z` address. With
**MagicDNS** enabled (it is by default, in the admin console under DNS) each
machine is also reachable by its hostname — `jarvis-box`, or the fully qualified
`jarvis-box.your-tailnet.ts.net` — from any other device in the tailnet,
anywhere in the world. No port forwarding, no dynamic DNS, no public exposure:
devices outside the tailnet cannot route to those addresses at all.

Once you are on the tailnet, everything in §3 works unchanged and is still the
best option:

```bash
ssh -N -L 8765:127.0.0.1:8765 user@jarvis-box
```

Tunnel over the VPN, browse to `localhost`, microphone works. Two layers, but
each is one command.

### Going straight to the box (and the HTTPS caveat)

If you would rather skip the tunnel and open `http://jarvis-box:8765` directly,
you hit the secure-context rule again: a `100.x.y.z` address or a `ts.net` name
over plain HTTP is *not* a secure context, so the microphone stays greyed out.
Text chat still works fine.

Tailscale can issue a **real, publicly-trusted certificate** for your machine's
MagicDNS name (enable HTTPS in the admin console first):

```bash
tailscale cert jarvis-box.your-tailnet.ts.net
jarvis serve --host 0.0.0.0 \
  --cert jarvis-box.your-tailnet.ts.net.crt \
  --key  jarvis-box.your-tailnet.ts.net.key
```

Now `https://jarvis-box.your-tailnet.ts.net:8765` is genuinely trusted, no
warning page, and the microphone works. Keep the token; `--host 0.0.0.0` means
the LAN can reach the port too, and TLS proves *who the server is*, not *who the
client is*.

Simpler still, `tailscale serve` puts Tailscale's own HTTPS front end over the
loopback-bound server, so JARVIS stays on `127.0.0.1` and Tailscale handles the
certificate:

```bash
jarvis serve                                    # still 127.0.0.1:8765
tailscale serve --bg http://127.0.0.1:8765      # syntax varies by version
tailscale serve status
```

Check `tailscale serve --help` — the sub-command's syntax has changed between
releases. **Do not use `tailscale funnel`**: that is the one that publishes to
the open internet, which for this service means publishing a shell.

### Plain WireGuard

Same shape without the coordination server: bring up a WireGuard tunnel, and the
JARVIS box has a stable VPN address that only your peers can reach. Everything
above applies, except that you supply your own certificate (or, again, just use
`ssh -L` over the tunnel and skip certificates entirely).

---

## 5. Terminal only — zero extra software

Sometimes you do not want a browser at all.

```bash
ssh user@jarvis-box jarvis chat        # interactive text conversation
ssh user@jarvis-box jarvis ask "what is the disk usage on /"
ssh -t user@jarvis-box jarvis voice    # -t: allocate a TTY for the live UI
```

`-t` forces a pseudo-terminal, which the interactive commands need for their
prompt and line editing.

**But be clear about `jarvis voice` over SSH.** It runs the voice loop *on the
server*, so it opens the **server's** microphone and the **server's** speakers.
You will be typing on your laptop while JARVIS listens to the empty room where
the Linux box lives, and answers out loud to that room. That is occasionally what
you want — talking to the box from across the room while the terminal is
elsewhere — and almost never what you meant.

That mismatch is exactly why the web client captures audio in the browser
instead. If you want to *speak* to JARVIS from another device, use `jarvis serve`
and the browser, not `jarvis voice` over SSH.

---

## 6. Plain LAN binding

If everything is on your own trusted network and you want the simplest possible
client experience — open a browser, done, no tunnel:

```bash
jarvis serve --host 0.0.0.0
```

The first run prints something like:

```
  URL     http://192.168.1.42:8765
  bind    0.0.0.0:8765
  token   Yb3n...                              (printed only when generated)
          saved to ~/.local/share/jarvis/server_token; the same one is reused next time
  LAN     http://192.168.1.42:8765

  This port is this machine. Whoever reaches it can read and write any
  file and run any command, exactly as you can. ...
```

Note what you get and what you give up:

* **You get:** any device on the Wi-Fi opens the URL and talks to JARVIS. No
  client software, no tunnel, works on a phone in two seconds.
* **You give up the microphone**, unless you also pass `--cert`/`--key`.
  `http://192.168.1.42:8765` is not a secure context, so the browser will not
  hand over the microphone. Text chat works; voice does not. See §6.3.
* **You give up isolation.** Every device on that network — including the guest
  laptop, the smart TV and whatever your ISP's router runs — can reach a port
  that drives this machine. The token is the only thing in the way, which is why
  `jarvis serve` refuses to start on a non-loopback address if you have
  explicitly blanked it.

### The token

Resolution order, first hit wins:

1. `--token VALUE` on the command line
2. `JARVIS_SERVER_TOKEN` in the environment
3. the `server.token` setting in the config file, if your build has one
4. `~/.local/share/jarvis/server_token` (`%LOCALAPPDATA%\Jarvis\server_token` on
   Windows), written mode `0600`
5. otherwise a fresh 32-byte URL-safe token is generated, saved to that file, and
   **printed once** — the only time the value appears on screen

On later runs the token is loaded from the file and the startup banner says only
`token   set, from the saved token file`. It is never printed again and never
written to the log. To read it back:

```bash
cat ~/.local/share/jarvis/server_token
```

To rotate it, delete that file and restart; a new one is generated and printed.

The browser client asks for the token the first time you connect and remembers
it, so you type it once per device.

### Firewall

Binding does not by itself get you through the host firewall. Scope the rule to
your own subnet rather than opening the port to everything the machine can see:

```bash
# ufw (Debian, Ubuntu, Mint)
sudo ufw allow from 192.168.1.0/24 to any port 8765 proto tcp
sudo ufw status verbose

# firewalld (Fedora, RHEL, openSUSE)
sudo firewall-cmd --permanent --add-rich-rule='rule family="ipv4" \
  source address="192.168.1.0/24" port port="8765" protocol="tcp" accept'
sudo firewall-cmd --reload
sudo firewall-cmd --list-all
```

To undo them later: `sudo ufw delete allow from 192.168.1.0/24 to any port 8765
proto tcp`, or re-run the `firewall-cmd` line with `--remove-rich-rule`.

If you use the SSH tunnel (§3) you need **none of this** — SSH is already open,
and JARVIS never touches the network stack beyond loopback.

---

## 7. Running it persistently

You want JARVIS answering when you open your laptop, not only when a terminal
happens to be open on the box.

`docs/OPERATIONS.md` §8 covers the systemd story in full — why it must be a
**user** unit and not a system one (a system unit has no `XDG_RUNTIME_DIR`, so no
PipeWire socket, so no microphone and no speakers), and the `jarvis.linux.service`
helpers that write and manage `~/.config/systemd/user/jarvis.service`. That unit
runs `jarvis voice`, the in-the-room assistant.

The server is a second, separate unit. Write
`~/.config/systemd/user/jarvis-serve.service` by hand:

```ini
[Unit]
Description=JARVIS remote access server
After=network-online.target
StartLimitIntervalSec=120
StartLimitBurst=5

[Service]
Type=simple
ExecStart=/home/you/JARVIS/.venv/bin/python -m jarvis serve
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1
# Uncomment if your data directory is not the default:
# Environment=JARVIS_HOME=/home/you/.local/share/jarvis

[Install]
WantedBy=default.target
```

`ExecStart` **must** be an absolute path — systemd rejects a relative command and
the unit then fails to load with an error you will only see in `systemctl --user
status`. Use the virtualenv's interpreter directly; no activation step is needed.

```bash
systemctl --user daemon-reload
systemctl --user enable --now jarvis-serve.service
systemctl --user status jarvis-serve.service
journalctl --user -u jarvis-serve.service -n 50 -f
```

### Lingering — the one that bites

A user manager starts when you log in and is torn down when your last session
ends. Close the lid, log out, or reboot to the login screen and your "always on"
server is simply gone, **with nothing logged anywhere**:

```bash
loginctl enable-linger "$USER"
loginctl show-user "$USER" | grep Linger      # expect Linger=yes
```

This is the same requirement as the voice unit and is explained at length in
`docs/OPERATIONS.md` §8. If you install one, you want this; if you install both,
you definitely want it.

If the server should hold the token from a config-managed place rather than the
saved file, add `Environment=JARVIS_SERVER_TOKEN=...` — but note that unit files
are world-readable by default, so `chmod 600` it, or leave the token in
`~/.local/share/jarvis/server_token` where it is already mode `0600`.

---

## 8. Troubleshooting

### 8.1 Connection refused

```bash
ssh user@jarvis-box 'ss -ltnp | grep 8765'      # is anything listening?
```

* **You just started it.** `jarvis serve` prints the URL *before* it builds the
  language model, speech and memory subsystems, and the port refuses connections
  until that finishes — on a CPU-only box, tens of seconds. The banner says so.
  Wait for `ss -ltnp` to show the port, then load the page.
* Nothing listening → the server is not running. Start it, or
  `systemctl --user status jarvis-serve.service`.
* Listening on `127.0.0.1:8765` and you are connecting from another machine
  *without* a tunnel → that is working as designed. Set up §3, or bind wider.
* Listening on `0.0.0.0:8765` but still refused from another device → host
  firewall (§6, Firewall), or client and server are on different subnets / the
  Wi-Fi has client isolation enabled (common on guest networks).
* Through a tunnel, `connection refused` on the *client* side usually means the
  tunnel died. Check the ssh terminal; add `ExitOnForwardFailure yes` and
  `ServerAliveInterval 30` as in §3.
* `bind: Address already in use` from ssh → something on the client already holds
  8765. Use a different local port: `-L 9000:127.0.0.1:8765`, then browse to
  `http://localhost:9000`.

### 8.2 Token rejected

* Confirm the value on the server: `cat ~/.local/share/jarvis/server_token`.
  Watch for a trailing newline if you are copying it by hand; leading and
  trailing whitespace is stripped, but a character dropped in the middle is not
  detectable.
* If you started the server with `--token` or `JARVIS_SERVER_TOKEN`, the file is
  **not** what is in force. Restart without them to fall back to the file.
* Clear the token the browser saved for this origin (the client stores it per
  origin, and `localhost:8765` and `192.168.1.42:8765` are different origins —
  switching transports means entering it again).
* Rotate: delete the token file, restart, use the freshly printed value.

### 8.3 The microphone button is greyed out — read this one first

This is the single most likely thing to confuse you, and it is not a bug.

Browsers expose `navigator.mediaDevices.getUserMedia` **only in a secure
context**. A secure context means:

* `https://` with a certificate the browser trusts, **or**
* an origin the browser treats as inherently local: `http://localhost`,
  `http://127.0.0.1`, `http://[::1]`.

`http://192.168.1.42:8765`, `http://jarvis-box:8765` and
`http://100.101.102.103:8765` are **none of those**, so the microphone API is not
merely denied — it is absent, and the button has nothing to call. Text chat keeps
working, which is what makes it look like a UI bug.

Three fixes, best first:

1. **Use the SSH tunnel (§3).** The page becomes `http://localhost:8765` on your
   device, which is a secure context by definition. No certificate, no flags,
   nothing to maintain. This is why §3 is the recommendation.
2. **Give it real TLS** — `tailscale cert` (§4) is the easy path to a genuinely
   trusted certificate. A self-signed certificate is *not* enough on its own: the
   browser must actually trust it, so you would have to install the CA on every
   client device.
3. **Override it in the browser**, for a LAN bind you cannot tunnel. In Chrome or
   Edge, open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add
   `http://192.168.1.42:8765`, enable it and restart. Firefox has equivalent
   `about:config` switches. Safari has no such override at all. Treat this as a
   last resort — you are telling the browser to relax an origin-security rule
   globally, and it will apply to whatever else answers on that address.

Also check the obvious: the browser's own site permission for the microphone (the
padlock/sliders icon in the address bar), and that the OS has not muted or
reassigned the input device.

### 8.4 No audio on iOS

Safari on iOS will not play audio that was not started by a user gesture — the
first tap on the page is what unlocks playback for the session. If replies are
silent until you tap something, that is why; tap once and it stays unlocked.

iOS also demands HTTPS for the microphone with no override flag available, so on
an iPhone you need either the SSH-tunnel route (an SSH client app with port
forwarding) or a real certificate via Tailscale. Check the hardware mute switch
and the volume too — the browser's media volume is separate from the ringer.

### 8.5 "ffmpeg not found" / uploaded audio is not transcribed

`MediaRecorder` in Chrome and Firefox produces WebM/Opus; Safari produces MP4/AAC.
The speech-to-text engines want PCM. Decoding that container is `ffmpeg`'s job,
and without it the upload arrives and nothing is transcribed.

```bash
sudo apt-get install ffmpeg      # Debian, Ubuntu, Mint
sudo dnf install ffmpeg          # Fedora  (RPM Fusion may be needed)
sudo pacman -S ffmpeg            # Arch
ffmpeg -version                  # confirm
```

`jarvis doctor` reports the rest of the audio stack. `install.sh` also checks for
`ffmpeg` and prints the correct package-manager line for your distribution.

### 8.6 It works on the LAN but not from outside

By design. Nothing here opens a path from the internet to your machine. Use
Tailscale (§4) — that is exactly the problem it solves — rather than forwarding a
port on the router.

---

## 9. The security position, stated once

JARVIS runs **unrestricted** on the owner's own machine, deliberately. Security
defaults to `mode="open"`: no protected paths, no confirmation prompts, no
command it will refuse. That is the point of the project and nothing in this
document changes it.

Who may **connect** is a completely different question from what JARVIS may do.
Putting an unrestricted shell-and-filesystem agent on a network port means
whoever reaches that port has the same unrestricted access to this machine that
you do — every file, every command, the desktop, the microphone. Not a
diminished, sandboxed version of it. The same.

So the network surface, and only the network surface, is deliberately narrow:

* **Loopback by default.** `jarvis serve` binds `127.0.0.1`. A wider bind is
  always an explicit `--host` you typed.
* **A token is required off loopback.** One is generated and saved if you have
  not set one; the server refuses to start on a non-loopback address with the
  token explicitly blanked, and says why.
* **Tokens are compared in constant time**, never logged, never echoed after the
  run that created them, and never placed in a URL.
* **Same-origin only.** No CORS wildcard.
* **The recommended route exposes nothing at all.** With `ssh -L`, JARVIS is
  never on the network; SSH is, and SSH is a lock you already maintain.

None of that limits you. It limits everyone who is not you.

---

## See also

* `docs/OPERATIONS.md` — running JARVIS day to day; §8 is systemd and lingering.
* `docs/TROUBLESHOOTING.md` — the audio stack, models, and everything not
  network-related.
* `docs/ARCHITECTURE.md` — how the subsystems fit together.
