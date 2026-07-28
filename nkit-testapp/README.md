# nkit-testapp

Sensor/gesture testbed for nkit — not the final Metro UI, just proof that
skeleton/hand/face/gesture/voice data flows end to end from the Kinect into
a web UI. See `nkit — Overview.md` §G at the repo root for the design
rationale.

## Run

Two processes, both inside `nix develop`:

```
# 1. the bridge — runs the actual sensor pipeline
python -m nkit.bridge --frame-stream --screen-w 1920 --screen-h 1080

# 2. the demo app
cd nkit-testapp
npm install          # first time only — fetches JS deps
electron .           # NOT `npm start` — see NixOS note below
```

**NixOS note:** run it with the `electron` binary from the nix devShell
(`pkgs.electron`, already on PATH inside `nix develop`), not `npm start` /
`node_modules/.bin/electron`. npm downloads a generic prebuilt Electron
binary that isn't patched for the Nix store's dynamic linker layout and
won't run as-is (`Could not start dynamically linked executable`) — this is
a standard NixOS thing, not specific to this app. `electron` is still a
devDependency in package.json for editor tooling; just don't launch through
it directly on NixOS.

`--screen-w`/`--screen-h` should match the Electron window's actual
resolution, since gesture coordinates arrive already mapped to that space.

Escape quits (or the Quit button in the sidebar). F12 toggles devtools.

## Gesture sounds

Drop WAV files (PCM16, 48kHz, mono, sub-1s — see the overview doc's
rationale) into `sounds/`, named to match `GestureEvent.kind`:

```
sounds/grab_start.wav
sounds/grab_end.wav
sounds/click.wav         # a "push" that landed on the test button
sounds/click_empty.wav   # a "push" that landed on empty space
sounds/swipe_edge.wav
sounds/grab_swipe.wav
```

Missing files are silently skipped (`Audio.play()` rejection is caught) —
the app works fine with none of these present, you just won't hear
anything.

## Not in scope here

No emulator/process spawning, no gamescope focus control — that's the
final app (§H), gated on the gamescope spike.
