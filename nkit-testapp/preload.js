// nkit-testapp/preload.js — the only bridge between renderer and Node.
// Renderer talks to nkit itself over a plain WebSocket (no Node needed for
// that) — this just exposes the one thing a sandboxed page can't do on its
// own: quitting the app, for the debug UI's Quit button.

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("nkitApp", {
  quit: () => ipcRenderer.send("quit"),
});
