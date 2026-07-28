// nkit-testapp/main.js — Electron main process
//
// This is the sensor/gesture testbed (§G in the overview doc), not the
// final Metro shell — deliberately minimal: one fullscreen window, no
// process spawning, no gamescope integration. Those land in the final app
// once the gamescope focus-control spike (§H) is done.
//
// Still built on Electron rather than plain kiosk Chromium (per the
// decision that landed once emulator embedding came up — see the overview
// doc) so the packaging/process model gets exercised here first.

const { app, BrowserWindow, globalShortcut, ipcMain } = require("electron");
const path = require("path");

let win = null;

function createWindow() {
  win = new BrowserWindow({
    fullscreen: true,
    frame: false,
    backgroundColor: "#0b0b0e",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.loadFile(path.join(__dirname, "renderer", "index.html"));

  // dev convenience — this is a debug harness, don't trap the window shut
  globalShortcut.register("Escape", () => app.quit());
  globalShortcut.register("F12", () => win.webContents.toggleDevTools());
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("will-quit", () => {
  globalShortcut.unregisterAll();
});

ipcMain.on("quit", () => app.quit());
