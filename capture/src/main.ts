import { mountAppShell } from "./shell.js";

const shellRoot = document.getElementById("shell-app");
if (!shellRoot) {
  throw new Error("main: #shell-app element not found in index.html");
}
// Return value not captured: nothing consumes it yet until C-05 exists to read/write it.
// mountAppShell still returns it for tests and for whenever C-05 is wired in here.
mountAppShell(shellRoot, { baseUrl: "" });
