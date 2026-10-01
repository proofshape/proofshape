import { mountGyroCheck } from "./gyro.js";
import { mountUploadPage } from "./page.js";

const root = document.getElementById("app");
if (!root) {
  throw new Error("main: #app element not found in index.html");
}
mountUploadPage(root, { baseUrl: "" });

const gyroRoot = document.getElementById("gyro-app");
if (!gyroRoot) {
  throw new Error("main: #gyro-app element not found in index.html");
}
mountGyroCheck(gyroRoot);
