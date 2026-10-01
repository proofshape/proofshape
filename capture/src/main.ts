import { mountCameraCheck } from "./camera.js";
import { mountUploadPage } from "./page.js";

const root = document.getElementById("app");
if (!root) {
  throw new Error("main: #app element not found in index.html");
}
mountUploadPage(root, { baseUrl: "" });

const cameraRoot = document.getElementById("camera-app");
if (!cameraRoot) {
  throw new Error("main: #camera-app element not found in index.html");
}
mountCameraCheck(cameraRoot);
