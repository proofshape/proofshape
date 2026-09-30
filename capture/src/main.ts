import { mountUploadPage } from "./page.js";

const root = document.getElementById("app");
if (!root) {
  throw new Error("main: #app element not found in index.html");
}
mountUploadPage(root, { baseUrl: "" });
