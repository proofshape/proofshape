import {
  HOSTED_RECON_UNAVAILABLE_MESSAGE,
  hostedReconUnavailable,
  resolveAuthToken,
  resolveBaseUrl,
} from "./config.js";
import { mountBuyerPage } from "./buyer.js";

const root = document.getElementById("buyer-app");
if (!root) throw new Error("buyer-main: #buyer-app element not found");

const baseUrl = resolveBaseUrl(import.meta.env.VITE_RECON_BASE_URL);
if (hostedReconUnavailable(baseUrl, import.meta.env.DEV)) {
  root.innerHTML = `<p role="status">${HOSTED_RECON_UNAVAILABLE_MESSAGE}</p>`;
} else {
  const authToken = resolveAuthToken(import.meta.env.VITE_RECON_TOKEN);
  mountBuyerPage(root, {
    baseUrl,
    ...(authToken !== undefined ? { authToken } : {}),
  });
}
