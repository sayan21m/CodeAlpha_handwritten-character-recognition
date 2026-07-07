/**
 * Frontend configuration — edit API_BASE_URL for your environment.
 */
window.HCR_CONFIG = {
  API_BASE_URL:
    window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
      ? "http://localhost:5000"
      : `${window.location.protocol}//${window.location.hostname}:5000`,
  LOW_CONFIDENCE_THRESHOLD: 70,
  CONFIDENCE_ANIMATION_MS: 600,
};
