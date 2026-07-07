/**
 * Frontend configuration — production points to Render backend.
 */
window.HCR_CONFIG = {
  API_BASE_URL:
    window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
      ? "http://localhost:5000"
      : "https://codealpha-handwritten-character-3jhq.onrender.com",
  LOW_CONFIDENCE_THRESHOLD: 70,
  CONFIDENCE_ANIMATION_MS: 600,
};
