/**
 * Handwritten Character Recognition — Frontend
 * Model selection, canvas drawing, upload, camera capture, API integration
 */

/* --------------------------------------------------------------------------
   Configuration (see config.js)
   -------------------------------------------------------------------------- */
const CONFIG = window.HCR_CONFIG || { API_BASE_URL: "http://localhost:5000", LOW_CONFIDENCE_THRESHOLD: 70 };
const API_BASE_URL = CONFIG.API_BASE_URL;
const API_ENDPOINTS = {
  digit: `${API_BASE_URL}/predict-digit`,
  character: `${API_BASE_URL}/predict-character`,
};

const MODEL_LABELS = {
  digit: "Digit Recognition",
  character: "Character Recognition",
};

const INPUT_LABELS = {
  canvas: "Canvas",
  upload: "Upload",
  camera: "Camera",
};

const CANVAS_SIZE = 280;
const MNIST_SIZE = 28;
// Scale stroke width relative to MNIST proportions on the drawing canvas
const STROKE_WIDTH = Math.max(12, Math.round((CANVAS_SIZE / MNIST_SIZE) * 2));
const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/jpg"];

/* --------------------------------------------------------------------------
   State
   -------------------------------------------------------------------------- */
const state = {
  activeModel: "digit",
  activeInput: "canvas",
  isDrawing: false,
  undoStack: [],
  uploadedImage: null,
  capturedImage: null,
  cameraStream: null,
  cameraDevices: [],
  activeCameraId: null,
  cameraActive: false,
  cameraCaptured: false,
};

/* --------------------------------------------------------------------------
   DOM references
   -------------------------------------------------------------------------- */
const canvas = document.getElementById("drawing-canvas");
const ctx = canvas.getContext("2d");
const modelCards = document.querySelectorAll(".model-card");
const inputTabs = document.querySelectorAll(".input-tab");
const inputPanels = document.querySelectorAll(".input-panel");
const activeModelLabel = document.getElementById("active-model-label");
const modelValue = document.getElementById("model-value");
const inputSourceValue = document.getElementById("input-source-value");
const clearBtn = document.getElementById("clear-btn");
const undoBtn = document.getElementById("undo-btn");
const predictBtn = document.getElementById("predict-btn");
const uploadZone = document.getElementById("upload-zone");
const fileInput = document.getElementById("file-input");
const uploadPreviewCard = document.getElementById("upload-preview-card");
const uploadPreview = document.getElementById("upload-preview");
const predictionCard = document.getElementById("prediction-card");
const predictionValue = document.getElementById("prediction-value");
const confidenceValue = document.getElementById("confidence-value");
const inferenceValue = document.getElementById("inference-value");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingMessage = document.getElementById("loading-message");
const toastContainer = document.getElementById("toast-container");

// Camera elements
const cameraPreview = document.getElementById("camera-preview");
const cameraLoading = document.getElementById("camera-loading");
const cameraError = document.getElementById("camera-error");
const cameraErrorText = document.getElementById("camera-error-text");
const cameraPlaceholder = document.getElementById("camera-placeholder");
const cameraSwitchWrap = document.getElementById("camera-switch-wrap");
const cameraSelect = document.getElementById("camera-select");
const startCameraBtn = document.getElementById("start-camera-btn");
const captureBtn = document.getElementById("capture-btn");
const retakeBtn = document.getElementById("retake-btn");
const closeCameraBtn = document.getElementById("close-camera-btn");
const cameraPreviewCard = document.getElementById("camera-preview-card");
const cameraCapturedPreview = document.getElementById("camera-captured-preview");
const cameraCropGuide = document.getElementById("camera-crop-guide");
const topPredictions = document.getElementById("top-predictions");
const topPredictionsList = document.getElementById("top-predictions-list");
const preprocessedPreviewCard = document.getElementById("preprocessed-preview-card");
const preprocessedPreview = document.getElementById("preprocessed-preview");
const confidenceWarning = document.getElementById("confidence-warning");

/* --------------------------------------------------------------------------
   Initialization
   -------------------------------------------------------------------------- */
function init() {
  setupCanvas();
  setupModelSelection();
  setupInputMethods();
  setupDrawing();
  setupToolbar();
  setupUpload();
  setupCamera();
  selectModel("digit");
  switchInputMethod("canvas", false);
}

/* --------------------------------------------------------------------------
   Input method switching — only one active at a time
   -------------------------------------------------------------------------- */
function setupInputMethods() {
  inputTabs.forEach((tab) => {
    tab.addEventListener("click", () => switchInputMethod(tab.dataset.input));
    tab.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        switchInputMethod(tab.dataset.input);
      }
    });
  });
}

function switchInputMethod(method, showNotification = true) {
  if (state.activeInput === method) return;

  clearCurrentInput(state.activeInput);
  resetPredictionResults();
  state.activeInput = method;

  inputTabs.forEach((tab) => {
    const isActive = tab.dataset.input === method;
    tab.classList.toggle("is-active", isActive);
    tab.setAttribute("aria-selected", String(isActive));
  });

  inputPanels.forEach((panel) => {
    const panelId = `panel-${method}`;
    panel.classList.toggle("is-active", panel.id === panelId);
    panel.classList.toggle("hidden", panel.id !== panelId);
  });

  inputSourceValue.textContent = INPUT_LABELS[method];

  if (showNotification) {
    showToast(`${INPUT_LABELS[method]} input selected.`, "info");
  }
}

function clearCurrentInput(previousMethod) {
  if (previousMethod === "canvas") {
    clearCanvas(false);
    state.undoStack = [];
  } else if (previousMethod === "upload") {
    state.uploadedImage = null;
    uploadPreviewCard.classList.add("hidden");
    uploadPreview.src = "";
    fileInput.value = "";
    clearCanvas(false);
  } else if (previousMethod === "camera") {
    stopCameraStream();
    resetCameraUI();
    state.capturedImage = null;
    cameraPreviewCard.classList.add("hidden");
    cameraCapturedPreview.src = "";
    clearCanvas(false);
  }
}

function resetPredictionResults() {
  predictionCard.classList.remove("has-result", "is-low-confidence");
  predictionValue.textContent = "—";
  confidenceValue.textContent = "0";
  inferenceValue.textContent = "0";
  confidenceWarning.classList.add("hidden");
  confidenceWarning.textContent = "";
  topPredictions.classList.add("hidden");
  topPredictionsList.innerHTML = "";
  preprocessedPreviewCard.classList.add("hidden");
  preprocessedPreview.src = "";
}

/* --------------------------------------------------------------------------
   Canvas
   -------------------------------------------------------------------------- */
function setupCanvas() {
  canvas.width = CANVAS_SIZE;
  canvas.height = CANVAS_SIZE;
  clearCanvas(false);
}

function clearCanvas(saveUndo = true) {
  if (saveUndo && hasCanvasContent()) {
    saveUndoState();
  }
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
}

function hasCanvasContent() {
  const pixels = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE).data;
  for (let i = 0; i < pixels.length; i += 4) {
    if (pixels[i] < 250 || pixels[i + 1] < 250 || pixels[i + 2] < 250) {
      return true;
    }
  }
  return false;
}

function saveUndoState() {
  state.undoStack.push(ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE));
  if (state.undoStack.length > 20) state.undoStack.shift();
}

function undoStroke() {
  if (state.undoStack.length === 0) {
    showToast("Nothing to undo.", "info");
    return;
  }
  ctx.putImageData(state.undoStack.pop(), 0, 0);
}

/* --------------------------------------------------------------------------
   Drawing
   -------------------------------------------------------------------------- */
function setupDrawing() {
  canvas.addEventListener("mousedown", startDrawing);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("mouseup", endDrawing);
  canvas.addEventListener("mouseleave", endDrawing);
  canvas.addEventListener("touchstart", handleTouchStart, { passive: false });
  canvas.addEventListener("touchmove", handleTouchMove, { passive: false });
  canvas.addEventListener("touchend", endDrawing);
}

function getPointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const clientX = event.clientX ?? event.touches?.[0]?.clientX;
  const clientY = event.clientY ?? event.touches?.[0]?.clientY;
  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top) * scaleY,
  };
}

function startDrawing(event) {
  if (state.activeInput !== "canvas") return;
  event.preventDefault();
  saveUndoState();
  resetPredictionResults();
  state.isDrawing = true;
  const pos = getPointerPosition(event);
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = STROKE_WIDTH;
}

function draw(event) {
  if (!state.isDrawing) return;
  event.preventDefault();
  const pos = getPointerPosition(event);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
}

function endDrawing() {
  state.isDrawing = false;
}

function handleTouchStart(event) {
  startDrawing(event);
}

function handleTouchMove(event) {
  draw(event);
}

/* --------------------------------------------------------------------------
   Model selection
   -------------------------------------------------------------------------- */
function setupModelSelection() {
  modelCards.forEach((card) => {
    const model = card.dataset.model;
    card.addEventListener("click", () => selectModel(model));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectModel(model);
      }
    });
    card.querySelector(".model-select-btn").addEventListener("click", (event) => {
      event.stopPropagation();
      selectModel(model);
    });
  });
}

function selectModel(model) {
  const changed = state.activeModel !== model;
  state.activeModel = model;
  modelCards.forEach((card) => {
    const isActive = card.dataset.model === model;
    card.classList.toggle("is-active", isActive);
    card.setAttribute("aria-checked", String(isActive));
  });
  activeModelLabel.textContent = MODEL_LABELS[model];
  modelValue.textContent = MODEL_LABELS[model];
  if (changed) {
    showToast(`${MODEL_LABELS[model]} selected.`, "info");
  }
}

/* --------------------------------------------------------------------------
   Toolbar
   -------------------------------------------------------------------------- */
function setupToolbar() {
  clearBtn.addEventListener("click", () => {
    clearCanvas(false);
    state.undoStack = [];
    resetPredictionResults();
    showToast("Canvas cleared.", "info");
  });
  undoBtn.addEventListener("click", undoStroke);
  predictBtn.addEventListener("click", handlePredict);
}

/* --------------------------------------------------------------------------
   Image upload
   -------------------------------------------------------------------------- */
function setupUpload() {
  uploadZone.addEventListener("click", () => fileInput.click());
  uploadZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) processFile(file);
    fileInput.value = "";
  });

  ["dragenter", "dragover"].forEach((type) => {
    uploadZone.addEventListener(type, (event) => {
      event.preventDefault();
      uploadZone.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach((type) => {
    uploadZone.addEventListener(type, (event) => {
      event.preventDefault();
      uploadZone.classList.remove("is-dragover");
    });
  });

  uploadZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) processFile(file);
  });
}

function processFile(file) {
  if (!ACCEPTED_TYPES.includes(file.type)) {
    showToast("Please upload a PNG, JPG, or JPEG image.", "error");
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      state.uploadedImage = img;
      drawImageToCanvas(img);
      uploadPreview.src = img.src;
      uploadPreviewCard.classList.remove("hidden");
      resetPredictionResults();
      showToast("Image uploaded successfully.", "success");
    };
    img.onerror = () => showToast("Failed to load image.", "error");
    img.src = reader.result;
  };
  reader.onerror = () => showToast("Failed to read file.", "error");
  reader.readAsDataURL(file);
}

function drawImageToCanvas(img) {
  clearCanvas(false);
  const scale = Math.min(CANVAS_SIZE / img.width, CANVAS_SIZE / img.height);
  const width = img.width * scale;
  const height = img.height * scale;
  const x = (CANVAS_SIZE - width) / 2;
  const y = (CANVAS_SIZE - height) / 2;

  const offscreen = document.createElement("canvas");
  offscreen.width = CANVAS_SIZE;
  offscreen.height = CANVAS_SIZE;
  const offCtx = offscreen.getContext("2d");
  offCtx.fillStyle = "#ffffff";
  offCtx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  offCtx.drawImage(img, x, y, width, height);

  const gray = offCtx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  convertToGrayscale(gray.data);
  offCtx.putImageData(gray, 0, 0);
  ctx.drawImage(offscreen, 0, 0);
}

function convertToGrayscale(pixels) {
  for (let i = 0; i < pixels.length; i += 4) {
    const gray = 0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2];
    pixels[i] = gray;
    pixels[i + 1] = gray;
    pixels[i + 2] = gray;
  }
}

/* --------------------------------------------------------------------------
   Camera capture
   -------------------------------------------------------------------------- */
function setupCamera() {
  if (!isCameraSupported()) {
    startCameraBtn.disabled = true;
    cameraErrorText.textContent = "Your browser does not support camera access.";
    cameraError.classList.remove("hidden");
    cameraPlaceholder.classList.add("hidden");
    return;
  }

  startCameraBtn.addEventListener("click", startCamera);
  captureBtn.addEventListener("click", capturePhoto);
  retakeBtn.addEventListener("click", retakePhoto);
  closeCameraBtn.addEventListener("click", closeCamera);
  cameraSelect.addEventListener("change", switchCamera);
}

function isCameraSupported() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function isMobileDevice() {
  return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
}

async function startCamera() {
  if (!isCameraSupported()) {
    showToast("Camera is not supported in this browser.", "error");
    return;
  }

  hideCameraError();
  cameraLoading.classList.remove("hidden");
  cameraPlaceholder.classList.add("hidden");
  startCameraBtn.disabled = true;

  try {
    const constraints = buildCameraConstraints();
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    attachStream(stream);

    // Enumerate after permission so device labels are available
    await enumerateCameras();
    if (!state.activeCameraId && state.cameraDevices.length > 0) {
      state.activeCameraId = pickPreferredCameraId();
    }
    if (state.activeCameraId) {
      cameraSelect.value = state.activeCameraId;
    }

    state.cameraActive = true;
    state.cameraCaptured = false;

    updateCameraControls("live");
    showToast("Camera started.", "success");
  } catch (error) {
    handleCameraError(error);
  } finally {
    cameraLoading.classList.add("hidden");
    startCameraBtn.disabled = false;
  }
}

async function enumerateCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  state.cameraDevices = devices.filter((d) => d.kind === "videoinput");

  if (state.cameraDevices.length === 0) return;

  cameraSelect.innerHTML = "";
  state.cameraDevices.forEach((device, index) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `Camera ${index + 1}`;
    cameraSelect.appendChild(option);
  });

  if (!state.activeCameraId) {
    state.activeCameraId = pickPreferredCameraId();
  }

  cameraSelect.value = state.activeCameraId;
  cameraSwitchWrap.classList.toggle("hidden", state.cameraDevices.length < 2);
}

function pickPreferredCameraId() {
  if (state.cameraDevices.length === 0) return null;

  if (isMobileDevice()) {
    const rear = state.cameraDevices.find((d) =>
      /back|rear|environment/i.test(d.label)
    );
    if (rear) return rear.deviceId;
  }

  return state.cameraDevices[0].deviceId;
}

function buildCameraConstraints() {
  const video = { width: { ideal: 1280 }, height: { ideal: 720 } };

  if (state.activeCameraId) {
    video.deviceId = { exact: state.activeCameraId };
  } else if (isMobileDevice()) {
    video.facingMode = { ideal: "environment" };
  } else {
    video.facingMode = "user";
  }

  return { video, audio: false };
}

function attachStream(stream) {
  stopCameraStream();
  state.cameraStream = stream;
  cameraPreview.srcObject = stream;
  cameraPreview.classList.add("is-live");
  cameraPlaceholder.classList.add("hidden");
  cameraCropGuide.classList.remove("hidden");
}

async function switchCamera() {
  state.activeCameraId = cameraSelect.value;
  if (!state.cameraActive || state.cameraCaptured) return;

  cameraLoading.classList.remove("hidden");
  try {
    const stream = await navigator.mediaDevices.getUserMedia(buildCameraConstraints());
    attachStream(stream);
    showToast("Camera switched.", "info");
  } catch (error) {
    handleCameraError(error);
  } finally {
    cameraLoading.classList.add("hidden");
  }
}

function capturePhoto() {
  if (!state.cameraStream || !cameraPreview.videoWidth) {
    showToast("Camera is not ready.", "error");
    return;
  }

  const videoW = cameraPreview.videoWidth;
  const videoH = cameraPreview.videoHeight;
  const cropSize = Math.min(videoW, videoH);
  const sx = (videoW - cropSize) / 2;
  const sy = (videoH - cropSize) / 2;

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = cropSize;
  tempCanvas.height = cropSize;
  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.drawImage(cameraPreview, sx, sy, cropSize, cropSize, 0, 0, cropSize, cropSize);

  const img = new Image();
  img.onload = () => {
    state.capturedImage = img;
    drawImageToCanvas(img);
    cameraCapturedPreview.src = tempCanvas.toDataURL("image/jpeg", 0.92);
    cameraPreviewCard.classList.remove("hidden");
    state.cameraCaptured = true;
    resetPredictionResults();

    stopCameraStream();
    cameraPreview.classList.remove("is-live");
    cameraPlaceholder.classList.remove("hidden");
    cameraCropGuide.classList.add("hidden");

    updateCameraControls("captured");
    showToast("Photo captured. Click Predict or Retake.", "success");
  };
  img.src = tempCanvas.toDataURL("image/jpeg", 0.92);
}

function retakePhoto() {
  state.capturedImage = null;
  cameraPreviewCard.classList.add("hidden");
  cameraCapturedPreview.src = "";
  clearCanvas(false);
  resetPredictionResults();
  state.cameraCaptured = false;
  startCamera();
}

function closeCamera() {
  stopCameraStream();
  resetCameraUI();
  state.capturedImage = null;
  state.cameraCaptured = false;
  cameraPreviewCard.classList.add("hidden");
  cameraCapturedPreview.src = "";
  clearCanvas(false);
  resetPredictionResults();
  showToast("Camera closed.", "info");
}

function stopCameraStream() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((track) => track.stop());
    state.cameraStream = null;
  }
  cameraPreview.srcObject = null;
  state.cameraActive = false;
}

function resetCameraUI() {
  cameraPreview.classList.remove("is-live");
  cameraPlaceholder.classList.remove("hidden");
  cameraCropGuide.classList.add("hidden");
  cameraLoading.classList.add("hidden");
  hideCameraError();
  updateCameraControls("idle");
}

function updateCameraControls(mode) {
  startCameraBtn.classList.toggle("hidden", mode === "live" || mode === "captured");
  captureBtn.classList.toggle("hidden", mode !== "live");
  retakeBtn.classList.toggle("hidden", mode !== "captured");
  closeCameraBtn.classList.toggle("hidden", mode === "idle");
  cameraSwitchWrap.classList.toggle("hidden", mode !== "live" || state.cameraDevices.length < 2);
}

function handleCameraError(error) {
  console.error("Camera error:", error);
  stopCameraStream();
  resetCameraUI();

  let message = "Could not access the camera.";
  if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
    message = "Camera permission denied. Please allow access in your browser settings.";
  } else if (error.name === "NotFoundError") {
    message = "No camera found on this device.";
  } else if (error.name === "NotReadableError") {
    message = "Camera is already in use by another application.";
  }

  cameraErrorText.textContent = message;
  cameraError.classList.remove("hidden");
  cameraPlaceholder.classList.add("hidden");
  showToast(message, "error");
}

function hideCameraError() {
  cameraError.classList.add("hidden");
}

/* --------------------------------------------------------------------------
   Prediction & API
   -------------------------------------------------------------------------- */
function hasActiveInput() {
  if (state.activeInput === "canvas") return hasCanvasContent();
  if (state.activeInput === "upload") return state.uploadedImage !== null && hasCanvasContent();
  if (state.activeInput === "camera") return state.capturedImage !== null && hasCanvasContent();
  return false;
}

async function handlePredict() {
  if (!hasActiveInput()) {
    const hints = {
      canvas: "Please draw on the canvas first.",
      upload: "Please upload an image first.",
      camera: "Please capture a photo first.",
    };
    showToast(hints[state.activeInput], "error");
    return;
  }

  const endpoint = API_ENDPOINTS[state.activeModel];
  const imageData = getGrayscaleImagePayload();

  setLoading(true, "Running inference…");

  try {
    const start = performance.now();
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: imageData,
        input_source: state.activeInput,
      }),
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const message = data.error || `Server responded with ${response.status}`;
      showToast(message, "error");
      return;
    }

    const elapsed = data.inference_time_ms ?? Math.round(performance.now() - start);
    const confidence = normalizeConfidence(data.confidence);
    const inputSource = data.input_source
      ? INPUT_LABELS[data.input_source] || data.input_source
      : INPUT_LABELS[state.activeInput];

    updatePredictionUI({
      prediction: formatPrediction(data.prediction),
      confidence,
      inferenceMs: elapsed,
      inputSource,
      topPredictions: data.top_predictions || [],
      preprocessedPreview: data.preprocessed_preview || null,
      lowConfidence: data.low_confidence || confidence < CONFIG.LOW_CONFIDENCE_THRESHOLD,
      warning: data.warning || null,
    });

    if (data.low_confidence || confidence < CONFIG.LOW_CONFIDENCE_THRESHOLD) {
      showToast(data.warning || "Low confidence — try redrawing larger and centered.", "error");
    } else {
      showToast("Prediction complete.", "success");
    }
  } catch (error) {
    console.error("Prediction error:", error);
    showToast("Could not reach the API. Is the backend running?", "error");
  } finally {
    setLoading(false);
  }
}

function getGrayscaleImagePayload() {
  const offscreen = document.createElement("canvas");
  offscreen.width = CANVAS_SIZE;
  offscreen.height = CANVAS_SIZE;
  const offCtx = offscreen.getContext("2d");
  offCtx.drawImage(canvas, 0, 0);
  const imageData = offCtx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  convertToGrayscale(imageData.data);
  offCtx.putImageData(imageData, 0, 0);
  return offscreen.toDataURL("image/png");
}

function formatPrediction(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return state.activeModel === "character"
      ? String.fromCharCode(65 + value)
      : String(value);
  }
  return String(value).toUpperCase();
}

function normalizeConfidence(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return 0;
  return value <= 1 ? value * 100 : value;
}

function updatePredictionUI({
  prediction,
  confidence,
  inferenceMs,
  inputSource,
  topPredictions: topList = [],
  preprocessedPreview: previewSrc = null,
  lowConfidence = false,
  warning = null,
}) {
  predictionCard.classList.add("has-result");
  predictionCard.classList.toggle("is-low-confidence", lowConfidence);
  animateText(predictionValue, prediction);
  animateNumber(confidenceValue, confidence, 2);
  animateNumber(inferenceValue, inferenceMs, 0);
  inputSourceValue.textContent = inputSource;

  if (lowConfidence && warning) {
    confidenceWarning.textContent = warning;
    confidenceWarning.classList.remove("hidden");
  } else {
    confidenceWarning.classList.add("hidden");
  }

  if (topList.length > 0) {
    topPredictionsList.innerHTML = topList
      .map((item) => `<li><strong>${item.label}</strong> — ${item.confidence}%</li>`)
      .join("");
    topPredictions.classList.remove("hidden");
  }

  if (previewSrc) {
    preprocessedPreview.src = previewSrc;
    preprocessedPreviewCard.classList.remove("hidden");
  }
}

/* --------------------------------------------------------------------------
   Animated transitions
   -------------------------------------------------------------------------- */
function animateNumber(element, target, decimals = 0) {
  const start = parseFloat(element.textContent) || 0;
  const duration = 600;
  const startTime = performance.now();
  element.classList.add("is-updating");

  function frame(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    element.textContent = (start + (target - start) * eased).toFixed(decimals);
    if (progress < 1) requestAnimationFrame(frame);
    else element.classList.remove("is-updating");
  }

  requestAnimationFrame(frame);
}

function animateText(element, text) {
  element.classList.add("is-updating");
  element.textContent = text;
  setTimeout(() => element.classList.remove("is-updating"), 300);
}

/* --------------------------------------------------------------------------
   Loading & toasts
   -------------------------------------------------------------------------- */
function setLoading(isLoading, message = "Running inference…") {
  loadingOverlay.classList.toggle("hidden", !isLoading);
  loadingOverlay.setAttribute("aria-busy", String(isLoading));
  loadingMessage.textContent = message;
  predictBtn.disabled = isLoading;
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(8px)";
    toast.style.transition = "0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3200);
}

/* --------------------------------------------------------------------------
   Boot
   -------------------------------------------------------------------------- */
document.addEventListener("DOMContentLoaded", init);

// Stop camera when user leaves the page
window.addEventListener("beforeunload", stopCameraStream);
