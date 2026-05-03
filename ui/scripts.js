'use strict';

// ── DOM refs ─────────────────────────────────────────────────────────────────
const uploadBtn        = document.getElementById('upload-btn');
const imageUpload      = document.getElementById('image-upload');
const previewImage     = document.getElementById('preview-image');
const viewerPlaceholder= document.getElementById('viewer-placeholder');
const loadingOverlay   = document.getElementById('loading-overlay');
const errorToast       = document.getElementById('error-toast');
const imageMeta        = document.getElementById('image-meta');
const imageFilename    = document.getElementById('image-filename');
const detectionCount   = document.getElementById('detection-count');
const navTabs          = document.querySelectorAll('.nav-tab');
const sections         = document.querySelectorAll('.section');
const viewRadios       = document.querySelectorAll('input[name="view"]');

// Metric card elements
const valClass        = document.getElementById('val-class');
const valClassSub     = document.getElementById('val-class-sub');
const barClass        = document.getElementById('bar-class');
const valYoloConf     = document.getElementById('val-yolo-conf');
const barYoloConf     = document.getElementById('bar-yolo-conf');
const valResnetConf   = document.getElementById('val-resnet-conf');
const barResnetConf   = document.getElementById('bar-resnet-conf');
const classChips      = document.getElementById('class-chips');
const statusLed       = document.getElementById('status-led');
const statusText      = document.getElementById('status-text');

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  imageLoaded:   false,
  currentView:   'gradcam',   // 'gradcam' | 'bbox'
  origSrc:       null,        // data URL from FileReader
  bboxSrc:       null,        // server-rendered bbox image
  gradcamSrc:    null,        // server-rendered gradcam image
  pendingFile:   null,        // File awaiting server response
};


// ══════════════════════════════════════════════════════════════════════════════
// 1. Image Upload → show locally + POST to backend
// ══════════════════════════════════════════════════════════════════════════════

uploadBtn.addEventListener('click', () => imageUpload.click());

imageUpload.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  imageUpload.value = '';   // allow re-selecting same file

  state.pendingFile = file;
  imageFilename.textContent = file.name;

  // Show original immediately so the UI doesn't feel slow
  const reader = new FileReader();
  reader.onload = (evt) => {
    state.origSrc    = evt.target.result;
    state.bboxSrc    = null;
    state.gradcamSrc = null;
    state.imageLoaded = true;

    showImage(state.origSrc);
    imageMeta.classList.remove('hidden');
    detectionCount.textContent = '—';

    postToBackend(file);
  };
  reader.readAsDataURL(file);
});


// ══════════════════════════════════════════════════════════════════════════════
// 2. API call
// ══════════════════════════════════════════════════════════════════════════════

async function postToBackend(file) {
  showLoading(true);
  hideError();

  const form = new FormData();
  form.append('image', file);

  try {
    const resp = await fetch('/api/inference', { method: 'POST', body: form });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: `HTTP ${resp.status}` }));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }

    const data = await resp.json();
    if (!data.success) throw new Error(data.error || 'Inference failed');

    handleInferenceResult(data);

  } catch (err) {
    showError(err.message);
    // fall back to showing the original image
    if (state.origSrc) showImage(state.origSrc);
  } finally {
    showLoading(false);
  }
}


// ══════════════════════════════════════════════════════════════════════════════
// 3. Handle server response
// ══════════════════════════════════════════════════════════════════════════════

function handleInferenceResult(data) {
  // Store rendered images
  state.bboxSrc    = data.bbox_image_b64 || state.origSrc;
  state.gradcamSrc = data.gradcam_b64    || state.origSrc;

  // Apply current view
  applyView();

  // Update metric cards
  updateMetrics(data);
}


function updateMetrics(data) {
  const topClass = data.top_class || '—';
  const yConf    = data.yolo_confidence   || 0;
  const rConf    = data.resnet_confidence;        // null = model not loaded, 0.0 = ran but low
  const boxes    = data.boxes             || [];

  // Top class card
  valClass.textContent    = topClass;
  valClassSub.textContent = `${boxes.length} object${boxes.length !== 1 ? 's' : ''} detected`;
  barClass.style.width    = `${Math.max(yConf, rConf ?? 0) * 100}%`;

  // YOLO confidence card
  valYoloConf.textContent   = yConf > 0 ? yConf.toFixed(2) : '—';
  barYoloConf.style.width   = `${yConf * 100}%`;

  // ResNet probability card
  valResnetConf.textContent = rConf !== null ? rConf.toFixed(2) : '—';
  barResnetConf.style.width = rConf !== null ? `${rConf * 100}%` : '0%';

  // Detection count tag
  detectionCount.textContent = boxes.length > 0
    ? `${boxes.length} detection${boxes.length !== 1 ? 's' : ''}`
    : 'no detections';

  // Class chips — top 6 unique detected classes
  const seen   = new Set();
  const unique = boxes.filter(b => !seen.has(b.class_name) && seen.add(b.class_name));
  classChips.innerHTML = unique.length > 0
    ? unique.slice(0, 6).map((b, i) =>
        `<span class="chip ${i === 0 ? 'active' : ''}">${b.class_name}</span>`
      ).join('')
    : '<span class="chip muted">—</span>';

  // Status banner
  const hasYolo   = yConf > 0;
  const hasResnet = rConf !== null;
  if (hasYolo && hasResnet) {
    statusText.textContent = 'DUAL-MODEL INFERENCE COMPLETE';
  } else if (hasYolo) {
    statusText.textContent = 'YOLO-LED DETECTION ACTIVE';
  } else if (hasResnet) {
    statusText.textContent = 'RESNET CLASSIFICATION ACTIVE';
  } else {
    statusText.textContent = 'INFERENCE COMPLETE';
  }
}


// ══════════════════════════════════════════════════════════════════════════════
// 4. View toggle
// ══════════════════════════════════════════════════════════════════════════════

viewRadios.forEach((radio) => {
  radio.addEventListener('change', () => {
    if (!radio.checked) return;
    state.currentView = radio.value;
    applyView();
  });
});

function applyView() {
  if (!state.imageLoaded) return;

  if (state.currentView === 'bbox' && state.bboxSrc) {
    showImage(state.bboxSrc);
  } else if (state.currentView === 'gradcam' && state.gradcamSrc) {
    showImage(state.gradcamSrc);
  } else {
    showImage(state.origSrc);
  }
}


// ══════════════════════════════════════════════════════════════════════════════
// 5. Tab navigation
// ══════════════════════════════════════════════════════════════════════════════

navTabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    const targetId = tab.dataset.section;
    navTabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    sections.forEach(sec => sec.classList.toggle('hidden', sec.id !== targetId));
    if (targetId !== 'section-inference') {
      console.log(`[VocAssist] tab: ${targetId}`);
    }
  });
});


// ══════════════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════════════

function showImage(src) {
  previewImage.src = src;
  previewImage.classList.remove('hidden');
  viewerPlaceholder.style.display = 'none';
}

function showLoading(on) {
  loadingOverlay.classList.toggle('hidden', !on);
}

function showError(msg) {
  errorToast.textContent = `Error: ${msg}`;
  errorToast.classList.remove('hidden');
  setTimeout(() => errorToast.classList.add('hidden'), 5000);
}

function hideError() {
  errorToast.classList.add('hidden');
}