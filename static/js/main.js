/* ============================================================
   main.js — Speech-Driven Robot Localization
   ============================================================ */
"use strict";

// ---------------------------------------------------------------------------
// Canvas renderer
// ---------------------------------------------------------------------------
const canvas = document.getElementById("mapCanvas");
const ctx    = canvas.getContext("2d");

const GRID   = 10;   // cells
const COLORS = {
  gridLine  : "#2e3455",
  cellBg    : "#1a1d27",
  cellAlt   : "#1e2131",
  robotBg   : "#4f8ef7",
  robotText : "#ffffff",
  trail     : "rgba(79,142,247,0.18)",
};

let robotState = { x: 5, y: 5, direction: "north", grid_size: GRID };
let trail      = [];   // array of {x, y}

function drawMap(state) {
  const W  = canvas.width;
  const H  = canvas.height;
  const cw = W / GRID;
  const ch = H / GRID;

  ctx.clearRect(0, 0, W, H);

  // Background cells
  for (let r = 0; r < GRID; r++) {
    for (let c = 0; c < GRID; c++) {
      ctx.fillStyle = (r + c) % 2 === 0 ? COLORS.cellBg : COLORS.cellAlt;
      ctx.fillRect(c * cw, r * ch, cw, ch);
    }
  }

  // Grid lines
  ctx.strokeStyle = COLORS.gridLine;
  ctx.lineWidth   = 0.5;
  for (let i = 0; i <= GRID; i++) {
    ctx.beginPath();
    ctx.moveTo(i * cw, 0);
    ctx.lineTo(i * cw, H);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i * ch);
    ctx.lineTo(W, i * ch);
    ctx.stroke();
  }

  // Trail
  trail.forEach((pt) => {
    ctx.fillStyle = COLORS.trail;
    ctx.fillRect(pt.x * cw + 2, pt.y * ch + 2, cw - 4, ch - 4);
  });

  // Robot cell highlight
  ctx.fillStyle = COLORS.robotBg;
  const rx = state.x * cw + 2;
  const ry = state.y * ch + 2;
  const rw = cw - 4;
  const rh = ch - 4;
  roundRect(ctx, rx, ry, rw, rh, 6);
  ctx.fill();

  // Robot emoji
  const fontSize = Math.floor(cw * 0.52);
  ctx.font      = `${fontSize}px serif`;
  ctx.textAlign    = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("🤖", state.x * cw + cw / 2, state.y * ch + ch / 2);

  // Direction arrow
  const arrow = { north: "↑", south: "↓", east: "→", west: "←" }[state.direction];
  const arrowSize = Math.floor(cw * 0.35);
  ctx.font      = `bold ${arrowSize}px sans-serif`;
  ctx.fillStyle = COLORS.robotText;
  ctx.fillText(arrow, state.x * cw + cw - arrowSize * 0.55, state.y * ch + arrowSize * 0.7);

  // Axis labels (col / row numbers)
  ctx.font      = "10px monospace";
  ctx.fillStyle = "#3e4a6a";
  ctx.textAlign = "center";
  for (let i = 0; i < GRID; i++) {
    ctx.fillText(i, i * cw + cw / 2, H - 3);
    ctx.fillText(i, 8, i * ch + ch / 2 + 4);
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

function updateMeta(state) {
  document.getElementById("posLabel").textContent =
    `Position: (${state.x}, ${state.y})`;
  const dir = state.direction;
  const arrow = { north: "↑", south: "↓", east: "→", west: "←" }[dir];
  const dirCap = dir.charAt(0).toUpperCase() + dir.slice(1);
  document.getElementById("dirLabel").textContent = `Facing: ${arrow} ${dirCap}`;
}

function applyState(state) {
  // Record trail
  const last = trail[trail.length - 1];
  if (!last || last.x !== robotState.x || last.y !== robotState.y) {
    trail.push({ x: robotState.x, y: robotState.y });
    if (trail.length > 30) trail.shift();
  }
  robotState = state;
  drawMap(state);
  updateMeta(state);
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------
async function processText(text) {
  const res = await fetch("/api/process", {
    method  : "POST",
    headers : { "Content-Type": "application/json" },
    body    : JSON.stringify({ text }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Server error ${res.status}`);
  }
  return res.json();
}

async function resetRobot() {
  const res = await fetch("/api/reset", { method: "POST" });
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  return res.json();
}

async function fetchState() {
  const res = await fetch("/api/state");
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  return res.json();
}

async function fetchSettings() {
  const res = await fetch("/api/settings");
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  return res.json();
}

async function saveSettings(payload) {
  const res = await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Server error ${res.status}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Log
// ---------------------------------------------------------------------------
const logList = document.getElementById("logList");

function addLog(text, commands) {
  const placeholder = logList.querySelector(".log-placeholder");
  if (placeholder) placeholder.remove();

  const now    = new Date();
  const time   = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const li     = document.createElement("li");
  li.className = "log-entry";
  li.innerHTML = `
    <div class="log-time">${time}</div>
    <div class="log-text">"${escHtml(text)}"</div>
    ${commands.length ? `<div class="log-cmds">▶ ${commands.join(", ")}</div>` : ""}
  `;
  logList.prepend(li);
}

function escHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ---------------------------------------------------------------------------
// Spatial context tags
// ---------------------------------------------------------------------------
const contextCard = document.getElementById("contextCard");
const contextTags = document.getElementById("contextTags");

function showContext(commands) {
  contextTags.innerHTML = "";
  if (!commands.length) {
    contextCard.hidden = true;
    return;
  }
  commands.forEach((cmd) => {
    const span       = document.createElement("span");
    span.className   = "tag";
    span.textContent = cmd;
    contextTags.appendChild(span);
  });
  contextCard.hidden = false;
}

// ---------------------------------------------------------------------------
// Submit handler (shared by button + Enter key + speech)
// ---------------------------------------------------------------------------
const textInput    = document.getElementById("textInput");
const transcriptEl = document.getElementById("transcript");
const transcriptTx = document.getElementById("transcriptText");
const dbPathInput = document.getElementById("dbPathInput");
const debugToggle = document.getElementById("debugToggle");
const saveSettingsBtn = document.getElementById("saveSettingsBtn");
const settingsStatus = document.getElementById("settingsStatus");
const effectiveDbPath = document.getElementById("effectiveDbPath");
const runtimeMode = document.getElementById("runtimeMode");
const settingsModeHint = document.getElementById("settingsModeHint");

function applySettingsState(settings) {
  const hostedReadonlyMode = Boolean(settings.hosted_readonly_mode);

  dbPathInput.value = settings.state_db_path || "";
  debugToggle.value = settings.flask_debug ? "1" : "0";
  effectiveDbPath.textContent = settings.effective_state_db_path || "Unavailable";
  runtimeMode.textContent = hostedReadonlyMode ? "Hosted read-only" : "Local writable";

  dbPathInput.disabled = hostedReadonlyMode;
  debugToggle.disabled = hostedReadonlyMode;
  saveSettingsBtn.disabled = hostedReadonlyMode;

  settingsModeHint.hidden = !hostedReadonlyMode;
  settingsModeHint.textContent = hostedReadonlyMode
    ? "Hosted deployments are read-only here. Update environment variables in your deployment settings instead."
    : "";

  if (hostedReadonlyMode) {
    settingsStatus.textContent = "Settings are managed by deployment environment variables.";
  } else if (settingsStatus.textContent === "Settings are managed by deployment environment variables.") {
    settingsStatus.textContent = "";
  }
}

async function handleSubmit(text) {
  if (!text.trim()) return;
  try {
    const data = await processText(text);
    applyState(data.robot_state);
    addLog(data.text, data.commands);
    showContext(data.commands);
  } catch (err) {
    console.error("Failed to process command:", err.message);
  }
}

document.getElementById("sendBtn").addEventListener("click", () => {
  const v = textInput.value.trim();
  if (!v) return;
  textInput.value = "";
  handleSubmit(v);
});

textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    const v = textInput.value.trim();
    if (!v) return;
    textInput.value = "";
    handleSubmit(v);
  }
});

// ---------------------------------------------------------------------------
// Clear log
// ---------------------------------------------------------------------------
document.getElementById("clearLog").addEventListener("click", () => {
  logList.innerHTML = '<li class="log-placeholder">No commands yet.</li>';
  contextCard.hidden = true;
  trail = [];
  drawMap(robotState);
});

// ---------------------------------------------------------------------------
// Reset button
// ---------------------------------------------------------------------------
document.getElementById("resetBtn").addEventListener("click", async () => {
  trail = [];
  try {
    const data = await resetRobot();
    applyState(data);
    addLog("[reset]", ["reset"]);
    showContext(["reset"]);
  } catch (err) {
    console.error("Failed to reset robot:", err.message);
  }
});

// ---------------------------------------------------------------------------
// Web Speech API
// ---------------------------------------------------------------------------
const micBtn    = document.getElementById("micBtn");
const micLabel  = document.getElementById("micLabel");
const statusDot = document.getElementById("statusDot");

const SpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition || null;

let recognition = null;
let recognising  = false;

if (SpeechRecognition) {
  recognition             = new SpeechRecognition();
  recognition.continuous  = false;
  recognition.lang        = "en-US";
  recognition.interimResults = true;

  recognition.onstart = () => {
    recognising = true;
    micBtn.classList.add("active");
    micLabel.textContent = "Listening…";
    statusDot.classList.add("listening");
  };

  recognition.onresult = (e) => {
    let interim = "";
    let final   = "";
    for (const result of e.results) {
      if (result.isFinal) {
        final += result[0].transcript;
      } else {
        interim += result[0].transcript;
      }
    }
    transcriptTx.textContent  = final || interim;
    transcriptEl.hidden       = false;
    if (final) handleSubmit(final.trim());
  };

  recognition.onerror = (e) => {
    console.error("Speech recognition error:", e.error);
    stopListening();
  };

  recognition.onend = () => stopListening();

  micBtn.addEventListener("click", () => {
    if (recognising) {
      recognition.stop();
    } else {
      transcriptEl.hidden = true;
      recognition.start();
    }
  });
} else {
  micBtn.disabled         = true;
  micLabel.textContent    = "Speech API not supported";
  micBtn.title            = "Your browser does not support the Web Speech API";
}

function stopListening() {
  recognising = false;
  micBtn.classList.remove("active");
  micLabel.textContent = "Start Listening";
  statusDot.classList.remove("listening");
}

saveSettingsBtn.addEventListener("click", async () => {
  if (saveSettingsBtn.disabled) {
    return;
  }

  const payload = {
    state_db_path: dbPathInput.value.trim(),
    flask_debug: debugToggle.value === "1",
  };
  settingsStatus.textContent = "Saving...";
  try {
    const saved = await saveSettings(payload);
    applySettingsState(saved);
    if (!saved.hosted_readonly_mode) {
      settingsStatus.textContent = "Saved. Restart server to apply debug mode.";
    }
  } catch (err) {
    settingsStatus.textContent = `Save failed: ${err.message}`;
  }
});

// ---------------------------------------------------------------------------
// Bootstrap: load current state
// ---------------------------------------------------------------------------
(async () => {
  try {
    const settings = await fetchSettings();
    applySettingsState(settings);

    const data = await fetchState();
    applyState(data);
  } catch (err) {
    console.error("Failed to load initial data:", err.message);
    // Render a default grid so the UI is not blank
    effectiveDbPath.textContent = "Unavailable";
    runtimeMode.textContent = "Unavailable";
    drawMap(robotState);
    updateMeta(robotState);
  }
})();
