/* OpenVINO Model Converter - front-end logic */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  info: null, modes: [], sources: [], converted: [],
  dlInfo: null, cvInfo: null, currentMode: null, currentTask: null,
};

/* ---------------------------------------------------------------- helpers */
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
function gb(bytes) { return (bytes / 1e9).toFixed(1); }

async function api(url, opts) {
  const r = await fetch(url, Object.assign({ headers: { "Content-Type": "application/json" } }, opts));
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch (e) {} throw new Error(m); }
  return r.json();
}

/* ---------------------------------------------------------------- init */
async function init() {
  await Promise.all([loadInfo(), loadModes(), loadModels()]);
  bindEvents();
  renderModes();
  renderModelSelect();
  bindTaskStream();
}
async function loadInfo() {
  state.info = await api("/api/info");
  $("paths").textContent =
    `cache: ${state.info.paths.cache}   ·   originals: ${state.info.paths.originals}   ·   output: ${state.info.paths.output}`;
  const chips = $("versions");
  chips.innerHTML = "";
  for (const [k, v] of Object.entries(state.info.versions)) {
    if (v == null) continue;
    chips.appendChild(el("span", "chip mono", `${k}=${v}`));
  }
}
async function loadModes() {
  state.modes = (await api("/api/modes")).modes;
}
async function loadModels() {
  const d = await api("/api/models");
  state.sources = d.sources;
  state.converted = d.converted;
  renderModelSelect();
}

/* ---------------------------------------------------------------- tabs */
function bindEvents() {
  document.querySelectorAll(".tab").forEach((t) => {
    t.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
      document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + t.dataset.panel));
    });
  });
  $("dl-validate").addEventListener("click", validateDownload);
  $("dl-run").addEventListener("click", runDownload);
  $("cv-refresh").addEventListener("click", loadModels);
  $("cv-model").addEventListener("change", onModelChange);
  $("cv-mode").addEventListener("change", onModeChange);
  $("cv-selftest").addEventListener("click", runSelfTest);
  $("cv-run").addEventListener("click", runConvert);
  $("task-cancel").addEventListener("click", () => api("/api/task/cancel", { method: "POST" }).catch(() => {}));
  $("task-clear").addEventListener("click", () => { $("task-log").innerHTML = ""; $("stages").innerHTML = ""; });
}

/* ---------------------------------------------------------------- Download tab */
function setInfo(container, text, kind) {
  const c = $(container);
  c.textContent = text || "";
  c.className = "info" + (text ? " show " + (kind || "ok") : "");
}
async function validateDownload() {
  const text = $("dl-text").value.trim();
  if (!text) return setInfo("dl-info", "Enter a model link / id / path.", "bad");
  setInfo("dl-info", "Validating…", "ok");
  try {
    const res = await api("/api/hf/validate", { method: "POST", body: JSON.stringify({ text, token: $("dl-token").value }) });
    state.dlInfo = res;
    renderDlInfo(res);
    updateDlChecks();
  } catch (e) { setInfo("dl-info", "Error: " + e.message, "bad"); state.dlInfo = null; }
}
function renderDlInfo(res) {
  const i = res.info;
  if (!i || i.ok === false) {
    // validation failed (e.g. HF API error / rate limit / model not found)
    state.dlInfo = null;
    state.dlNeededBytes = 0;
    setInfo("dl-info", "Validation failed: " + (i && i.error ? i.error : "unknown error"), "bad");
    updateDlChecks();
    return;
  }
  if (res.kind === "local") {
    setInfo("dl-info",
      `Local model: ${i.name}\nmodel_type: ${i.model_type} · task: ${i.task} · size: ${i.size_gb} GB` +
      (i.is_quantized ? "\n⚠ already quantized (not convertible)" : "") +
      (i.is_ov ? "\n⚠ already an OpenVINO model" : "") +
      (i.is_gguf ? "\n⚠ GGUF file (not convertible)" : ""),
      i.ok ? "ok" : "bad");
    $("dl-dest-input").value = i.path;
    state.dlNeededBytes = i.size_bytes;
  } else {
    setInfo("dl-info",
      `Model: ${i.id} (${i.pipeline_tag || "?"})\nfiles: ${i.files} · total: ${i.total_gb} GB · license: ${i.license || "?"}` +
      (i.gated ? "\n⚠ GATED — needs an HF token" : ""),
      i.ok ? "ok" : "bad");
    const org = i.id.split("/")[0];
    const dest = `T:\\models\\${org}\\${i.id.split("/")[1]}`;
    $("dl-dest-input").value = dest;
    state.dlNeededBytes = i.total_bytes;
  }
}
function updateDlChecks() {
  const i = state.dlInfo;
  if (!i) { $("dl-run").disabled = true; return; }
  const needed = (state.dlNeededBytes || 0) * 1.05;
  const free = (state.info.disk_free_gb || 0) * 1e9;
  const okDisk = free >= needed;
  $("dl-disk").innerHTML = "";
  $("dl-disk").appendChild(el("div", okDisk ? "checkline ok" : "checkline bad",
    okDisk ? `Disk: OK (free ${gb(free)} GB ≥ needed ${gb(needed)} GB)` : `Disk: NOT ENOUGH (free ${gb(free)} GB < needed ${gb(needed)} GB)`));
  const dlBtn = $("dl-run");
  dlBtn.disabled = !(okDisk && i.kind === "hf" && i.info.ok);
}
async function runDownload() {
  if (!state.dlInfo || state.dlInfo.kind !== "hf") return;
  const id = state.dlInfo.info.id;
  const body = {
    model_id: id,
    dest: $("dl-dest-input").value.trim() || null,
    revision: $("dl-rev").value || null,
    token: $("dl-token").value || null,
    include_only: $("dl-include").checked,
  };
  await startTask("download", body, "/api/download");
}

/* ---------------------------------------------------------------- Convert tab */
function renderModelSelect() {
  const sel = $("cv-model");
  const prev = sel.value;
  sel.innerHTML = "";
  sel.appendChild(el("option", null, "— select a scanned model —"));
  for (const s of state.sources) {
    const o = el("option", null, `${s.name}  (${s.size_gb} GB, ${s.task}, ${s.model_type || "?"})`);
    o.value = s.path;
    sel.appendChild(o);
  }
  if (state.converted.length) {
    const og = document.createElement("optgroup");
    og.label = "already converted (OV)";
    for (const c of state.converted) {
      const o = el("option", null, `${c.name}  (${c.size_gb} GB, OV)`);
      o.value = "ov:" + c.path;
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
  onModelChange();
}
async function onModelChange() {
  const v = $("cv-model").value;
  $("cv-meta").innerHTML = "";
  if (!v) { state.cvInfo = null; updateCvChecks(); return; }
  if (v.startsWith("ov:")) {
    state.cvInfo = null;
    setInfo("cv-meta", "This is an already-converted OpenVINO model. Pick a dense source instead.", "bad");
    updateCvChecks();
    return;
  }
  const src = state.sources.find((s) => s.path === v);
  if (src) {
    state.cvInfo = src;
    setInfo("cv-meta",
      `${src.name} · ${src.task} · ${src.model_type || "?"} · ${src.size_gb} GB · tokenizer: ${src.has_tokenizer ? "yes" : "no"}` +
      (src.is_moe ? "\n· MoE model (int2-mix available)" : ""), "ok");
    $("cv-task").value = src.task;
    $("cv-download-first").checked = false;
    state.cvParams = src.size_bytes / 2;
  } else {
    // custom / pasted path
    const p = v.replace(/^[A-Za-z]:[\\/]/, "");
    $("cv-text").value = p || "";
  }
  updateCvName();
  updateCvChecks();
}
function onModeChange() {
  const m = state.modes.find((x) => x.id === $("cv-mode").value);
  state.currentMode = m;
  const gsel = $("cv-group");
  gsel.innerHTML = "";
  for (const g of m.group_size_choices) {
    const o = el("option", null, g === -1 ? "-1 (per-channel)" : String(g));
    o.value = g;
    gsel.appendChild(o);
  }
  gsel.value = m.default_group_size;
  $("cv-ratio").disabled = m.moe_only;
  setInfo("cv-mode-detail",
    `${m.label} · ${m.bits ? m.bits + " bits" : ""} · symmetric: ${m.symmetric === null ? "n/a" : m.symmetric}` +
    (m.requires_per_channel ? "\n⚠ requires per-channel (group_size = -1)" : "") +
    (m.moe_only ? "\nMoE models only: experts int2, rest int4 (AutoRound-style)" : "") +
    (m.help ? "\n" + m.help : ""), "ok");
  updateCvName();
  updateCvChecks();
}
async function runSelfTest() {
  $("cv-selftest").disabled = true;
  $("cv-selftest").textContent = "Testing…";
  try {
    const r = await api("/api/modes/self-test", { method: "POST" });
    const lines = Object.entries(r.result).map(([k, v]) => `${k}: ${v}`);
    setInfo("cv-mode-detail", "Self-test (tiny compress+compile on CPU):\n" + lines.join("\n"), "ok");
  } catch (e) { setInfo("cv-mode-detail", "Self-test error: " + e.message, "bad"); }
  $("cv-selftest").disabled = false;
  $("cv-selftest").textContent = "Self-test modes";
}
function updateCvName() {
  const base = currentBase();
  const mode = state.currentMode ? state.currentMode.id : "int4_sym";
  const root = state.info.paths.output;
  $("cv-outdir").value = `${root}\\${base}-${modeToken(mode)}-ov`;
}
function currentBase() {
  const v = $("cv-model").value;
  if (state.cvInfo) return state.cvInfo.name;
  const t = $("cv-text").value.trim();
  if (t) return t.split(/[\\/]/).pop() || t;
  return "model";
}
function modeToken(mode) {
  const map = { int8_sym: "int8", int8_asym: "int8", int4_sym: "int4", int4_asym: "int4",
    int3_sym: "int3", int2_sym: "int2", nf4: "nf4", mxfp4: "mxfp4", mxfp8_e4m3: "mxfp8",
    fp8_e4m3: "fp8e4m3", cb4: "cb4", int2_mix: "int2-mix", int3_mix: "int3-mix", none: "fp16" };
  return map[mode] || mode;
}
function updateCvChecks() {
  const m = state.currentMode;
  const params = state.cvParams || 0;
  const bits = m ? (m.bits || 16) : 16;
  const wrap = $("cv-errors");
  wrap.innerHTML = "";
  if (!m) { $("cv-run").disabled = true; return; }
  const needs = estimateConvertNeeded(params, bits);
  const free = (state.info.disk_free_gb || 0) * 1e9;
  const okDisk = free >= needs;
  const line = $("cv-disk");
  line.innerHTML = "";
  line.appendChild(el("div", okDisk ? "checkline ok" : "checkline bad",
    okDisk ? `Disk: OK (free ${gb(free)} GB ≥ needed ~${gb(needs)} GB)` : `Disk: NOT ENOUGH (free ${gb(free)} GB < needed ~${gb(needs)} GB)`));

  const ramNeeded = params * 4 * 1.2;
  const vm = state.info.virtual_memory;
  const lineR = $("cv-ram");
  lineR.innerHTML = "";
  if (vm) {
    const avail = vm.avail_virtual_gb * 1e9;
    const okRam = avail >= ramNeeded;
    lineR.appendChild(el("div", okRam ? "checkline ok" : "checkline bad",
      okRam ? `Virtual memory: OK (avail ${vm.avail_virtual_gb} GB ≥ peak ~${gb(ramNeeded)} GB)` :
        `Virtual memory: NOT ENOUGH (avail ${vm.avail_virtual_gb} GB < peak ~${gb(ramNeeded)} GB)`));
  } else {
    lineR.appendChild(el("div", "checkline ok", "Virtual memory: non-Windows, check skipped"));
  }

  const errors = [];
  if (m.requires_per_channel && Number($("cv-group").value) !== -1) errors.push("INT8 mode requires group_size = -1.");
  if (m.moe_only && state.cvInfo && !state.cvInfo.is_moe) errors.push("This mode requires a MoE model.");
  if (m.available === false) errors.push("Mode not available in the installed NNCF.");
  if (!state.cvInfo && !$("cv-text").value.trim() && !$("cv-model").value) errors.push("Choose a model.");
  if (errors.length) { wrap.innerHTML = ""; errors.forEach((e) => wrap.appendChild(el("div", null, "⚠ " + e))); }
  $("cv-run").disabled = !(okDisk && errors.length === 0);
}
function estimateConvertNeeded(params, bits) {
  if (!params) return 0;
  const fp16 = params * 2;
  const result = params * (bits / 8);
  return fp16 * 1.15 + result * 1.1 + fp16; // + keep source
}

async function runConvert() {
  const cfg = {
    model_id: currentBase(),
    model_path: $("cv-model").value && !$("cv-model").value.startsWith("ov:") ? $("cv-model").value : null,
    download: $("cv-download-first").checked,
    task: $("cv-task").value.trim(),
    mode: state.currentMode ? state.currentMode.id : "int4_sym",
    group_size: Number($("cv-group").value),
    all_layers: $("cv-all-layers").checked,
    ratio: $("cv-ratio").value ? Number($("cv-ratio").value) : null,
    backup: $("cv-backup").value,
    data_aware: {
      awq: $("cv-awq").checked, scale_estimation: $("cv-scale").checked,
      gptq: $("cv-gptq").checked, lora_correction: $("cv-lora").checked,
      dataset: $("cv-dataset").value.trim() || null,
      num_samples: Number($("cv-nsamples").value) || 128,
    },
    only_text: $("cv-only-text").checked,
    delete_intermediate: $("cv-delete-int").checked,
    output_dir: $("cv-outdir").value.trim() || null,
    run_genai_test: $("cv-genai").checked,
  };
  await startTask("convert", cfg, "/api/convert");
}

/* ---------------------------------------------------------------- Task streaming */
const STAGES = ["validate", "download", "export", "compress", "package", "tokenizer", "genai_test"];
function bindTaskStream() {
  pollStatus();
}
function renderStages(kind) {
  const box = $("stages");
  box.innerHTML = "";
  const list = kind === "download" ? ["download"] : STAGES;
  for (const s of list) {
    const n = el("div", "stage", "");
    n.dataset.stage = s;
    n.appendChild(el("span", "dot"));
    n.appendChild(document.createTextNode(s));
    box.appendChild(n);
  }
}
function stageNode(name) {
  return document.querySelector(`.stage[data-stage="${name}"]`);
}
function setStage(name, status) {
  const n = stageNode(name);
  if (!n) return;
  n.classList.remove("running", "done", "fail");
  n.classList.add(status);
}
function appendLog(text) {
  const log = $("task-log");
  const line = el("div", "log-line", text);
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}
function parseEvent(line) {
  const m = line.match(/^@@(\w+)\s+(\S+)\s*\|\s*(.*)$/);
  if (!m) return null;
  return { ev: m[1], stage: m[2], payload: m[3] };
}
async function startTask(kind, body, url) {
  let id;
  try {
    id = (await api(url, { method: "POST", body: JSON.stringify(body) })).task_id;
  } catch (e) { appendLog("ERROR: " + e.message); return; }
  state.currentTask = id;
  renderStages(kind);
  appendLog("task started: " + id + " (" + kind + ")");
  $("task-status").textContent = "running";
  $("task-status").className = "chip busy";
  $("task-log").innerHTML = "";
  const es = new EventSource("/api/task/stream?task_id=" + id);
  es.onmessage = (e) => {
    try { const d = JSON.parse(e.data); onTaskLine(d.line); } catch (err) { appendLog(e.data); }
  };
  es.addEventListener("done", (e) => {
    es.close();
    try { const d = JSON.parse(e.data); $("task-status").textContent = "done rc=" + d.returncode; }
    catch (err) { $("task-status").textContent = "done"; }
    $("task-status").className = "chip done";
    loadModels(); loadInfo();
  });
  es.onerror = () => { es.close(); $("task-status").textContent = "stream closed"; $("task-status").className = "chip"; };
}
function onTaskLine(line) {
  if (line == null) return;
  const ev = parseEvent(line);
  if (ev) {
    if (ev.ev === "STAGE") {
      const [status, ...rest] = ev.payload.split(" ");
      if (status === "start") { renderStages(undefined); } // ensure stages exist
      if (status === "done") { setStage(ev.stage, "done"); appendLog("✔ " + ev.stage + " " + rest.join(" ")); }
      else if (status === "fail") { setStage(ev.stage, "fail"); appendLog("✕ " + ev.stage + " " + rest.join(" ")); }
      else { setStage(ev.stage, "running"); appendLog("▶ " + ev.stage); }
    } else if (ev.ev === "LOG") {
      appendLog(ev.payload);
    } else if (ev.ev === "META") {
      try { const r = JSON.parse(ev.payload); appendLog("\n— result —\noutput: " + (r.output_dir || "?") + "\n" + JSON.stringify(r.genai_test || {}, null, 1)); }
      catch (err) { appendLog(ev.payload); }
    }
  } else {
    appendLog(line);
  }
}
async function pollStatus() {
  try {
    const s = await api("/api/task/status");
    if (s.task && s.task.kind) {
      $("task-status").textContent = s.busy ? "running" : (s.done ? "finished" : "idle");
      $("task-status").className = "chip " + (s.busy ? "busy" : s.done ? "done" : "");
    }
  } catch (e) {}
  setTimeout(pollStatus, 3000);
}

init();
