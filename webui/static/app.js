/* OpenVINO Model Converter - front-end logic */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  info: null, modes: [], sources: [], converted: [],
  dlInfo: null, cvInfo: null, currentMode: null, currentTask: null,
  taskActive: false, dlFiles: null, dlSelected: null, dlLocalComplete: null,
};
let activeProgressId = null;
let dlLocalDebounce = null;
let hfEnvDebounce = null;

/* ---------------------------------------------------------------- helpers */
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
function gb(bytes) { return (bytes / 1e9).toFixed(1); }
function humanSize(bytes) {
  const b = Number(bytes) || 0;
  if (b >= 1e9) return (b / 1e9).toFixed(2) + " GB";
  if (b >= 1e6) return Math.round(b / 1e6) + " MB";
  if (b >= 1e3) return Math.round(b / 1e3) + " KB";
  return b + " B";
}
function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function short(s) {
  return String(s).length === 64 ? String(s).slice(0, 12) + "\u2026" : String(s);
}
function filePurpose(name) {
  if (name === "config.json") return "Model config";
  if (name === "model.safetensors.index.json") return "Weights index";
  if (name.endsWith(".safetensors") && !name.includes(".safetensors.index.json")) return "Model weights";
  if (name === "tokenizer.json") return "Tokenizer";
  if (name === "tokenizer_config.json") return "Tokenizer config";
  if (name === "merges.txt") return "BPE merges";
  if (name === "vocab.json") return "Vocabulary";
  if (name === "preprocessor_config.json") return "Image preprocessor";
  if (name === "processor_config.json") return "Processor";
  if (name === "video_preprocessor_config.json") return "Video preprocessor";
  if (name === "chat_template.jinja") return "Chat template";
  if (name === "generation_config.json") return "Generation config";
  if (name === "README.md") return "Model card";
  if (name === "LICENSE") return "License";
  if (name === ".gitattributes") return "Git metadata";
  return "Other";
}

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
  $("hf-home").value = state.info.paths.cache;
  $("hf-hub-cache").value = state.info.paths.cache + "\\hub";
  validateHfEnv();
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
async function validateHfEnv() {
  const res = $("hf-env-res");
  if (!res) return;
  const paths = { "HF_HOME": $("hf-home").value.trim(), "HF_HUB_CACHE": $("hf-hub-cache").value.trim() };
  const bad = Object.entries(paths).find(([, v]) => v === "" || !/^[A-Za-z]:[\\/]/.test(v));
  if (bad) {
    res.className = "banner show bad";
    res.textContent = "Invalid path: " + (bad[1] || "(" + bad[0] + " is empty)");
    return;
  }
  try {
    const entries = Object.entries(paths);
    const results = await Promise.all(entries.map(([, p]) => api("/api/disk?path=" + encodeURIComponent(p))));
    const driveOf = (p) => (String(p).match(/^([A-Za-z]):/) || [])[1] || "?";
    for (let idx = 0; idx < entries.length; idx++) {
      const p = entries[idx][1];
      const r = results[idx];
      if (r.ok === false) {
        let driveOk = false;
        const drive = driveOf(p);
        if (drive !== "?") {
          try { driveOk = (await api("/api/disk?path=" + encodeURIComponent(drive + ":\\"))).ok === true; } catch (e) {}
        }
        if (!driveOk) {
          res.className = "banner show bad";
          res.textContent = "Insufficient disk (invalid drive)";
          return;
        }
        res.className = "banner show bad";
        res.textContent = "Directory does not exist: " + p;
        const create = el("button", "btn btn-small", "Create");
        create.id = "hf-create";
        create.onclick = async () => {
          try {
            await api("/api/mkdir", { method: "POST", body: JSON.stringify({ path: p }) });
            validateHfEnv();
          } catch (e2) {
            res.textContent = "Create failed: " + e2.message;
          }
        };
        res.appendChild(create);
        return;
      }
    }
    const free = Math.min(...results.map((r) => Number(r.free_gb) || 0));
    if (free < 10) {
      res.className = "banner show bad";
      res.textContent = "Insufficient disk (" + free.toFixed(1) + " GB free on cache drive)";
      return;
    }
    const drive = driveOf(Object.values(paths)[0]);
    res.className = "banner show ok";
    res.textContent = "Cache paths OK — drive " + drive + ": has " + free.toFixed(1) + " GB free";
  } catch (e) {
    res.className = "banner show bad";
    res.textContent = "Error checking cache disk: " + e.message;
  }
}

/* ---------------------------------------------------------------- tabs */
function bindEvents() {
  document.querySelectorAll(".tab").forEach((t) => {
    t.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
      document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + t.dataset.panel));
    });
  });
  if (!document.getElementById("dl-path-spinner")) {
    const sp = document.createElement("span");
    sp.className = "spinner";
    sp.id = "dl-path-spinner";
    sp.hidden = true;
    const dir = document.querySelector(".dirline");
    if (dir) dir.appendChild(sp);
  }
  $("dl-validate").addEventListener("click", validateDownload);
  $("dl-run").addEventListener("click", runDownload);
  const tagsEl = $("dl-tags");
  if (tagsEl) tagsEl.addEventListener("click", (e) => {
    const t = e.target && e.target.closest ? e.target.closest("[data-fill]") : null;
    if (t && t.dataset.fill) {
      $("dl-text").value = t.dataset.fill;
      validateDownload();
    }
  });
  const onDestInput = () => {
    clearTimeout(dlLocalDebounce);
    dlLocalDebounce = setTimeout(() => {
      const sp = $("dl-path-spinner");
      if (sp) sp.hidden = false;
      updateDirLink();
      updateLocalStatus();
    }, 250);
  };
  $("dl-root").addEventListener("input", onDestInput);
  $("dl-sub").addEventListener("input", onDestInput);
  $("dl-files").addEventListener("change", () => {
    state.dlSelected = new Set();
    document.querySelectorAll("#dl-files input[type=checkbox]:checked:not(:disabled)").forEach((cb) => state.dlSelected.add(cb.value));
    updateDlChecks();
  });
  $("dl-all").addEventListener("click", () => setFilesAll(true));
  $("dl-none").addEventListener("click", () => setFilesAll(false));
  const onHfEnvInput = () => {
    clearTimeout(hfEnvDebounce);
    hfEnvDebounce = setTimeout(validateHfEnv, 300);
  };
  $("hf-home").addEventListener("input", onHfEnvInput);
  $("hf-hub-cache").addEventListener("input", onHfEnvInput);
  $("dl-dir-link").addEventListener("click", (e) => {
    e.preventDefault();
    api("/api/open-dir", { method: "POST", body: JSON.stringify({ path: composeDest() }) })
      .then((r) => { if (!r.ok) appendLog("open folder failed: " + (r.error || "?")); })
      .catch((err) => appendLog("open folder error: " + err.message));
  });
  $("dl-verify").addEventListener("click", verifyHashes);
  $("cv-refresh").addEventListener("click", loadModels);
  $("cv-model").addEventListener("change", onModelChange);
  $("cv-mode").addEventListener("change", onModeChange);
  $("cv-selftest").addEventListener("click", runSelfTest);
  $("cv-run").addEventListener("click", runConvert);
  const cancelTask = () => api("/api/task/cancel", { method: "POST" }).catch(() => {});
  $("task-cancel").addEventListener("click", cancelTask);
  $("task-cancel").disabled = true;
  $("dl-cancel").addEventListener("click", cancelTask);
  $("cv-cancel").addEventListener("click", cancelTask);
  $("task-clear").addEventListener("click", () => { $("task-log").innerHTML = ""; $("stages").innerHTML = ""; });
  $("task-collapse").addEventListener("click", () => {
    $("task-panel").classList.toggle("collapsed");
    const collapsed = $("task-panel").classList.contains("collapsed");
    $("task-collapse").textContent = collapsed ? "▾" : "▴";
    $("task-collapse").title = collapsed ? "Expand" : "Collapse";
  });
}

/* ---------------------------------------------------------------- Download tab */
function setInfo(container, text, kind) {
  const c = $(container);
  c.textContent = text || "";
  c.className = "info" + (text ? " show " + (kind || "ok") : "");
}
function ensureTaskPanel() {
  $("task-panel").classList.remove("collapsed");
  const btn = $("task-collapse");
  btn.textContent = "▴";
  btn.title = "Collapse";
  renderStages("validate");
  const chip = $("task-status");
  chip.textContent = "validating…";
  chip.className = "chip busy";
}
async function validateDownload() {
  const text = $("dl-text").value.trim();
  if (!text) return setInfo("dl-info", "Enter a model link / id / path.", "bad");
  setInfo("dl-info", "Validating…", "ok");
  ensureTaskPanel();
  setStage("validate", "running");
  appendLog("validating " + text + " ...");
  try {
    const res = await api("/api/hf/validate", { method: "POST", body: JSON.stringify({ text, token: $("dl-token").value || null }) });
    state.dlInfo = res;
    renderDlInfo(res);
    await updateLocalStatus();
    updateDlChecks();
    if (res.info && res.info.ok) {
      setStage("validate", "done");
      $("task-status").textContent = "validated";
      $("task-status").className = "chip done";
      appendLog(`validate: ${res.info.id} (${res.info.total_gb} GB, ${(res.info.files || []).length} files)`);
      if (res.info.local_complete) appendLog("already downloaded locally: " + res.info.local_dir);
      if (res.info.local_exists && !res.info.local_complete) appendLog("local copy exists but incomplete (missing: " + res.info.local_missing.length + " files)");
    } else {
      setStage("validate", "fail");
      $("task-status").textContent = "failed";
      $("task-status").className = "chip";
    }
  } catch (e) {
    setInfo("dl-info", "Error: " + e.message, "bad");
    state.dlInfo = null;
    state.dlFiles = null;
    state.dlSelected = null;
    setStage("validate", "fail");
    $("task-status").textContent = "failed";
    $("task-status").className = "chip";
    appendLog("validate failed: " + e.message);
  }
  const log = $("task-log");
  log.scrollTop = log.scrollHeight;
  $("task-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}
function setSelectBtns() {
  const hasFiles = !!(state.dlFiles && state.dlFiles.length);
  const disabled = state.dlLocalComplete === true || !hasFiles;
  $("dl-all").disabled = disabled;
  $("dl-none").disabled = disabled;
}
function renderFilePicker(filesMeta, presentFiles) {
  const box = $("dl-files");
  if (!box) return;
  box.innerHTML = "";
  state.dlSelected = new Set();
  const files = filesMeta || [];
  if (!files.length) { box.classList.remove("show"); setSelectBtns(); return; }
  const present = new Set(presentFiles || []);
  for (const f of files) {
    const isPresent = present.has(f.name);
    const label = el("label", "check " + (isPresent ? "present" : "missing"));
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = f.name;
    cb.checked = true;
    if (isPresent) cb.disabled = true;
    label.appendChild(cb);
    label.appendChild(el("span", "fname", f.name));
    label.appendChild(el("span", "fsize", "(" + humanSize(f.size) + ")"));
    box.appendChild(label);
    if (cb.checked && !cb.disabled) state.dlSelected.add(f.name);
  }
  box.classList.add("show");
  setSelectBtns();
}
function setFilesAll(value) {
  const box = $("dl-files");
  if (!box) return;
  state.dlSelected = new Set();
  box.querySelectorAll("input[type=checkbox]").forEach((cb) => {
    if (cb.disabled) return;
    cb.checked = value;
    if (value) state.dlSelected.add(cb.value);
  });
  updateDlChecks();
}
function renderDlInfo(res) {
  const i = res.info;
  if (!i || i.ok === false) {
    // validation failed (e.g. HF API error / rate limit / model not found)
    state.dlInfo = null;
    state.dlFiles = null;
    state.dlSelected = null;
    state.dlNeededBytes = 0;
    setInfo("dl-info", "Validation failed: " + (i && i.error ? i.error : "unknown error"), "bad");
    $("dl-files").classList.remove("show");
    $("dl-files").textContent = "";
    setSelectBtns();
    $("dl-tags").innerHTML = "";
    $("dl-tags").hidden = true;
    $("dl-hash-card").hidden = true;
    updateDirLink();
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
    const parts = String(i.path || "").split(/[\\/]+/).filter(Boolean);
    $("dl-sub").value = parts[parts.length - 1] || "";
    $("dl-root").value = parts.slice(0, -1).join("\\") || "";
    $("dl-tags").innerHTML = "";
    $("dl-tags").hidden = true;
    updateDirLink();
    state.dlNeededBytes = i.size_bytes;
  } else {
    const files = i.files_meta || [];
    let html = "";
    if (i.gated) html += "\n⚠ GATED — needs an HF token";
    if (files.length) {
      const totalBytes = files.reduce((s, f) => s + (Number(f.size) || 0), 0);
      html += "<table class=\"mini-table\"><thead><tr><th>File</th><th>Size</th><th>Purpose</th></tr></thead><tbody>";
      for (const f of files) {
        html += "<tr><td>" + esc(f.name) + "</td><td>" + esc(humanSize(f.size)) + "</td><td>" + esc(filePurpose(f.name)) + "</td></tr>";
      }
      html += "</tbody><tfoot><tr><td>Total</td><td>" + esc(humanSize(totalBytes)) + "</td><td></td></tr></tfoot></table>";
    }
    const infoBox = $("dl-info");
    infoBox.innerHTML = html;
    infoBox.className = "info show " + (i.ok ? "ok" : "bad");
    $("dl-tags").innerHTML =
      "<span class=\"tag tag-link\" title=\"Click to fill and validate\" data-fill=\"" + esc(i.id) + "\">" + esc(i.id) + "</span> " +
      "<span class=\"tag\">" + esc(i.pipeline_tag || "?") + "</span> " +
      "<span class=\"tag\">" + esc(i.total_gb) + " GB</span> " +
      "<span class=\"tag\">" + esc(i.license || "?") + "</span> " +
      "<a class=\"hf-link\" href=\"https://huggingface.co/" + esc(i.id) + "\" target=\"_blank\" rel=\"noopener noreferrer\">Hugging Face <sup>↗</sup></a>";
    $("dl-tags").hidden = false;
    const org = i.id.split("/")[0];
    const name = i.id.split("/")[1];
    $("dl-sub").value = org + "/" + name;
    const root = $("dl-root").value.trim();
    if (root === "" || root === "T:\\models") $("dl-root").value = "T:\\models";
    updateDirLink();
    state.dlNeededBytes = i.total_bytes;
    state.dlFiles = i.files || [];
    renderFilePicker(i.files_meta || []);
  }
}
function composeDest() {
  const root = $("dl-root").value.trim().replace(/[\\/]+$/, "");
  const sub = $("dl-sub").value.trim().replace(/^[\\/]+/, "").replace(/\//g, "\\");
  return root + "\\" + sub;
}
function updateDirLink() {
  const link = $("dl-dir-link");
  if (!link) return;
  const show = !!(state.dlInfo && state.dlInfo.kind === "hf");
  const active = show && state.dlLocalExists === true;
  link.hidden = !show;
  link.textContent = show ? composeDest() : "";
  link.classList.toggle("active", active);
  link.style.pointerEvents = active ? "auto" : "none";
}
async function updateLocalStatus() {
  const box = $("dl-local");
  const sp = $("dl-path-spinner");
  if (!state.dlInfo || state.dlInfo.kind !== "hf" || !state.dlFiles) {
    if (sp) sp.hidden = true;
    box.className = "banner";
    box.textContent = "";
    state.dlLocalComplete = null;
    state.dlLocalExists = false;
    $("dl-dest-card").classList.toggle("done", false);
    $("dl-hash-card").hidden = true;
    updateDirLink();
    updateDlChecks();
    return;
  }
  let r;
  if (sp) sp.hidden = false;
  try {
    r = await api("/api/model/local-check", { method: "POST", body: JSON.stringify({ path: composeDest(), files: state.dlFiles }) });
  } catch (e) {
    if (sp) sp.hidden = true;
    box.className = "banner";
    box.textContent = "";
    state.dlLocalComplete = false;
    state.dlLocalExists = false;
    $("dl-dest-card").classList.toggle("done", false);
    $("dl-hash-card").hidden = true;
    updateDirLink();
    updateDlChecks();
    return;
  }
  if (sp) sp.hidden = true;
  const M = state.dlFiles.length;
  const missingN = (r.missing && r.missing.length) || 0;
  if (r.complete) {
    box.className = "banner show ok";
    box.textContent = "Already downloaded locally — Download disabled";
  } else if (r.present > 0) {
    box.className = "banner show bad";
    box.textContent = "Local copy exists but incomplete — missing " + missingN + " of " + M + " files: " + ((r.missing || []).join(", "));
  } else {
    box.className = "banner show bad";
    box.textContent = "Not present locally — will download " + M + " files";
  }
  state.dlLocalComplete = r.complete;
  state.dlLocalExists = r.exists === true;
  updateDirLink();
  $("dl-dest-card").classList.toggle("done", state.dlLocalComplete === true);
  $("dl-hash-card").hidden = !(state.dlInfo && state.dlInfo.kind === "hf" && r.present > 0);
  renderFilePicker(state.dlInfo.info.files_meta || [], r.present_files || []);
  updateDlChecks();
}
function updateDlChecks() {
  const i = state.dlInfo;
  setSelectBtns();
  $("dl-verify").disabled = !$("dl-hash-card") || $("dl-hash-card").hidden;
  if (!i) { $("dl-run").disabled = true; return; }
  const meta = (state.dlInfo.info.files_meta) || [];
  const byName = new Map(meta.map((f) => [f.name, f]));
  let selBytes = 0;
  if (state.dlSelected) {
    for (const name of state.dlSelected) {
      const m = byName.get(name);
      if (m) selBytes += Number(m.size) || 0;
    }
  }
  const needed = selBytes * 1.05;
  const free = (state.info.disk_free_gb || 0) * 1e9;
  const okDisk = free >= needed;
  $("dl-disk").innerHTML = "";
  $("dl-disk").appendChild(el("div", okDisk ? "banner show ok" : "banner show bad",
    okDisk ? `Disk: OK (free ${gb(free)} GB ≥ needed ${gb(needed)} GB)` : `Disk: NOT ENOUGH (free ${gb(free)} GB < needed ${gb(needed)} GB)`));
  const dlBtn = $("dl-run");
  const hasSel = state.dlSelected && state.dlSelected.size > 0;
  dlBtn.disabled = !(okDisk && i.kind === "hf" && i.info.ok && state.dlLocalComplete === false && hasSel);
}
async function runDownload() {
  if (!state.dlInfo || state.dlInfo.kind !== "hf") return;
  const id = state.dlInfo.info.id;
  const body = {
    model_id: id,
    dest: composeDest() || null,
    revision: $("dl-rev").value || null,
    token: $("dl-token").value || null,
    hf_home: $("hf-home").value.trim() || null,
    hf_hub_cache: $("hf-hub-cache").value.trim() || null,
    files: state.dlSelected ? [...state.dlSelected] : null,
  };
  await startTask("download", body, "/api/download");
}
async function verifyHashes() {
  const btn = $("dl-verify");
  const spin = $("dl-vspinner");
  const box = $("dl-hashres");
  const filesMeta = (state.dlInfo && state.dlInfo.info.files_meta) || [];
  btn.disabled = true;
  spin.hidden = false;
  box.textContent = "Verifying hashes…";
  box.className = "checkline";
  try {
    const r = await api("/api/model/verify-hash", { method: "POST", body: JSON.stringify({ path: composeDest(), files: filesMeta }) });
    const results = r.results || [];
    const lines = [];
    for (const x of results) {
      if (x.present === true && x.ok === true) {
        lines.push(esc(x.name) + " — ok (ref sha256: " + short(x.sha) + ")");
      } else if (x.present === true && x.ok === false) {
        lines.push(esc(x.name) + " — MISMATCH");
      } else {
        lines.push(esc(x.name) + " — not present");
      }
    }
    const corrupt = results.filter((x) => !(x.present === true && x.ok === true)).length;
    const header = corrupt === 0
      ? "Hashes verified (" + results.length + " files)"
      : "Corrupt / missing (" + corrupt + ")";
    box.className = "checkline " + (corrupt === 0 ? "ok" : "bad");
    box.innerHTML = header + "<br>" + lines.join("<br>");
    appendLog(header);
    lines.forEach((l) => appendLog(l));
  } catch (e) {
    box.className = "checkline bad";
    box.textContent = "Verify hashes error: " + e.message;
    appendLog("verify hashes error: " + e.message);
  }
  btn.disabled = false;
  spin.hidden = true;
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
  line.appendChild(el("div", okDisk ? "banner show ok" : "banner show bad",
    okDisk ? `Disk: OK (free ${gb(free)} GB ≥ needed ~${gb(needs)} GB)` : `Disk: NOT ENOUGH (free ${gb(free)} GB < needed ~${gb(needs)} GB)`));

  const ramNeeded = params * 4 * 1.2;
  const vm = state.info.virtual_memory;
  const lineR = $("cv-ram");
  lineR.innerHTML = "";
  if (vm) {
    const avail = vm.avail_virtual_gb * 1e9;
    const okRam = avail >= ramNeeded;
    lineR.appendChild(el("div", okRam ? "banner show ok" : "banner show bad",
      okRam ? `Virtual memory: OK (avail ${vm.avail_virtual_gb} GB ≥ peak ~${gb(ramNeeded)} GB)` :
        `Virtual memory: NOT ENOUGH (avail ${vm.avail_virtual_gb} GB < peak ~${gb(ramNeeded)} GB)`));
  } else {
    lineR.appendChild(el("div", "banner show ok", "Virtual memory: non-Windows, check skipped"));
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
    hf_home: $("hf-home").value.trim() || null,
    hf_hub_cache: $("hf-hub-cache").value.trim() || null,
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
  const list = STAGES;
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
function ensureStage(name) {
  if (stageNode(name)) return;
  const box = $("stages");
  const n = el("div", "stage", "");
  n.dataset.stage = name;
  n.appendChild(el("span", "dot"));
  n.appendChild(document.createTextNode(name));
  box.appendChild(n);
}
function setStage(name, status) {
  ensureStage(name);
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
  $("task-panel").classList.remove("collapsed");
  $("task-collapse").textContent = "▴";
  $("task-collapse").title = "Collapse";
  appendLog("task started: " + id + " (" + kind + ")");
  appendLog("\n— task " + id + " (" + kind + ") —");
  $("task-status").textContent = "running";
  $("task-status").className = "chip busy";
  state.taskActive = true;
  $("task-cancel").disabled = false;
  const isDl = kind === "download";
  $("dl-spinner").hidden = !isDl;
  $("cv-spinner").hidden = isDl;
  $("dl-cancel").hidden = !isDl;
  $("cv-cancel").hidden = isDl;
  ["dl-run", "cv-run"].forEach((b) => { const el = $(b); el.disabled = true; el.classList.add("working"); });
  activeProgressId = isDl ? "dl-progress" : "cv-progress";
  ["dl-progress", "cv-progress"].forEach((id) => {
    const bar = $(id);
    bar.querySelector(".fill").style.width = "0%";
    bar.classList.toggle("indeterminate", id === activeProgressId);
  });
  const es = new EventSource("/api/task/stream?task_id=" + id);
  es.onmessage = (e) => {
    try { const d = JSON.parse(e.data); onTaskLine(d.line); } catch (err) { appendLog(e.data); }
  };
  es.addEventListener("done", (e) => {
    es.close();
    try { const d = JSON.parse(e.data); $("task-status").textContent = d.returncode === 0 ? "completed" : "failed (exit " + d.returncode + ")"; }
    catch (err) { $("task-status").textContent = "done"; }
    $("task-status").className = "chip done";
    resetTaskUi();
    loadModels(); loadInfo();
  });
  es.onerror = () => { es.close(); $("task-status").textContent = "stream closed"; $("task-status").className = "chip"; resetTaskUi(); };
}
function resetTaskUi() {
  state.taskActive = false;
  $("task-cancel").disabled = true;
  activeProgressId = null;
  $("dl-spinner").hidden = true;
  $("cv-spinner").hidden = true;
  $("dl-cancel").hidden = true;
  $("cv-cancel").hidden = true;
  ["dl-run", "cv-run"].forEach((b) => { const el = $(b); el.disabled = false; el.classList.remove("working"); });
  ["dl-progress", "cv-progress"].forEach((id) => {
    const bar = $(id);
    bar.classList.remove("indeterminate");
    bar.querySelector(".fill").style.width = "0%";
  });
}
function onTaskLine(line) {
  if (line == null) return;
  const ev = parseEvent(line);
  if (ev) {
    if (ev.ev === "STAGE") {
      const [status, ...rest] = ev.payload.split(" ");
      if (status === "done") { setStage(ev.stage, "done"); appendLog("✔ " + ev.stage + " " + rest.join(" ")); }
      else if (status === "fail") { setStage(ev.stage, "fail"); appendLog("✕ " + ev.stage + " " + rest.join(" ")); }
      else { setStage(ev.stage, "running"); appendLog("▶ " + ev.stage); }
    } else if (ev.ev === "LOG") {
      appendLog(ev.payload);
    } else if (ev.ev === "PROGRESS") {
      const pct = parseFloat(ev.payload);
      if (activeProgressId) {
        const el = document.getElementById(activeProgressId);
        if (el) {
          el.classList.remove("indeterminate");
          el.querySelector(".fill").style.width = Math.max(0, Math.min(100, pct)) + "%";
        }
      }
    } else if (ev.ev === "META") {
      try {
        const r = JSON.parse(ev.payload);
        if (r.genai_test) {
          appendLog("\n— result —\noutput: " + (r.output_dir || "?") + "\n" + JSON.stringify(r.genai_test, null, 1));
        } else {
          appendLog("\n— result —\noutput: " + (r.output_dir || "download complete"));
        }
      } catch (err) { appendLog(ev.payload); }
    }
  } else {
    appendLog(line);
  }
}
async function pollStatus() {
  try {
    const s = await api("/api/task/status");
    $("task-cancel").disabled = !(s.busy);
    if (s.task && s.task.kind) {
      $("task-status").textContent = s.busy ? "running" : (s.done ? "finished" : "idle");
      $("task-status").className = "chip " + (s.busy ? "busy" : s.done ? "done" : "");
      const isDl = s.task.kind === "download";
      $("dl-spinner").hidden = !(s.busy && isDl);
      $("cv-spinner").hidden = !(s.busy && !isDl);
      $("dl-cancel").hidden = !(s.busy && isDl);
      $("cv-cancel").hidden = !(s.busy && !isDl);
    } else {
      $("dl-spinner").hidden = true;
      $("cv-spinner").hidden = true;
      $("dl-cancel").hidden = true;
      $("cv-cancel").hidden = true;
    }
  } catch (e) {}
  setTimeout(pollStatus, 3000);
}

init();
