/* OpenVINO Model Converter - front-end logic */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  info: null, modes: [], sources: [], converted: [],
  dlInfo: null, cvInfo: null, currentMode: null, currentTask: null,
  selfTest: null, selfTestBusy: false, selfTestLast: null, prevCompressMode: null,
  cvTfreq: null, tfBusy: false,
  taskActive: false, taskKind: null, dlFiles: null, dlSelected: null, dlLocalComplete: null, dlResources: null,
  validatedSub: null, dlSubDirty: false, validating: false, drives: [], hfEnvState: null,
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
function fmtGb(v) { return v == null ? "—" : (Number(v) || 0).toFixed(1) + " GB"; }
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
  await loadInfo();
  await Promise.all([loadModes(), loadModels()]);
  bindEvents();
  updateGenaiDevice();
  await loadDrives();
  loadHfDrives();
  initHfDrive();
  renderTaskSelect();
  renderModes();
  renderModelSelect();
  onTaskChange();
  bindTaskStream();
}
async function loadInfo() {
  state.info = await api("/api/info");
  $("paths").textContent =
    `cache: ${state.info.paths.cache}   ·   originals: ${state.info.paths.originals}   ·   output: ${state.info.paths.output}`;
  $("hf-base").value = "";
  const chips = $("versions");
  chips.innerHTML = "";
  const v = state.info.resolved_versions && Object.keys(state.info.resolved_versions).length ? state.info.resolved_versions : state.info.versions;
  for (const [k, val] of Object.entries(v)) {
    if (val == null) continue;
    chips.appendChild(el("span", "chip mono", `${k}=${val}`));
  }
}
async function loadModes() {
  state.modes = (await api("/api/modes")).modes;
  renderModes();
}
async function loadModels() {
  const d = await api("/api/models");
  state.sources = d.sources;
  state.converted = d.converted;
  renderModelSelect();
}
function composeHfBase() {
  const drv = $("hf-drive").value.trim().replace(/[\\/]+$/, "");
  const path = $("hf-base").value.trim().replace(/^[\\/]+/, "");
  return path ? drv + osSep() + path : drv;
}
function updateHfDerived() {
  const base = composeHfBase().replace(/[\\/]+$/, "");
  const home = base ? base + osSep() + ".hf-cache" : "";
  const hub = home ? home + osSep() + "hub" : "";
  $("hf-derived").textContent = base ? ("HF_HOME=" + home + "\nHF_HUB_CACHE=" + hub) : "";
  return { home, hub };
}
async function loadHfDrives() {
  const sel = $("hf-drive");
  if (!sel) return;
  sel.innerHTML = "";
  (state.drives || []).forEach((drv) => {
    const o = document.createElement("option");
    o.value = drv;
    o.textContent = drv;
    sel.appendChild(o);
  });
}
function initHfDrive() {
  const orig = (state.info && state.info.paths && state.info.paths.originals) || "";
  const m = /^([A-Za-z]:)[\\/]/.exec(orig);
  const sel = $("hf-drive");
  let drive = null;
  let rel = "";
  if (m) { drive = m[1] + osSep(); rel = orig.slice(m[0].length); }
  else if (osSep() === "/" && orig.startsWith("/")) { drive = "/"; rel = orig.replace(/^\/+/, ""); }
  if (sel && drive) {
    if (![...sel.options].some((o) => o.value === drive)) {
      const o = document.createElement("option");
      o.value = drive;
      o.textContent = drive;
      sel.appendChild(o);
    }
    sel.value = drive;
  }
  $("hf-base").value = rel || "";
  updateHfDerived();
  validateHfEnv();
}
function syncHfDriveFromBase() {
  const sel = $("hf-drive");
  if (!sel) return;
  const path = $("hf-base").value.trim();
  const m = /^([A-Za-z]:)[\\/]/.exec(path);
  if (!m) return;
  const drive = m[1];
  if (![...sel.options].some((o) => o.value === drive)) {
    const o = document.createElement("option");
    o.value = drive;
    o.textContent = drive;
    sel.appendChild(o);
  }
  sel.value = drive;
  $("hf-base").value = path.slice(m[0].length).replace(/^[\\/]+/, "");
}
function syncHfBaseFromDrive() {
  const sel = $("hf-drive");
  if (!sel) return;
  const newDrive = sel.value;
  if (!newDrive) return;
  $("hf-base").value = $("hf-base").value.trim().replace(/^[A-Za-z]:[\\/]/, "").replace(/^[\\/]+/, "");
}
function applyOptions() {
  const card = $("dl-options-card");
  const res = $("hf-env-res");
  if (!card || !res) return;
  card.classList.remove("done", "pending", "bad");
  const env = state.hfEnvState || { cls: "", msg: "" };
  const token = ($("dl-token").value || "").trim();
  const tokenBad = token !== "" && !/^hf_[A-Za-z0-9_]+$/.test(token);
  let cls = env.cls;
  let msg = env.msg;
  if (tokenBad) { cls = "bad"; msg = "HF token looks invalid — should start with hf_ (gated models will fail to download)."; }
  const cardCls = cls === "bad" ? "bad" : cls === "warn" ? "pending" : cls === "ok" ? "done" : "";
  if (cardCls) card.classList.add(cardCls);
  res.className = "banner show " + (cls || "neutral");
  res.textContent = msg;
}
async function validateHfEnv() {
  const res = $("hf-env-res");
  if (!res) return;
  document.querySelectorAll(".create-btn").forEach((b) => b.remove());
  const base = composeHfBase().trim();
  const drive = (base.match(/^([A-Za-z]:)/) || [])[1] || (osSep() === "/" ? "/" : "?");
  const driveRoot = (drive === "/" || drive === "?") ? drive : drive.replace(/:$/, "") + osSep();
  const driveLabel = drive.replace(/:$/, "");
  if (!base || !/^[A-Za-z]:[\\/]/.test(base) && !(osSep() === "/" && base.startsWith("/"))) {
    state.hfEnvState = { cls: "bad", msg: base ? "Invalid base path" : "HF base path is empty" };
    applyOptions();
    return;
  }
  const hf = updateHfDerived();
  try {
    const r = await api("/api/disk?path=" + encodeURIComponent(base));
    if (r.ok === false) {
      let driveOk = false;
      if (drive !== "?") {
        try { driveOk = (await api("/api/disk?path=" + encodeURIComponent(driveRoot))).ok === true; } catch (e) {}
      }
      if (!driveOk) {
        state.hfEnvState = { cls: "bad", msg: "Insufficient disk (invalid drive)" };
        applyOptions();
        return;
      }
      const group = $("hf-base").closest(".root-group");
      if (group && !group.querySelector(".create-btn")) {
        const create = el("button", "create-btn", "Create now?");
        create.id = "create-btn-hf-base";
        create.onclick = async () => {
          try {
            await api("/api/mkdir", { method: "POST", body: JSON.stringify({ path: base }) });
            create.remove();
            validateHfEnv();
          } catch (e2) {
            res.textContent = "Create failed: " + e2.message;
          }
        };
        group.appendChild(create);
      }
      state.hfEnvState = { cls: "warn", msg: 'Missing folder — use "Create now?"' };
      applyOptions();
      return;
    }
    const free = Number(r.free_gb) || 0;
    if (free < 10) {
      state.hfEnvState = { cls: "bad", msg: "Insufficient disk (" + free.toFixed(1) + " GB free on cache drive)" };
      applyOptions();
      return;
    }
    state.hfEnvState = { cls: "ok", msg: "HF paths OK — drive " + driveLabel + ": has " + free.toFixed(1) + " GB free\nHF_HOME=" + hf.home + "\nHF_HUB_CACHE=" + hf.hub };
    applyOptions();
  } catch (e) {
    state.hfEnvState = { cls: "bad", msg: "Error checking cache disk: " + e.message };
    applyOptions();
  }
}

/* ---------------------------------------------------------------- tabs */
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x.dataset.panel === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + name));
}
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
  document.addEventListener("click", (e) => {
    const t = e.target && e.target.closest ? e.target.closest("[data-fill]") : null;
    if (t && t.dataset.fill) { $("dl-text").value = t.dataset.fill; validateDownload(); }
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
  $("dl-root").addEventListener("input", () => { syncDriveFromRoot(); onDestInput(); });
  $("dl-drive").addEventListener("change", syncRootFromDrive);
  $("dl-sub").addEventListener("input", () => {
    const v = $("dl-sub").value;
    const n = normalizeSub(v);
    if (n !== v) $("dl-sub").value = n;
    if (state.validatedSub && state.validatedSub !== n && $("dl-info") && state.dlInfo && state.dlInfo.kind === "hf") {
      state.dlSubDirty = true;
      const box = $("dl-local");
      if (box) {
        box.className = "banner show bad";
        box.textContent = "Model folder changed from " + state.validatedSub + " to " + n + " — Download is blocked. Re-run Validate to confirm the change.";
      }
      updateDlChecks();
    }
    updateDirLink();
    clearTimeout(dlLocalDebounce);
    dlLocalDebounce = setTimeout(() => {
      const sp = $("dl-path-spinner");
      if (sp) sp.hidden = false;
      updateLocalStatus();
    }, 250);
  });
  $("dl-files").addEventListener("change", () => {
    state.dlSelected = new Set();
    document.querySelectorAll("#dl-files input[type=checkbox]:checked:not(:disabled)").forEach((cb) => state.dlSelected.add(cb.value));
    updateDlChecks();
  });
  $("dl-all").addEventListener("click", () => setFilesAll(true));
  $("dl-none").addEventListener("click", () => setFilesAll(false));
  const onHfEnvInput = () => {
    syncHfDriveFromBase();
    updateHfDerived();
    clearTimeout(hfEnvDebounce);
    hfEnvDebounce = setTimeout(validateHfEnv, 300);
  };
  $("hf-base").addEventListener("input", onHfEnvInput);
  $("dl-token").addEventListener("input", applyOptions);
  $("hf-drive").addEventListener("change", () => {
    syncHfBaseFromDrive();
    updateHfDerived();
    clearTimeout(hfEnvDebounce);
    hfEnvDebounce = setTimeout(validateHfEnv, 300);
  });
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
  $("cv-task").addEventListener("change", onTaskChange);
  $("cv-group").addEventListener("change", () => { updateCvChecks(); updateFieldState(); });
  $("cv-backup").addEventListener("change", updateFieldState);
  $("cv-text").addEventListener("input", () => { updateCvChecks(); updateFieldState(); });
  $("cv-dataset").addEventListener("input", () => { updateDataAware(); updateCvChecks(); });
  $("cv-gptq").addEventListener("change", () => { updateDataAware(); updateCvChecks(); });
  $("cv-lora").addEventListener("change", () => { updateDataAware(); updateCvChecks(); });
  $("cv-awq").addEventListener("change", updateCvChecks);
  $("cv-scale").addEventListener("change", updateCvChecks);
  $("cv-nsamples").addEventListener("change", updateCvChecks);
  $("cv-tf-install").addEventListener("click", async () => {
    const t = state.cvTfreq;
    if (!t || !t.recommended) return;
    setTfBusy(true);
    appendLog("installing transformers " + t.recommended + " ...");
    try {
      const r = await api("/api/tf/switch", { method: "POST", body: JSON.stringify({ version: t.recommended }) });
      if (r.ok) { appendLog("switched to transformers " + r.version + " — reloading"); await loadModels(); }
      else appendLog("switch failed: " + (r.error || r.output || "?"));
      await loadInfo();
      await loadModes();
    } catch (e) { appendLog("switch error: " + e.message); }
    setTfBusy(false);
    renderCvTfreq();
    updateCvChecks();
  });
  $("cv-tf-restore").addEventListener("click", async () => {
    setTfBusy(true);
    appendLog("restoring pinned transformers version ...");
    try {
      const r = await api("/api/tf/restore", { method: "POST" });
      if (r.ok) { appendLog("restored pinned versions"); await loadModels(); }
      else appendLog("restore failed: " + (r.error || r.output || "?"));
      await loadInfo();
      await loadModes();
    } catch (e) { appendLog("restore error: " + e.message); }
    setTfBusy(false);
    renderCvTfreq();
    updateCvChecks();
  });
  $("cv-tfreq-auto").addEventListener("change", updateCvChecks);
  $("cv-run").addEventListener("click", runConvert);
  const genaiCheck = $("cv-genai");
  if (genaiCheck) genaiCheck.addEventListener("change", updateGenaiDevice);
  document.querySelectorAll('input[name="cv-genai-device"]').forEach((r) => {
    r.addEventListener("change", updateGenaiDevice);
  });
  const cancelTask = () => api("/api/task/cancel", { method: "POST" }).catch(() => {});
  $("task-cancel").addEventListener("click", cancelTask);
  $("task-cancel").disabled = true;
  $("dl-cancel").addEventListener("click", cancelTask);
  $("cv-cancel").addEventListener("click", cancelTask);
  $("task-clear").addEventListener("click", () => { $("task-log").innerHTML = ""; $("stages").innerHTML = ""; });
  $("task-collapse").addEventListener("click", () => {
    $("task-panel").classList.toggle("collapsed");
    const collapsed = $("task-panel").classList.contains("collapsed");
    $("task-collapse").textContent = collapsed ? "▴" : "▾";
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
  btn.textContent = "▾";
  btn.title = "Collapse";
  renderStages("validate");
  const chip = $("task-status");
  chip.textContent = "validating…";
  chip.className = "chip busy";
}
async function validateDownload() {
  if (state.validating) return;
  state.validating = true;
  $("dl-validate").disabled = true;
  try {
    if ($("dl-tags")) { $("dl-tags").innerHTML = ""; $("dl-tags").hidden = true; }
    if ($("dl-support")) { $("dl-support").innerHTML = ""; }
    if ($("dl-tfreq")) $("dl-tfreq").innerHTML = "";
    const mc = $("dl-model-card");
    if (mc) mc.classList.remove("done", "bad", "pending");
    if ($("dl-info")) { $("dl-info").innerHTML = ""; $("dl-info").className = "info"; }
    if ($("dl-files")) { $("dl-files").innerHTML = ""; $("dl-files").classList.remove("show"); }
    if ($("dl-local")) { $("dl-local").className = "banner"; $("dl-local").textContent = ""; }
    if ($("dl-hashres")) { $("dl-hashres").textContent = ""; $("dl-hashres").className = "checkline"; }
    if ($("dl-hash-card")) $("dl-hash-card").hidden = true;
    if ($("dl-resources-card")) { $("dl-resources-card").hidden = true; }
    state.dlResources = null;
    state.dlFiles = null;
    state.dlSelected = null;
    state.dlLocalComplete = null;
    state.dlLocalExists = null;
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
  } finally {
    state.validating = false;
    $("dl-validate").disabled = false;
  }
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
function renderSupportBadge(i) {
  const card = $("dl-model-card");
  const box = $("dl-support");
  if (!box) return;
  box.innerHTML = "";
  if (card) card.classList.remove("done", "bad", "pending");
  const s = i && i.support;
  if (!s) return;
  const chip = el("span", "tag", "");
  if (s.state === "supported" || s.state === "task_mismatch") {
    chip.className = "tag support-ok";
    chip.textContent = "✓ SUPPORTED" + (s.model_type ? "  (" + s.model_type + ")" : "");
    box.appendChild(chip);
    if (card) card.classList.add("done");
    if (s.state === "task_mismatch") {
      const note = el("span", "tag", "");
      note.style.background = "#fdf6e3";
      note.style.color = "#8a6d1a";
      note.textContent = "⚠ task \u201c" + (s.task || "?") + "\u201d not among supported tasks for this arch — supported: " + ((s.supported_tasks || []).join(", "));
      box.appendChild(note);
    }
  } else if (s.state === "unsupported") {
    chip.className = "tag support-bad";
    chip.textContent = "✕ NOT SUPPORTED" + (s.model_type ? "  (" + s.model_type + ")" : "");
    box.appendChild(chip);
    if (card) card.classList.add("bad");
  } else {
    chip.className = "tag support-unknown";
    chip.textContent = s.ok === false ? "support check unavailable (environment)" : "support unknown";
    box.appendChild(chip);
  }
}
function renderTfreqBadge(i) {
  const box = $("dl-tfreq");
  if (!box) return;
  box.innerHTML = "";
  const t = i && i.tfreq;
  if (!t || t.mode === "unknown") {
    if (t) {
      const chip = el("span", "tag support-unknown", "transformers version unknown");
      box.appendChild(chip);
    }
    return;
  }
  if (t.ok) {
    const chip = el("span", "tag support-ok", "✓ transformers " + (t.installed || "?") + " OK (needs " + t.required + ")");
    box.appendChild(chip);
  } else {
    const chip = el("span", "tag", "");
    chip.style.background = "#fdf6e3"; chip.style.color = "#8a6d1a";
    chip.textContent = "⚠ needs transformers " + t.required + (t.recommended ? " (install " + t.recommended + ")" : "") + " — installed " + (t.installed || "none") + ", export will fail";
    box.appendChild(chip);
  }
}
function renderResources(i) {
  const card = $("dl-resources-card");
  const box = $("dl-resources");
  if (!card || !box) return;
  const r = i && i.resources;
  if (!r || !r.stages) { card.hidden = true; box.innerHTML = ""; state.dlResources = null; return; }
  card.hidden = false;
  state.dlResources = r;
  box.innerHTML = "";
  const names = { download: "Download", convert: "Convert", compress: "Compress" };
  for (const key of ["download", "convert", "compress"]) {
    const s = r.stages[key];
    const row = el("div", "res-row state-" + (s.state || "unknown"), "");
    row.appendChild(el("span", "res-dot"));
    row.appendChild(el("span", "res-name", names[key]));
    const body = el("span", "res-body", "");
    body.textContent = resRowText(key, s);
    row.appendChild(body);
    box.appendChild(row);
  }
}
function resRowText(key, s) {
  const fd = fmtGb(s.free_disk_gb);
  const fr = fmtGb(s.avail_ram_gb);
  const need = fmtGb(s.need_disk_gb);
  const ram = fmtGb(s.need_ram_gb);
  const verdict = s.state === "fail" ? "NOT ENOUGH" : s.state === "warn" ? "tight/estimated" : s.state === "ok" ? "OK" : "unknown";
  let t;
  if (key === "download") {
    if (s.state === "fail") t = "Need " + need + " disk, only " + fd + " free — NOT ENOUGH";
    else if (s.state === "warn") t = "Need " + need + " disk, " + fd + " free (tight)";
    else if (s.state === "ok") t = "Need " + need + " disk, " + fd + " free — OK";
    else t = "size unknown — cannot estimate";
  } else {
    t = (key === "compress" ? "result ≈ " + fmtGb(s.result_gb) + " — " : "") +
      "Need ~" + need + " disk + ~" + ram + " RAM — free " + fd + " / avail " + fr + " (" + verdict + ")";
    if (s.state === "fail") {
      const parts = [];
      if (s.need_disk_gb != null && s.free_disk_gb != null && Number(s.free_disk_gb) < Number(s.need_disk_gb)) parts.push("DISK: need " + need + " free " + fd);
      if (s.need_ram_gb != null && s.avail_ram_gb != null && Number(s.avail_ram_gb) < Number(s.need_ram_gb)) parts.push("RAM: need " + ram + " avail " + fr);
      if (parts.length) t += " " + parts.join(" ");
    }
  }
  if (s.issue) t += " — " + s.issue;
  return t;
}
function renderDlInfo(res) {
  const i = res.info;
  if (!i || i.ok === false) {
    // validation failed (e.g. HF API error / rate limit / model not found)
    state.dlInfo = null;
    state.dlFiles = null;
    state.dlSelected = null;
    state.dlNeededBytes = 0;
    state.dlSubDirty = false;
    state.validatedSub = null;
    setInfo("dl-info", "Validation failed: " + (i && i.error ? i.error : "unknown error"), "bad");
    $("dl-files").classList.remove("show");
    $("dl-files").textContent = "";
    setSelectBtns();
    $("dl-tags").innerHTML = "";
    $("dl-tags").hidden = true;
    if ($("dl-support")) { $("dl-support").innerHTML = ""; }
    if ($("dl-tfreq")) $("dl-tfreq").innerHTML = "";
    const mc = $("dl-model-card");
    if (mc) mc.classList.remove("done", "bad", "pending");
    $("dl-hash-card").hidden = true;
    if ($("dl-resources-card")) { $("dl-resources-card").hidden = true; }
    state.dlResources = null;
    updateDirLink();
    updateDlChecks();
    return;
  }
  if (res.kind === "local") {
    const files = i.files_meta || [];
    const parts = String(i.path || "").split(/[\\/]+/).filter(Boolean);
    $("dl-sub").value = parts[parts.length - 1] || "";
    $("dl-root").value = parts.slice(0, -1).join("\\") || "";
    state.validatedSub = normalizeSub($("dl-sub").value);
    state.dlSubDirty = false;
    if (files.length) {
      const totalBytes = files.reduce((s, f) => s + (Number(f.size) || 0), 0);
      let html = "<table class=\"mini-table\"><thead><tr><th>File</th><th>Size</th><th>Purpose</th></tr></thead><tbody>";
      for (const f of files) {
        html += "<tr><td>" + esc(f.name) + "</td><td>" + esc(humanSize(f.size)) + "</td><td>" + esc(filePurpose(f.name)) + "</td></tr>";
      }
      html += "</tbody><tfoot><tr><td>Total</td><td>" + esc(humanSize(totalBytes)) + "</td><td></td></tr></tfoot></table>";
      const infoBox = $("dl-info");
      infoBox.innerHTML = html;
      infoBox.className = "info show " + (i.ok ? "ok" : "bad");
      const tagId = i.id || i.name;
      let tagsHtml =
        "<span class=\"tag\">" + esc(tagId) + "</span> " +
        "<span class=\"tag\">" + esc(i.model_type || "?") + "</span> " +
        "<span class=\"tag\">" + esc(i.task || "?") + "</span> " +
        "<span class=\"tag\">" + esc(i.total_gb ?? i.size_gb) + " GB</span> " +
        "<span class=\"tag\">" + esc(i.license || "?") + "</span> ";
      if (i.id && /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(String(i.id))) {
        tagsHtml += "<a class=\"hf-link\" href=\"https://huggingface.co/" + esc(i.id) + "\" target=\"_blank\" rel=\"noopener noreferrer\">Hugging Face <sup>↗</sup></a>";
      }
      $("dl-tags").innerHTML = tagsHtml;
      $("dl-tags").hidden = false;
    } else {
      setInfo("dl-info",
        `Local model: ${i.name}\nmodel_type: ${i.model_type} · task: ${i.task} · size: ${i.size_gb} GB` +
        (i.is_quantized ? "\n⚠ already quantized (not convertible)" : "") +
        (i.is_ov ? "\n⚠ already an OpenVINO model" : "") +
        (i.is_gguf ? "\n⚠ GGUF file (not convertible)" : ""),
        i.ok ? "ok" : "bad");
      $("dl-tags").innerHTML = "";
      $("dl-tags").hidden = true;
    }
    renderSupportBadge(i);
    renderTfreqBadge(i);
    renderResources(i);
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
      "<span class=\"tag\">" + esc(i.id) + "</span> " +
      "<span class=\"tag\">" + esc(i.pipeline_tag || "?") + "</span> " +
      "<span class=\"tag\">" + esc(i.total_gb) + " GB</span> " +
      "<span class=\"tag\">" + esc(i.license || "?") + "</span> " +
      "<a class=\"hf-link\" href=\"https://huggingface.co/" + esc(i.id) + "\" target=\"_blank\" rel=\"noopener noreferrer\">Hugging Face <sup>↗</sup></a>";
    $("dl-tags").hidden = false;
    const org = i.id.split("/")[0];
    const name = i.id.split("/")[1];
    $("dl-sub").value = normalizeSub(org + "/" + name);
    state.validatedSub = normalizeSub($("dl-sub").value);
    state.dlSubDirty = false;
    const root = $("dl-root").value.trim();
    if (!root) {
      const orig = (state.info && state.info.paths && state.info.paths.originals) || "";
      const m = /^([A-Za-z]:)[\\/]/.exec(orig);
      if (m) {
        setDriveOption(m[1]);
        $("dl-root").value = orig.slice(m[0].length).replace(/^[\\/]+/, "");
      } else {
        $("dl-root").value = orig.replace(/^[\\/]+/, "");
      }
    }
    updateDirLink();
    state.dlNeededBytes = i.total_bytes;
    state.dlFiles = i.files || [];
    renderFilePicker(i.files_meta || []);
    renderSupportBadge(i);
    renderTfreqBadge(i);
    renderResources(i);
  }
}
function osSep() { return ((navigator.platform || "") + " " + (navigator.userAgent || "")).indexOf("Win") !== -1 ? "\\" : "/"; }
function composeDest() {
  const drv = $("dl-drive").value.trim().replace(/[\\/]+$/, "");
  const root = $("dl-root").value.trim().replace(/^[\\/]+/, "");
  const sub = normalizeSub($("dl-sub").value);
  let base = root ? drv + osSep() + root : drv;
  return sub ? base + osSep() + sub : base;
}
async function loadDrives() {
  const d = await api("/api/drives");
  state.drives = (d.drives || []).filter(Boolean);
  const sel = $("dl-drive");
  if (!sel) return;
  sel.innerHTML = "";
  const drvList = state.drives;
  drvList.forEach((drv) => {
    const o = document.createElement("option");
    o.value = drv;
    o.textContent = drv;
    sel.appendChild(o);
  });
  // default drive to the drive of originals, and keep only the relative path in #dl-root
  const orig = (state.info && state.info.paths && state.info.paths.originals) || "";
  const m = /^([A-Za-z]:)[\\/]/.exec(orig);
  let defDrive = null;
  let defRoot = "";
  if (m) {
    defDrive = m[1] + osSep();
    defRoot = orig.slice(m[0].length);
  } else if (osSep() === "/" && orig.startsWith("/")) {
    defDrive = "/";
    defRoot = orig.replace(/^\/+/, "");
  }
  if (defDrive) {
    if (![...sel.options].some((o) => o.value === defDrive)) {
      const o = document.createElement("option");
      o.value = defDrive;
      o.textContent = defDrive;
      sel.appendChild(o);
    }
    sel.value = defDrive;
  }
  if (defRoot !== "") $("dl-root").value = defRoot;
  syncDriveFromRoot();
}
function setDriveOption(drive) {
  const sel = $("dl-drive");
  if (!sel) return;
  if (drive && ![...sel.options].some((o) => o.value === drive)) {
    const o = document.createElement("option");
    o.value = drive;
    o.textContent = drive;
    sel.appendChild(o);
  }
  if (drive) sel.value = drive;
}
function syncDriveFromRoot() {
  const sel = $("dl-drive");
  if (!sel) return;
  const root = $("dl-root").value.trim();
  const m = /^([A-Za-z]:)[\\/]/.exec(root);
  if (!m) {
    if (osSep() === "/" && root.startsWith("/")) {
      const drive = "/";
      if (![...sel.options].some((o) => o.value === drive)) {
        const o = document.createElement("option");
        o.value = drive;
        o.textContent = drive;
        sel.appendChild(o);
      }
      sel.value = drive;
    }
    return;
  }
  const drive = m[1];
  if (![...sel.options].some((o) => o.value === drive)) {
    const o = document.createElement("option");
    o.value = drive;
    o.textContent = drive;
    sel.appendChild(o);
  }
  sel.value = drive;
}
function syncRootFromDrive() {
  const sel = $("dl-drive");
  if (!sel) return;
  const newDrive = sel.value;
  if (!newDrive) return;
  const root = $("dl-root").value.trim();
  let rest = root.replace(/^[A-Za-z]:[\\/]/, "").replace(/^[\\/]+/, "");
  $("dl-root").value = rest;
  updateDirLink();
  clearTimeout(dlLocalDebounce);
  dlLocalDebounce = setTimeout(updateLocalStatus, 250);
}
function updateDirLink() {
  const link = $("dl-dir-link");
  if (!link) return;
  const show = !!(state.dlInfo && state.dlInfo.kind === "hf");
  const active = show && state.dlLocalExists === true;
  const lab = document.querySelector(".dir-label");
  if (lab) lab.hidden = !(show && active);
  link.hidden = !show;
  link.textContent = show ? composeDest() + (active ? "" : "  (folder not found)") : "";
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
    $("dl-dest-card").classList.toggle("pending", false);
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
    $("dl-dest-card").classList.toggle("pending", false);
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
    box.className = "banner show warn";
    box.textContent = "Not present locally — will download " + M + " files";
  }
  state.dlLocalComplete = r.complete;
  state.dlLocalExists = r.exists === true;
  if (state.dlInfo && state.dlInfo.kind === "hf") {
    if (state.dlLocalComplete === true) setStage("download", "done");
    else clearStage("download");
  }
  updateDirLink();
  $("dl-dest-card").classList.toggle("done", state.dlLocalComplete === true);
  $("dl-dest-card").classList.toggle("pending", state.dlInfo && state.dlInfo.kind === "hf" && state.dlLocalComplete !== true);
  $("dl-hash-card").hidden = !(state.dlInfo && state.dlInfo.kind === "hf" && r.present > 0);
  renderFilePicker(state.dlInfo.info.files_meta || [], r.present_files || []);
  updateDlChecks();
}
function normalizeSub(v) { return v.trim().replace(/[\\/]+/g, osSep()); }
function subValid(v) { return /^[A-Za-z0-9_.-]+([/\\])[A-Za-z0-9_.-]+$/.test(v.trim()); }
function updateDlChecks() {
  const i = state.dlInfo;
  setSelectBtns();
  $("dl-verify").disabled = !$("dl-hash-card") || $("dl-hash-card").hidden;
  const subOk = subValid($("dl-sub").value);
  $("dl-sub").classList.toggle("invalid", !subOk && $("dl-sub").value.trim() !== "");
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
  const resFail = !!(state.dlResources && Object.values(state.dlResources.stages).some((s) => s.state === "fail"));
  $("dl-disk").innerHTML = "";
  $("dl-disk").appendChild(el("div", okDisk ? "banner show ok" : "banner show bad",
    okDisk ? `Disk: OK (free ${gb(free)} GB ≥ needed ${gb(needed)} GB)` : `Disk: NOT ENOUGH (free ${gb(free)} GB < needed ${gb(needed)} GB)`));
  if (resFail) {
    $("dl-disk").appendChild(el("div", "banner show bad", "Resources: NOT ENOUGH — see Resources check above."));
  }
  const dlBtn = $("dl-run");
  const hasSel = state.dlSelected && state.dlSelected.size > 0;
  const blocked = state.dlSubDirty === true;
  dlBtn.disabled = !(okDisk && !resFail && i.kind === "hf" && i.info.ok && state.dlLocalComplete === false && hasSel && subOk && !blocked);
}
async function runDownload() {
  if (!state.dlInfo || state.dlInfo.kind !== "hf") return;
  const id = state.dlInfo.info.id;
  const hf = updateHfDerived();
  const body = {
    model_id: id,
    dest: composeDest() || null,
    revision: $("dl-rev").value || null,
    token: $("dl-token").value || null,
    hf_home: hf.home || null,
    hf_hub_cache: hf.hub || null,
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
  box.className = "checkline";
  box.innerHTML = "";
  const pct = el("div", null, "Verifying hashes… 0%");
  pct.id = "hash-pct";
  box.appendChild(pct);
  const list = el("div", null, "");
  list.id = "hash-list";
  filesMeta.forEach((f, i) => {
    const row = el("div", "hash-row", "");
    row.id = "hash-" + i;
    row.appendChild(el("span", "hname", f.name));
    row.appendChild(el("span", "hstat", "…"));
    list.appendChild(row);
  });
  box.appendChild(list);
  let sawDone = false;
  try {
    const resp = await fetch("/api/model/verify-hash", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: composeDest(), files: filesMeta }),
    });
    if (!resp.ok || !resp.body) {
      let msg = resp.statusText;
      try { msg = (await resp.text()) || msg; } catch (e) {}
      throw new Error(msg);
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buffer = "";
    let total = 0;
    let okCount = 0;
    let issuesCount = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += dec.decode(value, { stream: true });
      const parts = buffer.split("\n");
      buffer = parts.pop();
      for (const line of parts) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        let obj;
        try { obj = JSON.parse(trimmed); } catch (e) { continue; }
        if (obj.event === "start") {
          total = obj.total || 0;
          pct.textContent = "Verifying hashes… 0%";
        } else if (obj.event === "file") {
          const r = obj.result || {};
          const name = obj.name || "";
          const row = document.getElementById("hash-" + obj.index);
          const stat = row ? row.querySelector(".hstat") : null;
          if (r.present === false) {
            if (stat) { stat.textContent = "not present"; stat.className = "hstat bad"; }
          } else if (r.present === true && r.ok === true && r.method === "sha256") {
            if (stat) { stat.textContent = "✓ ok (ref sha256: " + short(r.expected) + ")"; stat.className = "hstat ok"; }
          } else if (r.present === true && r.ok === true && r.method === "size") {
            if (stat) { stat.textContent = "✓ ok (size " + humanSize(r.actual) + ")"; stat.className = "hstat ok"; }
          } else if (r.present === true && r.ok === false && r.method === "sha256") {
            if (stat) { stat.textContent = "✕ MISMATCH"; stat.className = "hstat bad"; }
          } else if (r.present === true && r.ok === false && r.method === "size") {
            if (stat) { stat.textContent = "✕ size mismatch"; stat.className = "hstat bad"; }
          } else if (r.present === true && r.ok == null) {
            if (stat) { stat.textContent = "no reference (not LFS)"; stat.className = "hstat muted"; }
          } else {
            if (stat) { stat.textContent = "not present"; stat.className = "hstat bad"; }
          }
          if (r.present === true && r.ok === true) okCount++;
          else if ((r.present === true && r.ok === false) || r.present === false) issuesCount++;
          const t = obj.total || total;
          if (t > 0) pct.textContent = "Verifying hashes… " + Math.round(((obj.index + 1) / t) * 100) + "%";
        } else if (obj.event === "done") {
          sawDone = true;
          if (issuesCount === 0) {
            box.className = "checkline ok";
            pct.textContent = "Hashes verified (" + okCount + " files OK)";
          } else {
            box.className = "checkline bad";
            pct.textContent = "Issues found (" + issuesCount + ")";
          }
          appendLog(pct.textContent);
          document.querySelectorAll("#hash-list .hash-row").forEach((row) => {
            const st = row.querySelector(".hstat");
            if (st && st.textContent !== "…") appendLog(st.textContent);
          });
        }
      }
    }
    if (!sawDone) {
      box.className = "checkline bad";
      pct.textContent = "Verifying hashes… stream ended unexpectedly (" + okCount + " ok, " + issuesCount + " issues) — try again";
      appendLog("verify hashes: stream ended before completion");
    }
  } catch (e) {
    box.className = "checkline bad";
    box.textContent = "Verify hashes error: " + e.message;
    appendLog("verify hashes error: " + e.message);
  } finally {
    btn.disabled = false;
    spin.hidden = true;
  }
}

/* ---------------------------------------------------------------- Convert tab */
function cvTask() {
  const sel = $("cv-task");
  return sel && (sel.value === "keep" || sel.value === "compress") ? sel.value : "compress";
}
function renderTaskSelect() {
  const sel = $("cv-task");
  if (!sel) return;
  const prev = sel.value;
  sel.innerHTML = "";
  const defs = [
    ["keep", "Convert to OpenVINO (keep original weights — no compression)"],
    ["compress", "Compress via NNCF"],
  ];
  for (const [value, label] of defs) {
    const o = el("option", null, label);
    o.value = value;
    sel.appendChild(o);
  }
  sel.value = prev === "keep" || prev === "compress" ? prev : "compress";
}
function modeOptionLabel(m) {
  const st = state.selfTest && state.selfTest[m.id];
  if (!st) return m.label;
  return st.startsWith("ok") ? `${m.label}  [self-test ok]` : `${m.label}  [self-test fail]`;
}
function renderModes() {
  const sel = $("cv-mode");
  if (!sel) return;
  const prev = sel.value;
  sel.innerHTML = "";
  for (const m of state.modes) {
    const o = el("option", null, modeOptionLabel(m));
    o.value = m.id;
    sel.appendChild(o);
  }
  const valid = (v) => v && [...sel.options].some((o) => o.value === v);
  if (cvTask() === "keep") {
    sel.value = valid("none") ? "none" : "";
  } else if (valid(prev)) {
    sel.value = prev;
  } else if (valid("int4_sym")) {
    sel.value = "int4_sym";
  }
  onModeChange();
}
function applyModeSuffixes() {
  const sel = $("cv-mode");
  if (!sel) return;
  for (const o of sel.options) {
    const m = state.modes.find((x) => x.id === o.value);
    if (m) o.textContent = modeOptionLabel(m);
  }
}
function onTaskChange() {
  const keep = cvTask() === "keep";
  const modeCard = $("cv-mode") ? $("cv-mode").closest(".card") : null;
  const compCard = $("cv-group") ? $("cv-group").closest(".card") : null;
  const controls = ["cv-mode", "cv-selftest", "cv-group", "cv-backup", "cv-ratio", "cv-all-layers"];
  if (keep && state.currentMode && state.currentMode.id !== "none") state.prevCompressMode = state.currentMode.id;
  const delRow = $("cv-delete-int-row");
  const delBox = $("cv-delete-int");
  const delHelp = $("cv-delete-int-help");
  if (delRow) {
    delRow.hidden = keep;
    delRow.style.display = keep ? "none" : "";
  }
  if (delHelp) {
    delHelp.hidden = keep;
    delHelp.style.display = keep ? "none" : "";
  }
  if (delBox) delBox.checked = !keep;
  controls.forEach((id) => { const n = $(id); if (n) n.disabled = keep; });
  if (modeCard) modeCard.hidden = keep;
  if (compCard) compCard.hidden = keep;
  const daBlock = document.querySelector("details.attach-above");
  if (daBlock) daBlock.hidden = keep;
  const daControls = ["cv-awq", "cv-scale", "cv-gptq", "cv-lora", "cv-dataset", "cv-nsamples"];
  daControls.forEach((id) => { const n = $(id); if (n) n.disabled = keep; });
  const sel = $("cv-mode");
  if (sel) {
    const want = keep ? "none"
      : (state.prevCompressMode && [...sel.options].some((o) => o.value === state.prevCompressMode)) ? state.prevCompressMode
        : ([...sel.options].some((o) => o.value === "int4_sym") ? "int4_sym" : "");
    sel.value = [...sel.options].some((o) => o.value === want) ? want : "";
  }
  onModeChange();
  updateDataAware();
  updateCvChecks();
  updateFieldState();
  updateTfreqStageStatus();
}
function updateFieldState() {
  const keep = cvTask() === "keep";
  const modelOk = !!(state.cvInfo || $("cv-text").value.trim());
  const modeOk = !!state.currentMode;
  const groupOk = !!($("cv-group") && $("cv-group").value !== "");
  const backupOk = !!($("cv-backup") && $("cv-backup").value !== "");
  const modelCard = $("cv-model") ? $("cv-model").closest(".card") : null;
  const modeCard = $("cv-mode") ? $("cv-mode").closest(".card") : null;
  const compCard = $("cv-group") ? $("cv-group").closest(".card") : null;
  const setOk = (node, ok) => { if (node) node.classList.toggle("field-ok", !!ok); };
  if (keep) {
    setOk($("cv-model"), modelOk);
    setOk(modelCard, modelOk);
    setOk($("cv-mode"), false);
    setOk(modeCard, false);
    setOk($("cv-group"), false);
    setOk($("cv-backup"), false);
    setOk(compCard, false);
  } else {
    const allOk = modelOk && modeOk && groupOk && backupOk;
    setOk($("cv-model"), modelOk);
    setOk(modelCard, modelOk);
    setOk($("cv-mode"), allOk);
    setOk(modeCard, allOk);
    setOk($("cv-group"), allOk);
    setOk($("cv-backup"), allOk);
    setOk(compCard, allOk);
  }
}
function updateDataAware() {
  const keep = cvTask() === "keep";
  const mode = keep ? null : state.currentMode;
  const incompatible = ["none", "int8_sym", "int8_asym", "cb4", "mxfp4", "mxfp8_e4m3", "int2_mix", "int3_mix"];
  const compat = !!mode && incompatible.indexOf(mode.id) === -1;
  const on = !keep && compat;
  const hasDs = !!$("cv-dataset").value.trim();
  const setMethod = (id, enabled) => {
    const cb = $(id);
    const label = cb ? cb.closest("label") : null;
    if (label) label.classList.toggle("data-unsupported", !enabled);
    if (cb) {
      cb.disabled = !enabled;
      if (!enabled) cb.checked = false;
    }
  };
  setMethod("cv-awq", on && hasDs);
  setMethod("cv-scale", on);
  setMethod("cv-gptq", on && hasDs);
  setMethod("cv-lora", on && hasDs);
  enforceDataAwareExclusivity();
  const ds = $("cv-dataset");
  const ns = $("cv-nsamples");
  if (ds) ds.disabled = !on;
  if (ns) ns.disabled = !on;
}
function enforceDataAwareExclusivity() {
  const gptq = $("cv-gptq");
  const lora = $("cv-lora");
  if (!gptq || !lora) return;
  const gLabel = gptq.closest("label");
  const lLabel = lora.closest("label");
  if (gptq.checked) {
    lora.checked = false;
    lora.disabled = true;
    if (lLabel) lLabel.classList.add("data-unsupported");
  } else if (lora.checked) {
    gptq.checked = false;
    gptq.disabled = true;
    if (gLabel) gLabel.classList.add("data-unsupported");
  }
}
function autoSelfTest(modelValue) {
  if (state.selfTestBusy) return;
  if (state.selfTest && state.selfTestLast === modelValue) return;
  state.selfTestLast = modelValue;
  runSelfTest();
}
function renderModelSelect() {
  const sel = $("cv-model");
  const prev = sel.value;
  sel.innerHTML = "";
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
function renderCvTfreq() {
  const box = $("cv-tfreq");
  if (!box) return;
  const t = state.cvTfreq;
  const installBtn = $("cv-tf-install");
  const restoreBtn = $("cv-tf-restore");
  const runCard = $("cv-run") ? $("cv-run").closest(".card") : null;
  if (runCard) {
    runCard.classList.toggle("state-ok", !!(t && t.ok));
    runCard.classList.toggle("state-warn", !!(t && t.ok === false));
  }
  if (!t || t.mode === "unknown") {
    box.className = "banner";
    box.textContent = "";
    if (installBtn) installBtn.disabled = true;
    if (restoreBtn) restoreBtn.disabled = state.tfBusy || false;
    return;
  }
  if (t.ok) {
    box.className = "banner show ok";
    box.textContent = "Transformers status: ✓ OK — " + (t.installed || "?") + " satisfies " + t.required;
    const cvTfreqAuto = $("cv-tfreq-auto");
    if (cvTfreqAuto) { cvTfreqAuto.disabled = true; cvTfreqAuto.checked = false; }
  } else {
    box.className = "banner show neutral";
    box.textContent = t.reason + (t.recommended ? "  —  use Install to switch to " + t.recommended : "");
    const cvTfreqAuto = $("cv-tfreq-auto");
    if (cvTfreqAuto) { cvTfreqAuto.disabled = false; if (!cvTfreqAuto.checked) cvTfreqAuto.checked = true; }
  }
  if (installBtn) installBtn.disabled = state.tfBusy || t.ok || !t.recommended;
  if (restoreBtn) restoreBtn.disabled = state.tfBusy || t.ok;
}
function updateTfreqStageStatus() {
  if (state.taskActive) return;
  const n = document.querySelector('[data-stage="tfreq"]');
  if (!n) return;
  const t = state.cvTfreq;
  n.classList.remove("done", "fail");
  if (t && t.ok) {
    n.classList.add("done");
    n.title = "Transformers version OK — " + (t.installed || "?") + " satisfies " + t.required;
  } else if (t && t.ok === false) {
    n.classList.add("fail");
    n.title = "Transformers version NOT OK — needs " + (t.required || "?") + ", installed " + (t.installed || "none");
  } else {
    n.title = "Transformers version unknown for the selected model";
  }
}
async function onModelChange() {
  const v = $("cv-model").value;
  $("cv-meta").innerHTML = "";
  if (!v) {
    state.cvInfo = null;
    state.cvTfreq = null;
    state.currentMode = null;
    $("cv-text").value = "";
    renderCvTfreq();
    updateTfreqStageStatus();
    updateCvChecks();
    updateFieldState();
    return;
  }
  if (v.startsWith("ov:")) {
    state.cvInfo = null;
    state.cvTfreq = null;
    state.currentMode = null;
    const modeSel = $("cv-mode");
    if (modeSel) {
      modeSel.value = "";
      modeSel.disabled = true;
      const gsel = $("cv-group");
      if (gsel) { gsel.innerHTML = ""; gsel.value = ""; }
      $("cv-ratio").disabled = true;
      $("cv-all-layers").disabled = true;
      setInfo("cv-mode-detail", "", "");
      updateDataAware();
    }
    setInfo("cv-meta", "This is an already-converted OpenVINO model. Pick a dense source instead.", "bad");
    renderCvTfreq();
    updateTfreqStageStatus();
    updateCvName();
    updateCvChecks();
    updateFieldState();
    return;
  }
  const src = state.sources.find((s) => s.path === v);
  if (src) {
    state.cvInfo = src;
    state.cvTfreq = src.tfreq || null;
    setInfo("cv-meta",
      `${src.name} · ${src.task} · ${src.model_type || "?"} · ${src.size_gb} GB · tokenizer: ${src.has_tokenizer ? "yes" : "no"}` +
      (src.is_moe ? "\n· MoE model (int2-mix available)" : ""), "ok");
    $("cv-download-first").checked = false;
    state.cvParams = src.size_bytes / 2;
    autoSelfTest(v);
  } else {
    // custom / pasted path
    state.cvTfreq = null;
    const p = v.replace(/^[A-Za-z]:[\\/]/, "");
    $("cv-text").value = p || "";
  }
  const modeSel = $("cv-mode");
  if (modeSel) modeSel.disabled = false;
  renderCvTfreq();
  updateTfreqStageStatus();
  updateCvName();
  updateCvChecks();
  updateFieldState();
}
function onModeChange() {
  const keep = cvTask() === "keep";
  const m = state.modes.find((x) => x.id === $("cv-mode").value);
  state.currentMode = m;
  const gsel = $("cv-group");
  if (!m) {
    if (gsel) { gsel.innerHTML = ""; gsel.value = ""; }
    $("cv-ratio").disabled = true;
    $("cv-all-layers").disabled = true;
    setInfo("cv-mode-detail", "", "");
    updateCvName();
    updateCvChecks();
    updateDataAware();
    updateFieldState();
    return;
  }
  gsel.innerHTML = "";
  for (const g of m.group_size_choices) {
    const o = el("option", null, g === -1 ? "-1 (per-channel)" : String(g));
    o.value = g;
    gsel.appendChild(o);
  }
  gsel.value = m.default_group_size;
  const isInt8 = isInt8Mode(m);
  $("cv-ratio").disabled = keep || m.moe_only || isInt8;
  $("cv-all-layers").disabled = keep || isInt8;
  if (isInt8) $("cv-all-layers").checked = false;
  setInfo("cv-mode-detail",
    `${m.label} · ${m.bits ? m.bits + " bits" : ""} · symmetric: ${m.symmetric === null ? "n/a" : m.symmetric}` +
    (m.requires_per_channel ? "\n⚠ requires per-channel (group_size = -1)" : "") +
    (m.moe_only ? "\nMoE models only: experts int2, rest int4 (AutoRound-style)" : "") +
    (m.help ? "\n" + m.help : ""), "ok");
  updateCvName();
  updateCvChecks();
  updateDataAware();
  updateFieldState();
}
async function runSelfTest() {
  if (state.selfTestBusy) return;
  state.selfTestBusy = true;
  $("cv-selftest").disabled = true;
  $("cv-selftest").textContent = "Testing…";
  try {
    const r = await api("/api/modes/self-test", { method: "POST" });
    state.selfTest = r.result || {};
    const lines = Object.entries(state.selfTest).map(([k, v]) => `${k}: ${v}`);
    setInfo("cv-selftest-res", "Self-test (tiny compress+compile on CPU):\n" + lines.join("\n"), "ok");
    applyModeSuffixes();
  } catch (e) {
    state.selfTest = null;
    setInfo("cv-selftest-res", "Self-test error: " + e.message, "bad");
  }
  state.selfTestBusy = false;
  $("cv-selftest").disabled = cvTask() === "keep";
  $("cv-selftest").textContent = "Self-test modes";
}
function updateCvName() {
  const base = currentBase();
  const mode = state.currentMode ? state.currentMode.id : "int4_sym";
  const root = state.info && state.info.paths ? state.info.paths.output : "";
  $("cv-outdir").value = `${root}\\${base}-${modeToken(mode)}-ov`;
  updateIntermediateHint();
}
function updateIntermediateHint() {
  const hint = $("cv-intermediate-hint");
  if (!hint) return;
  const base = currentBase();
  const root = state.info && state.info.paths ? state.info.paths.output : "";
  hint.textContent = root ? `${root}\\${base}-fp16-ov` : "";
}
function updateGenaiDevice() {
  const script = $("cv-genai-script");
  const row = $("cv-genai-device-row");
  const genai = $("cv-genai");
  const radio = document.querySelector('input[name="cv-genai-device"]:checked');
  const device = (radio && radio.value) || "CPU";
  if (script) {
    script.textContent =
      'from ov_converter.genai_test import run_test, format_result; res = run_test(r"T:\\models\\savvadesogle\\<Base>-<mode>-ov", max_new_tokens=24, device="' +
      device + '"); print(format_result(res))';
  }
  if (row && genai) {
    row.querySelectorAll("input").forEach((el) => { el.disabled = !genai.checked; });
  }
}
function currentBase() {
  const v = $("cv-model").value;
  if (state.cvInfo) return state.cvInfo.name;
  if (v.startsWith("ov:")) {
    const p = v.slice(3).replace(/[\\/]+$/, "");
    let base = p.split(/[\\/]/).pop() || p;
    const tokens = ["-fp16", "-int2", "-int3", "-int4", "-int8", "-nf4", "-mxfp4", "-mxfp8", "-fp8e4m3", "-cb4", "-int2-mix", "-int3-mix", "-ov"];
    let changed = true;
    while (changed) {
      changed = false;
      for (const tok of tokens) {
        if (base.endsWith(tok)) { base = base.slice(0, -tok.length); changed = true; }
      }
    }
    return base;
  }
  const t = $("cv-text").value.trim();
  if (t) return t.split(/[\\/]/).pop() || t;
  return "model";
}
function normalizeHfId(v) {
  let t = String(v || "").trim().replace(/^['"]+|['"]+$/g, "");
  if (!t) return null;
  t = t.replace(/\\/g, "/");
  t = t.replace(/^https?:\/\//, "").replace(/^\/+/, "");
  if (t.startsWith("huggingface.co/")) t = t.slice("huggingface.co/".length);
  else if (t.startsWith("hf.co/")) t = t.slice("hf.co/".length);
  t = t.split(/\/(?:tree|blob|resolve|blame|raw)\//)[0];
  const m = /^([A-Za-z0-9_.-]+)\/([A-Za-z0-9_.-]+)/.exec(t);
  return m ? m[0] : null;
}
function isInt8Mode(m) { return !!m && (m.id === "int8_sym" || m.id === "int8_asym"); }
function cvModelId() {
  const v = $("cv-model").value;
  const isOv = v.startsWith("ov:");
  const source = state.cvInfo ? v : $("cv-text").value.trim();
  if (!source || isOv) return { id: null, path: null };
  const looksLocal = /^[A-Za-z]:[\\/]/.test(source) || source.startsWith("/") || source.startsWith("\\");
  if (!looksLocal) {
    const hf = normalizeHfId(source);
    if (hf) return { id: hf, path: null };
  }
  return { id: source, path: source };
}
function modeToken(mode) {
  const map = { int8_sym: "int8", int8_asym: "int8", int4_sym: "int4", int4_asym: "int4",
    int3_sym: "int3", int2_sym: "int2", nf4: "nf4", mxfp4: "mxfp4", mxfp8_e4m3: "mxfp8",
    fp8_e4m3: "fp8e4m3", cb4: "cb4", int2_mix: "int2-mix", int3_mix: "int3-mix", none: "fp16" };
  return map[mode] || mode;
}
function updateCvChecks() {
  const keep = cvTask() === "keep";
  const m = keep ? null : state.currentMode;
  const params = state.cvParams || 0;
  const bits = m ? (m.bits || 16) : 16;
  const wrap = $("cv-errors");
  wrap.innerHTML = "";
  if (!keep && !m) { $("cv-run").disabled = true; return; }
  const needs = estimateConvertNeeded(params, bits);
  const free = ((state.info && state.info.disk_free_gb) || 0) * 1e9;
  const okDisk = free >= needs;
  const line = $("cv-disk");
  line.innerHTML = "";
  line.appendChild(el("div", okDisk ? "banner show ok" : "banner show bad",
    okDisk ? `Disk: OK (free ${gb(free)} GB ≥ needed ~${gb(needs)} GB)` : `Disk: NOT ENOUGH (free ${gb(free)} GB < needed ~${gb(needs)} GB)`));

  const ramNeeded = params * 4 * 1.2;
  const vm = state.info ? state.info.virtual_memory : null;
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
  if (m && m.requires_per_channel && Number($("cv-group").value) !== -1) errors.push("INT8 mode requires group_size = -1.");
  if (m && m.moe_only && state.cvInfo && !state.cvInfo.is_moe) errors.push("Mode " + m.id + " (int2-mix/int3-mix) requires a Mixture-of-Experts model — pick a non-MoE mode (e.g. INT4 SYM) or choose a MoE source model.");
  if (m && m.available === false) errors.push("Mode \"" + m.id + "\" not available in the installed NNCF — upgrade NNCF (pip install -r requirements.txt) or pick a different mode. Run 'Self-test modes' to see which work. This is about the weight-compression mode, not your model.");
  if (!state.cvInfo && !$("cv-text").value.trim() && !$("cv-model").value) errors.push("Choose a model.");
  const tf = state.cvTfreq;
  if (tf && tf.ok === false && tf.mode !== "unknown" && !$("cv-tfreq-auto").checked) {
    errors.push("Transformers " + (tf.required || "?") + " required but " + (tf.installed || "none") + " installed — enable auto-install or install it first.");
  }
  if (!keep && state.currentMode && ["none", "int8_sym", "int8_asym", "cb4", "mxfp4", "mxfp8_e4m3", "int2_mix", "int3_mix"].indexOf(state.currentMode.id) === -1) {
    const dsVal = ($("cv-dataset").value || "").trim();
    if (($("cv-awq").checked || $("cv-gptq").checked || $("cv-lora").checked) && !dsVal) {
      errors.push("AWQ/GPTQ/LoRA correction need a calibration dataset (.npy) — provide one (see Data-aware methods: numpy.save('inputs.npy', arr) with shape (N, ...)) or disable the method.");
    }
    const ns = $("cv-nsamples");
    const nsVal = ns ? ns.value : "";
    if (dsVal && ns && !ns.disabled && nsVal.trim() !== "" && (!Number.isInteger(Number(nsVal)) || Number(nsVal) < 1)) {
      errors.push("Num samples must be a positive integer (≥ 1).");
    }
  }
  if (errors.length) { wrap.innerHTML = ""; errors.forEach((e) => wrap.appendChild(el("div", null, "⚠ " + e))); }
  $("cv-run").disabled = !(okDisk && errors.length === 0);
}
function setTfBusy(v) {
  state.tfBusy = v;
  $("cv-tf-install").disabled = v || (state.cvTfreq ? state.cvTfreq.ok : true);
  $("cv-tf-restore").disabled = v;
  renderCvTfreq();
}
function estimateConvertNeeded(params, bits) {
  if (!params) return 0;
  const fp16 = params * 2;
  const result = params * (bits / 8);
  return fp16 * 1.15 + result * 1.1 + fp16; // + keep source
}

async function runConvert() {
  const src = cvModelId();
  const task = cvTask();
  if ((task === "compress" || task === "keep") && src.id && !src.path && !state.cvInfo) {
    switchTab("download");
    $("dl-text").value = src.id;
    appendLog("Model not present locally — validating on the Download tab.");
    validateDownload();
    return;
  }
  const hf = updateHfDerived();
  const m = state.currentMode;
  const isInt8 = isInt8Mode(m);
  const da = {};
  if ($("cv-awq").checked && !$("cv-awq").disabled) da.awq = true;
  if ($("cv-scale").checked && !$("cv-scale").disabled) da.scale_estimation = true;
  if ($("cv-gptq").checked && !$("cv-gptq").disabled) da.gptq = true;
  if ($("cv-lora").checked && !$("cv-lora").disabled) da.lora_correction = true;
  if (da.gptq && da.lora_correction) delete da.lora_correction;
  const ds = $("cv-dataset");
  const dsEnabled = ds && !ds.disabled;
  const ns = $("cv-nsamples");
  const dsVal = dsEnabled ? (ds.value || "").trim() : "";
  if (dsVal) { da.dataset = dsVal; da.num_samples = ns && !ns.disabled ? Math.max(1, Math.floor(Number(ns.value)) || 128) : 128; }
  const cfg = {
    model_id: src.id || currentBase(),
    model_path: src.path,
    download: $("cv-download-first").checked,
    hf_home: hf.home || null,
    hf_hub_cache: hf.hub || null,
    task: state.cvInfo ? state.cvInfo.task : "",
    mode: task === "keep" ? "none" : (m ? m.id : "int4_sym"),
    group_size: Number($("cv-group").value),
    all_layers: isInt8 ? false : $("cv-all-layers").checked,
    ratio: $("cv-ratio").value ? Number($("cv-ratio").value) : null,
    backup: $("cv-backup").value,
    data_aware: da,
    only_text: $("cv-only-text").checked,
    delete_intermediate: task === "keep" ? false : $("cv-delete-int").checked,
    output_dir: $("cv-outdir").value.trim() || null,
    run_genai_test: $("cv-genai").checked,
    genai_device: (document.querySelector('input[name="cv-genai-device"]:checked') || { value: "CPU" }).value || "CPU",
    tfreq_auto_install: $("cv-tfreq-auto").checked,
  };
  await startTask("convert", cfg, "/api/convert");
}

/* ---------------------------------------------------------------- Task streaming */
const STAGES = ["validate", "download", "tfreq", "export", "compress", "package", "tokenizer", "genai_test"];
const STAGE_DONE_LABEL = {
  validate: "validated", download: "downloaded", tfreq: "transformers ok",
  export: "exported",
  compress: "compressed", package: "packaged", tokenizer: "tokenizer ok", genai_test: "genai test ok"
};
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
    n.title = "Run progress marker — lit green when the running task passes this step";
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
  n.title = "Run progress marker — lit green when the running task passes this step";
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
function clearStage(name) {
  const n = stageNode(name);
  if (n) n.classList.remove("running", "done", "fail");
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
function setTaskBusy(kind) {
  state.taskActive = true;
  state.taskKind = kind;
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
}
async function startTask(kind, body, url) {
  setTaskBusy(kind);
  let id;
  try {
    id = (await api(url, { method: "POST", body: JSON.stringify(body) })).task_id;
  } catch (e) { appendLog("ERROR: " + e.message); resetTaskUi(); return; }
  state.currentTask = id;
  renderStages(kind);
  $("task-panel").classList.remove("collapsed");
  $("task-collapse").textContent = "▾";
  $("task-collapse").title = "Collapse";
  appendLog("task started: " + id + " (" + kind + ")");
  appendLog("\n— task " + id + " (" + kind + ") —");
  $("task-status").textContent = "running";
  $("task-status").className = "chip busy";
  const es = new EventSource("/api/task/stream?task_id=" + id);
  es.onmessage = (e) => {
    try { const d = JSON.parse(e.data); onTaskLine(d.line); } catch (err) { appendLog(e.data); }
  };
  es.addEventListener("done", async (e) => {
    es.close();
    try {
      const d = JSON.parse(e.data);
      if (d.returncode === 0) {
        $("task-status").textContent = (state.taskKind ? state.taskKind + " completed" : "completed");
        $("task-status").className = "chip done";
      } else {
        $("task-status").textContent = (state.taskKind ? state.taskKind + " FAILED" : "FAILED");
        $("task-status").className = "chip failed";
      }
    }
    catch (err) { $("task-status").textContent = "done"; $("task-status").className = "chip done"; }
    const wasDl = state.taskKind === "download";
    resetTaskUi();
    if (wasDl) { await updateLocalStatus(); updateDlChecks(); }
    loadModels(); loadInfo(); loadModes();
  });
  es.onerror = () => { es.close(); $("task-status").textContent = "stream closed"; $("task-status").className = "chip"; resetTaskUi(); };
}
function resetTaskUi() {
  state.taskActive = false;
  state.taskKind = null;
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
      if (status === "done") { setStage(ev.stage, "done"); appendLog("✔ " + (STAGE_DONE_LABEL[ev.stage] || ev.stage) + " " + rest.join(" ")); }
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
