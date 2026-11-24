/* Phishing detection UI — vanilla JS, light mode only, no build step. */
"use strict";

const state = {
  models: null,
  dataset: null,
  importance: null,
  selectedModels: new Set(),
  selectedMethods: new Set(),
  results: null,
  sortKey: "f1",
  sortDir: "desc",
};

/* ------------------------------------------------------------------ utils */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const trunc = (text, n) => (String(text).length > n ? String(text).slice(0, n - 1) + "…" : String(text));
const pct = (v) => (v === null || v === undefined ? "—" : (v * 100).toFixed(1) + "%");
const num = (v, d = 3) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));
const msFmt = (v) => (v === null || v === undefined ? "—" : v < 1 ? v.toFixed(3) + " ms" : v.toFixed(2) + " ms");

async function api(path, options) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail || JSON.stringify(body);
    } catch (_) { /* keep the status text */ }
    throw new Error(detail);
  }
  return response.json();
}

function banner(text, tone = "") {
  return `<div class="banner ${tone}">${text}</div>`;
}

/* ----------------------------------------------------------------- charts */
function hBars(rows, opts = {}) {
  const width = opts.width || 620;
  const labelW = opts.labelW || 168;
  const rowH = opts.rowH || 23;
  const color = opts.color || "#1d4ed8";
  const fmt = opts.fmt || ((v) => num(v, 3));
  const max = opts.max || Math.max(...rows.map((r) => r.value), 1e-9);
  const chartW = width - labelW - 74;
  const height = rows.length * rowH + 10;
  let out = `<svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" role="img">`;
  rows.forEach((row, i) => {
    const y = 5 + i * rowH;
    const w = Math.max(2, (Math.max(0, row.value) / max) * chartW);
    out += `<text x="${labelW - 8}" y="${y + 14}" text-anchor="end" font-size="11.4" fill="#475569">${esc(trunc(row.label, 24))}</text>`;
    out += `<rect x="${labelW}" y="${y + 3}" width="${w}" height="13" rx="3" fill="${row.color || color}"/>`;
    out += `<text x="${labelW + w + 6}" y="${y + 14}" font-size="10.8" fill="#64748b">${esc(row.display || fmt(row.value))}</text>`;
  });
  return out + "</svg>";
}

function contribBars(rows, opts = {}) {
  const width = opts.width || 620;
  const labelW = opts.labelW || 168;
  const rowH = 24;
  const chartW = width - labelW - 80;
  const height = rows.length * rowH + 12;
  const max = Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);
  const centre = labelW + chartW / 2;
  let out = `<svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" role="img">`;
  out += `<line x1="${centre}" y1="2" x2="${centre}" y2="${height - 2}" stroke="#e2e8f0"/>`;
  rows.forEach((row, i) => {
    const y = 6 + i * rowH;
    const w = (Math.abs(row.value) / max) * (chartW / 2);
    const positive = row.value >= 0;
    const x = positive ? centre : centre - w;
    out += `<text x="${labelW - 8}" y="${y + 14}" text-anchor="end" font-size="11.4" fill="#475569">${esc(trunc(row.label, 24))}</text>`;
    out += `<rect x="${x}" y="${y + 3}" width="${Math.max(1.5, w)}" height="13" rx="3" fill="${positive ? "#c0392b" : "#15803d"}"/>`;
    out += `<text x="${positive ? centre + w + 6 : centre - w - 6}" y="${y + 14}" font-size="10.8" text-anchor="${positive ? "start" : "end"}" fill="#64748b">${row.value >= 0 ? "+" : ""}${num(row.value, 3)}</text>`;
  });
  return out + "</svg>";
}

function distributionChart(counts) {
  const total = counts.reduce((acc, c) => acc + c.value, 0) || 1;
  const rows = counts.map((c) => ({
    label: c.label,
    value: c.value,
    color: c.color,
    display: `${c.value.toLocaleString()} · ${((c.value / total) * 100).toFixed(1)}%`,
  }));
  return hBars(rows, { width: 560, labelW: 120, max: Math.max(...rows.map((r) => r.value)), fmt: (v) => v.toLocaleString() });
}

/* ------------------------------------------------------------------ router */
const PAGES = ["dashboard", "analyze", "playground", "features", "explain", "dataset", "methodology"];

function showPage(name) {
  if (!PAGES.includes(name)) name = "dashboard";
  $$(".nav-item").forEach((btn) => btn.classList.toggle("active", btn.dataset.page === name));
  $$(".page").forEach((page) => page.classList.toggle("active", page.id === "page-" + name));
  window.scrollTo({ top: 0, behavior: "instant" });
  if (name === "methodology") renderMethodology();
}

/* --------------------------------------------------------------- dashboard */
function renderDashboard() {
  const dataset = state.dataset;
  const models = state.models;
  if (!dataset || !models) return;

  const kpis = [
    { label: "Dataset rows", value: dataset.n_samples.toLocaleString(), sub: "URL + 87 features + label" },
    { label: "Modelled features", value: dataset.n_features_modelled, sub: `${dataset.constant_features_removed.length} zero-variance columns dropped` },
    { label: "Models trained", value: models.models.length, sub: models.models.map((m) => m.label).join(" · ") },
    { label: "Held-out test rows", value: (dataset.splits.find((s) => s.split === "test") || {}).n_samples?.toLocaleString() || "—", sub: "Used only for reported metrics" },
  ];
  $("#dash-kpis").innerHTML = kpis.map((k) => `
    <div class="kpi">
      <div class="kpi-label">${esc(k.label)}</div>
      <div class="kpi-value">${esc(k.value)}</div>
      <div class="kpi-sub">${esc(k.sub)}</div>
    </div>`).join("");

  const phishing = dataset.class_counts_clean.phishing;
  const legitimate = dataset.class_counts_clean.legitimate;
  $("#dash-dist").innerHTML = distributionChart([
    { label: "Legitimate", value: legitimate, color: "#15803d" },
    { label: "Phishing", value: phishing, color: "#b91c1c" },
  ]) + `<div class="legend"><span><span class="dot" style="background:#15803d"></span>Legitimate ${legitimate.toLocaleString()}</span><span><span class="dot" style="background:#b91c1c"></span>Phishing ${phishing.toLocaleString()}</span><span>Cleaned: ${(phishing + legitimate).toLocaleString()} rows</span></div>`;

  $("#dash-status").innerHTML = `
    <dl class="kv">
      <dt>Live URL features</dt><dd>${dataset.n_features_served} extracted from the URL string</dd>
      <dt>Feature selectors</dt><dd>${models.methods.filter((m) => m.available).map((m) => m.label).join(", ")}</dd>
      <dt>Evaluation split</dt><dd>${esc(models.dataset.n_eval_samples.toLocaleString())} test rows</dd>
      <dt>Explanation</dt><dd>Exact tree SHAP for tree models; LIME for every model</dd>
      <dt>Models</dt><dd>${models.models.map((m) => `${m.label}${m.tree_shap ? "" : " (no tree SHAP)"}`).join(", ")}</dd>
    </dl>`;
}

async function dashboardQuickCheck() {
  const url = $("#dash-url").value.trim();
  const target = $("#dash-quick");
  if (!url) { target.innerHTML = banner("Enter a URL first.", "amber"); return; }
  target.innerHTML = `<div class="loading-box"><span class="spinner dark"></span>Running the trained models…</div>`;
  try {
    const data = await api("/api/analyze", { method: "POST", body: JSON.stringify({ url }) });
    const verdictClass = data.headline.prediction === "Suspicious" ? "suspicious" : "legitimate";
    target.innerHTML = `
      <div class="result-head" style="margin-bottom:10px">
        <span class="verdict ${verdictClass}">${esc(data.headline.prediction)}</span>
        <span class="small muted">${esc(data.headline.model_label)} · P(phishing) = ${num(data.headline.probability, 3)} · ${msFmt(data.headline.time_ms)}</span>
      </div>
      <div class="prob-bar"><div class="prob-fill ${verdictClass === "legitimate" ? "safe" : ""}" style="width:${Math.round(data.headline.probability * 100)}%"></div></div>
      <div class="small muted" style="margin-top:10px">Open <a href="#analyze">URL Detection</a> for the per-model table and the SHAP explanation.</div>`;
  } catch (error) {
    target.innerHTML = banner(esc(error.message), "red");
  }
}

/* ----------------------------------------------------------- URL detection */
function modelOptions() {
  const models = state.models ? state.models.models : [];
  return models.map((m) => ({ name: m.name, label: m.label, note: m.note }));
}

function renderAnalyzeModels() {
  const host = $("#analyze-models");
  const selected = state.selectedModels.size ? state.selectedModels : new Set(["RandomForest"]);
  host.innerHTML = modelOptions().map((m) => `
    <label class="pick ${selected.has(m.name) ? "on" : ""}" data-model="${esc(m.name)}">
      <input type="checkbox" ${selected.has(m.name) ? "checked" : ""} />
      <span class="pick-body">
        <span class="pick-name">${esc(m.label)}</span>
        <span class="pick-desc">${esc(m.name === "KNN" ? "No tree SHAP available" : "Supports exact tree SHAP")}</span>
      </span>
    </label>`).join("");

  $$("#analyze-models .pick").forEach((card) => {
    card.addEventListener("click", (event) => {
      if (event.target.tagName !== "INPUT") {
        const box = $("input", card);
        box.checked = !box.checked;
      }
      const name = card.dataset.model;
      const box = $("input", card);
      if (box.checked) state.selectedModels.add(name); else state.selectedModels.delete(name);
      card.classList.toggle("on", box.checked);
    });
  });
}

function whyRows(data, rows) {
  const byName = Object.fromEntries(data.features.map((f) => [f.feature, f]));
  return (rows || [])
    .filter((r) => r.value > 0)
    .slice(0, 6)
    .map((r) => {
      const meta = byName[r.feature] || {};
      return `<div class="why-item">
        <div class="why-head">
          <span class="why-feature">${esc(r.feature)}</span>
          <span class="why-val">+${num(r.value, 3)}</span>
        </div>
        <div class="why-text">${esc(meta.interpretation || "")} · extracted value <span class="mono">${num(r.feature_value ?? meta.value, 3)}</span></div>
      </div>`;
    });
}

function renderAnalysis(host, data) {
  const verdict = data.headline.prediction === "Suspicious" ? "suspicious" : "legitimate";
  const hasShap = Boolean(data.shap.rows && data.shap.rows.length);
  const contribRows = hasShap ? data.shap.rows : (data.lime.rows || []);
  const why = whyRows(data, contribRows);
  const sourceLabel = hasShap ? "SHAP" : "LIME";

  const modelRows = data.predictions.map((p) => `
    <tr>
      <td>${esc(p.model_label)}</td>
      <td><span class="chip ${p.prediction === "Suspicious" ? "red" : "green"}">${esc(p.prediction)}</span></td>
      <td class="num">${num(p.probability, 4)}</td>
      <td class="num">${msFmt(p.time_ms)}</td>
    </tr>`).join("");

  const shapRows = contribRows.slice(0, 10).map((r) => ({ label: r.feature, value: r.value }));

  host.innerHTML = `
    <div class="card">
      <div class="result-head">
        <span class="verdict ${verdict}">${esc(data.headline.prediction)}</span>
        <div>
          <div style="font-weight:640;color:var(--navy)">P(phishing) = ${num(data.headline.probability, 4)}</div>
          <div class="small muted">Flagged by ${esc(data.headline.model_label)} · analysis took ${msFmt(data.headline.time_ms)}</div>
        </div>
        <div style="flex:1;min-width:220px">
          <div class="prob-bar"><div class="prob-fill ${verdict === "legitimate" ? "safe" : ""}" style="width:${Math.round(data.headline.probability * 100)}%"></div></div>
          <div class="small muted" style="margin-top:4px">${esc(data.url)}</div>
        </div>
      </div>
      ${banner("This is a single-URL prediction, not a benchmark score. Accuracy is a dataset-level metric and is reported only in Model Playground.", "amber")}
    </div>

    <div class="grid cols-2">
      <div class="card">
        <div class="card-head"><h2 class="card-title">Model comparison</h2><span class="chip blue">${data.predictions.length} model${data.predictions.length > 1 ? "s" : ""}</span></div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Model</th><th>Prediction</th><th class="num">Probability</th><th class="num">Predict time</th></tr></thead>
            <tbody>${modelRows}</tbody>
          </table>
        </div>
        <div class="card-note">Each row is that model's own output for the same extracted feature vector.</div>
      </div>

      <div class="card">
        <div class="card-head"><h2 class="card-title">Why was it flagged?</h2></div>
        ${why.length ? why.join("") : `<div class="empty">No feature pushed this prediction towards “suspicious”.</div>`}
        <div class="card-note">Ranked by ${sourceLabel} contribution for ${esc(data.shap.model || data.headline.model_label)}.</div>
      </div>
    </div>

    <div class="card">
      <div class="card-head"><h2 class="card-title">Feature contributions (${sourceLabel})</h2></div>
      ${data.shap.note ? banner(esc(data.shap.note), "amber") : ""}
      ${shapRows.length ? `<div class="chart-wrap">${contribBars(shapRows)}</div>
      <div class="legend"><span><span class="dot" style="background:#c0392b"></span>Pushes towards suspicious</span><span><span class="dot" style="background:#15803d"></span>Pushes towards legitimate</span></div>` : ""}
    </div>

    ${data.lime.rows && data.lime.rows.length ? `
    <div class="card">
      <div class="card-head"><h2 class="card-title">LIME explanation</h2><span class="chip">${esc(data.lime.model)}</span></div>
      <div class="chart-wrap">${contribBars(data.lime.rows.slice(0, 10).map((r) => ({ label: r.feature, value: r.value })))}</div>
      <div class="card-note">Local surrogate fitted around this prediction; weights are in probability units, not SHAP units.</div>
    </div>` : (data.lime.note ? `<div class="card">${banner(esc(data.lime.note), "amber")}</div>` : "")}

    <div class="card tight">
      ${data.notes.map((n) => `<p class="small muted">${esc(n)}</p>`).join("")}
    </div>`;
}

async function analyzeUrl() {
  const host = $("#analyze-result");
  const url = $("#analyze-url").value.trim();
  if (!url) { host.innerHTML = banner("Enter a URL to analyse.", "amber"); return; }
  const models = Array.from(state.selectedModels);
  host.innerHTML = `<div class="card"><div class="loading-box"><span class="spinner dark"></span>Extracting features and running ${models.length || "the"} model(s)…</div></div>`;
  try {
    const data = await api("/api/analyze", { method: "POST", body: JSON.stringify({ url, models }) });
    renderAnalysis(host, data);
  } catch (error) {
    host.innerHTML = `<div class="card">${banner(esc(error.message), "red")}</div>`;
  }
}

/* --------------------------------------------------------- model playground */
function renderPlaygroundPickers() {
  const modelHost = $("#pg-models");
  modelHost.innerHTML = modelOptions().map((m) => `
    <label class="pick ${state.selectedModels.has(m.name) ? "on" : ""}" data-pg-model="${esc(m.name)}">
      <input type="checkbox" ${state.selectedModels.has(m.name) ? "checked" : ""} />
      <span class="pick-body">
        <span class="pick-name">${esc(m.label)}</span>
        <span class="pick-desc">${esc(m.name === "KNN" ? "Distance-based; no tree SHAP" : m.name === "LightGBM" ? "Leaf-wise boosting; slowest to fit" : m.note.split(";")[0])}</span>
      </span>
    </label>`).join("");

  $$("#pg-models .pick").forEach((card) => {
    card.addEventListener("click", (event) => {
      const box = $("input", card);
      if (event.target.tagName !== "INPUT") box.checked = !box.checked;
      const name = card.dataset.pgModel;
      if (box.checked) state.selectedModels.add(name); else state.selectedModels.delete(name);
      card.classList.toggle("on", box.checked);
      $("#pg-model-count").textContent = `${state.selectedModels.size} selected`;
    });
  });

  const methods = state.models.methods.filter((m) => m.available);
  const methodHost = $("#pg-methods");
  methodHost.innerHTML = methods.map((m) => `
    <label class="pick ${state.selectedMethods.has(m.name) ? "on" : ""}" data-pg-method="${esc(m.name)}">
      <input type="checkbox" ${state.selectedMethods.has(m.name) ? "checked" : ""} />
      <span class="pick-body">
        <span class="pick-name"><span data-tip="${esc(m.description)}">${esc(m.label)}</span></span>
      </span>
    </label>`).join("");

  $$("#pg-methods .pick").forEach((card) => {
    card.addEventListener("click", (event) => {
      const box = $("input", card);
      if (event.target.tagName !== "INPUT") box.checked = !box.checked;
      const name = card.dataset.pgMethod;
      if (box.checked) state.selectedMethods.add(name); else state.selectedMethods.delete(name);
      card.classList.toggle("on", box.checked);
      $("#pg-method-count").textContent = `${state.selectedMethods.size} selected`;
    });
  });

  $("#pg-model-count").textContent = `${state.selectedModels.size} selected`;
  $("#pg-method-count").textContent = `${state.selectedMethods.size} selected`;
}

function comparisonTable(results) {
  const bestF1 = Math.max(...results.map((r) => r.f1));
  const minFeatures = Math.min(...results.map((r) => r.n_features));
  const leanest = results.filter((r) => r.n_features === minFeatures).sort((a, b) => b.f1 - a.f1)[0];

  const sorted = [...results].sort((a, b) => {
    const key = state.sortKey;
    const dir = state.sortDir === "asc" ? 1 : -1;
    if (key === "model") return a.model_label.localeCompare(b.model_label) * dir;
    return ((a[key] ?? 0) - (b[key] ?? 0)) * dir;
  });

  const header = (key, label) => {
    const arrow = state.sortKey === key ? (state.sortDir === "asc" ? "▲" : "▼") : "";
    return `<th class="sortable" data-sort="${key}">${label}<span class="arrow">${arrow}</span></th>`;
  };

  const rows = sorted.map((r) => {
    const isBest = r.f1 === bestF1;
    const tag = [];
    if (isBest) tag.push('<span class="chip green">best F1</span>');
    if (r === leanest && !isBest) tag.push('<span class="chip amber">least features</span>');
    return `<tr class="${isBest ? "best" : ""}">
      <td>${esc(r.model_label)}${tag.length ? `<div class="tag-row" style="margin-top:4px">${tag.join("")}</div>` : ""}</td>
      <td>${esc(r.method_label)}</td>
      <td class="num">${pct(r.accuracy)}</td>
      <td class="num">${pct(r.precision)}</td>
      <td class="num">${pct(r.recall)}</td>
      <td class="num"><strong>${pct(r.f1)}</strong></td>
      <td class="num">${r.roc_auc === null ? "—" : num(r.roc_auc, 4)}</td>
      <td class="num">${r.n_features}</td>
      <td class="num">${msFmt(r.predict_time_ms)}</td>
    </tr>`;
  }).join("");

  return `<div class="table-wrap">
    <table>
      <thead><tr>
        <th class="sortable" data-sort="model">Model<span class="arrow">${state.sortKey === "model" ? (state.sortDir === "asc" ? "▲" : "▼") : ""}</span></th>
        <th>Feature selection</th>
        ${header("accuracy", "Accuracy")}
        <th class="num">Precision</th>
        <th class="num">Recall</th>
        ${header("f1", "F1")}
        <th class="num">ROC-AUC</th>
        <th class="num">Features</th>
        ${header("predict_time_ms", "Pred. time")}
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </div>`;
}

function tradeoffText(results) {
  const full = results.filter((r) => r.method === "Full");
  const best = results.reduce((a, b) => (b.f1 > a.f1 ? b : a), results[0]);
  if (!full.length) {
    return `Best F1 is ${pct(best.f1)} (${esc(best.model_label)} · ${esc(best.method_label)}) with ${best.n_features} features. Add the Full Feature Set baseline to measure the cost of selection directly.`;
  }
  const baseline = full.reduce((a, b) => (b.f1 > a.f1 ? b : a), full[0]);
  const delta = best.f1 - baseline.f1;
  const saved = Math.round((1 - best.n_features / baseline.n_features) * 100);
  return `Full-set baseline reaches F1 ${pct(baseline.f1)} with ${baseline.n_features} features. The best row (${esc(best.model_label)} · ${esc(best.method_label)}) uses ${best.n_features} features — ${saved > 0 ? `${saved}% fewer` : "no reduction"} — and changes F1 by ${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(2)} points. Fewer features mean a smaller attack surface for data drift and cheaper extraction, so a tiny F1 loss can still be the better engineering choice; accuracy alone is not the deciding metric.`;
}

function renderPlaygroundResults() {
  const host = $("#pg-results");
  const payload = state.results;
  if (!payload || !payload.results.length) {
    host.innerHTML = `<div class="card">${banner("No result rows were produced. Check the configuration and try again.", "amber")}</div>`;
    return;
  }
  const results = payload.results;
  const label = (r) => `${r.model_label.replace("K-Nearest Neighbours", "KNN")} · ${r.method_label}`;

  const accuracyRows = results.map((r) => ({ label: label(r), value: r.accuracy }));
  const f1Rows = results.map((r) => ({ label: label(r), value: r.f1 }));
  const aucRows = results.map((r) => ({ label: label(r), value: r.roc_auc || 0 }));
  const featRows = results.map((r) => ({ label: label(r), value: r.n_features, color: "#b45309", display: `${r.n_features}` }));
  const timeRows = results.map((r) => ({ label: label(r), value: r.predict_time_ms, color: "#64748b", display: msFmt(r.predict_time_ms) }));

  host.innerHTML = `
    <div class="card">
      <div class="card-head">
        <h2 class="card-title">Comparison results</h2>
        <span class="chip">${results.length} configurations · ${payload.elapsed_s < 0.1 ? "cached" : payload.elapsed_s.toFixed(1) + "s"} · ${payload.n_eval_samples.toLocaleString()} test rows</span>
      </div>
      ${comparisonTable(results)}
      <div class="card-note">Held-out test split of ${payload.n_eval_samples.toLocaleString()} labelled URLs (${esc(payload.eval_split)}). Click Model, Accuracy, F1 or Pred. time to sort. Metrics are dataset-level; no accuracy is computed for a single URL.</div>
    </div>

    <div class="card">
      <div class="card-head"><h2 class="card-title">Performance vs. feature reduction</h2></div>
      <p class="small" style="color:#334155;margin:0">${tradeoffText(results)}</p>
    </div>

    <div class="grid cols-2">
      <div class="card"><div class="card-head"><h2 class="card-title">Accuracy</h2></div><div class="chart-wrap">${hBars(accuracyRows, { color: "#1d4ed8", fmt: (v) => pct(v) })}</div></div>
      <div class="card"><div class="card-head"><h2 class="card-title">F1 score</h2></div><div class="chart-wrap">${hBars(f1Rows, { color: "#15803d", fmt: (v) => pct(v) })}</div></div>
      <div class="card"><div class="card-head"><h2 class="card-title">ROC-AUC</h2></div><div class="chart-wrap">${hBars(aucRows, { color: "#0b2545", fmt: (v) => num(v, 4) })}</div></div>
      <div class="card"><div class="card-head"><h2 class="card-title">Features used</h2></div><div class="chart-wrap">${hBars(featRows, { color: "#b45309", fmt: (v) => `${v}` })}</div></div>
      <div class="card"><div class="card-head"><h2 class="card-title">Prediction time (per sample)</h2></div><div class="chart-wrap">${hBars(timeRows, { color: "#64748b", fmt: (v) => msFmt(v) })}</div>
        <div class="card-note">Measured scoring the test split with the fitted model; training time is excluded.</div></div>
      <div class="card">
        <div class="card-head"><h2 class="card-title">Feature importance by method</h2></div>
        <div id="pg-importance"></div>
      </div>
    </div>
    ${payload.skipped.length ? `<div class="card">${banner("Skipped: " + payload.skipped.map((s) => esc(`${s.model} · ${s.method} (${s.reason})`)).join("; "), "amber")}</div>` : ""}`;

  renderImportanceComparison($("#pg-importance"));
  $$("#pg-results th.sortable").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (state.sortKey === key) state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
      else { state.sortKey = key; state.sortDir = "desc"; }
      renderPlaygroundResults();
    });
  });
}

function renderImportanceComparison(host) {
  if (!host) return;
  const rankings = state.importance ? state.importance.rankings : null;
  const methods = Array.from(state.selectedMethods).filter((m) => rankings && rankings.methods[m]);
  if (!methods.length) {
    host.innerHTML = `<div class="empty">Select a ranked method (SHAP, LIME, Permutation, …) to see its top features.</div>`;
    return;
  }
  host.innerHTML = methods.map((m) => `
    <div style="margin-bottom:14px">
      <div class="small" style="font-weight:640;color:var(--navy);margin-bottom:6px">${esc(rankings.methods[m].label)} <span class="muted" style="font-weight:400">· ${rankings.methods[m].elapsed_s}s</span></div>
      <div class="tag-row">${rankings.methods[m].top.slice(0, 10).map((f, i) => `<span class="chip ${i === 0 ? "blue" : ""}">${i + 1}. ${esc(f)}</span>`).join("")}</div>
    </div>`).join("");
}

async function runComparison() {
  const status = $("#pg-status");
  const host = $("#pg-results");
  const models = Array.from(state.selectedModels);
  const methods = Array.from(state.selectedMethods);
  if (!models.length) { status.innerHTML = `<span class="chip red">Select at least one model</span>`; return; }
  if (!methods.length) { status.innerHTML = `<span class="chip red">Select at least one method</span>`; return; }

  const k = Number($("#pg-k").value);
  const epsilon = Number($("#pg-eps").value);
  const button = $("#pg-run");
  button.disabled = true;
  status.innerHTML = `<span class="small muted"><span class="spinner dark" style="vertical-align:-2px"></span> Fitting ${models.length * methods.length} configurations…</span>`;
  host.innerHTML = `<div class="card"><div class="loading-box"><span class="spinner dark"></span>Training and evaluating on the held-out split. The first run of a configuration can take a few seconds per model.</div></div>`;

  try {
    const data = await api("/api/compare-models", {
      method: "POST",
      body: JSON.stringify({ models, methods, k, epsilon }),
    });
    state.results = data;
    renderPlaygroundResults();
    status.innerHTML = `<span class="chip green">${data.elapsed_s < 0.1 ? "Completed instantly — models cached" : `Completed in ${data.elapsed_s.toFixed(1)}s`}</span>`;
  } catch (error) {
    host.innerHTML = `<div class="card">${banner(esc(error.message), "red")}</div>`;
    status.innerHTML = "";
  } finally {
    button.disabled = false;
  }
}

/* --------------------------------------------------------- feature analysis */
function featureTable(features, sort = true) {
  const rows = [...features];
  if (sort) {
    rows.sort((a, b) => {
      if (a.importance === null && b.importance === null) return a.feature.localeCompare(b.feature);
      if (a.importance === null) return 1;
      if (b.importance === null) return -1;
      return b.importance - a.importance;
    });
  }
  return `<div class="table-wrap"><table>
    <thead><tr><th>Feature</th><th class="num">Extracted value</th><th class="num">Importance</th><th>Interpretation</th></tr></thead>
    <tbody>${rows.map((f) => `<tr>
      <td><span class="mono">${esc(f.feature)}</span></td>
      <td class="num">${num(f.value, 4)}</td>
      <td class="num">${f.importance === null ? "—" : num(f.importance, 4)}</td>
      <td class="small">${esc(f.interpretation || f.description || "")}</td>
    </tr>`).join("")}</tbody>
  </table></div>`;
}

async function runFeatureAnalysis() {
  const host = $("#fa-result");
  const url = $("#fa-url").value.trim();
  if (!url) { host.innerHTML = `<div class="card">${banner("Enter a URL to extract its features.", "amber")}</div>`; return; }
  const model = $("#fa-model").value;
  host.innerHTML = `<div class="card"><div class="loading-box"><span class="spinner dark"></span>Extracting features…</div></div>`;
  try {
    const data = await api("/api/feature-analysis", { method: "POST", body: JSON.stringify({ url, model }) });
    const ranked = data.features.filter((f) => f.importance !== null).sort((a, b) => b.importance - a.importance);
    const chart = ranked.slice(0, 12).map((f) => ({ label: f.feature, value: f.importance }));
    const source = data.importance_source === "lime" ? "LIME weights" : "SHAP values";

    host.innerHTML = `
      <div class="card">
        <div class="result-head">
          <span class="verdict ${data.prediction.prediction === "Suspicious" ? "suspicious" : "legitimate"}">${esc(data.prediction.prediction)}</span>
          <div>
            <div style="font-weight:640;color:var(--navy)">${esc(data.model_label)} · P(phishing) = ${num(data.prediction.probability, 4)}</div>
            <div class="small muted">${esc(data.url)}</div>
          </div>
        </div>
      </div>

      <div class="grid cols-2">
        <div class="card">
          <div class="card-head"><h2 class="card-title">Most influential features</h2><span class="chip">${esc(source)}</span></div>
          ${chart.length ? `<div class="chart-wrap">${hBars(chart, { width: 560, labelW: 168, color: "#1d4ed8", fmt: (v) => num(v, 4) })}</div>` : `<div class="empty">No importance values available for this model.</div>`}
          <div class="card-note">Magnitude of each feature's contribution to this single prediction.</div>
        </div>
        <div class="card">
          <div class="card-head"><h2 class="card-title">Not retrieved for live URLs</h2></div>
          <p class="small muted">These dataset columns need the page itself or a third-party service. They are excluded from live analysis rather than guessed.</p>
          <div class="tag-row">${data.external_not_retrieved.map((f) => `<span class="chip">${esc(f)}</span>`).join("")}</div>
          <div class="section-sep"></div>
          <p class="small muted">${esc(data.note)}</p>
        </div>
      </div>

      <div class="card">
        <div class="card-head"><h2 class="card-title">Extracted features (${data.features.length})</h2><span class="chip">${esc(data.model_label)}</span></div>
        ${featureTable(data.features)}
        <div class="card-note">Importance is ${esc(source)}, sorted by magnitude. Values are exactly what the model received, in the dataset's own units.</div>
      </div>`;
  } catch (error) {
    host.innerHTML = `<div class="card">${banner(esc(error.message), "red")}</div>`;
  }
}

/* ------------------------------------------------------------ explainability */
async function runExplain() {
  const model = $("#ex-model").value;
  const url = $("#ex-url").value.trim();
  const globalHost = $("#ex-global");
  const instanceHost = $("#ex-instance");
  globalHost.innerHTML = `<div class="card"><div class="loading-box"><span class="spinner dark"></span>Computing global SHAP importance…</div></div>`;
  instanceHost.innerHTML = "";

  try {
    const data = await api("/api/explain", { method: "POST", body: JSON.stringify({ model, url: url || null }) });

    if (data.global && data.global.error) {
      globalHost.innerHTML = `<div class="card">${banner(esc(data.global.error) + " Global importance is unavailable for this model; individual LIME explanations still work.", "amber")}</div>`;
    } else {
      const rows = data.global.rows.slice(0, 15).map((r) => ({ label: r.feature, value: r.importance }));
      globalHost.innerHTML = `
        <div class="card">
          <div class="card-head"><h2 class="card-title">Global SHAP feature importance</h2><span class="chip blue">${esc(data.model_label)}</span></div>
          <div class="card-note" style="margin:-6px 0 12px">Mean absolute SHAP value over ${data.global.n_samples} test-split instances. This describes the model across the dataset, not any single URL.</div>
          <div class="chart-wrap">${hBars(rows, { width: 700, labelW: 180, color: "#1d4ed8", fmt: (v) => num(v, 5) })}</div>
          <div class="table-wrap" style="margin-top:14px"><table>
            <thead><tr><th class="num">Rank</th><th>Feature</th><th class="num">Mean |SHAP|</th></tr></thead>
            <tbody>${data.global.rows.slice(0, 12).map((r) => `<tr><td class="num">${r.rank}</td><td class="mono">${esc(r.feature)}</td><td class="num">${num(r.importance, 5)}</td></tr>`).join("")}</tbody>
          </table></div>
        </div>`;
    }

    if (!url) {
      instanceHost.innerHTML = `<div class="card">${banner("No URL supplied. Enter one above to see an individual explanation; global and individual explanations are deliberately kept separate.", "")}</div>`;
      return;
    }

    const inst = data.instance;
    const shapNote = inst.shap.note;
    const positive = inst.contributions.positive;
    const negative = inst.contributions.negative;

    const listBlock = (rows, tone, title) => `
      <div class="card">
        <div class="card-head"><h2 class="card-title">${title}</h2><span class="chip ${tone === "red" ? "red" : "green"}">${rows.length}</span></div>
        ${rows.length ? rows.map((r) => `<div class="why-item">
          <div class="why-head"><span class="why-feature">${esc(r.feature)}</span>
          <span class="why-val ${tone === "green" ? "neg" : ""}">${r.value >= 0 ? "+" : ""}${num(r.value, 4)}</span></div>
          <div class="why-text">extracted value <span class="mono">${num(r.feature_value, 4)}</span></div>
        </div>`).join("") : `<div class="empty">None for this prediction.</div>`}
      </div>`;

    instanceHost.innerHTML = `
      <div class="card">
        <div class="result-head">
          <span class="verdict ${inst.prediction.prediction === "Suspicious" ? "suspicious" : "legitimate"}">${esc(inst.prediction.prediction)}</span>
          <div>
            <div style="font-weight:640;color:var(--navy)">Individual explanation · ${esc(data.model_label)} · P(phishing) = ${num(inst.prediction.probability, 4)}</div>
            <div class="small muted">${esc(inst.url)}</div>
          </div>
        </div>
      </div>
      <div class="grid cols-2">
        ${shapNote ? `<div class="card">${banner(esc(shapNote), "amber")}</div>` : listBlock(positive, "red", "Positive contributions (→ suspicious)")}
        ${shapNote ? "" : listBlock(negative, "green", "Negative contributions (→ legitimate)")}
      </div>
      ${inst.lime.rows.length ? `<div class="card">
        <div class="card-head"><h2 class="card-title">LIME — local surrogate</h2><span class="chip">${esc(inst.lime.model)}</span></div>
        <div class="chart-wrap">${contribBars(inst.lime.rows.slice(0, 10).map((r) => ({ label: r.feature, value: r.value })))}</div>
        <div class="card-note">LIME fits a linear model around this instance; weights are not comparable to SHAP values.</div>
      </div>` : ""}`;
  } catch (error) {
    globalHost.innerHTML = `<div class="card">${banner(esc(error.message), "red")}</div>`;
  }
}

/* ---------------------------------------------------------------- dataset */
function renderDataset() {
  const d = state.dataset;
  if (!d) return;
  const test = d.splits.find((s) => s.split === "test") || {};
  const splitRows = d.splits.map((s) => `<tr><td>${esc(s.split)}</td><td class="num">${s.n_samples.toLocaleString()}</td><td class="num">${s.n_phishing.toLocaleString()}</td><td class="num">${s.n_legitimate.toLocaleString()}</td><td class="num">${(s.phishing_rate * 100).toFixed(1)}%</td></tr>`).join("");

  const fidelityRows = Object.entries(d.fidelity)
    .sort((a, b) => b[1].exact_match - a[1].exact_match)
    .map(([name, s]) => `<tr>
      <td class="mono">${esc(name)}</td>
      <td class="num">${(s.exact_match * 100).toFixed(1)}%</td>
      <td class="num">${num(s.correlation, 3)}</td>
      <td class="num">${s.mae < 0.001 ? "0" : num(s.mae, 3)}</td>
      <td>${s.exact_match >= 0.95 ? '<span class="chip green">reproduced</span>' : '<span class="chip amber">approximate</span>'}</td>
    </tr>`).join("");

  $("#ds-body").innerHTML = `
    <div class="grid cols-4">
      <div class="kpi"><div class="kpi-label">Samples</div><div class="kpi-value">${d.n_samples.toLocaleString()}</div><div class="kpi-sub">raw rows loaded</div></div>
      <div class="kpi"><div class="kpi-label">Features (raw)</div><div class="kpi-value">${d.n_features_raw}</div><div class="kpi-sub">${d.n_features_modelled} after cleaning</div></div>
      <div class="kpi"><div class="kpi-label">Phishing</div><div class="kpi-value">${d.class_counts_clean.phishing.toLocaleString()}</div><div class="kpi-sub">label = 1</div></div>
      <div class="kpi"><div class="kpi-label">Legitimate</div><div class="kpi-value">${d.class_counts_clean.legitimate.toLocaleString()}</div><div class="kpi-sub">label = 0</div></div>
    </div>

    <div class="card">
      <div class="card-head"><h2 class="card-title">Dataset identity</h2>
        <span class="chip ${d.matches_paper ? "green" : "amber"}">${d.matches_paper ? "matches the paper dimensions" : "differs from the paper"}</span></div>
      <dl class="kv">
        <dt>Name</dt><dd>${esc(d.name)}</dd>
        <dt>Source</dt><dd>${esc(d.source)}</dd>
        <dt>Raw shape</dt><dd>${d.raw_shape[0].toLocaleString()} rows × ${d.raw_shape[1]} columns (URL + ${d.n_features_raw} features + label)</dd>
        <dt>Missing values</dt><dd>${d.missing_values}</dd>
        <dt>Duplicate feature rows removed</dt><dd>${d.duplicate_feature_rows}</dd>
        <dt>Zero-variance columns dropped</dt><dd>${d.constant_features_removed.length ? d.constant_features_removed.map((c) => `<span class="mono">${esc(c)}</span>`).join(", ") : "none"}</dd>
        <dt>Paper reference</dt><dd>11,430 samples · 87 features</dd>
      </dl>
      <div class="section-sep"></div>
      <p class="small" style="color:#334155;margin:0">
        ${d.matches_paper
          ? "The implementation loads the same 11,430 × 87 benchmark quoted in the reference paper, so no substitution has taken place. The counts shown elsewhere differ only because five rows are ambiguous duplicates and six columns are constant, which the cleaning stage removes before modelling."
          : "The loaded file differs from the dimensions quoted in the reference paper; the counts above are the real ones and are used everywhere in the application."}
        The live analyser additionally uses only the ${d.n_features_served} URL-derived columns, because the remaining features require fetching the page or querying third-party services.
      </p>
    </div>

    <div class="grid cols-2">
      <div class="card">
        <div class="card-head"><h2 class="card-title">Class distribution</h2></div>
        ${distributionChart([
          { label: "Legitimate", value: d.class_counts_clean.legitimate, color: "#15803d" },
          { label: "Phishing", value: d.class_counts_clean.phishing, color: "#b91c1c" },
        ])}
        <div class="card-note">Balanced by construction: the benchmark pairs each phishing URL with a legitimate one.</div>
      </div>
      <div class="card">
        <div class="card-head"><h2 class="card-title">Train / validation / test</h2><span class="chip">${test.n_samples ? test.n_samples.toLocaleString() + " test rows" : ""}</span></div>
        <div class="table-wrap"><table>
          <thead><tr><th>Split</th><th class="num">Rows</th><th class="num">Phishing</th><th class="num">Legitimate</th><th class="num">Rate</th></tr></thead>
          <tbody>${splitRows}</tbody>
        </table></div>
        <div class="card-note">Stratified 60/20/20 with a fixed seed. Feature selection sees the training split only.</div>
      </div>
    </div>

    <div class="card">
      <div class="card-head"><h2 class="card-title">Feature dictionary (${d.feature_dictionary_all.length} columns)</h2>
        <input type="text" id="ds-filter" placeholder="Filter features…" style="max-width:240px" /></div>
      <div class="table-wrap" style="max-height:520px;overflow-y:auto">
        <table>
          <thead><tr><th>Feature</th><th>Group</th><th>Description</th><th>Live</th></tr></thead>
          <tbody id="ds-dict-body">${dictRows(d.feature_dictionary_all)}</tbody>
        </table>
      </div>
      <div class="card-note">“Live” marks the columns the URL analyser can compute from a URL string alone.</div>
    </div>

    <div class="card">
      <div class="card-head"><h2 class="card-title">Extractor fidelity</h2><span class="chip">measured on 400 random dataset rows</span></div>
      <p class="small muted">The live extractor re-implements the dataset's own features. This table reports how often it reproduces the published column values: exact-match rate for discrete features and correlation for continuous ones. Only features at or above the threshold are served.</p>
      <div class="table-wrap" style="max-height:360px;overflow-y:auto"><table>
        <thead><tr><th>Feature</th><th class="num">Exact match</th><th class="num">Correlation</th><th class="num">MAE</th><th>Status</th></tr></thead>
        <tbody>${fidelityRows}</tbody>
      </table></div>
    </div>`;

  const filter = $("#ds-filter");
  if (filter) {
    filter.addEventListener("input", () => {
      $("#ds-dict-body").innerHTML = dictRows(d.feature_dictionary_all, filter.value);
    });
  }
}

function dictRows(rows, query = "") {
  const q = query.trim().toLowerCase();
  const filtered = q
    ? rows.filter((r) => r.name.toLowerCase().includes(q) || r.description.toLowerCase().includes(q) || r.group.toLowerCase().includes(q))
    : rows;
  if (!filtered.length) return `<tr><td colspan="4" class="empty">No feature matches that filter.</td></tr>`;
  return filtered.map((r) => `<tr>
    <td class="mono">${esc(r.name)}</td>
    <td><span class="chip">${esc(r.group)}</span></td>
    <td class="small">${esc(r.description)}</td>
    <td>${r.computed_live ? '<span class="chip green">yes</span>' : '<span class="chip">no</span>'}</td>
  </tr>`).join("");
}

/* ------------------------------------------------------------- methodology */
function renderMethodology() {
  const host = $("#ms-ranks");
  const rankings = state.importance ? state.importance.rankings : null;
  if (!host) return;
  if (!rankings || !Object.keys(rankings.methods).length) {
    host.innerHTML = `<div class="empty">Rankings are still being computed.</div>`;
    return;
  }
  host.innerHTML = `<div class="table-wrap"><table>
    <thead><tr><th>Method</th><th class="num">Fit time (s)</th><th>Top 5 features</th></tr></thead>
    <tbody>${Object.entries(rankings.methods).map(([name, m]) => `<tr>
      <td>${esc(m.label)}</td><td class="num">${m.elapsed_s}</td>
      <td class="small mono">${m.top.slice(0, 5).map(esc).join(", ")}</td>
    </tr>`).join("")}</tbody>
  </table></div>
  <div class="card-note">One-off cost on the 32-feature URL space; every playground run reuses these rankings.</div>`;
}

/* ------------------------------------------------------------------- boot */
async function boot() {
  $$(".nav-item").forEach((btn) => btn.addEventListener("click", () => { location.hash = btn.dataset.page; }));

  const initial = (location.hash || "#dashboard").replace("#", "");
  showPage(initial);

  $("#dash-go").addEventListener("click", dashboardQuickCheck);
  $("#dash-url").addEventListener("keydown", (e) => { if (e.key === "Enter") dashboardQuickCheck(); });
  $("#analyze-go").addEventListener("click", analyzeUrl);
  $("#analyze-url").addEventListener("keydown", (e) => { if (e.key === "Enter") analyzeUrl(); });
  $("#pg-run").addEventListener("click", runComparison);
  $("#fa-go").addEventListener("click", runFeatureAnalysis);
  $("#fa-url").addEventListener("keydown", (e) => { if (e.key === "Enter") runFeatureAnalysis(); });
  $("#ex-run").addEventListener("click", runExplain);
  $("#ex-url").addEventListener("keydown", (e) => { if (e.key === "Enter") runExplain(); });

  $("#pg-k").addEventListener("input", (e) => { $("#pg-k-out").textContent = e.target.value; });
  $("#pg-eps").addEventListener("input", (e) => { $("#pg-eps-out").textContent = Number(e.target.value).toFixed(3); });

  window.addEventListener("hashchange", () => showPage((location.hash || "#dashboard").replace("#", "")));

  try {
    const [models, dataset] = await Promise.all([api("/api/models"), api("/api/dataset-info")]);
    state.models = models;
    state.dataset = dataset;

    state.selectedModels = new Set(models.defaults.models.filter((m) => models.models.some((x) => x.name === m)));
    state.selectedMethods = new Set(models.defaults.methods);

    $("#pg-k").max = models.defaults.max_k;
    $("#pg-k").value = Math.min(models.defaults.k, models.defaults.max_k);
    $("#pg-k-out").textContent = $("#pg-k").value;
    $("#pg-eps").value = models.defaults.epsilon;
    $("#pg-eps-out").textContent = Number(models.defaults.epsilon).toFixed(3);

    const modelSelects = [$("#ex-model"), $("#fa-model")];
    modelSelects.forEach((select) => {
      select.innerHTML = models.models.map((m) => `<option value="${esc(m.name)}">${esc(m.label)}</option>`).join("");
    });

    renderDashboard();
    renderAnalyzeModels();
    renderPlaygroundPickers();
    renderDataset();

    $("#sidebar-status").innerHTML = `<strong style="color:var(--navy)">${models.models.length} models ready</strong><br>${dataset.n_features_served} live URL features · seed ${models.defaults.seed}`;
    $("#dash-health").textContent = "online";

    api("/api/feature-importance?limit=20").then((imp) => {
      state.importance = imp;
      renderMethodology();
    }).catch(() => { /* rankings may still be computing */ });
  } catch (error) {
    $("#sidebar-status").innerHTML = `<span style="color:var(--red)">Backend unavailable</span><br>${esc(error.message)}`;
    $("#dash-health").textContent = "offline";
    $("#dash-health").className = "chip red";
    $("#dash-kpis").innerHTML = `<div class="card">${banner("Could not reach the API. Start the backend with <span class='mono'>python prototype/backend/api.py</span>.", "red")}</div>`;
  }
}

document.addEventListener("DOMContentLoaded", boot);
