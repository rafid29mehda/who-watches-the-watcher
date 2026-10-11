"use strict";

const VIEWS = ["V1", "V2", "V3", "V4", "V4raw"];
const VIEW_INFO = {
  V1: ["Action only", "The tool call and its arguments, nothing else."],
  V2: ["Action and request", "Adds what the user asked for."],
  V3: ["Full transcript", "Adds every earlier step, including raw tool outputs, so it also reads any injected text."],
  V4: ["Provenance", "The user's request, the names of earlier tools, and a label on each argument saying where its value came from. Raw tool outputs are withheld and long text is summarized."],
  V4raw: ["Provenance, long text shown", "Like V4, but long arguments such as email bodies are shown in full."],
};
const ATTACKS = ["important_instructions", "monitor_aware"];
const ATTACK_LABEL = { important_instructions: "Standard injection", monitor_aware: "Monitor-aware injection" };
const CATEGORY_LABEL = {
  v4_only: "the provenance view caught it and the full transcript missed it",
  v3_only: "the full transcript caught it and the provenance view missed it",
  both: "both views caught it",
  neither: "both views missed it",
};
const SVG = "http://www.w3.org/2000/svg";

const pct = (x, d = 1) => (x == null ? "n/a" : `${(100 * x).toFixed(d)}%`);
const num = (x, d = 3) => (x == null ? "n/a" : Number(x).toFixed(d));
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const viewColor = (v) => `var(--${v.toLowerCase()})`;

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
}
function svgEl(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
}
function row(summary, model, view, attack) {
  return summary.results.find((r) => r.model === model && r.view === view && r.attack === attack);
}

/* ---------- static copy filled from data ---------- */

function fillText(summary) {
  const m = summary.meta.primary_monitor;
  const second = [...new Set(summary.results.map((r) => r.model))].find((x) => x !== m && x !== "rule");
  const v3 = row(summary, m, "V3", ATTACKS[0]);
  const v4 = row(summary, m, "V4", ATTACKS[0]);
  const bua = (v) => summary.secondary.benign_under_attack_flag_rate.find((r) => r.view === v && r.attack === ATTACKS[0]);
  const big = second && row(summary, second, "V3", ATTACKS[0]);
  document.getElementById("finding").textContent =
    `With the full transcript, a small monitor caught ${pct(v3.catch_5)} of attacks at a budget of 5% false alarms. ` +
    `The provenance view caught ${pct(v4.catch_5)}, but wrongly flagged ${pct(bua("V4").rate)} of benign actions near an injection instead of ${pct(bua("V3").rate)}.` +
    (big ? ` A larger monitor reading the full transcript caught ${pct(big.catch_5)}.` : "");
  document.querySelectorAll('[data-bind="monitor"]').forEach((n) => (n.textContent = m));
  const v1 = row(summary, m, "V1", ATTACKS[0]);
  const v2 = row(summary, m, "V2", ATTACKS[0]);
  document.getElementById("results-intro").textContent =
    `Showing the monitor more context helps. From the action alone it separated harmful from legitimate calls with an AUROC of ${num(v1.auroc)}; ` +
    `adding the user's request raised that to ${num(v2.auroc)}, and the full transcript gave ${num(v3.auroc)}. ` +
    `The provenance view came within 0.03 of the transcript in AUROC but caught fewer attacks at a strict threshold. ` +
    `A fixed appeal addressed to the monitor hurt the transcript view a little more than the provenance view, but not by a statistically reliable margin.`;
  const dl = document.getElementById("views");
  for (const v of VIEWS) {
    const dt = el("dt");
    const sw = el("span", { class: "swatch", "aria-hidden": "true" });
    sw.style.background = viewColor(v);
    dt.append(sw, `${v} ${VIEW_INFO[v][0]}`);
    dl.append(dt, el("dd", {}, VIEW_INFO[v][1]));
  }
  const repoBase = location.hostname.endsWith(".github.io")
    ? `https://github.com/${location.hostname.split(".")[0]}/${location.pathname.split("/").filter(Boolean)[0] || ""}`
    : null;
  const repoLink = document.getElementById("repo-link");
  if (repoBase) repoLink.href = repoBase;
  else repoLink.remove();
}

function fillChecks(summary) {
  const v = summary.validity;
  const meta = summary.meta;
  const items = [
    `Every recorded trajectory was re-executed against the benchmark. All tool outputs and outcomes reproduced except ${meta.n_retried} episodes that AgentDojo had silently re-run, which we excluded and report.`,
    v.online && v.online.n
      ? `A live rerun of the banking suite with the provenance monitor gating each call reached ${pct(v.online.online_observed_asr)} attack success against an offline prediction of ${pct(v.online.offline_predicted_asr)}; the interval of the difference includes zero.`
      : null,
    v.test_retest_spearman != null
      ? `Re-scoring ${v.test_retest_n.toLocaleString()} monitor calls gave a Spearman correlation of ${v.test_retest_spearman.toFixed(2)}, so individual scores are noisy and we report intervals throughout.`
      : null,
    `Hypotheses and analyses were registered before the main monitor runs, and every result here, null or not, is reported as computed.`,
    `The whole study cost $${meta.total_spend_usd.toFixed(2)} in API fees. All traces, labels and scores are public, so every number can be regenerated without API access.`,
  ].filter(Boolean);
  const ul = document.getElementById("checks");
  for (const t of items) ul.append(el("li", {}, t));
}

function fillHypotheses(summary) {
  const q = {
    H1: "Does the full transcript separate attacks better than the action alone?",
    H2: "Does the monitor-aware appeal hurt the transcript view more than the provenance view?",
    H3a: "Is the provenance view no worse than the transcript view (margin 0.05 AUROC)?",
    H3b: "Is the provenance view better than the transcript view against the monitor-aware appeal?",
  };
  const tbody = document.querySelector("#hyp-table tbody");
  for (const h of summary.hypotheses) {
    const tr = el("tr");
    tr.append(el("td", {}, `${h.name}: ${q[h.name] || h.description}`));
    tr.append(el("td", { class: "num" }, num(h.estimate)));
    tr.append(el("td", { class: "num" }, `[${num(h.ci_low)}, ${num(h.ci_high)}]`));
    tr.append(el("td", { class: "num" }, h.p_holm < 0.001 ? "< 0.001" : num(h.p_holm)));
    tr.append(el("td", { class: h.supported ? "yes" : "no" }, h.supported ? "Supported" : "Not supported"));
    tbody.append(tr);
  }
}

/* ---------- charts (inline SVG, hover + focus tooltips) ---------- */

function chartWidth(container) {
  return Math.max(320, Math.round(container.clientWidth || 760));
}

function chartFrame(container, width, height, margin) {
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
  const tip = el("div", { class: "tooltip", role: "status" });
  container.append(svg, tip);
  const inner = { x0: margin.l, x1: width - margin.r, y0: margin.t, y1: height - margin.b };
  const show = (text, x, y) => {
    const box = svg.getBoundingClientRect();
    tip.textContent = text;
    tip.style.left = `${(x / width) * box.width}px`;
    tip.style.top = `${(y / height) * box.height}px`;
    tip.classList.add("show");
  };
  const hide = () => tip.classList.remove("show");
  return { svg, inner, show, hide };
}

function yAxis(svg, inner, ticks, scale, fmt) {
  const g = svgEl("g", { class: "axis" });
  for (const t of ticks) {
    const y = scale(t);
    g.append(svgEl("line", { class: "gridline", x1: inner.x0, x2: inner.x1, y1: y, y2: y }));
    g.append(svgEl("text", { x: inner.x0 - 8, y: y + 4, "text-anchor": "end" }, fmt(t)));
  }
  svg.append(g);
}

function attachHit(svg, frame, rect, text) {
  const hit = svgEl("rect", { ...rect, class: "hit", tabindex: "0", "aria-label": text });
  const cx = rect.x + rect.width / 2;
  hit.addEventListener("mouseenter", () => frame.show(text, cx, rect.y));
  hit.addEventListener("focus", () => frame.show(text, cx, rect.y));
  hit.addEventListener("mouseleave", frame.hide);
  hit.addEventListener("blur", frame.hide);
  svg.append(hit);
}

function legend(id, items) {
  const box = document.getElementById(id);
  for (const [label, color] of items) {
    const s = el("span");
    const sw = el("span", { class: "swatch", "aria-hidden": "true" });
    sw.style.background = color;
    s.append(sw, label);
    box.append(s);
  }
}

function catchChart(summary) {
  const m = summary.meta.primary_monitor;
  const box = document.getElementById("catch-chart");
  box.replaceChildren();
  const W = chartWidth(box), H = 300, M = { l: 52, r: 12, t: 14, b: 34 };
  const f = chartFrame(box, W, H, M);
  f.svg.setAttribute("aria-label", "Bar chart of attacks caught at 5% false positive rate, by view and attack");
  const y = (v) => f.inner.y1 - v * (f.inner.y1 - f.inner.y0);
  yAxis(f.svg, f.inner, [0, 0.25, 0.5, 0.75, 1], y, (t) => `${t * 100}%`);
  const band = (f.inner.x1 - f.inner.x0) / VIEWS.length;
  const bw = Math.min(46, band * 0.34);
  VIEWS.forEach((v, i) => {
    const cx = f.inner.x0 + band * (i + 0.5);
    f.svg.append(svgEl("text", { x: cx, y: H - 10, "text-anchor": "middle", class: "axis-label", fill: cssVar("--ink-2"), "font-size": 13 }, v));
    ATTACKS.forEach((a, j) => {
      const r = row(summary, m, v, a);
      if (!r || r.catch_5 == null) return;
      const x = cx + (j - 1) * (bw + 2) + 1;
      const top = y(r.catch_5);
      f.svg.append(svgEl("path", {
        d: `M${x},${y(0)} V${top + 4} q0,-4 4,-4 h${bw - 8} q4,0 4,4 V${y(0)} Z`,
        fill: a === ATTACKS[0] ? "var(--a-ii)" : "var(--a-ma)",
      }));
      const [lo, hi] = r.catch_5_ci;
      if (lo != null) {
        const wx = x + bw / 2;
        f.svg.append(svgEl("line", { class: "whisker", x1: wx, x2: wx, y1: y(lo), y2: y(hi) }));
        f.svg.append(svgEl("line", { class: "whisker", x1: wx - 4, x2: wx + 4, y1: y(hi), y2: y(hi) }));
        f.svg.append(svgEl("line", { class: "whisker", x1: wx - 4, x2: wx + 4, y1: y(lo), y2: y(lo) }));
      }
      attachHit(f.svg, f, { x, y: f.inner.y0, width: bw, height: f.inner.y1 - f.inner.y0 },
        `${v}, ${ATTACK_LABEL[a].toLowerCase()}: ${pct(r.catch_5)} caught (95% CI ${pct(lo, 0)} to ${pct(hi, 0)})`);
    });
  });
  if (!document.getElementById("catch-legend").childElementCount) legend("catch-legend", ATTACKS.map((a) => [ATTACK_LABEL[a], a === ATTACKS[0] ? "var(--a-ii)" : "var(--a-ma)"]));
}

function aurocChart(summary) {
  const models = [summary.meta.primary_monitor, ...new Set(summary.results.map((r) => r.model))]
    .filter((m, i, arr) => m !== "rule" && arr.indexOf(m) === i).slice(0, 2);
  const views = ["V1", "V3", "V4"];
  const box = document.getElementById("auroc-chart");
  box.replaceChildren();
  const W = chartWidth(box), H = 280, M = { l: 44, r: 8, t: 14, b: 50 };
  const f = chartFrame(box, W, H, M);
  f.svg.setAttribute("aria-label", "Dot plot of AUROC by view, attack and monitor model");
  const lo0 = 0.6, hi0 = 1.0;
  const y = (v) => f.inner.y1 - ((v - lo0) / (hi0 - lo0)) * (f.inner.y1 - f.inner.y0);
  yAxis(f.svg, f.inner, [0.6, 0.7, 0.8, 0.9, 1.0], y, (t) => t.toFixed(1));
  const groups = ATTACKS.flatMap((a) => views.map((v) => [a, v]));
  const band = (f.inner.x1 - f.inner.x0) / groups.length;
  groups.forEach(([a, v], i) => {
    const cx = f.inner.x0 + band * (i + 0.5);
    f.svg.append(svgEl("text", { x: cx, y: H - 30, "text-anchor": "middle", fill: cssVar("--ink-2"), "font-size": 13 }, v));
    if (i % views.length === 1) {
      f.svg.append(svgEl("text", { x: cx, y: H - 10, "text-anchor": "middle", fill: cssVar("--ink"), "font-size": 12 }, ATTACK_LABEL[a]));
    }
    models.forEach((m, k) => {
      const r = row(summary, m, v, a);
      if (!r) return;
      const x = cx + (k - 0.5) * 18;
      const [lo, hi] = r.auroc_ci;
      const color = k === 0 ? "var(--m1)" : "var(--m2)";
      f.svg.append(svgEl("line", { x1: x, x2: x, y1: y(Math.max(lo, lo0)), y2: y(Math.min(hi, hi0)), stroke: color, "stroke-width": 2 }));
      const mark = k === 0
        ? svgEl("circle", { cx: x, cy: y(r.auroc), r: 5.5, fill: color, stroke: cssVar("--surface"), "stroke-width": 2 })
        : svgEl("rect", { x: x - 5, y: y(r.auroc) - 5, width: 10, height: 10, rx: 2, fill: color, stroke: cssVar("--surface"), "stroke-width": 2 });
      f.svg.append(mark);
      attachHit(f.svg, f, { x: x - 9, y: f.inner.y0, width: 18, height: f.inner.y1 - f.inner.y0 },
        `${m}, ${v}, ${ATTACK_LABEL[a].toLowerCase()}: AUROC ${num(r.auroc)} (95% CI ${num(lo)} to ${num(hi)})`);
    });
  });
  if (!document.getElementById("auroc-legend").childElementCount) legend("auroc-legend", models.map((m, k) => [m, k === 0 ? "var(--m1)" : "var(--m2)"]));
}

function buaChart(summary) {
  const rows = summary.secondary.benign_under_attack_flag_rate;
  const box = document.getElementById("bua-chart");
  box.replaceChildren();
  const W = chartWidth(box), rowH = 26, M = { l: 62, r: 56, t: 6, b: 6 };
  const H = M.t + M.b + VIEWS.length * rowH * 2 + VIEWS.length * 8;
  const f = chartFrame(box, W, H, M);
  f.svg.setAttribute("aria-label", "Bar chart of benign actions flagged in episodes with an injection, by view");
  const max = Math.max(0.15, ...rows.map((r) => r.rate || 0));
  const x = (v) => f.inner.x0 + (v / max) * (f.inner.x1 - f.inner.x0);
  let yy = M.t;
  for (const v of VIEWS) {
    f.svg.append(svgEl("text", { x: M.l - 10, y: yy + rowH, "text-anchor": "end", fill: cssVar("--ink"), "font-size": 13 }, v));
    ATTACKS.forEach((a) => {
      const r = rows.find((q) => q.view === v && q.attack === a);
      if (!r) return;
      const w = Math.max(1, x(r.rate) - x(0));
      f.svg.append(svgEl("rect", { x: x(0), y: yy + 3, width: w, height: rowH - 6, rx: 3, fill: a === ATTACKS[0] ? "var(--a-ii)" : "var(--a-ma)" }));
      f.svg.append(svgEl("text", { x: x(0) + w + 6, y: yy + rowH / 2 + 4, fill: cssVar("--ink-2"), "font-size": 12 }, pct(r.rate)));
      attachHit(f.svg, f, { x: x(0), y: yy, width: f.inner.x1 - x(0), height: rowH },
        `${v}, ${ATTACK_LABEL[a].toLowerCase()}: ${pct(r.rate)} of ${r.n} benign actions flagged`);
      yy += rowH;
    });
    yy += 8;
  }
}

/* ---------- trace explorer ---------- */

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function highlight(text) {
  return escapeHtml(text)
    .replace(/&lt;INFORMATION&gt;[\s\S]*?(&lt;\/INFORMATION&gt;|$)/g, (m) => `<mark class="inj">${m}</mark>`)
    .replace(/\[(sources: [^\]\n]*|TRIVIAL)\]/g, (m) => `<span class="prov">${m}</span>`);
}

function explorer(data) {
  const state = { ex: null, step: null, view: "V3" };
  const select = document.getElementById("example-select");
  ATTACKS.forEach((a) => {
    const group = el("optgroup", { label: ATTACK_LABEL[a] });
    data.examples.forEach((ex, i) => {
      if (ex.attack !== a) return;
      const suite = ex.suite[0].toUpperCase() + ex.suite.slice(1);
      group.append(el("option", { value: String(i) }, `${suite}: ${CATEGORY_LABEL[ex.category]}`));
    });
    select.append(group);
  });
  const tabs = document.getElementById("tabs");
  VIEWS.forEach((v) => {
    const b = el("button", { class: "tab", role: "tab", id: `tab-${v}`, "aria-controls": "panel", type: "button" });
    const sw = el("span", { class: "swatch", "aria-hidden": "true" });
    sw.style.background = viewColor(v);
    b.append(sw, v);
    b.title = VIEW_INFO[v][0];
    b.style.setProperty("--tab-color", viewColor(v));
    b.addEventListener("click", () => { state.view = v; render(); });
    b.addEventListener("keydown", (e) => {
      const i = VIEWS.indexOf(state.view);
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        state.view = VIEWS[(i + (e.key === "ArrowRight" ? 1 : VIEWS.length - 1)) % VIEWS.length];
        render();
        document.getElementById(`tab-${state.view}`).focus();
        e.preventDefault();
      }
    });
    tabs.append(b);
  });

  function pickExample(i) {
    state.ex = data.examples[i];
    const harm = state.ex.steps.findIndex((s) => s.is_harm);
    state.step = harm >= 0 ? harm : 0;
    render();
  }

  function render() {
    const ex = state.ex;
    const step = ex.steps[state.step];
    document.getElementById("ex-task").textContent = ex.user_prompt;
    const harmStep = ex.steps.find((s) => s.is_harm);
    document.getElementById("ex-meta").textContent =
      `${ATTACK_LABEL[ex.attack]}. The attack succeeded without a monitor; at the 5% false-alarm threshold, ${CATEGORY_LABEL[ex.category]}.` +
      (harmStep ? ` The harm happens at call #${harmStep.call_index}, ${harmStep.function}.` : "");
    const steps = document.getElementById("steps");
    steps.replaceChildren();
    ex.steps.forEach((s, i) => {
      const b = el("button", { class: "step", type: "button", "aria-pressed": String(i === state.step) });
      b.append(`#${s.call_index} ${s.function}`);
      if (s.is_harm) b.append(el("span", { class: "harm" }, "harm step"));
      b.addEventListener("click", () => { state.step = i; render(); });
      steps.append(b);
    });
    VIEWS.forEach((v) => {
      const t = document.getElementById(`tab-${v}`);
      const selected = v === state.view;
      t.setAttribute("aria-selected", String(selected));
      t.tabIndex = selected ? 0 : -1;
    });
    document.getElementById("panel").setAttribute("aria-labelledby", `tab-${state.view}`);
    const cell = step.views[state.view];
    const tau = (data.thresholds[ex.attack] || {})[state.view];
    const score = document.getElementById("score");
    score.replaceChildren();
    const meter = el("div", { class: "meter", role: "img",
      "aria-label": `Suspicion ${cell.score ?? "not scored"} out of 100; flag threshold ${tau}` });
    const fill = el("div", { class: "fill" });
    fill.style.width = `${cell.score ?? 0}%`;
    fill.style.background = viewColor(state.view);
    const tick = el("div", { class: "tick" });
    tick.style.left = `calc(${tau}% - 1px)`;
    meter.append(fill, tick);
    const flagged = cell.score != null && cell.score >= tau;
    score.append(el("span", {}, `Suspicion ${cell.score ?? "n/a"} / 100`), meter,
      el("span", { class: `verdict ${flagged ? "flag" : "allow"}` }, flagged ? "Flagged: agent halted" : "Allowed"));
    score.append(el("span", { class: "note" }, `${state.view} ${VIEW_INFO[state.view][0].toLowerCase()}. The line marks this view's flag threshold (${tau}), set so that 5% of benign tasks would be halted.`));
    const pre = document.getElementById("evidence");
    pre.innerHTML = highlight(cell.evidence);  // all evidence text is HTML-escaped first
    const firstMark = pre.querySelector("mark.inj");
    pre.scrollTop = firstMark ? Math.max(0, firstMark.offsetTop - pre.offsetTop - 48) : 0;
  }

  select.addEventListener("change", () => pickExample(Number(select.value)));
  const firstV4 = data.examples.findIndex((e) => e.attack === "monitor_aware" && e.category === "v4_only");
  const start = firstV4 >= 0 ? firstV4 : 0;
  select.value = String(start);
  pickExample(start);
}

/* ---------- boot ---------- */

Promise.all([fetch("data/summary.json").then((r) => r.json()), fetch("data/examples.json").then((r) => r.json())])
  .then(([summary, examples]) => {
    fillText(summary);
    catchChart(summary);
    aurocChart(summary);
    buaChart(summary);
    fillHypotheses(summary);
    let resizeTimer;
    let lastWidth = window.innerWidth;
    window.addEventListener("resize", () => {
      if (window.innerWidth === lastWidth) return;
      lastWidth = window.innerWidth;
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => { catchChart(summary); aurocChart(summary); buaChart(summary); }, 150);
    });
    fillChecks(summary);
    explorer(examples);
  })
  .catch((err) => {
    document.getElementById("finding").textContent =
      "The results files could not be loaded. Serve this folder over HTTP (for example with `python3 -m http.server`) rather than opening the file directly.";
    console.error(err);
  });
