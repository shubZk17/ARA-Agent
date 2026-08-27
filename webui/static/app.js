// ARA-1 web UI. Vanilla JS on purpose — this page has one form and one
// results panel; a framework would be more code to load than it saves.

const $ = (id) => document.getElementById(id);

async function checkBackendStatus() {
  const el = $("backend-status");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.backend_up) {
      el.textContent = "backend online";
      el.className = "status status--up";
    } else {
      el.textContent = "backend unreachable";
      el.className = "status status--down";
    }
  } catch {
    el.textContent = "gateway unreachable";
    el.className = "status status--down";
  }
}

function setRunning(isRunning) {
  $("submit-btn").disabled = isRunning;
  $("run-status").classList.toggle("hidden", !isRunning);
  $("error-banner").classList.add("hidden");
}

function showError(message) {
  const banner = $("error-banner");
  banner.textContent = message;
  banner.classList.remove("hidden");
}

function renderResult(data) {
  $("result-empty").classList.add("hidden");
  $("result-content").classList.remove("hidden");

  $("result-ticker").textContent = data.ticker || "—";
  $("result-status").textContent = data.status || "unknown";

  $("stat-iterations").textContent = data.iterations_used ?? "–";
  $("stat-tools").textContent = data.tool_calls ?? "–";
  $("stat-time").textContent = data.execution_time_seconds ?? "–";
  $("stat-eval").textContent =
    data.evaluation_score != null ? `${Math.round(data.evaluation_score * 100)}%` : "–";

  $("result-answer").textContent = data.final_answer || "(no answer produced)";

  const errorsBox = $("result-errors");
  const errorsList = $("result-errors-list");
  errorsList.innerHTML = "";
  if (data.errors && data.errors.length > 0) {
    data.errors.forEach((e) => {
      const li = document.createElement("li");
      li.textContent = e;
      errorsList.appendChild(li);
    });
    errorsBox.classList.remove("hidden");
  } else {
    errorsBox.classList.add("hidden");
  }

  renderReport(data.report);
}

// The full Phase 3 SynthesisReport, when synthesis ran and succeeded — same
// fields the CLI's Rich table shows (outlook, confidence, risk,
// invalidation condition, review date).
function renderReport(report) {
  const box = $("result-report");
  if (!report) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");

  const rec = report.recommendation || {};
  const confidence = report.confidence || {};
  const risk = report.risk || {};

  $("report-outlook").textContent = (report.outlook || "—").toUpperCase().replace(/_/g, " ");
  $("report-confidence").textContent =
    confidence.overall != null
      ? `${Math.round(confidence.overall * 100)}% (${confidence.label || ""})`
      : "—";
  $("report-risk").textContent = risk.overall_risk_level || "—";
  $("report-invalidation").textContent = rec.invalidation_condition || "Not specified.";
  $("report-review-date").textContent = rec.review_by_date || "—";
}

async function loadReports() {
  const container = $("reports-list");
  try {
    const res = await fetch("/api/reports");
    const data = await res.json();
    const reports = data.reports || [];
    if (reports.length === 0) {
      container.innerHTML = '<div class="empty">No reports generated yet.</div>';
      return;
    }
    container.innerHTML = "";
    reports.slice(0, 12).forEach((r) => {
      const a = document.createElement("a");
      a.className = "report-row";
      a.href = `/api/reports/${encodeURIComponent(r.filename)}`;
      a.target = "_blank";
      a.rel = "noopener";
      a.innerHTML = `<span class="name">${r.filename}</span><span class="open">${r.format.toUpperCase()} ↗</span>`;
      container.appendChild(a);
    });
  } catch {
    container.innerHTML = '<div class="empty">Could not load reports — is the backend running?</div>';
  }
}

$("analyze-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = $("query").value.trim();
  if (!query) return;

  setRunning(true);
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        horizon: $("horizon").value,
        risk_profile: $("risk_profile").value,
        enable_evaluation: true,
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      showError(data.detail || data.error || `Analysis failed (HTTP ${res.status})`);
      return;
    }
    renderResult(data);
    loadReports();
  } catch (err) {
    showError(`Could not reach the backend: ${err.message}`);
  } finally {
    setRunning(false);
  }
});

checkBackendStatus();
loadReports();
setInterval(checkBackendStatus, 15000);
