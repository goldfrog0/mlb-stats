const COLOR1 = "crimson";
const COLOR2 = "steelblue";

const statSelect = document.getElementById("stat");
const errorEl = document.getElementById("error");
const chartEl = document.getElementById("chart");
const tableContainer = document.getElementById("tableContainer");

async function loadStats() {
  const resp = await fetch("/api/stats");
  const stats = await resp.json();
  statSelect.innerHTML = "";
  for (const [key, info] of Object.entries(stats)) {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = `${info.label} (${info.group})`;
    if (key === "era") option.selected = true;
    statSelect.appendChild(option);
  }
}

function showError(message) {
  errorEl.textContent = message || "";
}

// Mask a cumulative series wherever the paired rolling value is null,
// so a noisy small-sample cumulative value early in the season doesn't
// dominate the y-axis in comparison mode (mirrors the CLI's --show-cumulative
// behavior).
function maskedCumulative(records) {
  return records.map((r) => (r.rolling === null ? null : r.cumulative));
}

function buildSingleTraces(records, label, showCumulative) {
  const traces = [
    {
      x: records.map((r) => r.date),
      y: records.map((r) => r.rolling),
      name: `Rolling ${label}`,
      mode: "lines",
      line: { color: COLOR1, width: 3 },
      text: records.map((r) => r.opponent),
      hovertemplate: "%{x}<br>%{text}<br>%{y:.3f}<extra>Rolling</extra>",
    },
  ];
  if (showCumulative) {
    traces.push({
      x: records.map((r) => r.date),
      y: records.map((r) => r.cumulative),
      name: `Season cumulative ${label}`,
      mode: "lines",
      line: { color: "gray", width: 1.5, dash: "dot" },
      opacity: 0.6,
      hovertemplate: "%{x}<br>%{y:.3f}<extra>Cumulative</extra>",
    });
  }
  return traces;
}

function buildComparisonTraces(records, name, color, showCumulative) {
  const traces = [
    {
      x: records.map((r) => r.date),
      y: records.map((r) => r.rolling),
      name,
      mode: "lines",
      line: { color, width: 3 },
      text: records.map((r) => r.opponent),
      hovertemplate: "%{x}<br>%{text}<br>%{y:.3f}<extra>" + name + "</extra>",
    },
  ];
  if (showCumulative) {
    traces.push({
      x: records.map((r) => r.date),
      y: maskedCumulative(records),
      name: `${name} season cumulative`,
      mode: "lines",
      line: { color, width: 1.5, dash: "dot" },
      opacity: 0.5,
      hovertemplate: "%{x}<br>%{y:.3f}<extra>" + name + " cumulative</extra>",
    });
  }
  return traces;
}

function renderTable(sections) {
  if (!document.getElementById("showTable").checked) {
    tableContainer.innerHTML = "";
    return;
  }

  let html = "";
  for (const { title, records } of sections) {
    html += `<h2>${title}</h2><table><thead><tr>
      <th>Date</th><th>Opponent</th><th>Game</th><th>Season</th><th>Rolling</th>
    </tr></thead><tbody>`;
    for (const r of records) {
      const fmt = (v) => (v === null || v === undefined ? "–" : v.toFixed(3));
      html += `<tr><td>${r.date}</td><td>${r.opponent}</td><td>${fmt(r.game)}</td><td>${fmt(r.cumulative)}</td><td>${fmt(r.rolling)}</td></tr>`;
    }
    html += "</tbody></table>";
  }
  tableContainer.innerHTML = html;
}

async function plotSingle(player, stat, season, window, showCumulative) {
  const url = `/api/player?name=${encodeURIComponent(player)}&stat=${stat}&season=${season}&window=${window}`;
  const resp = await fetch(url);
  const payload = await resp.json();
  if (!resp.ok) throw new Error(payload.detail || "Request failed");

  const traces = buildSingleTraces(payload.data, payload.label, showCumulative);
  Plotly.newPlot(chartEl, traces, {
    title: `${payload.name} — ${payload.label} Over Time (${season} Season)`,
    yaxis: { title: payload.label },
    xaxis: { title: "Date" },
  }, { responsive: true });

  renderTable([{ title: payload.name, records: payload.data }]);
}

async function plotCompare(player1, player2, stat, season, window, showCumulative) {
  const url = `/api/compare?player1=${encodeURIComponent(player1)}&player2=${encodeURIComponent(player2)}&stat=${stat}&season=${season}&window=${window}`;
  const resp = await fetch(url);
  const payload = await resp.json();
  if (!resp.ok) throw new Error(payload.detail || "Request failed");

  const traces = [
    ...buildComparisonTraces(payload.player1.data, payload.player1.name, COLOR1, showCumulative),
    ...buildComparisonTraces(payload.player2.data, payload.player2.name, COLOR2, showCumulative),
  ];
  Plotly.newPlot(chartEl, traces, {
    title: `${payload.player1.name} vs ${payload.player2.name} — ${payload.label} Rolling ${window}-Game Average (${season} Season)`,
    yaxis: { title: payload.label },
    xaxis: { title: "Date" },
  }, { responsive: true });

  renderTable([
    { title: payload.player1.name, records: payload.player1.data },
    { title: payload.player2.name, records: payload.player2.data },
  ]);
}

document.getElementById("controls").addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("");

  const player1 = document.getElementById("player1").value.trim();
  const player2 = document.getElementById("player2").value.trim();
  const stat = statSelect.value;
  const season = document.getElementById("season").value;
  const window = document.getElementById("window").value;
  const showCumulative = document.getElementById("showCumulative").checked;

  try {
    if (player2) {
      await plotCompare(player1, player2, stat, season, window, showCumulative);
    } else {
      await plotSingle(player1, stat, season, window, showCumulative);
    }
  } catch (err) {
    showError(err.message);
    Plotly.purge(chartEl);
    tableContainer.innerHTML = "";
  }
});

loadStats();
