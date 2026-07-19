const COLOR1 = "crimson";
const COLOR2 = "steelblue";

// Horizontal legend below the chart instead of Plotly's default vertical
// legend on the right, which eats horizontal space the chart needs on
// narrow/mobile viewports.
const HORIZONTAL_LEGEND = {
  orientation: "h",
  x: 0.5,
  xanchor: "center",
  y: -0.18,
  yanchor: "top",
};

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

// Builds a player's rolling (+ optional cumulative) traces, routed to a
// given subplot via axisIds ({xaxis, yaxis}, Plotly ids like "x"/"y2").
function buildComparisonTraces(records, name, color, showCumulative, axisIds) {
  const traces = [
    {
      x: records.map((r) => r.date),
      y: records.map((r) => r.rolling),
      name,
      mode: "lines",
      line: { color, width: 3 },
      text: records.map((r) => r.opponent),
      hovertemplate: "%{x}<br>%{text}<br>%{y:.3f}<extra>" + name + "</extra>",
      xaxis: axisIds.xaxis,
      yaxis: axisIds.yaxis,
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
      xaxis: axisIds.xaxis,
      yaxis: axisIds.yaxis,
    });
  }
  return traces;
}

// Player1's rolling value minus player2's, reindexed onto the union of
// both players' game dates with forward-fill (mirrors the CLI's
// _reindexed_rolling_diff in plots.py).
function computeDiff(records1, records2) {
  const map1 = new Map(records1.map((r) => [r.date, r.rolling]));
  const map2 = new Map(records2.map((r) => [r.date, r.rolling]));
  const allDates = Array.from(new Set([...map1.keys(), ...map2.keys()])).sort();

  let last1 = null;
  let last2 = null;
  const diff = [];
  for (const d of allDates) {
    if (map1.has(d) && map1.get(d) !== null) last1 = map1.get(d);
    if (map2.has(d) && map2.get(d) !== null) last2 = map2.get(d);
    diff.push(last1 !== null && last2 !== null ? last1 - last2 : null);
  }
  return { dates: allDates, diff };
}

// Diff panel as a black reference line plus two "fill to zero" traces
// (one clipped to the positive side in color1, one to the negative side
// in color2) -- mirrors the CLI's two fill_between calls plus its plot line.
function buildDiffTraces(dates, diff, color1, color2, axisIds) {
  const positive = diff.map((v) => (v === null ? null : Math.max(v, 0)));
  const negative = diff.map((v) => (v === null ? null : Math.min(v, 0)));

  return [
    {
      x: dates, y: positive, mode: "none", fill: "tozeroy",
      fillcolor: withAlpha(color1, 0.3), showlegend: false, hoverinfo: "skip",
      xaxis: axisIds.xaxis, yaxis: axisIds.yaxis,
    },
    {
      x: dates, y: negative, mode: "none", fill: "tozeroy",
      fillcolor: withAlpha(color2, 0.3), showlegend: false, hoverinfo: "skip",
      xaxis: axisIds.xaxis, yaxis: axisIds.yaxis,
    },
    {
      x: dates, y: diff, mode: "lines", line: { color: "black", width: 1 },
      showlegend: false, hovertemplate: "%{x}<br>%{y:.3f}<extra>Diff</extra>",
      xaxis: axisIds.xaxis, yaxis: axisIds.yaxis,
    },
  ];
}

function withAlpha(cssColor, alpha) {
  const named = { crimson: "220,20,60", steelblue: "70,130,180" };
  return `rgba(${named[cssColor] || "0,0,0"},${alpha})`;
}

// Computes subplot axis ids/domains for every (layout, showDiff) combo, plus
// title annotations for the per-panel names shown in stacked/side-by-side.
// Mirrors the exact gridspec layout in plots.py's plot_stat_comparison.
function computeComparisonLayout(layoutMode, showDiff, name1, name2) {
  const annotations = [];
  let axes = {};
  const plotlyLayout = {};

  const annotate = (text, x, y) => annotations.push({
    text, x, y, xref: "paper", yref: "paper", showarrow: false,
    xanchor: "center", yanchor: "bottom", font: { size: 13 },
  });

  if (layoutMode === "stacked") {
    const p1Y = showDiff ? [0.70, 1] : [0.55, 1];
    const p2Y = showDiff ? [0.38, 0.62] : [0, 0.45];
    const diffY = [0, 0.22];

    axes = { p1: { xaxis: "x", yaxis: "y" }, p2: { xaxis: "x2", yaxis: "y2" } };
    plotlyLayout.xaxis = { domain: [0, 1], anchor: "y" };
    plotlyLayout.yaxis = { domain: p1Y };
    plotlyLayout.xaxis2 = { domain: [0, 1], anchor: "y2", matches: "x" };
    plotlyLayout.yaxis2 = { domain: p2Y };
    annotate(name1, 0.5, p1Y[1]);
    annotate(name2, 0.5, p2Y[1]);

    if (showDiff) {
      axes.diff = { xaxis: "x3", yaxis: "y3" };
      plotlyLayout.xaxis3 = { domain: [0, 1], anchor: "y3", matches: "x", title: "Date" };
      plotlyLayout.yaxis3 = { domain: diffY };
    } else {
      plotlyLayout.xaxis2.title = "Date";
    }
  } else if (layoutMode === "side-by-side") {
    const mainY = showDiff ? [0.34, 1] : [0, 1];
    const gap = 0.06;
    const p1X = [0, (1 - gap) / 2];
    const p2X = [(1 - gap) / 2 + gap, 1];

    axes = { p1: { xaxis: "x", yaxis: "y" }, p2: { xaxis: "x2", yaxis: "y2" } };
    plotlyLayout.xaxis = { domain: p1X, anchor: "y" };
    plotlyLayout.yaxis = { domain: mainY };
    plotlyLayout.xaxis2 = { domain: p2X, anchor: "y2" };
    plotlyLayout.yaxis2 = { domain: mainY, matches: "y" };
    annotate(name1, (p1X[0] + p1X[1]) / 2, mainY[1]);
    annotate(name2, (p2X[0] + p2X[1]) / 2, mainY[1]);

    if (showDiff) {
      axes.diff = { xaxis: "x3", yaxis: "y3" };
      plotlyLayout.xaxis3 = { domain: [0, 1], anchor: "y3", title: "Date" };
      plotlyLayout.yaxis3 = { domain: [0, 0.22] };
    } else {
      plotlyLayout.xaxis2.title = "Date";
    }
  } else {
    // overlay
    const mainY = showDiff ? [0.34, 1] : [0, 1];
    axes = { p1: { xaxis: "x", yaxis: "y" }, p2: { xaxis: "x", yaxis: "y" } };
    plotlyLayout.xaxis = { domain: [0, 1], anchor: "y", title: "Date" };
    plotlyLayout.yaxis = { domain: mainY };

    if (showDiff) {
      axes.diff = { xaxis: "x2", yaxis: "y2" };
      plotlyLayout.xaxis2 = { domain: [0, 1], anchor: "y2", matches: "x", title: "Date" };
      plotlyLayout.yaxis2 = { domain: [0, 0.22] };
    }
  }

  plotlyLayout.annotations = annotations;
  plotlyLayout.legend = HORIZONTAL_LEGEND;
  plotlyLayout.margin = { b: 110 };

  // Mirrors the varying figsize() calls in the CLI's matplotlib version --
  // more panels need more vertical room. The +70 accounts for the
  // horizontal legend row below the chart (see HORIZONTAL_LEGEND) so it
  // doesn't eat into the plotted area itself.
  const heights = {
    "overlay": [500, 650],
    "stacked": [700, 850],
    "side-by-side": [500, 650],
  };
  plotlyLayout.height = heights[layoutMode][showDiff ? 1 : 0] + 70;

  return { axes, plotlyLayout };
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

// Standings are a single current snapshot across several teams, not a
// per-game time series for one or two "subjects" like everything else
// in this app -- so this gets its own bar-chart renderer and table
// (with its own columns) rather than reusing buildSingleTraces/renderTable.
async function plotStandings(division, season) {
  const url = `/api/standings?division=${encodeURIComponent(division)}&season=${season}`;
  const resp = await fetch(url);
  const payload = await resp.json();
  if (!resp.ok) throw new Error(payload.detail || "Request failed");

  const teams = payload.teams; // already rank-sorted, best team first
  // Plotly's horizontal bars plot the first entry at the bottom, so
  // reverse to put the division leader at the top (mirrors plots.py).
  const ordered = [...teams].reverse();

  const trace = {
    type: "bar",
    orientation: "h",
    x: ordered.map((t) => t.pct),
    y: ordered.map((t) => t.team),
    marker: { color: ordered.map((t) => (t.rank === 1 ? COLOR1 : COLOR2)) },
    text: ordered.map((t) => `${t.wins}-${t.losses}  (${t.pct.toFixed(3)})`),
    textposition: "outside",
    hovertemplate: "%{y}<br>Win%% %{x:.3f}<extra></extra>",
  };

  const maxPct = Math.max(...teams.map((t) => t.pct));

  Plotly.newPlot(chartEl, [trace], {
    title: `${payload.division} Standings (${season} Season)`,
    xaxis: { title: "Win%", range: [0, maxPct * 1.3] },
    margin: { l: 110, r: 90 },
    height: 140 + 60 * teams.length,
    showlegend: false,
  }, { responsive: true });

  renderStandingsTable(payload.division, teams);
}

function renderStandingsTable(divisionName, teams) {
  if (!document.getElementById("showStandingsTable").checked) {
    tableContainer.innerHTML = "";
    return;
  }

  let html = `<h2>${divisionName}</h2><table><thead><tr>
    <th>Rank</th><th>Team</th><th>W</th><th>L</th><th>PCT</th><th>GB</th><th>Streak</th>
  </tr></thead><tbody>`;
  for (const t of teams) {
    html += `<tr><td>${t.rank}</td><td>${t.team}</td><td>${t.wins}</td><td>${t.losses}</td>`
          + `<td>${t.pct.toFixed(3)}</td><td>${t.games_back}</td><td>${t.streak}</td></tr>`;
  }
  html += "</tbody></table>";
  tableContainer.innerHTML = html;
}

// Career WAR: one stacked bar per season (batting + pitching -- both
// matter for two-way players), grouped side by side when comparing two
// players. Mirrors the CLI's plot_career_war. The grouping/stacking is
// done manually (numeric x offsets, explicit widths, and a computed
// "base" for the pitching segment) rather than with Plotly barmodes:
// offsetgroup-based grouped stacks don't take effect in this Plotly
// build, and manual placement handles the mixed-sign seasons the same
// way the CLI does (each component spans its own contribution from 0
// when signs differ).
async function fetchCareerWar(player) {
  const resp = await fetch(`/api/career-war?name=${encodeURIComponent(player)}`);
  const payload = await resp.json();
  if (!resp.ok) throw new Error(payload.detail || "Request failed");
  return payload;
}

function buildWarTraces(payload, colors, xOffset, width, withName) {
  const seasons = payload.seasons;
  const prefix = withName ? `${payload.name} ` : "";
  const x = seasons.map((s) => s.season + xOffset);
  const pitchingBase = seasons.map((s) => (s.batting * s.pitching >= 0 ? s.batting : 0));
  return [
    {
      x, y: seasons.map((s) => s.batting), width,
      name: `${prefix}batting`, type: "bar",
      marker: { color: colors.batting },
      // text carries the un-offset season for hover; textposition "none"
      // keeps Plotly from also printing it on the bars themselves.
      text: seasons.map((s) => s.season), textposition: "none",
      hovertemplate: "%{text}<br>%{y:.1f} batting WAR<extra>" + payload.name + "</extra>",
    },
    {
      x, y: seasons.map((s) => s.pitching), width, base: pitchingBase,
      name: `${prefix}pitching`, type: "bar",
      marker: { color: colors.pitching },
      text: seasons.map((s) => s.season), textposition: "none",
      hovertemplate: "%{text}<br>%{y:.1f} pitching WAR<extra>" + payload.name + "</extra>",
    },
  ];
}

async function plotWar(player, player2) {
  const payload1 = await fetchCareerWar(player);
  const payload2 = player2 ? await fetchCareerWar(player2) : null;

  let traces;
  if (payload2 === null) {
    traces = buildWarTraces(payload1, { batting: "steelblue", pitching: "darkorange" }, 0, 0.7, false);
  } else {
    traces = [
      ...buildWarTraces(payload1, { batting: "crimson", pitching: "rgba(220,20,60,0.45)" }, -0.2, 0.38, true),
      ...buildWarTraces(payload2, { batting: "steelblue", pitching: "rgba(70,130,180,0.45)" }, 0.2, 0.38, true),
    ];
  }

  const title = payload2 === null
    ? `${payload1.name} — WAR by Season`
    : `${payload1.name} vs ${payload2.name} — WAR by Season`;

  Plotly.newPlot(chartEl, traces, {
    title,
    barmode: "overlay",
    xaxis: { title: "Season", dtick: 1 },
    yaxis: { title: "WAR" },
    legend: HORIZONTAL_LEGEND,
    margin: { b: 110 },
  }, { responsive: true });

  renderWarTable([payload1, payload2].filter(Boolean));
}

function renderWarTable(payloads) {
  if (!document.getElementById("showWarTable").checked) {
    tableContainer.innerHTML = "";
    return;
  }

  let html = "";
  for (const { name, seasons } of payloads) {
    html += `<h2>${name}</h2><table><thead><tr>
      <th>Season</th><th>Batting</th><th>Pitching</th><th>Total</th>
    </tr></thead><tbody>`;
    const career = { batting: 0, pitching: 0, total: 0 };
    for (const s of seasons) {
      html += `<tr><td>${s.season}</td><td>${s.batting.toFixed(1)}</td>`
            + `<td>${s.pitching.toFixed(1)}</td><td>${s.total.toFixed(1)}</td></tr>`;
      career.batting += s.batting;
      career.pitching += s.pitching;
      career.total += s.total;
    }
    html += `<tr><td>Career</td><td>${career.batting.toFixed(1)}</td>`
          + `<td>${career.pitching.toFixed(1)}</td><td>${career.total.toFixed(1)}</td></tr>`;
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
    legend: HORIZONTAL_LEGEND,
    margin: { b: 110 },
  }, { responsive: true });

  renderTable([{ title: payload.name, records: payload.data }]);
}

async function plotCompare(player1, player2, stat, season, window, showCumulative, layoutMode, showDiff) {
  const url = `/api/compare?player1=${encodeURIComponent(player1)}&player2=${encodeURIComponent(player2)}&stat=${stat}&season=${season}&window=${window}`;
  const resp = await fetch(url);
  const payload = await resp.json();
  if (!resp.ok) throw new Error(payload.detail || "Request failed");

  const name1 = payload.player1.name;
  const name2 = payload.player2.name;
  const { axes, plotlyLayout } = computeComparisonLayout(layoutMode, showDiff, name1, name2);

  const traces = [
    ...buildComparisonTraces(payload.player1.data, name1, COLOR1, showCumulative, axes.p1),
    ...buildComparisonTraces(payload.player2.data, name2, COLOR2, showCumulative, axes.p2),
  ];

  const label = payload.label;
  plotlyLayout.yaxis.title = label;
  if (plotlyLayout.yaxis2 && layoutMode !== "overlay") plotlyLayout.yaxis2.title = label;

  if (showDiff) {
    const { dates, diff } = computeDiff(payload.player1.data, payload.player2.data);
    traces.push(...buildDiffTraces(dates, diff, COLOR1, COLOR2, axes.diff));
    plotlyLayout.yaxis3 ? (plotlyLayout.yaxis3.title = `${label} diff<br>(${name1} − ${name2})`)
                        : (plotlyLayout.yaxis2.title = `${label} diff<br>(${name1} − ${name2})`);
    plotlyLayout.shapes = [{
      type: "line", xref: "paper", x0: 0, x1: 1,
      yref: axes.diff.yaxis, y0: 0, y1: 0,
      line: { color: "gray", width: 1 },
    }];
  }

  plotlyLayout.title = layoutMode === "overlay"
    ? `${name1} vs ${name2} — ${label} Rolling ${window}-Game Average (${season} Season)`
    : `${label} Rolling ${window}-Game Average (${season} Season)`;

  Plotly.newPlot(chartEl, traces, plotlyLayout, { responsive: true });

  renderTable([
    { title: name1, records: payload.player1.data },
    { title: name2, records: payload.player2.data },
  ]);
}

// Mode toggle: swaps which field group is visible and which plotting
// path the submit handler takes.
const modeButtons = document.querySelectorAll(".mode-btn");
const statFields = document.getElementById("statFields");
const standingsFields = document.getElementById("standingsFields");
const warFields = document.getElementById("warFields");
const player1Input = document.getElementById("player1");
const warPlayerInput = document.getElementById("warPlayer");

function setMode(mode) {
  modeButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.mode === mode));
  statFields.hidden = mode !== "stats";
  standingsFields.hidden = mode !== "standings";
  warFields.hidden = mode !== "war";
  // A field's "required" doesn't reliably get exempted from constraint
  // validation just because an ancestor is hidden (browser-dependent),
  // so toggle it directly rather than relying on that.
  player1Input.required = mode === "stats";
  warPlayerInput.required = mode === "war";
}

modeButtons.forEach((btn) => btn.addEventListener("click", () => setMode(btn.dataset.mode)));

document.getElementById("controls").addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("");

  const mode = document.querySelector(".mode-btn.active").dataset.mode;

  try {
    if (mode === "standings") {
      const division = document.getElementById("division").value;
      const season = document.getElementById("standingsSeason").value;
      await plotStandings(division, season);
    } else if (mode === "war") {
      const player = document.getElementById("warPlayer").value.trim();
      const player2 = document.getElementById("warPlayer2").value.trim();
      await plotWar(player, player2);
    } else {
      const player1 = document.getElementById("player1").value.trim();
      const player2 = document.getElementById("player2").value.trim();
      const stat = statSelect.value;
      const season = document.getElementById("season").value;
      const window = document.getElementById("window").value;
      const showCumulative = document.getElementById("showCumulative").checked;
      const layoutMode = document.getElementById("layout").value;
      const showDiff = document.getElementById("showDiff").checked;

      if (player2) {
        await plotCompare(player1, player2, stat, season, window, showCumulative, layoutMode, showDiff);
      } else {
        await plotSingle(player1, stat, season, window, showCumulative);
      }
    }
  } catch (err) {
    showError(err.message);
    Plotly.purge(chartEl);
    // purge empties the div but leaves the js-plotly-plot class behind,
    // which would keep the chart card visible as an empty white box
    chartEl.className = "";
    tableContainer.innerHTML = "";
  }
});

function debounce(fn, delayMs) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delayMs);
  };
}

// Populates a <datalist> from /api/search-players as the user types, so
// the input field's native autocomplete dropdown suggests real player
// names instead of the user having to guess exact spelling and find out
// via a 404 on submit. Matches the server-side minimum length in web.py.
const MIN_SEARCH_LENGTH = 2;

async function updatePlayerSuggestions(inputId, datalistId) {
  const query = document.getElementById(inputId).value.trim();
  const datalist = document.getElementById(datalistId);

  if (query.length < MIN_SEARCH_LENGTH) {
    datalist.replaceChildren();
    return;
  }

  const resp = await fetch(`/api/search-players?q=${encodeURIComponent(query)}`);
  if (!resp.ok) return;
  const players = await resp.json();

  datalist.replaceChildren(...players.map((p) => {
    const option = document.createElement("option");
    option.value = p.name;
    return option;
  }));
}

function wireAutocomplete(inputId, datalistId) {
  const debounced = debounce(() => updatePlayerSuggestions(inputId, datalistId), 250);
  document.getElementById(inputId).addEventListener("input", debounced);
}

wireAutocomplete("player1", "player1-suggestions");
wireAutocomplete("player2", "player2-suggestions");
wireAutocomplete("warPlayer", "warPlayer-suggestions");
wireAutocomplete("warPlayer2", "warPlayer2-suggestions");

// Default the season inputs to the current year (the backend also falls
// back to the current season server-side when the param is omitted).
document.getElementById("season").value = new Date().getFullYear();
document.getElementById("standingsSeason").value = new Date().getFullYear();

loadStats();
