import { useState, useRef, useCallback, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, AreaChart, Area, BarChart, Bar, Legend,
} from "recharts";

// ─── DESIGN TOKENS ────────────────────────────────────────────────────────────
const C = {
  bg: "#07090f", panel: "#0d1117", border: "#161d2a", border2: "#1e2d40",
  text: "#c9d5e0", muted: "#3a5060", dim: "#1a2535",
  green: "#34d399", blue: "#38bdf8", amber: "#fbbf24",
  red: "#f87171", purple: "#a78bfa", teal: "#2dd4bf",
};

// ─── CONFIG (mirrors config.py) ───────────────────────────────────────────────
const CFG = {
  LEAD_TIME: 3,
  DEFAULT_SL: 0.95,
  WRITE_OFF_RATE: 0.00143,
  WRITE_OFF_FREQ: 7,
  HISTO_DAYS: 365,
  SIM_DAYS: 730,
  DECISION_INTERVAL: 5,
  MEMORY_SIZE: 200,
  SELLING_PRICE: 25.0,
  UNIT_COST: 10.0,
  FIXED_ORDER_COST: 150.0,
  HOLDING_RATE: 0.02,
};

// ─── MATH HELPERS ─────────────────────────────────────────────────────────────
function normalRandom() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function gammaRandom(shape, scale) {
  if (shape < 1) return gammaRandom(1 + shape, scale) * Math.pow(Math.random(), 1 / shape);
  const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x, v;
    do { x = normalRandom(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v * scale;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scale;
  }
}
function poissonRandom(lambda) {
  let L = Math.exp(-lambda), k = 0, p = 1;
  do { k++; p *= Math.random(); } while (p > L);
  return k - 1;
}
function expRandom(rate) { return -Math.log(Math.random()) / rate; }
function arr_mean(a) { return a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0; }
function arr_std(a) {
  if (a.length < 2) return 0;
  const m = arr_mean(a);
  return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1));
}
function quantile(sorted, q) { return sorted[Math.floor(sorted.length * q)]; }

// ─── DEMAND ENVIRONMENTS ──────────────────────────────────────────────────────
const ENVS = {
  gamma_poisson: {
    label: "Gamma–Poisson", tag: "MODERATE", color: C.green,
    desc: "90% Gamma(7,16) + 10% Poisson(80). Stable with rare spikes.",
    sample: () => Math.random() < 0.9 ? Math.max(0, Math.round(gammaRandom(7, 16))) : poissonRandom(80),
    demMean: 112, demStd: 38,
  },
  bimodal_hv: {
    label: "Bimodal High-Var", tag: "HARD", color: C.amber,
    desc: "50% low-mean Gamma + 50% high-mean Gamma. Extremely unpredictable.",
    sample: () => Math.random() < 0.5
      ? Math.max(0, Math.round(gammaRandom(7, 3)))
      : Math.max(0, Math.round(gammaRandom(7, 29))),
    demMean: 112, demStd: 95,
  },
  spiking: {
    label: "Sporadic Spiking", tag: "EXTREME", color: C.red,
    desc: "95% zero demand, 5% large Exponential bursts.",
    sample: () => Math.random() < 0.95 ? 0 : Math.max(0, Math.round(expRandom(0.05))),
    demMean: 20, demStd: 55,
  },
  gamma_stable: {
    label: "Stable Gamma", tag: "EASY", color: C.blue,
    desc: "Single Gamma(7,16), low variance. Baseline environment.",
    sample: () => Math.max(0, Math.round(gammaRandom(7, 16))),
    demMean: 112, demStd: 35,
  },
};

// ─── BASELINE AGENTS ──────────────────────────────────────────────────────────
const BASELINES = {
  base: { label: "Base", color: C.muted, compute: (h) => arr_mean(h) * CFG.LEAD_TIME },
  safety_stock: {
    label: "Safety Stock", color: C.blue,
    compute: (h) => arr_mean(h) * CFG.LEAD_TIME + 1.645 * arr_std(h) * Math.sqrt(CFG.LEAD_TIME),
  },
  forecast: {
    label: "Oracle Forecast", color: C.green,
    compute: (h, dm, ds) => dm * CFG.LEAD_TIME + 1.645 * ds * Math.sqrt(CFG.LEAD_TIME),
  },
  monte_carlo: {
    label: "Monte Carlo", color: C.purple,
    compute: (h) => {
      const s = [];
      for (let i = 0; i < 500; i++) {
        let t = 0;
        for (let j = 0; j < CFG.LEAD_TIME; j++)
          t += h[Math.floor(Math.random() * h.length)] * (0.8 + Math.random() * 0.4);
        s.push(t);
      }
      s.sort((a, b) => a - b);
      return quantile(s, 0.95);
    },
  },
};

// ─── SIMULATION ENGINE ────────────────────────────────────────────────────────
function buildDemandSeries(envKey, n) {
  return Array.from({ length: n }, () => ENVS[envKey].sample());
}

function runOneSimulation(computeROP, demandSeries, envKey) {
  const env = ENVS[envKey];
  const n = demandSeries.length;
  let inventory = 0;
  const orders = [];
  let totDemand = 0, totFulfilled = 0, totWriteOff = 0, stockOuts = 0, lostSales = 0, totProfit = 0, servicedays = 0;
  const timeline = [];

  for (let day = 0; day < n; day++) {
    const demand = demandSeries[day];
    const hist = demandSeries.slice(Math.max(0, day - CFG.HISTO_DAYS), day);
    const arrivals = orders.filter(o => o.arr === day);
    const delivered = arrivals.reduce((s, o) => s + o.qty, 0);
    inventory += delivered;
    orders.splice(0, orders.length, ...orders.filter(o => o.arr > day));
    const preInv = inventory;
    const fulfilled = Math.min(demand, inventory);
    inventory = Math.max(0, inventory - demand);
    const lost = Math.max(0, demand - fulfilled);
    if (lost > 0) stockOuts++; else servicedays++;
    lostSales += lost;
    let rop = 0, ordered = 0;
    if (hist.length >= 5 && day < n - CFG.LEAD_TIME) {
      rop = Math.max(0, computeROP(hist, env.demMean, env.demStd));
      if (inventory <= rop) {
        const qty = Math.ceil(rop - inventory + arr_mean(hist) * CFG.LEAD_TIME);
        orders.push({ arr: day + CFG.LEAD_TIME, qty });
        ordered = qty;
      }
    }
    let wo = 0;
    if (day % CFG.WRITE_OFF_FREQ === 0) {
      wo = Math.floor(inventory * CFG.WRITE_OFF_RATE);
      inventory -= wo;
      totWriteOff += wo;
    }
    totDemand += demand;
    totFulfilled += fulfilled;
    const revenue = fulfilled * CFG.SELLING_PRICE;
    const holdingCost = inventory * CFG.UNIT_COST * CFG.HOLDING_RATE;
    const stockoutPenalty = lost * (CFG.SELLING_PRICE - CFG.UNIT_COST);
    const orderCost = (ordered > 0 ? CFG.FIXED_ORDER_COST : 0) + ordered * CFG.UNIT_COST;
    const writeoffCost = wo * CFG.UNIT_COST;
    totProfit += revenue - holdingCost - stockoutPenalty - orderCost - writeoffCost;
    const fillRateCum = totDemand > 0 ? totFulfilled / totDemand : 0;
    timeline.push({ day, demand, inventory: preInv, inventoryAfter: inventory, fulfilled, lost, rop: Math.round(rop), ordered, wo, delivered, fillRateCum });
  }
  const daysElapsed = n;
  return {
    timeline,
    metrics: { fillRate: totDemand > 0 ? totFulfilled / totDemand : 0, stockOuts, lostSales, totWriteOff, totDemand, totFulfilled, profit: totProfit, serviceLevel: daysElapsed > 0 ? servicedays / daysElapsed : 0 },
  };
}

// ─── HF INFERENCE API ─────────────────────────────────────────────────────────
async function callQwen(messages, modelId, hfToken) {
  const url = `https://api-inference.huggingface.co/models/${modelId}/v1/chat/completions`;
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(hfToken ? { Authorization: `Bearer ${hfToken}` } : {}),
    },
    body: JSON.stringify({ model: modelId, messages, max_tokens: 600, temperature: 0.7 }),
  });
  if (!resp.ok) throw new Error(`API error ${resp.status}: ${await resp.text()}`);
  const data = await resp.json();
  return data.choices?.[0]?.message?.content || "";
}

// ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are an expert inventory optimization agent in a stochastic supply-chain simulation.

YOUR OBJECTIVE:
Maximize profit while maintaining fill rate >= 95% over a 365-day decision horizon (days 365–730 of the simulation, after a 365-day warm-up).

ENVIRONMENT RULES:
- Orders arrive exactly 3 days after placement (LEAD_TIME = 3)
- An order fires whenever inventory <= your reorder_point
- Order quantity = reorder_point - current_inventory + mean_demand * LEAD_TIME
- Every 7 days, ~0.14% of inventory is written off (spoilage/expiry)
- Reward = daily P&L: revenue - holding_cost - stockout_penalty - order_cost - writeoff_cost

REASONING REQUIREMENTS — all 4:
1. SUBGOAL DECOMPOSITION: Break into subgoals (e.g., "rebuild buffer", "reduce overstock")
2. STATE ANALYSIS: Interpret inventory level, demand trend, pending orders, fill rate trajectory
3. DECISION: Output a specific numeric reorder_point with clear justification
4. RECOVERY PLAN: If fill rate < 95% or recent stockouts, state recovery strategy

Think 3+ days ahead — your ROP today only shows effect after lead time.

OUTPUT FORMAT — valid JSON only, no markdown fences:
{"subgoals":["...","..."],"state_analysis":"...","recovery_plan":"...","reorder_point":<number>,"confidence":"high|medium|low","reasoning_depth":"..."}`;

// ─── BUILD SNAPSHOT FOR LLM ───────────────────────────────────────────────────
function buildSnapshot(demandSeries, timeline, day, memory) {
  const hist = demandSeries.slice(Math.max(0, day - CFG.HISTO_DAYS), day);
  const last5 = timeline.slice(Math.max(0, day - 5), day);
  const curInv = timeline[day - 1]?.inventoryAfter ?? 0;
  return {
    day,
    days_remaining: CFG.SIM_DAYS - day,
    current_inventory: Math.round(curInv),
    demand_mean_30d: Math.round(arr_mean(demandSeries.slice(Math.max(0, day - 30), day)) * 10) / 10,
    demand_std_30d: Math.round(arr_std(demandSeries.slice(Math.max(0, day - 30), day)) * 10) / 10,
    fill_rate_so_far: timeline[day - 1]?.fillRateCum
      ? `${(timeline[day - 1].fillRateCum * 100).toFixed(1)}%` : "N/A",
    recent_stockouts: last5.filter(d => d.lost > 0).length,
    recent_lost_sales: last5.reduce((s, d) => s + d.lost, 0),
    last_5_days: last5.map(d => ({
      day: d.day, demand: d.demand, inv: d.inventoryAfter, lost: d.lost, rop: d.rop,
    })),
    memory_bank: memory.slice(-CFG.MEMORY_SIZE),
  };
}

// ─── SHARED SIMULATION RUNNER ─────────────────────────────────────────────────
async function runAgentLoop({ envKey, modelId, hfToken, onDay, onDecision, onStatus, abortRef }) {
  const demandSeries = buildDemandSeries(envKey, CFG.SIM_DAYS);
  const env = ENVS[envKey];
  let inventory = 0;
  const orders = [];
  let totDemand = 0, totFulfilled = 0, totWriteOff = 0, stockOuts = 0, lostSales = 0, totProfit = 0, servicedays = 0;
  const timeline = [];
  let currentROP = env.demMean * CFG.LEAD_TIME;
  let memory = [];
  let convo = [];

  for (let day = 0; day < CFG.SIM_DAYS; day++) {
    if (abortRef.current) break;
    const demand = demandSeries[day];
    const hist = demandSeries.slice(Math.max(0, day - CFG.HISTO_DAYS), day);
    const arrivals = orders.filter(o => o.arr === day);
    const delivered = arrivals.reduce((s, o) => s + o.qty, 0);
    inventory += delivered;
    orders.splice(0, orders.length, ...orders.filter(o => o.arr > day));
    const preInv = inventory;
    const fulfilled = Math.min(demand, inventory);
    inventory = Math.max(0, inventory - demand);
    const lost = Math.max(0, demand - fulfilled);
    if (lost > 0) stockOuts++; else servicedays++;
    lostSales += lost;
    let ordered = 0;
    if (hist.length >= 5 && day < CFG.SIM_DAYS - CFG.LEAD_TIME && inventory <= currentROP) {
      const qty = Math.ceil(currentROP - inventory + arr_mean(hist) * CFG.LEAD_TIME);
      orders.push({ arr: day + CFG.LEAD_TIME, qty });
      ordered = qty;
    }
    let wo = 0;
    if (day % CFG.WRITE_OFF_FREQ === 0) {
      wo = Math.floor(inventory * CFG.WRITE_OFF_RATE);
      inventory -= wo;
      totWriteOff += wo;
    }
    totDemand += demand;
    totFulfilled += fulfilled;
    const revenue = fulfilled * CFG.SELLING_PRICE;
    const holdingCost = inventory * CFG.UNIT_COST * CFG.HOLDING_RATE;
    const stockoutPenalty = lost * (CFG.SELLING_PRICE - CFG.UNIT_COST);
    const orderCost = (ordered > 0 ? CFG.FIXED_ORDER_COST : 0) + ordered * CFG.UNIT_COST;
    const writeoffCost = wo * CFG.UNIT_COST;
    totProfit += revenue - holdingCost - stockoutPenalty - orderCost - writeoffCost;
    const fillRateCum = totDemand > 0 ? totFulfilled / totDemand : 0;
    const entry = { day, demand, inventory: preInv, inventoryAfter: inventory, fulfilled, lost, rop: Math.round(currentROP), ordered, wo, delivered, fillRateCum };
    timeline.push(entry);
    onDay(day, [...timeline]);

    if (day >= CFG.HISTO_DAYS && day % CFG.DECISION_INTERVAL === 0 && day < CFG.SIM_DAYS - CFG.LEAD_TIME) {
      onStatus(`Day ${day}/${CFG.SIM_DAYS}: agent reasoning...`);
      const snapshot = buildSnapshot(demandSeries, timeline, day, memory);
      const userMsg = {
        role: "user",
        content: `SNAPSHOT Day ${day}/${CFG.SIM_DAYS}\n${JSON.stringify(snapshot)}\n\nSet reorder_point for next ${CFG.DECISION_INTERVAL} days.`,
      };
      const msgs = [...convo.slice(-6), userMsg];
      try {
        const raw = await callQwen(msgs, modelId, hfToken);
        let decision;
        try {
          decision = JSON.parse(raw.replace(/```json|```/g, "").trim());
        } catch {
          const m = raw.match(/"reorder_point"\s*:\s*(\d+\.?\d*)/);
          decision = { subgoals: ["parse error"], state_analysis: raw.slice(0, 200), recovery_plan: "N/A", reorder_point: m ? parseFloat(m[1]) : currentROP, confidence: "low", reasoning_depth: "parse failed" };
        }
        currentROP = Math.max(0, decision.reorder_point || currentROP);
        convo = [...convo, userMsg, { role: "assistant", content: raw }];
        memory = [...memory, {
          day,
          rop: Math.round(currentROP),
          confidence: decision.confidence,
          fill_rate: `${(fillRateCum * 100).toFixed(1)}%`,
          inventory: Math.round(preInv),
          demand_mean: Math.round(arr_mean(demandSeries.slice(Math.max(0, day - 30), day))),
          stockouts_cumulative: stockOuts,
          lost_sales_cumulative: Math.round(lostSales),
          key_insight: decision.state_analysis?.slice(0, 100),
        }].slice(-CFG.MEMORY_SIZE);
        onDecision({ day, snapshot, decision, rop: currentROP, fillRateCum, memory: [...memory] });
      } catch (e) {
        onStatus(`Day ${day}: API error — ${e.message}`);
        onDecision({ day, snapshot, decision: { subgoals: [], state_analysis: `API error: ${e.message}`, recovery_plan: "N/A", reorder_point: currentROP, confidence: "low", reasoning_depth: "error" }, rop: currentROP, fillRateCum, memory: [...memory] });
      }
      await new Promise(r => setTimeout(r, 150));
    }
  }
  return {
    timeline,
    metrics: { fillRate: totDemand > 0 ? totFulfilled / totDemand : 0, stockOuts, lostSales, totWriteOff, totDemand, totFulfilled, profit: totProfit, serviceLevel: CFG.SIM_DAYS > 0 ? servicedays / CFG.SIM_DAYS : 0 },
    memory,
  };
}

// ─── SHARED UI COMPONENTS ─────────────────────────────────────────────────────
function Panel({ title, children, style = {} }) {
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 10, padding: "16px 18px", ...style }}>
      {title && <div style={{ fontSize: 9, letterSpacing: 4, color: C.muted, marginBottom: 12, textTransform: "uppercase" }}>{title}</div>}
      {children}
    </div>
  );
}

function FillBadge({ rate }) {
  const color = rate >= 0.95 ? C.green : rate >= 0.85 ? C.amber : C.red;
  return <span style={{ color, fontWeight: 700 }}>{rate ? `${(rate * 100).toFixed(1)}%` : "—"}</span>;
}

function MetricBox({ label, value, highlight, color }) {
  return (
    <div style={{ background: highlight ? "#0d1f18" : C.panel, border: `1px solid ${(color || C.green) + (highlight ? "30" : "15")}`, borderRadius: 8, padding: "10px 16px", textAlign: "center" }}>
      <div style={{ fontSize: 9, letterSpacing: 3, color: C.muted, marginBottom: 3 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 600 }}>{value}</div>
    </div>
  );
}

function SimTabs({ tabs, active, onSelect }) {
  return (
    <div style={{ display: "flex", gap: 6, marginBottom: 14, flexWrap: "wrap" }}>
      {tabs.map(({ id, label }) => {
        const isActive = active === id;
        return (
          <button key={id} onClick={() => onSelect(id)} style={{
            background: isActive ? C.border2 : "transparent",
            border: `1px solid ${isActive ? C.border2 : "transparent"}`,
            borderRadius: 6, padding: "7px 14px",
            color: isActive ? C.text : C.muted, fontFamily: "inherit",
            fontSize: 11, cursor: "pointer", letterSpacing: 1,
          }}>{label}</button>
        );
      })}
    </div>
  );
}

function LiveSimCharts({ timeline }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <Panel title="Inventory · Demand · Reorder Point">
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={timeline} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="ig" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={C.blue} stopOpacity={0.25} />
                <stop offset="95%" stopColor={C.blue} stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="day" tick={{ fontSize: 9, fill: C.muted }} />
            <YAxis tick={{ fontSize: 9, fill: C.muted }} width={45} />
            <Tooltip contentStyle={{ background: "#0a0f18", border: `1px solid ${C.border2}`, fontSize: 10, borderRadius: 6 }} labelFormatter={d => `Day ${d}`} />
            <Area type="monotone" dataKey="inventory" stroke={C.blue} strokeWidth={1.5} fill="url(#ig)" dot={false} name="Inventory" />
            <Line type="monotone" dataKey="demand" stroke={C.red} strokeWidth={1} dot={false} name="Demand" />
            <Line type="monotone" dataKey="rop" stroke={C.amber} strokeWidth={1} strokeDasharray="5 3" dot={false} name="ROP" />
          </AreaChart>
        </ResponsiveContainer>
      </Panel>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Panel title="Cumulative Fill Rate">
          <ResponsiveContainer width="100%" height={130}>
            <LineChart data={timeline} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <XAxis dataKey="day" tick={{ fontSize: 9, fill: C.muted }} />
              <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 9, fill: C.muted }} width={38} />
              <ReferenceLine y={0.95} stroke={C.amber} strokeDasharray="4 3" />
              <Tooltip contentStyle={{ background: "#0a0f18", border: `1px solid ${C.border2}`, fontSize: 10 }} formatter={v => `${(v * 100).toFixed(1)}%`} />
              <Line type="monotone" dataKey="fillRateCum" stroke={C.teal} strokeWidth={2} dot={false} name="Fill Rate" />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="Lost Sales Per Day">
          <ResponsiveContainer width="100%" height={130}>
            <BarChart data={timeline} barSize={2} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <XAxis dataKey="day" tick={{ fontSize: 9, fill: C.muted }} />
              <YAxis tick={{ fontSize: 9, fill: C.muted }} width={38} />
              <Tooltip contentStyle={{ background: "#0a0f18", border: `1px solid ${C.border2}`, fontSize: 10 }} />
              <Bar dataKey="lost" fill={C.red} opacity={0.8} name="Lost Sales" />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </div>
    </div>
  );
}

function ReasoningLog({ log, logEndRef }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, maxHeight: "72vh", overflowY: "auto", paddingRight: 4 }}>
      {log.length === 0 && <div style={{ color: C.muted, fontSize: 12, padding: 20, textAlign: "center" }}>Waiting for first LLM decision (after day {CFG.HISTO_DAYS})…</div>}
      {log.map((entry, i) => {
        const d = entry.decision;
        const isLatest = i === log.length - 1;
        return (
          <div key={i} style={{ background: isLatest ? "#0c1a24" : C.panel, border: `1px solid ${isLatest ? C.teal + "40" : C.border}`, borderRadius: 10, padding: "14px 16px", borderLeft: `3px solid ${isLatest ? C.teal : C.border2}` }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10, flexWrap: "wrap", gap: 6 }}>
              <div style={{ fontSize: 11, color: C.teal, fontWeight: 600 }}>Day {entry.day} — Decision #{i + 1}</div>
              <div style={{ display: "flex", gap: 8 }}>
                <span style={{ fontSize: 10, color: C.muted }}>ROP: <span style={{ color: C.amber, fontWeight: 600 }}>{Math.round(entry.rop)}</span></span>
                <span style={{ fontSize: 10, color: C.muted }}>Fill: <FillBadge rate={entry.fillRateCum} /></span>
                <span style={{ fontSize: 9, padding: "2px 7px", borderRadius: 3, background: d.confidence === "high" ? "#0d1f18" : d.confidence === "medium" ? "#1f1a0d" : "#1f0d0d", color: d.confidence === "high" ? C.green : d.confidence === "medium" ? C.amber : C.red, border: "1px solid currentColor", opacity: 0.8 }}>{(d.confidence || "?").toUpperCase()}</span>
              </div>
            </div>
            {d.subgoals?.length > 0 && (
              <div style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 9, letterSpacing: 3, color: C.muted, marginBottom: 6 }}>SUBGOAL DECOMPOSITION</div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {d.subgoals.map((sg, j) => (
                    <div key={j} style={{ fontSize: 10, background: C.dim, border: `1px solid ${C.border2}`, borderRadius: 4, padding: "4px 9px", color: C.blue }}>{j + 1}. {sg}</div>
                  ))}
                </div>
              </div>
            )}
            <div style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 9, letterSpacing: 3, color: C.muted, marginBottom: 5 }}>STATE ANALYSIS</div>
              <div style={{ fontSize: 11, color: C.text, lineHeight: 1.7, background: C.dim, borderRadius: 6, padding: "8px 10px" }}>{d.state_analysis}</div>
            </div>
            {d.recovery_plan && d.recovery_plan !== "N/A" && (
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 9, letterSpacing: 3, color: C.muted, marginBottom: 5 }}>RECOVERY PLAN</div>
                <div style={{ fontSize: 11, color: C.amber, lineHeight: 1.6, background: "#1a1400", borderRadius: 6, padding: "8px 10px", border: `1px solid ${C.amber}20` }}>{d.recovery_plan}</div>
              </div>
            )}
            {d.reasoning_depth && <div style={{ fontSize: 10, color: C.muted }}><span style={{ color: C.purple }}>Reasoning: </span>{d.reasoning_depth}</div>}
          </div>
        );
      })}
      <div ref={logEndRef} />
    </div>
  );
}

function ComparePanel({ agentMetrics, agentLog, simTimeline, baselineResults }) {
  const agentFillRates = simTimeline.map(t => ({ day: t.day, agent: t.fillRateCum }));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 10 }}>
        {agentMetrics && (
          <div style={{ background: "#0a1e18", border: `2px solid ${C.teal}40`, borderRadius: 10, padding: 14 }}>
            <div style={{ fontSize: 9, color: C.teal, letterSpacing: 3, marginBottom: 8 }}>🤖 LLM AGENT</div>
            {[["Profit", `$${Math.round(agentMetrics.profit).toLocaleString()}`], ["Service Level", <FillBadge rate={agentMetrics.serviceLevel} />], ["Fill Rate", <FillBadge rate={agentMetrics.fillRate} />], ["Stockouts", agentMetrics.stockOuts]].map(([l, v]) => (
              <div key={l} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 5 }}>
                <span style={{ color: C.muted }}>{l}</span><span style={{ fontWeight: 600 }}>{v}</span>
              </div>
            ))}
          </div>
        )}
        {Object.entries(baselineResults).map(([bk, br]) => (
          <div key={bk} style={{ background: C.panel, border: `1px solid ${BASELINES[bk].color}30`, borderRadius: 10, padding: 14 }}>
            <div style={{ fontSize: 9, color: BASELINES[bk].color, letterSpacing: 3, marginBottom: 8 }}>{BASELINES[bk].label.toUpperCase()}</div>
            {[["Profit", `$${Math.round(br.metrics.profit).toLocaleString()}`], ["Service Level", <FillBadge rate={br.metrics.serviceLevel} />], ["Fill Rate", <FillBadge rate={br.metrics.fillRate} />], ["Stockouts", br.metrics.stockOuts]].map(([l, v]) => (
              <div key={l} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 5 }}>
                <span style={{ color: C.muted }}>{l}</span><span style={{ fontWeight: 600 }}>{v}</span>
              </div>
            ))}
          </div>
        ))}
      </div>
      {Object.keys(baselineResults).length > 0 && (
        <Panel title="Fill Rate Convergence — Agent vs All Baselines">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
              <XAxis dataKey="day" type="number" domain={[0, CFG.SIM_DAYS]} tick={{ fontSize: 9, fill: C.muted }} />
              <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 9, fill: C.muted }} width={40} />
              <ReferenceLine y={0.95} stroke={C.amber} strokeDasharray="5 3" label={{ value: "95% target", fontSize: 9, fill: C.amber }} />
              <Tooltip contentStyle={{ background: "#0a0f18", border: `1px solid ${C.border2}`, fontSize: 10 }} formatter={v => `${(v * 100).toFixed(1)}%`} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Line data={agentFillRates} type="monotone" dataKey="agent" stroke={C.teal} strokeWidth={2.5} dot={false} name="LLM Agent" />
              {Object.entries(baselineResults).map(([bk, br]) => (
                <Line key={bk} data={br.timeline.map(t => ({ day: t.day, fillRate: t.fillRateCum }))} type="monotone" dataKey="fillRate" stroke={BASELINES[bk].color} strokeWidth={1} strokeDasharray="3 2" dot={false} name={BASELINES[bk].label} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Panel>
      )}
    </div>
  );
}

function MemoryBankPanel({ memory }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <Panel>
        <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.8, marginBottom: 12 }}>
          The memory bank stores the last {CFG.MEMORY_SIZE} decisions with full context — inventory level, demand signal, fill rate, and cumulative losses. This enables the agent to reason across the full {CFG.SIM_DAYS - CFG.HISTO_DAYS}-day horizon beyond the LLM's context window.
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(220px,1fr))", gap: 8 }}>
          {memory.map((m, i) => (
            <div key={i} style={{ background: C.dim, border: `1px solid ${C.border}`, borderRadius: 7, padding: "10px 12px" }}>
              <div style={{ fontSize: 10, color: C.teal, fontWeight: 600, marginBottom: 6 }}>Day {m.day}</div>
              {[["ROP Set", m.rop], ["Confidence", m.confidence], ["Fill Rate", m.fill_rate || "—"], ["Inventory", m.inventory], ["Demand Mean", m.demand_mean], ["Stockouts ∑", m.stockouts_cumulative], ["Lost Sales ∑", m.lost_sales_cumulative]].map(([l, v]) => (
                <div key={l} style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 3 }}>
                  <span style={{ color: C.muted }}>{l}</span>
                  <span style={{ color: C.text }}>{v}</span>
                </div>
              ))}
              {m.key_insight && <div style={{ fontSize: 9, color: C.muted, marginTop: 6, lineHeight: 1.5, borderTop: `1px solid ${C.border}`, paddingTop: 5 }}>{m.key_insight}</div>}
            </div>
          ))}
          {memory.length === 0 && <div style={{ color: C.muted, fontSize: 11 }}>Memory builds as agent makes decisions…</div>}
        </div>
      </Panel>
    </div>
  );
}

// ─── SINGLE-AGENT SIMULATION VIEW ─────────────────────────────────────────────
function AgentSimView({ label, accentColor, modelId, hfToken, envKey, baselineResults }) {
  const [phase, setPhase] = useState("idle"); // idle | running | done
  const [timeline, setTimeline] = useState([]);
  const [log, setLog] = useState([]);
  const [memory, setMemory] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [activeTab, setActiveTab] = useState("live");
  const [status, setStatus] = useState("");
  const [runningDay, setRunningDay] = useState(0);
  const abortRef = useRef(false);
  const logEndRef = useRef(null);
  useEffect(() => { if (logEndRef.current) logEndRef.current.scrollIntoView({ behavior: "smooth" }); }, [log]);

  const start = useCallback(async () => {
    abortRef.current = false;
    setPhase("running"); setTimeline([]); setLog([]); setMemory([]); setMetrics(null); setRunningDay(0);
    try {
      const result = await runAgentLoop({
        envKey, modelId, hfToken, abortRef,
        onDay: (day, tl) => { setTimeline(tl); setRunningDay(day); },
        onDecision: ({ day, snapshot, decision, rop, fillRateCum, memory: mem }) => {
          setLog(prev => [...prev, { day, snapshot, decision, rop, fillRateCum }]);
          setMemory(mem);
        },
        onStatus: setStatus,
      });
      setMetrics(result.metrics);
      setMemory(result.memory);
    } catch (e) {
      setStatus(`Error: ${e.message}`);
    }
    setPhase("done");
  }, [envKey, modelId, hfToken]);

  const stop = () => { abortRef.current = true; setPhase("done"); setStatus("Stopped."); };
  const reset = () => { setPhase("idle"); setTimeline([]); setLog([]); setMemory([]); setMetrics(null); };

  const tabs = [
    { id: "live", label: "LIVE SIM" },
    { id: "reasoning", label: `REASONING (${log.length})` },
    { id: "compare", label: "COMPARE" },
    { id: "memory", label: `MEMORY (${memory.length})` },
  ];

  return (
    <div>
      {phase === "idle" && (
        <button onClick={start} style={{ background: "#0d1f18", border: `1px solid ${accentColor}60`, borderRadius: 7, padding: "12px 24px", color: accentColor, fontFamily: "inherit", fontSize: 13, cursor: "pointer", letterSpacing: 2, fontWeight: 600 }}>
          ▶ RUN {label.toUpperCase()}
        </button>
      )}
      {(phase === "running" || phase === "done") && (
        <>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, flexWrap: "wrap", gap: 8 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 11 }}>
              {phase === "running" && <span style={{ color: C.amber }}>●</span>}
              <span style={{ color: C.muted }}>{status}</span>
              {phase === "running" && (
                <div style={{ width: 160, height: 4, background: C.border, borderRadius: 2, overflow: "hidden" }}>
                  <div style={{ height: "100%", width: `${(runningDay / CFG.SIM_DAYS) * 100}%`, background: accentColor, transition: "width 0.3s", borderRadius: 2 }} />
                </div>
              )}
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              {phase === "running" && <button onClick={stop} style={{ background: "#2a0f0f", border: `1px solid ${C.red}40`, borderRadius: 6, padding: "6px 14px", color: C.red, fontFamily: "inherit", fontSize: 11, cursor: "pointer" }}>■ STOP</button>}
              <button onClick={reset} style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 6, padding: "6px 14px", color: C.muted, fontFamily: "inherit", fontSize: 11, cursor: "pointer" }}>↺ RESET</button>
            </div>
          </div>
          {metrics && (
            <div style={{ display: "flex", gap: 10, marginBottom: 14, flexWrap: "wrap" }}>
              <MetricBox label="PROFIT" value={`$${Math.round(metrics.profit).toLocaleString()}`} highlight color={accentColor} />
              <MetricBox label="SERVICE LEVEL" value={<FillBadge rate={metrics.serviceLevel} />} />
              <MetricBox label="FILL RATE" value={<FillBadge rate={metrics.fillRate} />} />
              <MetricBox label="STOCKOUTS" value={metrics.stockOuts} />
              <MetricBox label="DECISIONS" value={log.length} />
            </div>
          )}
          <SimTabs tabs={tabs} active={activeTab} onSelect={setActiveTab} />
          {activeTab === "live" && <LiveSimCharts timeline={timeline} />}
          {activeTab === "reasoning" && <ReasoningLog log={log} logEndRef={logEndRef} />}
          {activeTab === "compare" && <ComparePanel agentMetrics={metrics} agentLog={log} simTimeline={timeline} baselineResults={baselineResults} />}
          {activeTab === "memory" && <MemoryBankPanel memory={memory} />}
        </>
      )}
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function StockOracle() {
  const [envKey, setEnvKey] = useState("gamma_poisson");
  const [hfToken, setHfToken] = useState("");
  const [grpoModelId, setGrpoModelId] = useState("");
  const [activeTopTab, setActiveTopTab] = useState("llm");
  const [baselineResults, setBaselineResults] = useState({});
  const [baselinesReady, setBaselinesReady] = useState(false);
  const env = ENVS[envKey];

  const runBaselines = useCallback(() => {
    setBaselinesReady(false);
    const demand = buildDemandSeries(envKey, CFG.SIM_DAYS);
    const results = {};
    Object.entries(BASELINES).forEach(([k, ag]) => {
      results[k] = runOneSimulation((h, dm, ds) => ag.compute(h, dm, ds), demand, envKey);
    });
    setBaselineResults(results);
    setBaselinesReady(true);
  }, [envKey]);

  const topTabs = [
    { id: "llm", label: "QWEN BASE AGENT" },
    { id: "grpo", label: "GRPO FINE-TUNED ★" },
    { id: "baselines", label: "BASELINES" },
  ];

  return (
    <div style={{ minHeight: "100vh", background: C.bg, fontFamily: "'JetBrains Mono',monospace", color: C.text, padding: "24px 16px" }}>
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet" />

      {/* HEADER */}
      <div style={{ maxWidth: 1280, margin: "0 auto" }}>
        <div style={{ marginBottom: 28 }}>
          <div style={{ fontSize: 9, letterSpacing: 5, color: C.muted, marginBottom: 6 }}>HACKATHON · LONG-HORIZON REASONING ENVIRONMENT</div>
          <h1 style={{ margin: 0, fontSize: "clamp(32px,5vw,52px)", fontWeight: 700, letterSpacing: -1, background: `linear-gradient(120deg,${C.teal},${C.blue},${C.purple})`, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", lineHeight: 1.1, fontFamily: "inherit" }}>
            STOCK ORACLE
          </h1>
          <div style={{ fontSize: 10, color: C.muted, marginTop: 5, letterSpacing: 2 }}>
            LLM AGENT · GRPO RL TRAINING · INVENTORY OPTIMIZATION · LONG-HORIZON PLANNING
          </div>
        </div>

        {/* GLOBAL CONFIG */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16, marginBottom: 24 }}>
          {/* Env selector */}
          <Panel title="Demand Environment">
            {Object.entries(ENVS).map(([k, e]) => (
              <button key={k} onClick={() => { setEnvKey(k); setBaselinesReady(false); }} style={{ display: "block", width: "100%", textAlign: "left", background: envKey === k ? "#0f1e2e" : "transparent", border: `1px solid ${envKey === k ? e.color + "50" : C.border}`, borderRadius: 6, padding: "9px 12px", marginBottom: 6, cursor: "pointer", fontFamily: "inherit", transition: "all 0.15s" }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 12, color: envKey === k ? e.color : C.muted, fontWeight: 500 }}>{e.label}</span>
                  <span style={{ fontSize: 9, color: e.color, border: `1px solid ${e.color}40`, borderRadius: 3, padding: "2px 6px" }}>{e.tag}</span>
                </div>
                <div style={{ fontSize: 10, color: C.dim, marginTop: 3, lineHeight: 1.5 }}>{e.desc}</div>
              </button>
            ))}
          </Panel>

          {/* HF Token */}
          <Panel title="HuggingFace Token">
            <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7, marginBottom: 12 }}>
              Required for Qwen2.5-72B inference via HF Inference API.<br />
              Get one at <span style={{ color: C.blue }}>huggingface.co/settings/tokens</span>
            </div>
            <input
              type="password"
              placeholder="hf_..."
              value={hfToken}
              onChange={e => setHfToken(e.target.value)}
              style={{ width: "100%", background: C.dim, border: `1px solid ${C.border2}`, borderRadius: 6, padding: "9px 12px", color: C.text, fontFamily: "inherit", fontSize: 12, outline: "none", marginBottom: 10 }}
            />
            <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7, marginBottom: 8, marginTop: 8 }}>
              <span style={{ color: C.purple, fontWeight: 600 }}>GRPO Fine-tuned Model ID</span><br />
              <span style={{ fontSize: 10 }}>HF model ID of the trained adapter (e.g. ademarteau/qwen-inventory-grpo-iter4). Leave blank while training.</span>
            </div>
            <input
              type="text"
              placeholder="ademarteau/qwen-inventory-grpo-iter4"
              value={grpoModelId}
              onChange={e => setGrpoModelId(e.target.value)}
              style={{ width: "100%", background: C.dim, border: `1px solid ${C.purple}40`, borderRadius: 6, padding: "9px 12px", color: C.text, fontFamily: "inherit", fontSize: 12, outline: "none" }}
            />
          </Panel>

          {/* Baselines */}
          <Panel title="Baseline Agents">
            <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7, marginBottom: 12 }}>
              Pre-compute all 4 rule-based baselines for comparison in the Compare tab.
            </div>
            <button onClick={runBaselines} style={{ width: "100%", background: C.dim, border: `1px solid ${C.border2}`, borderRadius: 6, padding: "10px", color: C.text, fontFamily: "inherit", fontSize: 11, cursor: "pointer", letterSpacing: 1, marginBottom: 10 }}>
              ▶ RUN BASELINES
            </button>
            {baselinesReady && Object.entries(baselineResults).map(([k, r]) => (
              <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 5 }}>
                <span style={{ color: BASELINES[k].color }}>{BASELINES[k].label}</span>
                <FillBadge rate={r.metrics.fillRate} />
              </div>
            ))}
          </Panel>
        </div>

        {/* TOP TABS */}
        <SimTabs tabs={topTabs} active={activeTopTab} onSelect={setActiveTopTab} />

        {/* QWEN BASE TAB */}
        {activeTopTab === "llm" && (
          <div>
            <div style={{ fontSize: 11, color: C.muted, marginBottom: 14 }}>
              <span style={{ color: C.blue, fontWeight: 600 }}>Qwen2.5-72B-Instruct</span> via HF Inference API · decisions every {CFG.DECISION_INTERVAL} days · {CFG.SIM_DAYS - CFG.HISTO_DAYS} decision steps · memory bank up to {CFG.MEMORY_SIZE} entries
            </div>
            <AgentSimView
              label="Qwen Base"
              accentColor={C.teal}
              modelId="Qwen/Qwen2.5-72B-Instruct"
              hfToken={hfToken}
              envKey={envKey}
              baselineResults={baselineResults}
            />
          </div>
        )}

        {/* GRPO TAB */}
        {activeTopTab === "grpo" && (
          <div>
            {/* Training status banner */}
            <div style={{ background: "#0d1a0d", border: `1px solid ${C.green}30`, borderRadius: 10, padding: "16px 20px", marginBottom: 18 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                <span style={{ color: C.green, fontSize: 11 }}>● TRAINING IN PROGRESS</span>
                <span style={{ color: C.muted, fontSize: 10 }}>Northflank · 16 vCPU / 196 GB · Qwen2.5-3B-Instruct + LoRA</span>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, fontSize: 10, color: C.muted }}>
                {[["Algorithm", "GRPO (Group Relative Policy Optimization)"], ["Reward", "Analytical P&L simulation — 30-day lookahead"], ["Base Model", "Qwen/Qwen2.5-3B-Instruct via Unsloth"], ["Status", "Iteration 1/5 · Rollout collection in progress"]].map(([l, v]) => (
                  <div key={l}>
                    <div style={{ color: C.muted, letterSpacing: 2, marginBottom: 3, fontSize: 9 }}>{l.toUpperCase()}</div>
                    <div style={{ color: C.text }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ fontSize: 11, color: C.muted, marginBottom: 14 }}>
              {grpoModelId
                ? <><span style={{ color: C.purple, fontWeight: 600 }}>Fine-tuned model:</span> {grpoModelId}</>
                : <span style={{ color: C.amber }}>⚠ Enter the GRPO model ID above once training completes to run inference.</span>}
            </div>

            {grpoModelId ? (
              <AgentSimView
                label="GRPO Fine-tuned"
                accentColor={C.purple}
                modelId={grpoModelId}
                hfToken={hfToken}
                envKey={envKey}
                baselineResults={baselineResults}
              />
            ) : (
              <Panel>
                <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.8 }}>
                  {[
                    ["What is GRPO?", "Group Relative Policy Optimization — reinforcement learning applied to the LLM. The model generates candidate reorder points, receives P&L rewards from the simulation, and updates weights to favor profitable decisions."],
                    ["Reward signal", "Analytical 30-day forward simulation from current state: revenue − holding_cost − stockout_penalty − order_cost − writeoff_cost, normalized by baseline profit. 60% P&L weight + 40% fill rate vs 95% target."],
                    ["vs Base Qwen", "The base model reasons generically. After GRPO training, the model should internalize inventory-specific heuristics: lead-time-aware ordering, demand volatility buffers, write-off avoidance at high inventory levels."],
                    ["Memory (200 entries)", "Unlike base Qwen (limited by context window), the GRPO-trained model was trained with full 200-entry memory banks, enabling true long-horizon reasoning across the 365-day decision horizon."],
                  ].map(([t, d]) => (
                    <div key={t} style={{ marginBottom: 12 }}>
                      <span style={{ color: C.purple, fontWeight: 600 }}>{t}: </span>
                      <span>{d}</span>
                    </div>
                  ))}
                </div>
              </Panel>
            )}
          </div>
        )}

        {/* BASELINES TAB */}
        {activeTopTab === "baselines" && (
          <div>
            {!baselinesReady ? (
              <div style={{ color: C.muted, fontSize: 12, padding: 20 }}>Run baselines from the config panel above first.</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10 }}>
                  {Object.entries(baselineResults).map(([k, r]) => (
                    <div key={k} style={{ background: C.panel, border: `1px solid ${BASELINES[k].color}30`, borderRadius: 10, padding: 16 }}>
                      <div style={{ fontSize: 9, color: BASELINES[k].color, letterSpacing: 3, marginBottom: 10 }}>{BASELINES[k].label.toUpperCase()}</div>
                      {[["Profit", `$${Math.round(r.metrics.profit).toLocaleString()}`], ["Service Level", <FillBadge rate={r.metrics.serviceLevel} />], ["Fill Rate", <FillBadge rate={r.metrics.fillRate} />], ["Stockouts", r.metrics.stockOuts]].map(([l, v]) => (
                        <div key={l} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 6 }}>
                          <span style={{ color: C.muted }}>{l}</span><span style={{ fontWeight: 600 }}>{v}</span>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
                <Panel title="Fill Rate Convergence — All Baselines">
                  <ResponsiveContainer width="100%" height={240}>
                    <LineChart margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                      <XAxis dataKey="day" type="number" domain={[0, CFG.SIM_DAYS]} tick={{ fontSize: 9, fill: C.muted }} />
                      <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 9, fill: C.muted }} width={40} />
                      <ReferenceLine y={0.95} stroke={C.amber} strokeDasharray="5 3" />
                      <Tooltip contentStyle={{ background: "#0a0f18", border: `1px solid ${C.border2}`, fontSize: 10 }} formatter={v => `${(v * 100).toFixed(1)}%`} />
                      <Legend wrapperStyle={{ fontSize: 10 }} />
                      {Object.entries(baselineResults).map(([k, r]) => (
                        <Line key={k} data={r.timeline.map(t => ({ day: t.day, fillRate: t.fillRateCum }))} type="monotone" dataKey="fillRate" stroke={BASELINES[k].color} strokeWidth={1.5} dot={false} name={BASELINES[k].label} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </Panel>
              </div>
            )}
          </div>
        )}

        {/* FOOTER */}
        <div style={{ marginTop: 32, paddingTop: 16, borderTop: `1px solid ${C.border}`, display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, fontSize: 10, color: C.dim }}>
          {[
            ["Environment", `Stochastic inventory simulation · ${CFG.SIM_DAYS}-day horizon · 4 demand regimes · lead time ${CFG.LEAD_TIME} days · spoilage ${(CFG.WRITE_OFF_RATE * 100).toFixed(2)}%/day`],
            ["Agent Architecture", `Qwen2.5-72B via HF Inference API · decisions every ${CFG.DECISION_INTERVAL} days · rolling 6-turn conversation · ${CFG.MEMORY_SIZE}-entry memory bank`],
            ["GRPO Training", "Qwen2.5-3B-Instruct fine-tuned with GRPO · analytical P&L reward · 30-day lookahead simulation · LoRA r=16 · currently training on Northflank"],
            ["Benchmarking", "LLM agent vs 4 rule-based baselines: Base, Safety Stock, Oracle Forecast, Monte Carlo · same demand series · identical simulation engine"],
          ].map(([t, d]) => (
            <div key={t}>
              <div style={{ color: C.muted, fontWeight: 600, marginBottom: 4, fontSize: 9, letterSpacing: 2 }}>{t.toUpperCase()}</div>
              <div style={{ lineHeight: 1.7 }}>{d}</div>
            </div>
          ))}
        </div>
      </div>
      <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}} input:focus{border-color:${C.teal}!important;} input::placeholder{color:${C.muted}}`}</style>
    </div>
  );
}
