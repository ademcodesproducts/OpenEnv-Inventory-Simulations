import { useState, useRef, useCallback, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, AreaChart, Area, BarChart, Bar, Legend } from "recharts";

// ─── DESIGN TOKENS ────────────────────────────────────────────────────────────
const C = {
  bg:       "#07090f",
  panel:    "#0d1117",
  border:   "#161d2a",
  border2:  "#1e2d40",
  text:     "#c9d5e0",
  muted:    "#3a5060",
  dim:      "#1a2535",
  green:    "#34d399",
  blue:     "#38bdf8",
  amber:    "#fbbf24",
  red:      "#f87171",
  purple:   "#a78bfa",
  teal:     "#2dd4bf",
};

// ─── CONFIG (mirrors config.py exactly) ───────────────────────────────────────
const CFG = {
  LEAD_TIME:           3,
  BASE_STOCK:          0,
  DEFAULT_SL:          0.95,
  WRITE_OFF_RATE:      0.01,
  WRITE_OFF_FREQ:      7,
  HISTO_DAYS:          30,
  SIM_DAYS:            120,
};

// ─── MATH HELPERS ─────────────────────────────────────────────────────────────
function normalRandom() {
  let u=0,v=0; while(!u)u=Math.random(); while(!v)v=Math.random();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
}
function gammaRandom(shape,scale){
  if(shape<1)return gammaRandom(1+shape,scale)*Math.pow(Math.random(),1/shape);
  const d=shape-1/3,c=1/Math.sqrt(9*d);
  while(true){let x,v;do{x=normalRandom();v=1+c*x;}while(v<=0);v=v*v*v;const u=Math.random();
    if(u<1-0.0331*x*x*x*x)return d*v*scale;
    if(Math.log(u)<0.5*x*x+d*(1-v+Math.log(v)))return d*v*scale;}
}
function poissonRandom(lambda){let L=Math.exp(-lambda),k=0,p=1;do{k++;p*=Math.random();}while(p>L);return k-1;}
function expRandom(rate){return-Math.log(Math.random())/rate;}
function arr_mean(a){return a.length?a.reduce((s,x)=>s+x,0)/a.length:0;}
function arr_std(a){if(a.length<2)return 0;const m=arr_mean(a);return Math.sqrt(a.reduce((s,x)=>s+(x-m)**2,0)/(a.length-1));}
function quantile(sorted,q){return sorted[Math.floor(sorted.length*q)];}

// ─── DEMAND ENVIRONMENTS (mirrors demand_environment.py) ──────────────────────
const ENVS = {
  gamma_poisson:{
    label:"Gamma–Poisson",tag:"MODERATE",color:C.green,
    desc:"90% Gamma(7,16) + 10% Poisson(80). Stable with rare spikes.",
    sample:()=>Math.random()<0.9?Math.max(0,Math.round(gammaRandom(7,16))):poissonRandom(80),
    demMean:112,demStd:38,
  },
  bimodal_hv:{
    label:"Bimodal High-Var",tag:"HARD",color:C.amber,
    desc:"50% Gamma(low mean) + 50% Gamma(high mean). Extremely unpredictable.",
    sample:()=>Math.random()<0.5?Math.max(0,Math.round(gammaRandom(7,3))):Math.max(0,Math.round(gammaRandom(7,29))),
    demMean:112,demStd:95,
  },
  spiking:{
    label:"Sporadic Spiking",tag:"EXTREME",color:C.red,
    desc:"95% zero demand, 5% large Exponential bursts. Hardest to plan.",
    sample:()=>Math.random()<0.95?0:Math.max(0,Math.round(expRandom(0.05))),
    demMean:20,demStd:55,
  },
  gamma_stable:{
    label:"Stable Gamma",tag:"EASY",color:C.blue,
    desc:"Single Gamma(7,16), low variance. Baseline environment.",
    sample:()=>Math.max(0,Math.round(gammaRandom(7,16))),
    demMean:112,demStd:35,
  },
};

// ─── BASELINE AGENTS (mirrors agent_environment.py) ───────────────────────────
const BASELINES = {
  base:{
    label:"Base",color:C.muted,
    compute:(hist)=>arr_mean(hist)*CFG.LEAD_TIME,
  },
  safety_stock:{
    label:"Safety Stock",color:C.blue,
    compute:(hist)=>{
      const m=arr_mean(hist),s=arr_std(hist);
      return m*CFG.LEAD_TIME+1.645*s*Math.sqrt(CFG.LEAD_TIME);
    },
  },
  forecast:{
    label:"Oracle Forecast",color:C.green,
    compute:(hist,dMean,dStd)=>dMean*CFG.LEAD_TIME+1.645*dStd*Math.sqrt(CFG.LEAD_TIME),
  },
  monte_carlo:{
    label:"Monte Carlo",color:C.purple,
    compute:(hist)=>{
      const s=[];
      for(let i=0;i<500;i++){
        let t=0;for(let j=0;j<CFG.LEAD_TIME;j++)t+=hist[Math.floor(Math.random()*hist.length)]*(0.8+Math.random()*0.4);
        s.push(t);
      }
      s.sort((a,b)=>a-b);return quantile(s,0.95);
    },
  },
};

// ─── SIMULATION ENGINE ────────────────────────────────────────────────────────
function buildDemandSeries(envKey, n){
  return Array.from({length:n},()=>ENVS[envKey].sample());
}

function runOneSimulation(computeROP, demandSeries, envKey){
  const env=ENVS[envKey];
  const n=demandSeries.length;
  let inventory=0;
  const orders=[];
  let totDemand=0,totFulfilled=0,totWriteOff=0,stockOuts=0,lostSales=0;
  const timeline=[];

  for(let day=0;day<n;day++){
    const demand=demandSeries[day];
    const hist=demandSeries.slice(Math.max(0,day-CFG.HISTO_DAYS),day);

    // Deliver orders
    const arrivals=orders.filter(o=>o.arr===day);
    const delivered=arrivals.reduce((s,o)=>s+o.qty,0);
    inventory+=delivered;
    orders.splice(0,orders.length,...orders.filter(o=>o.arr>day));

    const preInv=inventory;

    // Fulfill demand
    const fulfilled=Math.min(demand,inventory);
    inventory=Math.max(0,inventory-demand);
    const lost=Math.max(0,demand-fulfilled);
    if(lost>0)stockOuts++;
    lostSales+=lost;

    // Reorder
    let rop=0,ordered=0;
    if(hist.length>=5&&day<n-CFG.LEAD_TIME){
      rop=Math.max(0,computeROP(hist,env.demMean,env.demStd));
      if(inventory<=rop){
        const qty=Math.ceil(rop-inventory+arr_mean(hist)*CFG.LEAD_TIME);
        orders.push({arr:day+CFG.LEAD_TIME,qty});
        ordered=qty;
      }
    }

    // Write-off
    let wo=0;
    if(day%CFG.WRITE_OFF_FREQ===0){wo=Math.floor(inventory*CFG.WRITE_OFF_RATE);inventory-=wo;totWriteOff+=wo;}

    totDemand+=demand;totFulfilled+=fulfilled;
    const fillRateCum=totDemand>0?totFulfilled/totDemand:0;
    timeline.push({day,demand,inventory:preInv,inventoryAfter:inventory,fulfilled,lost,rop:Math.round(rop),ordered,wo,delivered,fillRateCum});
  }
  return{timeline,metrics:{fillRate:totDemand>0?totFulfilled/totDemand:0,stockOuts,lostSales,totWriteOff,totDemand,totFulfilled}};
}

// ─── BUILD ENVIRONMENT SNAPSHOT FOR LLM ───────────────────────────────────────
function buildEnvSnapshot(demandSeries, timeline, day){
  const recent=demandSeries.slice(Math.max(0,day-10),day);
  const hist=demandSeries.slice(Math.max(0,day-CFG.HISTO_DAYS),day);
  const last5=timeline.slice(Math.max(0,day-5),day);
  const curInv=timeline[day-1]?.inventoryAfter??0;
  const pendingOrders=[];
  // Reconstruct pending from timeline (simplified)
  const fillSoFar=timeline[day-1]?.fillRateCum??null;

  return {
    day,
    current_inventory: curInv,
    lead_time: CFG.LEAD_TIME,
    write_off_rate: CFG.WRITE_OFF_RATE,
    service_level_target: CFG.DEFAULT_SL,
    sim_days_total: CFG.SIM_DAYS,
    days_remaining: CFG.SIM_DAYS-day,
    recent_demand_10d: recent,
    demand_mean_30d: Math.round(arr_mean(hist)*10)/10,
    demand_std_30d:  Math.round(arr_std(hist)*10)/10,
    fill_rate_so_far: fillSoFar ? Math.round(fillSoFar*1000)/10+"%" : "N/A",
    last_5_days: last5.map(d=>({day:d.day,demand:d.demand,inv:d.inventoryAfter,lost:d.lost,rop:d.rop,ordered:d.ordered})),
    recent_stockouts: last5.filter(d=>d.lost>0).length,
    recent_lost_sales: last5.reduce((s,d)=>s+d.lost,0),
  };
}

// ─── LLM CALL ─────────────────────────────────────────────────────────────────
async function callClaude(messages, systemPrompt){
  const resp=await fetch("https://api.anthropic.com/v1/messages",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({
      model:"claude-sonnet-4-20250514",
      max_tokens:1000,
      system:systemPrompt,
      messages,
    }),
  });
  const data=await resp.json();
  const text=data.content?.find(b=>b.type==="text")?.text||"";
  return text;
}

// ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are an expert inventory optimization agent embedded in a stochastic simulation environment.

YOUR ROLE:
You receive a JSON snapshot of the current simulation state and must decide the REORDER POINT (ROP) — the inventory threshold that triggers a new order.

ENVIRONMENT RULES:
- Orders arrive exactly LEAD_TIME=3 days after placement
- You place an order whenever inventory <= your ROP
- Order quantity = ROP - current_inventory + mean_demand * LEAD_TIME (already handled)
- Every 7 days, 1% of inventory is written off (waste/expiry)
- Reward = fill_rate at end of simulation (target: >=95%)
- Reward is SPARSE: fill rate only stabilizes after ~50 days

REASONING REQUIREMENTS - you MUST do all 4:
1. SUBGOAL DECOMPOSITION: Break the problem into explicit subgoals (e.g., "build buffer", "survive spike risk", "minimize waste")
2. STATE ANALYSIS: Interpret current inventory, demand trend, stockout risk, fill rate trajectory
3. DECISION: Output a specific numeric ROP with clear justification
4. RECOVERY PLAN: If fill rate < 95% or recent stockouts occurred, state your recovery strategy

CRITICAL: You must reason BEYOND the next step. Consider that your ROP today affects inventory 3+ days from now.
For spiking demand: ROP must account for rare but catastrophic spikes.
For high-variance: wider safety buffers needed.
For stable demand: tighter ROP to avoid write-offs.

OUTPUT FORMAT — respond with this exact JSON (no markdown fences):
{
  "subgoals": ["subgoal 1", "subgoal 2", "subgoal 3"],
  "state_analysis": "2-3 sentence analysis of current state and risks",
  "recovery_plan": "what you're doing to recover or maintain performance",
  "reorder_point": <number>,
  "confidence": "high|medium|low",
  "reasoning_depth": "brief note on what makes this decision non-trivial"
}`;

// ─── MAIN COMPONENT ───────────────────────────────────────────────────────────
export default function StockOracleAgent() {
  const [envKey, setEnvKey]           = useState("gamma_poisson");
  const [phase, setPhase]             = useState("config"); // config | running | done
  const [agentLog, setAgentLog]       = useState([]);       // [{day, snapshot, decision, rop}]
  const [simTimeline, setSimTimeline] = useState([]);
  const [baselineResults, setBaselineResults] = useState({});
  const [agentMetrics, setAgentMetrics]       = useState(null);
  const [runningDay, setRunningDay]   = useState(0);
  const [statusMsg, setStatusMsg]     = useState("");
  const [memoryBank, setMemoryBank]   = useState([]);       // persistent cross-turn memory
  const [conversationHistory, setConversationHistory] = useState([]);
  const [activeTab, setActiveTab]     = useState("live");   // live | reasoning | compare | memory
  const abortRef = useRef(false);
  const logEndRef = useRef(null);

  useEffect(()=>{if(logEndRef.current)logEndRef.current.scrollIntoView({behavior:"smooth"});},[agentLog]);

  // ── Run baselines (instant, no API) ──
  const runBaselines = useCallback((demandSeries) => {
    const results = {};
    Object.entries(BASELINES).forEach(([k,ag])=>{
      results[k]=runOneSimulation((h,dm,ds)=>ag.compute(h,dm,ds), demandSeries, envKey);
    });
    setBaselineResults(results);
    return results;
  },[envKey]);

  // ── Build persistent memory summary ──
  function updateMemory(prevMemory, decision, day, metrics){
    const entry = {
      day,
      rop: decision.reorder_point,
      confidence: decision.confidence,
      fill_rate: metrics?.fillRate ? Math.round(metrics.fillRate*1000)/10 : null,
      stockouts_in_window: metrics?.stockOuts??0,
      key_insight: decision.state_analysis?.slice(0,80)+"...",
    };
    // Keep last 15 memory entries as compressed state
    const newMem = [...prevMemory.slice(-14), entry];
    return newMem;
  }

  // ── Main simulation loop ──
  const runAgentSimulation = useCallback(async () => {
    abortRef.current = false;
    setPhase("running");
    setAgentLog([]);
    setSimTimeline([]);
    setAgentMetrics(null);
    setMemoryBank([]);
    setConversationHistory([]);
    setRunningDay(0);

    const demandSeries = buildDemandSeries(envKey, CFG.SIM_DAYS);

    // Run baselines in background
    setStatusMsg("Computing baseline agents...");
    runBaselines(demandSeries);

    // Agent-driven simulation
    // We step through the sim, calling Claude every DECISION_INTERVAL days
    const DECISION_INTERVAL = 5; // Claude decides ROP every 5 days
    let inventory = 0;
    const orders = [];
    let totDemand=0, totFulfilled=0, totWriteOff=0, stockOuts=0, lostSales=0;
    const timeline = [];
    let currentROP = arr_mean(demandSeries.slice(0,CFG.HISTO_DAYS)) * CFG.LEAD_TIME; // initial ROP
    let localMemory = [];
    let localConvo = [];
    let localLog = [];

    for(let day=0; day<CFG.SIM_DAYS; day++){
      if(abortRef.current) break;

      const demand = demandSeries[day];
      const hist   = demandSeries.slice(Math.max(0,day-CFG.HISTO_DAYS), day);

      // Deliver orders
      const arrivals = orders.filter(o=>o.arr===day);
      const delivered = arrivals.reduce((s,o)=>s+o.qty,0);
      inventory += delivered;
      orders.splice(0,orders.length,...orders.filter(o=>o.arr>day));

      const preInv = inventory;

      // Fulfill demand
      const fulfilled = Math.min(demand, inventory);
      inventory = Math.max(0, inventory-demand);
      const lost = Math.max(0, demand-fulfilled);
      if(lost>0) stockOuts++;
      lostSales += lost;

      // Reorder check using current ROP
      let ordered=0;
      if(hist.length>=5 && day<CFG.SIM_DAYS-CFG.LEAD_TIME){
        if(inventory<=currentROP){
          const qty=Math.ceil(currentROP-inventory+arr_mean(hist)*CFG.LEAD_TIME);
          orders.push({arr:day+CFG.LEAD_TIME,qty});
          ordered=qty;
        }
      }

      // Write-off
      let wo=0;
      if(day%CFG.WRITE_OFF_FREQ===0){wo=Math.floor(inventory*CFG.WRITE_OFF_RATE);inventory-=wo;totWriteOff+=wo;}

      totDemand+=demand; totFulfilled+=fulfilled;
      const fillRateCum = totDemand>0?totFulfilled/totDemand:0;
      const tEntry = {day,demand,inventory:preInv,inventoryAfter:inventory,fulfilled,lost,rop:Math.round(currentROP),ordered,wo,delivered,fillRateCum};
      timeline.push(tEntry);

      setSimTimeline([...timeline]);
      setRunningDay(day);

      // ── LLM Decision every DECISION_INTERVAL days ──
      if(day>=CFG.HISTO_DAYS && day%DECISION_INTERVAL===0 && day<CFG.SIM_DAYS-CFG.LEAD_TIME){
        setStatusMsg(`Day ${day}: Agent reasoning...`);

        const snapshot = buildEnvSnapshot(demandSeries, timeline, day);

        // Build memory context
        const memoryContext = localMemory.length>0
          ? `\nYOUR MEMORY FROM PREVIOUS DECISIONS:\n${JSON.stringify(localMemory.slice(-8),null,2)}`
          : "";

        const userMsg = {
          role:"user",
          content: `ENVIRONMENT SNAPSHOT — Day ${day}/${CFG.SIM_DAYS}\n${JSON.stringify(snapshot,null,2)}${memoryContext}\n\nDecide your reorder_point for the next ${DECISION_INTERVAL} days.`
        };

        // Maintain rolling conversation (last 6 turns to stay in context)
        const trimmedConvo = localConvo.slice(-6);
        const fullMessages = [...trimmedConvo, userMsg];

        try {
          const rawResp = await callClaude(fullMessages, SYSTEM_PROMPT);
          let decision;
          try {
            const cleaned = rawResp.replace(/```json|```/g,"").trim();
            decision = JSON.parse(cleaned);
          } catch {
            // Fallback: extract reorder_point with regex
            const match = rawResp.match(/"reorder_point"\s*:\s*(\d+\.?\d*)/);
            decision = {
              subgoals:["parse error — fallback"],
              state_analysis: rawResp.slice(0,200),
              recovery_plan:"N/A",
              reorder_point: match ? parseFloat(match[1]) : currentROP,
              confidence:"low",
              reasoning_depth:"parse failed",
            };
          }

          currentROP = Math.max(0, decision.reorder_point||currentROP);

          // Update conversation history
          const assistantMsg = {role:"assistant", content:rawResp};
          localConvo = [...localConvo, userMsg, assistantMsg];
          setConversationHistory([...localConvo]);

          // Update memory bank
          localMemory = updateMemory(localMemory, decision, day, {fillRate:fillRateCum, stockOuts});
          setMemoryBank([...localMemory]);

          // Add to agent log
          const logEntry = {day, snapshot, decision, rop:currentROP, fillRateCum};
          localLog = [...localLog, logEntry];
          setAgentLog([...localLog]);

        } catch(e) {
          setStatusMsg(`Day ${day}: API error — ${e.message}`);
        }

        // Small pause to not slam API
        await new Promise(r=>setTimeout(r,200));
      }
    }

    // Final metrics
    const finalMetrics = {
      fillRate:totDemand>0?totFulfilled/totDemand:0,
      stockOuts, lostSales, totWriteOff, totDemand, totFulfilled
    };
    setAgentMetrics(finalMetrics);
    setSimTimeline([...timeline]);
    setPhase("done");
    setStatusMsg("Simulation complete.");
    setActiveTab("compare");
  }, [envKey, runBaselines]);

  const stopSim = () => { abortRef.current=true; setStatusMsg("Stopped by user."); setPhase("done"); };

  // ── Render helpers ──
  const env = ENVS[envKey];
  const latestLog = agentLog[agentLog.length-1];

  function FillBadge({rate}){
    const c=rate>=0.95?C.green:rate>=0.85?C.amber:C.red;
    return <span style={{color:c,fontWeight:700}}>{rate?(rate*100).toFixed(1)+"%":"—"}</span>;
  }

  function Panel({title,children,style={}}){
    return(
      <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:10,padding:"16px 18px",...style}}>
        {title&&<div style={{fontSize:9,letterSpacing:4,color:C.muted,marginBottom:12,textTransform:"uppercase"}}>{title}</div>}
        {children}
      </div>
    );
  }

  function Tab({id,label}){
    const active=activeTab===id;
    return(
      <button onClick={()=>setActiveTab(id)} style={{
        background:active?C.border2:"transparent",
        border:`1px solid ${active?C.border2:"transparent"}`,
        borderRadius:6,padding:"7px 14px",
        color:active?C.text:C.muted,fontFamily:"inherit",
        fontSize:11,cursor:"pointer",letterSpacing:1,transition:"all 0.15s",
      }}>{label}</button>
    );
  }

  const agentTimelineFillRates = simTimeline.map(t=>({day:t.day,agent:t.fillRateCum}));

  return(
    <div style={{minHeight:"100vh",background:C.bg,fontFamily:"'JetBrains Mono',monospace",color:C.text,padding:"24px 16px"}}>
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Clash+Display:wght@600;700&display=swap" rel="stylesheet"/>

      {/* ── HEADER ── */}
      <div style={{maxWidth:1200,margin:"0 auto"}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:28,flexWrap:"wrap",gap:12}}>
          <div>
            <div style={{fontSize:9,letterSpacing:5,color:C.muted,marginBottom:6}}>HACKATHON · LONG-HORIZON REASONING ENVIRONMENT</div>
            <h1 style={{margin:0,fontSize:"clamp(32px,5vw,52px)",fontWeight:700,letterSpacing:-1,
              background:`linear-gradient(120deg,${C.teal},${C.blue},${C.purple})`,
              WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",lineHeight:1.1,
              fontFamily:"'JetBrains Mono',monospace",
            }}>STOCK ORACLE</h1>
            <div style={{fontSize:10,color:C.muted,marginTop:5,letterSpacing:2}}>
              LLM AGENT · INVENTORY OPTIMIZATION · SPARSE REWARD · MULTI-STEP PLANNING
            </div>
          </div>
          {phase==="done"&&agentMetrics&&(
            <div style={{display:"flex",gap:10,flexWrap:"wrap"}}>
              {[
                {label:"AGENT FILL RATE",val:<FillBadge rate={agentMetrics.fillRate}/>,highlight:true},
                {label:"STOCKOUTS",val:agentMetrics.stockOuts},
                {label:"LOST SALES",val:agentMetrics.lostSales.toLocaleString()},
                {label:"LLM DECISIONS",val:agentLog.length},
              ].map(({label,val,highlight})=>(
                <div key={label} style={{background:highlight?"#0d1f18":C.panel,border:`1px solid ${highlight?C.green+"30":C.border}`,borderRadius:8,padding:"10px 16px",textAlign:"center"}}>
                  <div style={{fontSize:9,letterSpacing:3,color:C.muted,marginBottom:3}}>{label}</div>
                  <div style={{fontSize:22,fontWeight:600,letterSpacing:1}}>{val}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── CONFIG ── */}
        {phase==="config"&&(
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16,marginBottom:20,maxWidth:800}}>
            <Panel title="Demand Environment">
              {Object.entries(ENVS).map(([k,e])=>(
                <button key={k} onClick={()=>setEnvKey(k)} style={{
                  display:"block",width:"100%",textAlign:"left",
                  background:envKey===k?"#0f1e2e":"transparent",
                  border:`1px solid ${envKey===k?e.color+"50":C.border}`,
                  borderRadius:6,padding:"10px 12px",marginBottom:6,cursor:"pointer",fontFamily:"inherit",
                  transition:"all 0.15s",
                }}>
                  <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                    <span style={{fontSize:12,color:envKey===k?e.color:C.muted,fontWeight:500}}>{e.label}</span>
                    <span style={{fontSize:9,color:e.color,border:`1px solid ${e.color}40`,borderRadius:3,padding:"2px 6px"}}>{e.tag}</span>
                  </div>
                  <div style={{fontSize:10,color:C.dim,marginTop:4,lineHeight:1.5}}>{e.desc}</div>
                </button>
              ))}
            </Panel>
            <Panel title="About This Environment">
              <div style={{fontSize:11,color:C.muted,lineHeight:1.8}}>
                {[
                  ["Sparse Reward","Fill rate only converges after 50+ days. No reward signal per individual decision."],
                  ["Multi-Step Planning","Each ROP decision affects inventory 3 days forward (lead time). Cascading errors are common."],
                  ["State Tracking","Agent maintains memory across 120 days: inventory levels, order pipeline, demand patterns."],
                  ["Error Recovery","Post-stockout, agent must over-order to rebuild buffer without triggering write-off waste."],
                  ["Extended Horizon","120 decisions × 5-day intervals. LLM conversation history managed via rolling window + memory bank."],
                ].map(([t,d])=>(
                  <div key={t} style={{marginBottom:10}}>
                    <span style={{color:C.teal,fontWeight:600}}>{t}: </span>
                    <span style={{color:C.muted}}>{d}</span>
                  </div>
                ))}
              </div>
              <button onClick={runAgentSimulation} style={{
                width:"100%",marginTop:16,
                background:"#0d1f18",border:`1px solid ${C.green}60`,
                borderRadius:7,padding:"14px",color:C.green,
                fontFamily:"inherit",fontSize:13,cursor:"pointer",
                letterSpacing:2,fontWeight:600,transition:"all 0.2s",
              }}>
                ▶ LAUNCH AGENT SIMULATION
              </button>
            </Panel>
          </div>
        )}

        {/* ── RUNNING / DONE ── */}
        {(phase==="running"||phase==="done")&&(
          <>
            {/* Status bar */}
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:16,flexWrap:"wrap",gap:8}}>
              <div style={{display:"flex",gap:8,alignItems:"center",fontSize:11}}>
                {phase==="running"&&<span style={{color:C.amber,animation:"pulse 1s infinite"}}
                  >●</span>}
                <span style={{color:C.muted}}>{statusMsg}</span>
                {phase==="running"&&(
                  <div style={{width:200,height:4,background:C.border,borderRadius:2,overflow:"hidden"}}>
                    <div style={{height:"100%",width:`${(runningDay/CFG.SIM_DAYS)*100}%`,background:C.teal,transition:"width 0.3s",borderRadius:2}}/>
                  </div>
                )}
              </div>
              <div style={{display:"flex",gap:8}}>
                {phase==="running"&&<button onClick={stopSim} style={{background:"#2a0f0f",border:`1px solid ${C.red}40`,borderRadius:6,padding:"6px 14px",color:C.red,fontFamily:"inherit",fontSize:11,cursor:"pointer"}}>■ STOP</button>}
                <button onClick={()=>{setPhase("config");setAgentLog([]);setSimTimeline([]);setBaselineResults({});setAgentMetrics(null);}} style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:6,padding:"6px 14px",color:C.muted,fontFamily:"inherit",fontSize:11,cursor:"pointer"}}>↺ RESET</button>
              </div>
            </div>

            {/* Tabs */}
            <div style={{display:"flex",gap:6,marginBottom:14,flexWrap:"wrap"}}>
              <Tab id="live"      label="LIVE SIM"/>
              <Tab id="reasoning" label={`AGENT REASONING (${agentLog.length})`}/>
              <Tab id="compare"   label="COMPARE AGENTS"/>
              <Tab id="memory"    label={`MEMORY BANK (${memoryBank.length})`}/>
            </div>

            {/* ── TAB: LIVE SIM ── */}
            {activeTab==="live"&&(
              <div style={{display:"flex",flexDirection:"column",gap:14}}>
                <Panel title="Inventory · Demand · Reorder Point">
                  <ResponsiveContainer width="100%" height={200}>
                    <AreaChart data={simTimeline} margin={{top:4,right:4,bottom:0,left:0}}>
                      <defs>
                        <linearGradient id="ig" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={C.blue} stopOpacity={0.25}/>
                          <stop offset="95%" stopColor={C.blue} stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="day" tick={{fontSize:9,fill:C.muted}}/>
                      <YAxis tick={{fontSize:9,fill:C.muted}} width={45}/>
                      <Tooltip contentStyle={{background:"#0a0f18",border:`1px solid ${C.border2}`,fontSize:10,borderRadius:6}} labelFormatter={d=>`Day ${d}`}/>
                      <Area type="monotone" dataKey="inventory" stroke={C.blue} strokeWidth={1.5} fill="url(#ig)" dot={false} name="Inventory"/>
                      <Line type="monotone" dataKey="demand" stroke={C.red} strokeWidth={1} dot={false} name="Demand"/>
                      <Line type="monotone" dataKey="rop" stroke={C.amber} strokeWidth={1} strokeDasharray="5 3" dot={false} name="Agent ROP"/>
                    </AreaChart>
                  </ResponsiveContainer>
                </Panel>
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
                  <Panel title="Cumulative Fill Rate">
                    <ResponsiveContainer width="100%" height={130}>
                      <LineChart data={simTimeline} margin={{top:4,right:4,bottom:0,left:0}}>
                        <XAxis dataKey="day" tick={{fontSize:9,fill:C.muted}}/>
                        <YAxis domain={[0,1]} tickFormatter={v=>`${(v*100).toFixed(0)}%`} tick={{fontSize:9,fill:C.muted}} width={38}/>
                        <ReferenceLine y={0.95} stroke={C.amber} strokeDasharray="4 3"/>
                        <Tooltip contentStyle={{background:"#0a0f18",border:`1px solid ${C.border2}`,fontSize:10}} formatter={v=>`${(v*100).toFixed(1)}%`}/>
                        <Line type="monotone" dataKey="fillRateCum" stroke={C.teal} strokeWidth={2} dot={false} name="Fill Rate"/>
                      </LineChart>
                    </ResponsiveContainer>
                  </Panel>
                  <Panel title="Lost Sales Per Day">
                    <ResponsiveContainer width="100%" height={130}>
                      <BarChart data={simTimeline} barSize={2} margin={{top:4,right:4,bottom:0,left:0}}>
                        <XAxis dataKey="day" tick={{fontSize:9,fill:C.muted}}/>
                        <YAxis tick={{fontSize:9,fill:C.muted}} width={38}/>
                        <Tooltip contentStyle={{background:"#0a0f18",border:`1px solid ${C.border2}`,fontSize:10}}/>
                        <Bar dataKey="lost" fill={C.red} opacity={0.8} name="Lost Sales"/>
                      </BarChart>
                    </ResponsiveContainer>
                  </Panel>
                </div>
              </div>
            )}

            {/* ── TAB: AGENT REASONING ── */}
            {activeTab==="reasoning"&&(
              <div style={{display:"flex",flexDirection:"column",gap:10,maxHeight:"72vh",overflowY:"auto",paddingRight:4}}>
                {agentLog.length===0&&<div style={{color:C.muted,fontSize:12,padding:20,textAlign:"center"}}>Waiting for first LLM decision (after day {CFG.HISTO_DAYS})...</div>}
                {agentLog.map((entry,i)=>{
                  const d=entry.decision;
                  const isLatest=i===agentLog.length-1;
                  return(
                    <div key={i} style={{
                      background:isLatest?"#0c1a24":C.panel,
                      border:`1px solid ${isLatest?C.teal+"40":C.border}`,
                      borderRadius:10,padding:"14px 16px",
                      borderLeft:`3px solid ${isLatest?C.teal:C.border2}`,
                    }}>
                      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10,flexWrap:"wrap",gap:6}}>
                        <div style={{fontSize:11,color:C.teal,fontWeight:600}}>Day {entry.day} — Decision #{i+1}</div>
                        <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
                          <span style={{fontSize:10,color:C.muted}}>ROP: <span style={{color:C.amber,fontWeight:600}}>{Math.round(entry.rop)}</span></span>
                          <span style={{fontSize:10,color:C.muted}}>Fill: <FillBadge rate={entry.fillRateCum}/></span>
                          <span style={{fontSize:9,padding:"2px 7px",borderRadius:3,
                            background:d.confidence==="high"?"#0d1f18":d.confidence==="medium"?"#1f1a0d":"#1f0d0d",
                            color:d.confidence==="high"?C.green:d.confidence==="medium"?C.amber:C.red,
                            border:`1px solid currentColor`,opacity:0.8,
                          }}>{d.confidence?.toUpperCase()||"?"}</span>
                        </div>
                      </div>

                      {/* Subgoals */}
                      {d.subgoals?.length>0&&(
                        <div style={{marginBottom:10}}>
                          <div style={{fontSize:9,letterSpacing:3,color:C.muted,marginBottom:6}}>SUBGOAL DECOMPOSITION</div>
                          <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
                            {d.subgoals.map((sg,j)=>(
                              <div key={j} style={{fontSize:10,background:C.dim,border:`1px solid ${C.border2}`,borderRadius:4,padding:"4px 9px",color:C.blue}}>
                                {j+1}. {sg}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* State analysis */}
                      <div style={{marginBottom:8}}>
                        <div style={{fontSize:9,letterSpacing:3,color:C.muted,marginBottom:5}}>STATE ANALYSIS</div>
                        <div style={{fontSize:11,color:C.text,lineHeight:1.7,background:C.dim,borderRadius:6,padding:"8px 10px"}}>{d.state_analysis}</div>
                      </div>

                      {/* Recovery */}
                      {d.recovery_plan&&d.recovery_plan!=="N/A"&&(
                        <div style={{marginBottom:8}}>
                          <div style={{fontSize:9,letterSpacing:3,color:C.muted,marginBottom:5}}>RECOVERY PLAN</div>
                          <div style={{fontSize:11,color:C.amber,lineHeight:1.6,background:"#1a1400",borderRadius:6,padding:"8px 10px",border:`1px solid ${C.amber}20`}}>{d.recovery_plan}</div>
                        </div>
                      )}

                      {/* Reasoning depth */}
                      {d.reasoning_depth&&(
                        <div style={{fontSize:10,color:C.muted,marginTop:6}}>
                          <span style={{color:C.purple}}>Reasoning: </span>{d.reasoning_depth}
                        </div>
                      )}
                    </div>
                  );
                })}
                <div ref={logEndRef}/>
              </div>
            )}

            {/* ── TAB: COMPARE ── */}
            {activeTab==="compare"&&(
              <div style={{display:"flex",flexDirection:"column",gap:14}}>
                {/* Scorecard */}
                <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:10}}>
                  {/* Agent */}
                  {agentMetrics&&(
                    <div style={{background:"#0a1e18",border:`2px solid ${C.teal}40`,borderRadius:10,padding:"14px",gridColumn:"1"}}>
                      <div style={{fontSize:9,color:C.teal,letterSpacing:3,marginBottom:8}}>🤖 LLM AGENT</div>
                      {[["Fill Rate",<FillBadge rate={agentMetrics.fillRate}/>],["Stockouts",agentMetrics.stockOuts],["Lost Sales",agentMetrics.lostSales.toLocaleString()],["Write-Offs",agentMetrics.totWriteOff.toLocaleString()]].map(([l,v])=>(
                        <div key={l} style={{display:"flex",justifyContent:"space-between",fontSize:11,marginBottom:5}}>
                          <span style={{color:C.muted}}>{l}</span><span style={{fontWeight:600}}>{v}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {/* Baselines */}
                  {Object.entries(baselineResults).map(([bk,br])=>(
                    <div key={bk} style={{background:C.panel,border:`1px solid ${BASELINES[bk].color}30`,borderRadius:10,padding:"14px"}}>
                      <div style={{fontSize:9,color:BASELINES[bk].color,letterSpacing:3,marginBottom:8}}>{BASELINES[bk].label.toUpperCase()}</div>
                      {[["Fill Rate",<FillBadge rate={br.metrics.fillRate}/>],["Stockouts",br.metrics.stockOuts],["Lost Sales",br.metrics.lostSales.toLocaleString()],["Write-Offs",br.metrics.totWriteOff.toLocaleString()]].map(([l,v])=>(
                        <div key={l} style={{display:"flex",justifyContent:"space-between",fontSize:11,marginBottom:5}}>
                          <span style={{color:C.muted}}>{l}</span><span style={{fontWeight:600}}>{v}</span>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>

                {/* Fill rate comparison chart */}
                {Object.keys(baselineResults).length>0&&(
                  <Panel title="Fill Rate Convergence — Agent vs All Baselines">
                    <div style={{fontSize:10,color:C.muted,marginBottom:10}}>
                      Dashed line = 95% target. The LLM agent ({C.teal}) must beat baselines through structured reasoning, not hard-coded rules.
                    </div>
                    <ResponsiveContainer width="100%" height={220}>
                      <LineChart margin={{top:4,right:8,bottom:0,left:0}}>
                        <XAxis dataKey="day" type="number" domain={[0,CFG.SIM_DAYS]} tick={{fontSize:9,fill:C.muted}}/>
                        <YAxis domain={[0,1]} tickFormatter={v=>`${(v*100).toFixed(0)}%`} tick={{fontSize:9,fill:C.muted}} width={40}/>
                        <ReferenceLine y={0.95} stroke={C.amber} strokeDasharray="5 3" label={{value:"95% target",fontSize:9,fill:C.amber}}/>
                        <Tooltip contentStyle={{background:"#0a0f18",border:`1px solid ${C.border2}`,fontSize:10}} formatter={v=>`${(v*100).toFixed(1)}%`}/>
                        <Legend wrapperStyle={{fontSize:10}}/>
                        {/* Agent line */}
                        <Line data={agentTimelineFillRates} type="monotone" dataKey="agent" stroke={C.teal} strokeWidth={2.5} dot={false} name="LLM Agent"/>
                        {/* Baselines */}
                        {Object.entries(baselineResults).map(([bk,br])=>(
                          <Line key={bk} data={br.timeline.map(t=>({day:t.day,fillRate:t.fillRateCum}))}
                            type="monotone" dataKey="fillRate" stroke={BASELINES[bk].color} strokeWidth={1}
                            strokeDasharray="3 2" dot={false} name={BASELINES[bk].label}/>
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </Panel>
                )}

                {/* ROP decisions overlay */}
                {agentLog.length>0&&(
                  <Panel title="Agent Reorder Point Over Time vs Demand Distribution">
                    <ResponsiveContainer width="100%" height={160}>
                      <AreaChart data={simTimeline} margin={{top:4,right:4,bottom:0,left:0}}>
                        <defs>
                          <linearGradient id="dg" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={C.red} stopOpacity={0.15}/>
                            <stop offset="95%" stopColor={C.red} stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <XAxis dataKey="day" tick={{fontSize:9,fill:C.muted}}/>
                        <YAxis tick={{fontSize:9,fill:C.muted}} width={45}/>
                        <Tooltip contentStyle={{background:"#0a0f18",border:`1px solid ${C.border2}`,fontSize:10}}/>
                        <Area type="monotone" dataKey="demand" stroke={C.red} strokeWidth={1} fill="url(#dg)" dot={false} name="Demand"/>
                        <Line type="monotone" dataKey="rop" stroke={C.amber} strokeWidth={2} dot={false} name="Agent ROP"/>
                      </AreaChart>
                    </ResponsiveContainer>
                  </Panel>
                )}
              </div>
            )}

            {/* ── TAB: MEMORY BANK ── */}
            {activeTab==="memory"&&(
              <div style={{display:"flex",flexDirection:"column",gap:10}}>
                <Panel>
                  <div style={{fontSize:11,color:C.muted,lineHeight:1.8,marginBottom:12}}>
                    The memory bank is a compressed rolling state passed to the LLM on every decision turn. It enables the agent to reason beyond its context window — tracking performance trends, past ROP decisions, and emerging patterns across the full 120-day horizon.
                  </div>
                  <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(200px,1fr))",gap:8}}>
                    {memoryBank.map((m,i)=>(
                      <div key={i} style={{background:C.dim,border:`1px solid ${C.border}`,borderRadius:7,padding:"10px 12px"}}>
                        <div style={{fontSize:10,color:C.teal,fontWeight:600,marginBottom:6}}>Day {m.day}</div>
                        {[
                          ["ROP Set",m.rop],
                          ["Confidence",m.confidence],
                          ["Fill Rate",m.fill_rate?(m.fill_rate+"%"):"—"],
                          ["Stockouts",m.stockouts_in_window],
                        ].map(([l,v])=>(
                          <div key={l} style={{display:"flex",justifyContent:"space-between",fontSize:10,marginBottom:4}}>
                            <span style={{color:C.muted}}>{l}</span>
                            <span style={{color:C.text}}>{v}</span>
                          </div>
                        ))}
                        <div style={{fontSize:9,color:C.muted,marginTop:6,lineHeight:1.5,borderTop:`1px solid ${C.border}`,paddingTop:5}}>
                          {m.key_insight}
                        </div>
                      </div>
                    ))}
                    {memoryBank.length===0&&<div style={{color:C.muted,fontSize:11}}>Memory builds as agent makes decisions...</div>}
                  </div>
                </Panel>
              </div>
            )}
          </>
        )}

        {/* ── FOOTER ── */}
        <div style={{marginTop:28,paddingTop:16,borderTop:`1px solid ${C.border}`,display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12,fontSize:10,color:C.dim}}>
          {[
            ["Environment","Stochastic inventory simulation with 4 demand regimes (Gamma-Poisson, Bimodal HV, Spiking, Stable Gamma). Mirrors real supply-chain uncertainty."],
            ["Agent Architecture","Claude Sonnet 4 called every 5 simulation days. Rolling 6-turn conversation + compressed memory bank enables reasoning beyond context window."],
            ["Reward Structure","Sparse: fill rate signal only meaningful after 50+ days. Agent must plan across 120-day horizon with no per-step guidance."],
            ["Benchmarking","LLM agent compared against 4 rule-based baselines: Base, Safety Stock, Oracle Forecast, Monte Carlo — all from the original Python codebase."],
          ].map(([t,d])=>(
            <div key={t}>
              <div style={{color:C.muted,fontWeight:600,marginBottom:4,fontSize:9,letterSpacing:2}}>{t.toUpperCase()}</div>
              <div style={{lineHeight:1.7}}>{d}</div>
            </div>
          ))}
        </div>
      </div>
      <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}`}</style>
    </div>
  );
}