const state = {
  route: 'dashboard',

  // Data loaded from FastAPI
  payments: [],
  events: [],
  selectedPayment: null,

  // Search / filters
  search: '',
  filter: 'all',

  // Backend metrics
  metrics: {
    atRisk: 0,
    recoverable: 0,
    recovered: 0,
    openFailures: 0,
    successfulRecoveries: 0,
    highConfidence: 0,
    rate: 0
  },

  // ML model information
  modelInfo: null,

  // Merchant controls
  settings: {
    autoRecovery: true,
    notifications: true,
    approval: true,
    learning: true
  },

  // Simulator state
  sim: {
    transactions: 100000,
    avgAmount: 1500,
    failRate: 8,
    recoverRate: 67
  },

  // Loading state
  loading: true
};

function money(value){
  return new Intl.NumberFormat('en-IN',{
    style:'currency',
    currency:'INR',
    maximumFractionDigits:0
  }).format(Number(value) || 0);
}

function initials(value){
  return String(value || '')
    .split(/\s+/)
    .filter(Boolean)
    .map(part => part[0])
    .join('')
    .slice(0,2)
    .toUpperCase();
}


// =========================================================
// BACKEND API CLIENT
// =========================================================

const API = '/api';


async function api(path, options = {}) {

  const response = await fetch(`${API}${path}`, {

    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    },

    ...options
  });


  if (!response.ok) {

    let message = `API error ${response.status}`;

    try {

      const body = await response.json();

      message =
        body.detail ||
        body.message ||
        message;

    } catch (error) {
      // Ignore JSON parsing errors.
    }

    throw new Error(message);
  }


  return response.json();
}



// =========================================================
// LOAD DATA FROM FASTAPI
// =========================================================

async function syncFromBackend() {

  state.loading = true;

  render();


  try {

    const [
      paymentData,
      eventData,
      metrics,
      modelInfo
    ] = await Promise.all([

      api('/payments?limit=5000'),

      api('/events?limit=30'),

      api('/metrics'),

      api('/model/info')

    ]);


    // Payments
    state.payments =
      paymentData.payments || [];


    // Recovery events
    state.events =
      eventData.events || [];


    // Dashboard metrics
    state.metrics =
      metrics || {};


    // ML model metadata
    state.modelInfo =
      modelInfo || null;


    state.loading = false;

    render();


  } catch (error) {

    console.error(
      'RecoverAI backend connection failed:',
      error
    );


    state.loading = false;

    render();


    toast(
      `Backend connection failed: ${error.message}`
    );
  }
}



// =========================================================
// LOADING SCREEN
// =========================================================

function renderLoading() {

  const page =
    document.querySelector('#page');


  if (!page) {
    return;
  }


  page.innerHTML = `

    <div
      style="
        min-height:420px;
        display:grid;
        place-items:center;
      "
    >

      <div
        style="
          text-align:center;
          max-width:420px;
        "
      >

        <div
          class="agent-orb"
          style="
            margin:0 auto 20px;
          "
        >
          ✦
        </div>


        <div
          class="section-title"
          style="
            font-size:18px;
          "
        >
          Connecting to RecoverAI engine
        </div>


        <p
          style="
            color:#7b8798;
            font-size:11px;
            line-height:1.6;
            margin-top:8px;
          "
        >
          Loading payment intelligence,
          ML predictions, recovery metrics
          and agent events…
        </p>


        <div
          style="
            margin:20px auto 0;
            width:180px;
            height:5px;
            border-radius:999px;
            background:#e9eef5;
            overflow:hidden;
          "
        >

          <div
            style="
              width:65%;
              height:100%;
              border-radius:999px;
              background:#53c9a5;
              animation:recoverai-loading 1.2s infinite ease-in-out;
            "
          ></div>

        </div>

      </div>

    </div>
  `;
}



function nav(){
  document.querySelectorAll('[data-route]').forEach(el=>{
    el.addEventListener('click',()=>navigate(el.dataset.route));
  });
}
function navigate(route){
  state.route=route; state.selectedPayment=null;
  renderNav(); render(); window.scrollTo({top:0,behavior:'smooth'});
}
function renderNav(){
  document.querySelectorAll('.nav-item').forEach(el=>el.classList.toggle('active', el.dataset.route===state.route));
  const titles={dashboard:['OVERVIEW','Revenue recovery command center'],payments:['FAILED PAYMENTS','Revenue at risk'],agent:['AI AGENT','Autonomous recovery control'],simulator:['SIMULATOR','Model your recovery upside'],settings:['SETTINGS','Recovery policy & controls']};
  document.querySelector('#page-kicker').textContent=titles[state.route][0];
  document.querySelector('#page-title').textContent=titles[state.route][1];
}
function render(){

  renderNav();

  const page = document.querySelector('#page');

  if(!page){
    return;
  }

  // Show loading screen while backend data is being fetched
  if(state.loading){
    renderLoading();
    return;
  }

  // Render current route
  if(state.route === 'dashboard'){
    page.innerHTML = dashboardPage();
  }

  if(state.route === 'payments'){
    page.innerHTML = paymentsPage();
  }

  if(state.route === 'agent'){
    page.innerHTML = agentPage();
  }

  if(state.route === 'simulator'){
    page.innerHTML = simulatorPage();
  }

  if(state.route === 'settings'){
    page.innerHTML = settingsPage();
  }

  // Attach event handlers for the newly rendered page
  bindPageEvents();
}

function kpis(){

  return {
    atRisk: state.metrics.atRisk || 0,
    recoverable: state.metrics.recoverable || 0,
    recovered: state.metrics.recovered || 0,
    rate: state.metrics.rate || 0
  };
}

function dashboardPage(){
  const k = kpis();

  const top = state.payments
    .filter(p => p.status !== 'Recovered')
    .sort((a,b) => (b.risk - a.risk))
    .slice(0,5);

  const highConfidence = state.payments.filter(
    p => p.status !== 'Recovered' && p.prob >= 80
  ).length;

  const highValue = state.payments.filter(
    p => p.status !== 'Recovered' && p.amount >= 50000
  ).length;

  return `
    <div class="hero">
      <div class="hero-card">
        <div class="eyebrow">AI REVENUE RECOVERY</div>

        <h2>
          Recover the revenue that usually slips through the cracks.
        </h2>

        <p>
          RecoverAI continuously analyzes failed payments, predicts recovery
          probability, checks merchant policies, and selects the safest
          next-best action.
        </p>

        <div class="hero-cta">
          <button data-route="agent">Open AI agent →</button>
          <button class="hero-secondary" data-route="payments">
            Review failed payments
          </button>
        </div>

        <div class="hero-stats">
          <div class="hero-stat">
            <label>Revenue recovered</label>
            <strong>${money(k.recovered)}</strong>
          </div>

          <div class="hero-stat">
            <label>AI recovery rate</label>
            <strong>${k.rate}%</strong>
          </div>
        </div>
      </div>

      <div class="pulse-card">
        <div class="pulse-head">
          <div>
            <div class="section-title">Recovery pulse</div>
            <div class="section-sub">Live opportunity detection</div>
          </div>

          <span class="pill green">● LIVE</span>
        </div>

        <div class="pulse-number">${money(k.recoverable)}</div>
        <div class="pulse-label">estimated recoverable revenue</div>

        <div class="pulse-bar">
          <span style="width:${Math.min(94,k.rate + 22)}%"></span>
        </div>

        <div class="pulse-foot">
          <span>AI opportunity identified</span>
          <strong>${highConfidence} high-confidence cases</strong>
        </div>
      </div>
    </div>

    <div class="section-row">
      <div>
        <div class="section-title">Revenue command center</div>
        <div class="section-sub">
          Monitor the full recovery pipeline from failure to outcome.
        </div>
      </div>
    </div>

    <div class="metric-grid">
      ${metric(
        'Revenue at risk',
        money(k.atRisk),
        '↑ 8.7%',
        'negative',
        '◒'
      )}

      ${metric(
        'Recoverable revenue',
        money(k.recoverable),
        '↑ 12.4%',
        'positive',
        '↗'
      )}

      ${metric(
        'Revenue recovered',
        money(k.recovered),
        '↑ 18.1%',
        'positive',
        '✓'
      )}

      ${metric(
        'Recovery rate',
        k.rate + '%',
        '↑ 6.8 pts',
        'positive',
        '✦'
      )}
    </div>

    <!-- AI PIPELINE -->
    <div class="section-row">
      <div>
        <div class="section-title">AI recovery pipeline</div>
        <div class="section-sub">
          How RecoverAI turns failed payments into recovered revenue.
        </div>
      </div>
    </div>

    <div class="pipeline-grid">
      <div class="pipeline-card">
        <span class="pipeline-icon">01</span>
        <div>
          <strong>Detect</strong>
          <p>${state.payments.length.toLocaleString('en-IN')} payment events analyzed</p>
        </div>
      </div>

      <div class="pipeline-arrow">→</div>

      <div class="pipeline-card">
        <span class="pipeline-icon">02</span>
        <div>
          <strong>Predict</strong>
          <p>${highConfidence} high-confidence recovery opportunities</p>
        </div>
      </div>

      <div class="pipeline-arrow">→</div>

      <div class="pipeline-card">
        <span class="pipeline-icon">03</span>
        <div>
          <strong>Decide</strong>
          <p>${highValue} high-value cases routed through policy checks</p>
        </div>
      </div>

      <div class="pipeline-arrow">→</div>

      <div class="pipeline-card">
        <span class="pipeline-icon">04</span>
        <div>
          <strong>Recover</strong>
          <p>AI executes the safest next-best action</p>
        </div>
      </div>
    </div>

    <!-- LIVE EVENTS -->
    <div class="section-row">
      <div>
        <div class="section-title">Live payment events</div>
        <div class="section-sub">
          Recent activity entering the recovery engine.
        </div>
      </div>

      <button class="link-btn" id="simulate-event-btn">
        + Simulate payment event
      </button>
    </div>

    <div class="card event-card">
      ${liveEvents()}
    </div>

    <!-- CHARTS -->
    <div class="section-row">
      <div>
        <div class="section-title">Recovery intelligence</div>
        <div class="section-sub">
          Patterns across failure reason, customer history and timing.
        </div>
      </div>

      <button class="link-btn" data-route="agent">
        View agent activity →
      </button>
    </div>

    <div class="charts-grid">
      <div class="card card-pad">
        <div class="chart-head">
          <div>
            <div class="chart-title">Revenue recovery trend</div>
            <div class="chart-sub">
              Estimated opportunity vs. recovered revenue
            </div>
          </div>

          <span class="pill green">AI impact</span>
        </div>

        <div class="chart-wrap">
          ${lineChart()}
        </div>

        <div class="legend-row">
          <div class="legend-item">
            <span class="legend-dot" style="background:#2d7ff9"></span>
            At risk
          </div>

          <div class="legend-item">
            <span class="legend-dot" style="background:#53c9a5"></span>
            Recovered
          </div>
        </div>
      </div>

      <div class="card card-pad">
        <div class="chart-head">
          <div>
            <div class="chart-title">Failure mix</div>
            <div class="chart-sub">
              Where recovery opportunities originate
            </div>
          </div>
        </div>

        <div class="chart-wrap">
          ${donutChart()}
        </div>
      </div>
    </div>

    <!-- AI INSIGHT -->
    <div class="ai-insight-card">
      <div class="ai-insight-symbol">✦</div>

      <div class="ai-insight-content">
        <div class="eyebrow">AI OPPORTUNITY DETECTED</div>

        <h3>
          ${highConfidence} payments are strong candidates for automated
          recovery.
        </h3>

        <p>
          RecoverAI is prioritizing transactions with high recovery
          probability, strong customer history and meaningful expected
          recoverable value.
        </p>
      </div>

      <button class="primary-btn" data-route="payments">
        Review opportunities
      </button>
    </div>

    <!-- PRIORITY QUEUE -->
    <div class="section-row">
      <div>
        <div class="section-title">Priority recovery queue</div>
        <div class="section-sub">
          Ranked by expected recoverable value.
        </div>
      </div>

      <button class="link-btn" data-route="payments">
        View all →
      </button>
    </div>

    <div class="card table-card">
      ${paymentTable(top,true)}
    </div>

    <!-- DECISION TRACE -->
    <div class="section-row">
      <div>
        <div class="section-title">How the AI decides</div>
        <div class="section-sub">
          Every recommendation is explainable and policy-aware.
        </div>
      </div>
    </div>

    <div class="decision-trace-card">
      <div class="trace-step">
        <span>01</span>
        <strong>Payment failure detected</strong>
        <small>Gateway emits a failed payment event.</small>
      </div>

      <div class="trace-line"></div>

      <div class="trace-step">
        <span>02</span>
        <strong>Recovery probability scored</strong>
        <small>Failure reason + customer history + amount.</small>
      </div>

      <div class="trace-line"></div>

      <div class="trace-step">
        <span>03</span>
        <strong>Merchant policy checked</strong>
        <small>Retry limits + high-value approval + safeguards.</small>
      </div>

      <div class="trace-line"></div>

      <div class="trace-step">
        <span>04</span>
        <strong>Best action selected</strong>
        <small>Retry, customer action, switch method or escalate.</small>
      </div>
    </div>
  `;
}

function metric(label,value,change,cls,icon){return `<div class="metric-card"><div class="metric-top"><span>${label}</span><span class="metric-icon">${icon}</span></div><div class="metric-value">${value}</div><div class="metric-change ${cls}">${change} vs previous period</div></div>`;}

function lineChart(){
  const w=640,h=210,p=22; const a=[62,70,65,82,74,89,92,86,104,98,112,118], b=[31,36,34,42,46,51,57,55,69,70,78,85];
  const max=125; const pt=arr=>arr.map((v,i)=>`${p+i*(w-2*p)/(arr.length-1)},${h-p-v*(h-2*p)/max}`).join(' ');
  let grid='';[0,25,50,75,100].forEach(v=>{const y=h-p-v*(h-2*p)/max; grid+=`<line x1="${p}" y1="${y}" x2="${w-p}" y2="${y}" stroke="#edf1f5" stroke-width="1"/>`;});
  return `<svg class="chart-svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">${grid}<polyline points="${pt(a)}" fill="none" stroke="#2d7ff9" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/><polyline points="${pt(b)}" fill="none" stroke="#53c9a5" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>${b.map((v,i)=>{const x=p+i*(w-2*p)/(b.length-1),y=h-p-v*(h-2*p)/max;return `<circle cx="${x}" cy="${y}" r="3.2" fill="#53c9a5"/>`;}).join('')}</svg>`;
}
function donutChart(){
  const vals=[28,23,18,14,10,7], total=100; let start=-90, paths=''; const cx=160,cy=100,r=64,ri=43; const cols=['#2d7ff9','#53c9a5','#f59d4b','#8f6ff0','#8da1b9','#d2dce8'];
  function arc(a1,a2){const toRad=a=>a*Math.PI/180; const x1=cx+r*Math.cos(toRad(a1)),y1=cy+r*Math.sin(toRad(a1)); const x2=cx+r*Math.cos(toRad(a2)),y2=cy+r*Math.sin(toRad(a2)); const xi1=cx+ri*Math.cos(toRad(a2)),yi1=cy+ri*Math.sin(toRad(a2)); const xi2=cx+ri*Math.cos(toRad(a1)),yi2=cy+ri*Math.sin(toRad(a1)); const large=a2-a1>180?1:0; return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} L ${xi1} ${yi1} A ${ri} ${ri} 0 ${large} 0 ${xi2} ${yi2} Z`;}
  vals.forEach((v,i)=>{const end=start+v/total*360;paths+=`<path d="${arc(start,end)}" fill="${cols[i]}"/>`;start=end;});
  return `<svg class="chart-svg" viewBox="0 0 320 200"><g transform="translate(0 0)">${paths}</g><text x="160" y="95" text-anchor="middle" font-size="21" font-weight="800" fill="#0d1730">1,284</text><text x="160" y="112" text-anchor="middle" font-size="9" fill="#7a8798">failed payments</text></svg>`;
}

function paymentsPage(){
  const filtered=state.payments.filter(p=>{
    const q=state.search.toLowerCase(); const match=!q || [p.id,p.customer,p.merchant,p.reason].some(v=>String(v).toLowerCase().includes(q));
    const fm=state.filter==='all' || (state.filter==='high' ? p.prob>=80 : state.filter==='action' ? p.status==='Action needed' : p.status===state.filter);
    return match&&fm;
  }).slice(0,30);
  return `<div class="section-row"><div><div class="section-title">Failed payments</div><div class="section-sub">Every failure is scored, explained and assigned a next-best recovery action.</div></div><div class="table-controls"><input class="search" id="payment-search" value="${state.search}" placeholder="Search payment, customer…"/><select class="filter-btn" id="payment-filter"><option value="all">All</option><option value="high">High probability</option><option value="action">Action needed</option><option value="Pending">Pending</option><option value="Recovered">Recovered</option></select></div></div>
  <div class="alert-strip"><span>✦</span><div><strong>${state.payments.filter(p=>p.prob>=80 && p.status!=='Recovered').length} payments</strong> are high-confidence recovery candidates. The agent can safely automate them under your current policy.</div><button class="link-btn" data-route="settings">Review policy →</button></div>
  <div class="card table-card">${paymentTable(filtered,false)}</div>`;
}
function paymentTable(rows,compact){
  return `<table class="table"><thead><tr><th>Payment</th><th>Customer</th><th>Amount</th><th>Failure</th><th>Recovery</th><th>Recommended action</th><th>Status</th></tr></thead><tbody>${rows.map(p=>`<tr data-payment="${p.id}"><td><span class="payment-id">${p.id}</span><div style="font-size:8px;color:#8b97aa;margin-top:3px">${p.method} · ${formatDate(p.timestamp)}</div></td><td><div class="customer"><span class="customer-dot">${initials(p.customer)}</span><div><strong style="font-size:10px">${p.customer}</strong><div style="font-size:8px;color:#8591a3">${p.previous} prior payments</div></div></div></td><td class="amount">${money(p.amount)}</td><td><span class="reason-chip">${p.reason}</span></td><td><span class="prob" style="color:${p.prob>=80?'#1e9b76':p.prob>=60?'#bc761f':'#ba5353'}">${p.prob}%</span></td><td><span class="action-chip">${p.recommended}</span></td><td>${p.status==='Recovered'?'<span class="success-chip">Recovered</span>':'<span class="warn-chip">'+p.status+'</span>'}</td></tr>`).join('')}</tbody></table>${rows.length?`<div style="padding:12px 14px;font-size:9px;color:#8490a4">Showing ${rows.length} priority records · click a row to open AI analysis</div>`:'<div class="empty">No payments match your filters.</div>'}`;
}
function formatDate(ts){const d=new Date(ts); return d.toLocaleDateString('en-IN',{day:'2-digit',month:'short'});}

function agentPage(){

  const metrics = state.metrics || {};

  const pending =
    metrics.openFailures || 0;

  const high =
    metrics.highConfidence || 0;

  const recovered =
    metrics.recovered || 0;

  return `<div class="agent-layout"><div class="card agent-big"><div class="agent-head"><div><div class="section-title">Recovery agent</div><div class="section-sub">Autonomous decisions under merchant policy</div></div><span class="pill green">● LIVE</span></div><div class="agent-orb">✦</div><div class="agent-state"><strong>RecoverAI agent is operating</strong><p>Scanning ${pending.toLocaleString('en-IN')} open failures · ${high} high-confidence actions queued</p></div><div class="agent-kpis">${mini('Open failures',pending.toLocaleString('en-IN'))}${mini('High-confidence',high.toLocaleString('en-IN'))}${mini('Recovered value',money(recovered))}</div><div class="agent-actions"><button class="btn-ghost" id="agent-pause">Pause agent</button><button class="btn-green" id="agent-run">Run recovery cycle</button></div></div><div class="card feed"><div class="section-title">Agent activity</div><div class="section-sub">Most recent decisions and executions</div><div style="margin-top:18px">${feedItem('10:42 AM','High-value opportunity identified','RP20518 · ₹84,000 · 88% recovery probability · gateway timeout.')}${feedItem('10:41 AM','Retry scheduled','RP20492 · Retry window selected for 10:00–11:00 AM based on merchant history.')}${feedItem('10:39 AM','Payment method recovery','RP20601 · Card expired → customer update link prepared.')}${feedItem('10:35 AM','Escalated to merchant','RP20577 · ₹4.5L · above approval threshold, manual review required.')}</div></div></div>
    <div class="section-row"><div><div class="section-title">What the agent considers</div><div class="section-sub">The model combines structured signals before taking action.</div></div></div>
    <div class="metric-grid">${signalCard('Failure reason','Classifies temporary vs. user-actionable failures','Bank decline, timeout, expired card')}${signalCard('Customer history','Weights repeat success and prior recoveries','31 successful payments, 94% success rate')}${signalCard('Transaction value','Escalates unusual or high-value transactions','₹4.5L → manual approval')}${signalCard('Timing','Selects the highest-likelihood recovery window','Tomorrow, 10–11 AM')}</div>`;
}
function mini(l,v){return `<div class="mini-kpi"><label>${l}</label><strong>${v}</strong></div>`}
function feedItem(time,title,copy){return `<div class="feed-item"><span class="feed-dot"></span><div class="feed-time">${time}</div><div class="feed-title">${title}</div><div class="feed-copy">${copy}</div></div>`}
function signalCard(t,h,d){return `<div class="metric-card"><div class="metric-top"><span>${t}</span><span class="pill green">AI</span></div><div style="font-weight:800;font-size:11px;margin-top:10px">${h}</div><div style="font-size:9px;color:#7b8798;margin-top:5px;line-height:1.45">${d}</div></div>`}

function simulatorPage(){
  const s=state.sim; const total=s.transactions*s.avgAmount; const failed=total*s.failRate/100; const recovered=failed*s.recoverRate/100; const current=failed*.18; const upside=Math.max(0,recovered-current);
  return `<div class="section-row"><div><div class="section-title">Recovery simulator</div><div class="section-sub">Model the revenue upside from intelligent recovery before connecting production data.</div></div><span class="pill green">Scenario mode</span></div><div class="sim-grid"><div class="card sim-card"><div class="section-title">Merchant inputs</div><div class="section-sub">Adjust the assumptions to see the impact.</div>${slider('Monthly transactions','transactions',s.transactions,10000,300000,10000,n=>n.toLocaleString('en-IN'))}${slider('Average payment','avgAmount',s.avgAmount,300,10000,100,n=>money(n))}${slider('Payment failure rate','failRate',s.failRate,2,15,.5,n=>n+'%')}${slider('AI recovery rate','recoverRate',s.recoverRate,20,90,1,n=>n+'%')}</div><div class="sim-result"><h3>Estimated monthly upside</h3><div class="sim-big">${money(upside)}</div><div class="sim-label">additional revenue recovered beyond your current baseline</div><div class="sim-bars"><div class="bar-line"><span>Current recovery</span><div class="bar-track"><div class="bar-fill current" style="width:${Math.min(100,current/failed*100)}%"></div></div><strong>${money(current)}</strong></div><div class="bar-line"><span>With RecoverAI</span><div class="bar-track"><div class="bar-fill recover" style="width:${Math.min(100,recovered/failed*100)}%"></div></div><strong>${money(recovered)}</strong></div></div><div class="roi-box"><strong>${money(failed)}</strong><p>revenue entering the failure pool each month · ${s.failRate}% failure rate across ${s.transactions.toLocaleString('en-IN')} payments</p></div></div></div>`;
}
function slider(label,key,val,min,max,step,fmt){return `<div class="form-row"><div class="form-label"><span>${label}</span><strong id="${key}-value">${fmt(val)}</strong></div><input class="slider" type="range" min="${min}" max="${max}" step="${step}" value="${val}" data-sim="${key}"/></div>`}

function settingsPage(){
  return `<div class="settings-grid"><div class="card setting-card"><div class="section-title">Automation policy</div><div class="section-sub">Define what RecoverAI can do without manual approval.</div>${setting('autoRecovery','Automatic recovery','Allow the agent to execute approved retry and payment-method recovery actions.',state.settings.autoRecovery)}${setting('approval','Manual approval for high-value payments','Escalate transactions above ₹1,00,000 instead of acting automatically.',state.settings.approval)}${setting('notifications','Customer notifications','Allow the agent to prepare customer-facing recovery notifications.',state.settings.notifications)}</div><div class="card setting-card"><div class="section-title">Learning & analytics</div><div class="section-sub">Control how outcome data improves future decisions.</div>${setting('learning','Outcome-based learning','Use recovery outcomes to recalibrate the scoring model over time.',state.settings.learning)}<div class="setting-row"><div class="setting-copy"><strong>Recovery threshold</strong><p>Only recommend automation when predicted probability is at least 60%.</p></div><div style="font-size:10px;font-weight:800">60%</div></div><div class="setting-row"><div class="setting-copy"><strong>High-value threshold</strong><p>Transactions above this amount are routed to the merchant for review.</p></div><div style="font-size:10px;font-weight:800">₹1,00,000</div></div><div class="setting-row"><div class="setting-copy"><strong>Max automatic retries</strong><p>Prevent repeated customer attempts from becoming noisy or costly.</p></div><div style="font-size:10px;font-weight:800">3</div></div></div></div><div class="card setting-card" style="margin-top:14px"><div class="section-title">AI guardrails</div><div class="section-sub">The agent is designed to optimize recovery without hiding important decisions.</div><div class="signal-grid" style="margin-top:14px">${signalCard('Safety','Escalate unusual value','High-value payments require approval.')}${signalCard('Explainability','Show every decision','Every action includes the signals behind it.')}${signalCard('Restraint','Stop low-confidence actions','The agent can recommend “do nothing”.')}${signalCard('Audit','Keep recovery history','All decisions and outcomes stay traceable.')}</div></div>`;
}
function setting(key,title,desc,on){return `<div class="setting-row"><div class="setting-copy"><strong>${title}</strong><p>${desc}</p></div><button class="toggle ${on?'on':''}" data-toggle="${key}"><span></span></button></div>`}

function bindPageEvents(){
  document.querySelectorAll('[data-route]').forEach(el=>el.onclick=()=>navigate(el.dataset.route));
  document.querySelectorAll('[data-payment]').forEach(el=>el.onclick=()=>openPayment(el.dataset.payment));
  const search=document.querySelector('#payment-search'); if(search) search.oninput=e=>{state.search=e.target.value;render()};
  const filter=document.querySelector('#payment-filter'); if(filter){filter.value=state.filter;filter.onchange=e=>{state.filter=e.target.value;render()}}
  document.querySelectorAll('[data-toggle]').forEach(btn=>btn.onclick=()=>{const k=btn.dataset.toggle;state.settings[k]=!state.settings[k];render();toast(`${labelFor(k)} ${state.settings[k]?'enabled':'disabled'}.`)});
  document.querySelectorAll('[data-sim]').forEach(input=>input.oninput=e=>{state.sim[e.target.dataset.sim]=Number(e.target.value);render()});
  const run=document.querySelector('#agent-run'); if(run) run.onclick=runAgentCycle;
  const pause=document.querySelector('#agent-pause'); if(pause) pause.onclick=()=>toast('Agent paused for this session. No automated actions will execute.');
  const simulateEvent =
  document.querySelector('#simulate-event-btn');

if(simulateEvent){

  simulateEvent.onclick = simulatePayment;
}
}
function labelFor(k){return ({autoRecovery:'Automatic recovery',approval:'High-value approval',notifications:'Customer notifications',learning:'Outcome learning'})[k]}
async function simulatePayment(){

  const button =
    document.querySelector('#simulate-failure-btn');

  if(button){
    button.disabled = true;
    button.textContent = 'Creating payment…';
  }

  try{

    const payment = await api('/payments/simulate', {
      method: 'POST',
      body: JSON.stringify({})
    });

    await syncFromBackend();

    toast(
      `Payment ${payment.id} failed — AI assigned ${payment.prob}% recovery probability.`
    );

  }catch(error){

    console.error(error);

    toast(
      `Simulation failed: ${error.message}`
    );

  }finally{

    const current =
      document.querySelector('#simulate-failure-btn');

    if(current){
      current.disabled = false;
      current.textContent = '+ Simulate failed payment';
    }

  }
}
async function runAgentCycle(){

  const button = document.querySelector('#agent-run');

  if(button){
    button.disabled = true;
    button.textContent = 'AI agent is working…';
  }

  toast(
    'Recovery engine is evaluating opportunities…'
  );

  try {

    const result = await api('/agent/cycle', {
      method: 'POST'
    });

    await syncFromBackend();

    toast(
      `Cycle complete — ${result.recovered} recovered, ${result.failed} failed, ${result.escalated} escalated.`
    );

  } catch(error){

    console.error(error);

    toast(
      `Recovery cycle failed: ${error.message}`
    );

    const current = document.querySelector('#agent-run');

    if(current){
      current.disabled = false;
      current.textContent = 'Run recovery cycle';
    }
  }
}
function openPayment(id){
  const p=state.payments.find(x=>x.id===id); if(!p) return; state.selectedPayment=p;
  document.querySelector('#modal-root').innerHTML=`<div class="modal-backdrop"><div class="modal"><div class="modal-head"><div><div class="eyebrow" style="color:#3b7bd6">AI PAYMENT ANALYSIS</div><h3>${p.id} · ${money(p.amount)}</h3><p>${p.customer} · ${p.method} · ${p.reason}</p></div><button class="modal-close" id="close-modal">✕</button></div><div class="modal-body"><div class="score-panel"><div class="score-ring"><span>${p.prob}%</span></div><div><div class="section-title">Recovery probability</div><div class="section-sub">Based on failure class, customer history, retry behavior and transaction value.</div><div class="signal-grid" style="margin-top:11px">${signal('Failure class',p.reason)}${signal('Customer success',Math.round(p.successRate*100)+'%')}${signal('Prior payments',p.previous)}${signal('Suggested timing',p.reason.includes('timeout')?'Now':'2 hours')}</div></div></div><div class="ai-box"><h4>✦ AI reasoning</h4><p>${reasoning(p)}</p></div><div class="ai-action"><div><strong>Recommended action: ${p.recommended}</strong><span>Expected recovered value: ${money(p.risk)} · Confidence: ${p.prob>=80?'High':p.prob>=60?'Medium':'Low'}</span></div><button class="primary-btn" id="modal-act">Execute action</button></div></div><div class="modal-footer"><button class="outline-btn" id="modal-dismiss">Close</button><button class="primary-btn" id="modal-escalate">Escalate to merchant</button></div></div></div>`;
  document.querySelector('#close-modal').onclick=closeModal;document.querySelector('#modal-dismiss').onclick=closeModal;
  document.querySelector('#modal-act').onclick = async () => {

  const button =
    document.querySelector('#modal-act');

  if(button){
    button.disabled = true;
    button.textContent = 'Executing…';
  }

  try {

    const result = await api(
      `/payments/${p.id}/action`,
      {
        method: 'POST',
        body: JSON.stringify({
          action: p.recommended
        })
      }
    );

    closeModal();

    await syncFromBackend();

    if(result.result === 'recovered'){

      toast(
        `${p.id} recovered — ${money(p.amount)} added to recovered revenue.`
      );

    } else if(result.result === 'escalated'){

      toast(
        `${p.id} escalated for merchant approval.`
      );

    } else {

      toast(
        `${p.id}: no automatic recovery executed.`
      );
    }

  } catch(error){

    console.error(error);

    toast(
      `Recovery action failed: ${error.message}`
    );

    if(button){
      button.disabled = false;
      button.textContent = 'Execute action';
    }
  }
};

  document.querySelector('#modal-escalate').onclick=()=>{closeModal();toast(`${p.id} escalated for manual review.`)};
}
function signal(l,v){return `<div class="signal"><label>${l}</label><strong>${v}</strong></div>`}
function reasoning(p){
  if(p.reason==='Card expired') return `The customer has a strong prior payment history (${p.successes} successful attempts). The failure is actionable rather than a credit issue, so a payment-method update has the highest expected recovery value.`;
  if(p.reason.includes('timeout')||p.reason.includes('Network')) return `The failure looks transient. Similar transactions recover at a high rate after a short delay, and this customer has a ${Math.round(p.successRate*100)}% historical payment success rate. The agent favors an immediate or delayed retry.`;
  if(p.amount>90000) return `This transaction has meaningful financial impact. Even with a high recovery probability, the amount is above the configured approval threshold, so the agent recommends merchant review before execution.`;
  return `The customer has ${p.previous} prior payment attempts with ${Math.round(p.successRate*100)}% success. Historical outcomes suggest a ${p.prob}% chance of recovery, making ${p.recommended.toLowerCase()} the best next action.`;
}
function closeModal(){document.querySelector('#modal-root').innerHTML=''}
function toast(msg){const root=document.querySelector('#toast-root');root.innerHTML=`<div class="toast"><strong>RecoverAI</strong> · ${msg}</div>`;setTimeout(()=>root.innerHTML='',3500)}


function liveEvents(){

  if(!state.events.length){

    return `
      <div class="empty">
        No recent recovery events.
      </div>
    `;
  }

  return state.events.map(event => {

    const icons = {
      failure: '!',
      analysis: '✦',
      recovery: '↗',
      success: '✓',
      escalation: '!',
      no_action: '—'
    };

    return `
      <div class="event-row">

        <div class="event-time">
          ${event.time}
        </div>

        <div class="event-icon event-${event.type}">
          ${icons[event.type] || '•'}
        </div>

        <div class="event-copy">
          <strong>${event.title}</strong>

          <span>
            ${event.detail}
          </span>
        </div>

        <div class="event-amount">
          ${money(event.amount || 0)}
        </div>

        <button
          class="event-view"
          onclick="openEventTrace(
            '${event.type}',
            '${String(event.detail).replaceAll("'", "\\'")}',
            ${event.amount || 0}
          )"
        >
          Inspect
        </button>

      </div>
    `;
  }).join('');
}


function openEventTrace(type, detail, amount){

  const probability =
    type === 'success'
      ? 96
      : type === 'analysis'
      ? 91
      : type === 'recovery'
      ? 88
      : type === 'escalation'
      ? 63
      : 84;

  const action =
    type === 'success'
      ? 'Recovery completed'
      : type === 'escalation'
      ? 'Escalate to merchant'
      : type === 'recovery'
      ? 'Smart retry'
      : probability >= 85
      ? 'Retry after 2 hours'
      : 'Request customer action';

  document.querySelector('#modal-root').innerHTML = `

    <div class="modal-backdrop">

      <div class="modal decision-modal">

        <div class="modal-head">

          <div>

            <div class="eyebrow" style="color:#3b7bd6">
              AI DECISION TRACE
            </div>

            <h3>
              ${detail}
            </h3>

            <p>
              Transaction value · ${money(amount)}
            </p>

          </div>

          <button
            class="modal-close"
            id="close-event-modal"
          >
            ✕
          </button>

        </div>


        <div class="modal-body">

          <div class="trace-score">

            <div class="trace-score-ring">
              <span>${probability}%</span>
            </div>

            <div>

              <div class="section-title">
                Recovery probability
              </div>

              <div class="section-sub">
                AI confidence score for successful recovery.
              </div>

            </div>

          </div>


          <div class="ai-box">

            <h4>✦ AI reasoning</h4>

            <p>
              The recovery engine evaluated failure type, customer payment
              history, transaction value, retry history and merchant policy.
              The model estimates a
              <strong>${probability}%</strong>
              probability of successful recovery.
            </p>

          </div>


          <div class="decision-flow">

            <div class="decision-flow-item">
              <span>1</span>
              <strong>Detect</strong>
              <small>Payment event received</small>
            </div>

            <div class="decision-flow-item">
              <span>2</span>
              <strong>Score</strong>
              <small>${probability}% probability</small>
            </div>

            <div class="decision-flow-item">
              <span>3</span>
              <strong>Guardrail</strong>
              <small>
                ${amount > 100000
                  ? 'Manual approval required'
                  : 'Policy passed'}
              </small>
            </div>

            <div class="decision-flow-item">
              <span>4</span>
              <strong>Action</strong>
              <small>${action}</small>
            </div>

          </div>


          <div class="ai-action">

            <div>

              <strong>
                Recommended action: ${action}
              </strong>

              <span>
                Expected recoverable value:
                ${money(amount * probability / 100)}
              </span>

            </div>

            <button
              class="primary-btn"
              id="execute-trace-action"
            >
              Execute action
            </button>

          </div>

        </div>


        <div class="modal-footer">

          <button
            class="outline-btn"
            id="dismiss-event-modal"
          >
            Close
          </button>

        </div>

      </div>

    </div>
  `;

  document.querySelector('#close-event-modal').onclick = closeModal;
  document.querySelector('#dismiss-event-modal').onclick = closeModal;

  document.querySelector('#execute-trace-action').onclick = () => {

    closeModal();

    toast(
      `AI action executed — ${action} for ${money(amount)} payment opportunity.`
    );

  };
}

// Global interactions

nav();

document.querySelector(
  '#simulate-failure-btn'
).onclick = simulatePayment;

render();

syncFromBackend();
