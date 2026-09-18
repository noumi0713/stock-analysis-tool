(() => {
  const RAW_BASE = "https://raw.githubusercontent.com/noumi0713/stock-analysis-tool/swing-data-120d-latest";
  const REC_BASE = "https://raw.githubusercontent.com/noumi0713/stock-analysis-tool/swipe-decisions/swipe_review/recommendations";
  const $ = (id) => document.getElementById(id);
  const els = {
    loading:$("loading"), loadingText:$("loadingText"), empty:$("emptyState"), deck:$("deck"),
    decisionBar:$("decisionBar"), card:$("card"), date:$("dateLabel"), progress:$("progressLabel"),
    sync:$("syncLabel"), code:$("stockCode"), name:$("stockName"), close:$("closePrice"),
    day:$("dayChange"), rank:$("bbsRank"), five:$("fiveDay"), volume:$("volumeValue"),
    rsi:$("rsiValue"), badge:$("recommendBadge"), yesOverlay:$("yesOverlay"), noOverlay:$("noOverlay"),
    interested:$("interestedCount"), rejected:$("rejectedCount"), remaining:$("remainingCount"),
    undo:$("undoBtn"), reject:$("rejectBtn"), interest:$("interestBtn"), review:$("reviewBtn"),
    summary:$("summaryText"), priceChart:$("priceChart"), volumeChart:$("volumeChart"), rsiChart:$("rsiChart")
  };
  const state = {date:null, all:[], queue:[], decisions:{}, recommendations:new Set(), lastAction:null, dragging:false, startX:0, dx:0, syncTimer:null};
  const yen = new Intl.NumberFormat("ja-JP", {maximumFractionDigits:1});
  const intFmt = new Intl.NumberFormat("ja-JP", {maximumFractionDigits:0});

  function parseCsv(text) {
    const rows=[]; let row=[], field="", quoted=false;
    for(let i=0;i<text.length;i++){
      const c=text[i], n=text[i+1];
      if(c === '"'){
        if(quoted && n === '"'){ field+='"'; i++; } else quoted=!quoted;
      } else if(c === "," && !quoted){ row.push(field); field=""; }
      else if((c === "\n" || c === "\r") && !quoted){
        if(c === "\r" && n === "\n") i++;
        row.push(field); field="";
        if(row.some(v=>v!=="")) rows.push(row);
        row=[];
      } else field+=c;
    }
    if(field || row.length){ row.push(field); rows.push(row); }
    if(rows.length < 2) return [];
    const head=rows[0].map(x=>x.trim());
    return rows.slice(1).map(r=>Object.fromEntries(head.map((h,i)=>[h,(r[i]??"").trim()])));
  }

  async function fetchText(url) {
    const r=await fetch(url,{cache:"no-store"});
    if(!r.ok) throw new Error(`${r.status} ${url}`);
    return r.text();
  }

  async function loadRecommendations(date) {
    try{
      const r=await fetch(`${REC_BASE}/${date}.json?t=${Date.now()}`,{cache:"no-store"});
      if(!r.ok) return new Set();
      const j=await r.json();
      return new Set((j.recommendations||[]).map(x=>String(x.stock_code||x.code||x)));
    }catch{return new Set();}
  }

  function localKey(){ return `stock-swipe:${state.date}`; }
  function readLocal(){ try{return JSON.parse(localStorage.getItem(localKey())||"{}");}catch{return {};} }
  function writeLocal(){ localStorage.setItem(localKey(), JSON.stringify(state.decisions)); }

  async function readRemote() {
    try{
      const r=await fetch(`/api/swipes?date=${encodeURIComponent(state.date)}`,{cache:"no-store"});
      if(!r.ok) return {};
      const j=await r.json();
      const out={};
      for(const d of (j.decisions||[])) if(d.stock_code) out[String(d.stock_code)]=d;
      return out;
    }catch{return {};}
  }

  const num = (v) => Number.isFinite(Number(v)) ? Number(v) : null;
  const pct = (v,d=1) => Number.isFinite(v) ? `${v>=0?"+":""}${v.toFixed(d)}%` : "--";

  async function loadStock(candidate){
    try{
      const text=await fetchText(`${RAW_BASE}/stocks/${candidate.stock_code}.csv?t=${Date.now()}`);
      const rows=parseCsv(text).map(r=>({
        date:r.date, open:num(r.open), high:num(r.high), low:num(r.low), close:num(r.close), volume:num(r.volume)||0
      })).filter(r=>r.close!==null && r.open!==null && r.high!==null && r.low!==null);
      const latest=rows.at(-1), prev=rows.at(-2), fiveAgo=rows.at(-6);
      candidate.series=rows;
      candidate.close=latest?.close ?? num(candidate.price);
      candidate.dayPct=(latest&&prev)?(latest.close/prev.close-1)*100:null;
      candidate.fiveDayPct=(latest&&fiveAgo)?(latest.close/fiveAgo.close-1)*100:null;
      candidate.latestVolume=latest?.volume ?? null;
      candidate.rsi14=computeRsi(rows.map(x=>x.close),14).at(-1) ?? null;
    }catch{
      candidate.series=[]; candidate.close=num(candidate.price); candidate.dayPct=null;
      candidate.fiveDayPct=null; candidate.latestVolume=null; candidate.rsi14=null;
    }
    return candidate;
  }

  async function mapLimit(items, limit, fn){
    let next=0, done=0;
    const workers=Array.from({length:Math.min(limit,items.length)}, async()=>{
      while(true){
        const i=next++; if(i>=items.length) return;
        await fn(items[i],i); done++;
        els.loadingText.textContent=`5日騰落率を計算中… ${done}/${items.length}`;
      }
    });
    await Promise.all(workers);
  }

  function computeRsi(values, period=14){
    const out=Array(values.length).fill(null);
    if(values.length<=period) return out;
    let gain=0, loss=0;
    for(let i=1;i<=period;i++){ const d=values[i]-values[i-1]; if(d>=0) gain+=d; else loss-=d; }
    let ag=gain/period, al=loss/period;
    out[period]=al===0?100:100-(100/(1+ag/al));
    for(let i=period+1;i<values.length;i++){
      const d=values[i]-values[i-1], g=Math.max(d,0), l=Math.max(-d,0);
      ag=(ag*(period-1)+g)/period; al=(al*(period-1)+l)/period;
      out[i]=al===0?100:100-(100/(1+ag/al));
    }
    return out;
  }

  function movingAverage(values, period){
    const out=Array(values.length).fill(null); let sum=0;
    for(let i=0;i<values.length;i++){
      sum+=values[i]; if(i>=period) sum-=values[i-period];
      if(i>=period-1) out[i]=sum/period;
    }
    return out;
  }

  async function init(){
    try{
      const rankingText=await fetchText(`${RAW_BASE}/bbs_ranking_latest.csv?t=${Date.now()}`);
      let ranking=parseCsv(rankingText).map(r=>({...r,rank:Number(r.rank)}))
        .filter(r=>r.rank>=1 && r.rank<=100).sort((a,b)=>a.rank-b.rank);
      if(!ranking.length) throw new Error("ランキングが空です");
      state.date=ranking[0].date; els.date.textContent=state.date;
      state.recommendations=await loadRecommendations(state.date);
      state.decisions={...await readRemote(),...readLocal()}; writeLocal();
      await mapLimit(ranking,8,loadStock);
      ranking.sort((a,b)=>{
        const av=Number.isFinite(a.fiveDayPct)?a.fiveDayPct:-Infinity;
        const bv=Number.isFinite(b.fiveDayPct)?b.fiveDayPct:-Infinity;
        return bv-av || a.rank-b.rank;
      });
      state.all=ranking;
      state.queue=ranking.filter(x=>!state.decisions[String(x.stock_code)]);
      els.loading.classList.add("hidden"); updateCounts();
      if(state.queue.length) showDeck(); else showEmpty();
    }catch(e){
      els.loadingText.textContent=`読み込み失敗: ${e.message}`;
      els.loading.querySelector(".spinner")?.classList.add("hidden");
    }
  }

  function showDeck(){ els.empty.classList.add("hidden"); els.deck.classList.remove("hidden"); els.decisionBar.classList.remove("hidden"); renderCurrent(); }
  function showEmpty(){
    els.deck.classList.add("hidden"); els.decisionBar.classList.add("hidden"); els.empty.classList.remove("hidden");
    const vals=Object.values(state.decisions);
    const a=vals.filter(x=>x.decision==="interested").length, n=vals.filter(x=>x.decision==="rejected").length;
    els.summary.textContent=`興味あり ${a} / 興味なし ${n}`; els.progress.textContent=`${state.all.length}/${state.all.length}`;
  }

  function renderCurrent(){
    const c=state.queue[0]; if(!c){showEmpty();return;} resetCard();
    els.code.textContent=c.stock_code; els.name.textContent=c.stock_name;
    els.close.textContent=c.close==null?"--":yen.format(c.close); els.day.textContent=pct(c.dayPct,2);
    els.day.className=`day-change ${c.dayPct>0?"up":c.dayPct<0?"down":"flat"}`;
    els.rank.textContent=`${c.rank}位`; els.five.textContent=pct(c.fiveDayPct,2);
    els.five.className=c.fiveDayPct>0?"up":c.fiveDayPct<0?"down":"flat";
    els.volume.textContent=c.latestVolume==null?"--":compact(c.latestVolume); els.rsi.textContent=c.rsi14==null?"--":c.rsi14.toFixed(1);
    els.badge.classList.toggle("hidden",!state.recommendations.has(String(c.stock_code)));
    drawAll(c.series||[]);
    const done=state.all.length-state.queue.length; els.progress.textContent=`${Math.min(done+1,state.all.length)} / ${state.all.length}`;
  }

  function compact(v){ if(v>=1e8)return `${(v/1e8).toFixed(1)}億`; if(v>=1e4)return `${(v/1e4).toFixed(1)}万`; return intFmt.format(v); }

  function updateCounts(){
    const vals=Object.values(state.decisions), codes=new Set(state.all.map(s=>String(s.stock_code)));
    els.interested.textContent=vals.filter(x=>codes.has(String(x.stock_code))&&x.decision==="interested").length;
    els.rejected.textContent=vals.filter(x=>codes.has(String(x.stock_code))&&x.decision==="rejected").length;
    els.remaining.textContent=Math.max(0,state.all.length-vals.filter(x=>codes.has(String(x.stock_code))).length);
  }

  function makeRecord(c,decision){
    return {date:state.date,stock_code:String(c.stock_code),stock_name:c.stock_name,bbs_rank:c.rank,
      five_day_return_pct:Number.isFinite(c.fiveDayPct)?Number(c.fiveDayPct.toFixed(4)):null,
      decision,recommended:state.recommendations.has(String(c.stock_code)),recorded_at:new Date().toISOString()};
  }

  function decide(decision){
    const c=state.queue[0]; if(!c)return;
    const key=String(c.stock_code), previous=state.decisions[key]||null;
    state.decisions[key]=makeRecord(c,decision); state.lastAction={candidate:c,previous};
    state.queue.shift(); writeLocal(); updateCounts(); els.undo.disabled=false;
    animateOut(decision,()=>{renderCurrent(); scheduleSync();});
  }

  function undo(){
    if(!state.lastAction)return;
    const {candidate,previous}=state.lastAction,key=String(candidate.stock_code);
    if(previous)state.decisions[key]=previous;else delete state.decisions[key];
    state.queue.unshift(candidate);state.lastAction=null;els.undo.disabled=true;writeLocal();updateCounts();showDeck();renderCurrent();scheduleSync();
  }

  function animateOut(decision,done){
    const dir=decision==="interested"?1:-1; els.card.style.transition="transform .18s ease,opacity .18s ease";
    els.card.style.transform=`translateX(${dir*120}%) rotate(${dir*10}deg)`; els.card.style.opacity="0"; setTimeout(done,190);
  }
  function resetCard(){ els.card.style.transition="none";els.card.style.transform="translateX(0) rotate(0)";els.card.style.opacity="1";els.yesOverlay.style.opacity="0";els.noOverlay.style.opacity="0"; }

  function scheduleSync(){ clearTimeout(state.syncTimer);state.syncTimer=setTimeout(syncNow,900); }
  async function syncNow(){
    setSync("pending","同期中");
    try{
      const r=await fetch("/api/swipes",{method:"POST",headers:{"content-type":"application/json"},
        body:JSON.stringify({date:state.date,decisions:Object.values(state.decisions)})});
      if(!r.ok)throw new Error(String(r.status)); setSync("ok","チャッピー共有済み");
    }catch{ setSync("error","同期待ち"); }
  }
  function setSync(cls,text){els.sync.className=`sync ${cls}`;els.sync.textContent=text;}

  function prep(canvas){const rect=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);return{ctx,w:rect.width,h:rect.height};}
  function drawAll(series){drawPrice(series);drawVolume(series);drawRsi(series);}
  function drawPrice(series){
    const {ctx,w,h}=prep(els.priceChart);ctx.clearRect(0,0,w,h);if(series.length<2){drawNoData(ctx,w,h);return;}
    const start=Math.max(0,series.length-75),d=series.slice(start),vals=series.map(x=>x.close);
    const ma5=movingAverage(vals,5).slice(start),ma25=movingAverage(vals,25).slice(start),ma75=movingAverage(vals,75).slice(start);
    const lo=Math.min(...d.map(x=>x.low)),hi=Math.max(...d.map(x=>x.high)),pad=(hi-lo||1)*.06,ymin=lo-pad,ymax=hi+pad,top=8,bottom=h-16,plotH=bottom-top,step=w/d.length,body=Math.max(2,step*.58),y=v=>top+(ymax-v)/(ymax-ymin)*plotH;
    ctx.strokeStyle="#27303a";ctx.lineWidth=1;for(let i=0;i<4;i++){const yy=top+i*plotH/3;ctx.beginPath();ctx.moveTo(0,yy);ctx.lineTo(w,yy);ctx.stroke();}
    d.forEach((r,i)=>{const x=(i+.5)*step,up=r.close>=r.open;ctx.strokeStyle=up?"#ff675f":"#f1f4f8";ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.moveTo(x,y(r.high));ctx.lineTo(x,y(r.low));ctx.stroke();const yy=Math.min(y(r.open),y(r.close)),hh=Math.max(1,Math.abs(y(r.open)-y(r.close)));ctx.fillRect(x-body/2,yy,body,hh);});
    drawLine(ctx,ma5,step,y,"#ff4fe1",1.5);drawLine(ctx,ma25,step,y,"#56d75f",1.5);drawLine(ctx,ma75,step,y,"#78a9ff",1.2);
    ctx.fillStyle="#8d98a8";ctx.font="10px sans-serif";ctx.textAlign="right";ctx.fillText(yen.format(hi),w-3,12);ctx.fillText(yen.format(lo),w-3,h-4);
  }
  function drawLine(ctx,arr,step,y,color,width){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();let started=false;arr.forEach((v,i)=>{if(v==null)return;const x=(i+.5)*step,yy=y(v);if(!started){ctx.moveTo(x,yy);started=true;}else ctx.lineTo(x,yy);});if(started)ctx.stroke();}
  function drawVolume(series){const {ctx,w,h}=prep(els.volumeChart);ctx.clearRect(0,0,w,h);if(!series.length){drawNoData(ctx,w,h);return;}const d=series.slice(-75),max=Math.max(...d.map(x=>x.volume),1),step=w/d.length,bw=Math.max(1,step*.65);d.forEach((r,i)=>{const bh=(r.volume/max)*(h-8),x=(i+.5)*step;ctx.fillStyle="#f5a623";ctx.fillRect(x-bw/2,h-bh,bw,bh);});}
  function drawRsi(series){const {ctx,w,h}=prep(els.rsiChart);ctx.clearRect(0,0,w,h);if(series.length<16){drawNoData(ctx,w,h);return;}const values=series.map(x=>x.close),r14=computeRsi(values,14).slice(-75),r5=computeRsi(values,5).slice(-75),n=Math.max(r14.length,r5.length),step=w/n,y=v=>h-(v/100)*h;ctx.strokeStyle="#333b46";ctx.lineWidth=1;[30,50,70].forEach(v=>{ctx.beginPath();ctx.moveTo(0,y(v));ctx.lineTo(w,y(v));ctx.stroke();});drawLine(ctx,r14,step,y,"#60e06a",1.5);drawLine(ctx,r5,step,y,"#ff4fe1",1.3);ctx.fillStyle="#7f8998";ctx.font="9px sans-serif";ctx.textAlign="right";ctx.fillText("70",w-2,y(70)-2);ctx.fillText("30",w-2,y(30)-2);}
  function drawNoData(ctx,w,h){ctx.fillStyle="#7d8795";ctx.font="12px sans-serif";ctx.textAlign="center";ctx.fillText("データなし",w/2,h/2);}

  function startDrag(e){if(!state.queue.length)return;state.dragging=true;state.startX=e.touches?e.touches[0].clientX:e.clientX;state.dx=0;els.card.style.transition="none";}
  function moveDrag(e){if(!state.dragging)return;const x=e.touches?e.touches[0].clientX:e.clientX;state.dx=x-state.startX;els.card.style.transform=`translateX(${state.dx}px) rotate(${state.dx/28}deg)`;const a=Math.min(Math.abs(state.dx)/100,1);els.yesOverlay.style.opacity=state.dx>0?String(a):"0";els.noOverlay.style.opacity=state.dx<0?String(a):"0";}
  function endDrag(){if(!state.dragging)return;state.dragging=false;if(state.dx>90)decide("interested");else if(state.dx<-90)decide("rejected");else resetCard();}

  els.card.addEventListener("touchstart",startDrag,{passive:true});els.card.addEventListener("touchmove",moveDrag,{passive:true});els.card.addEventListener("touchend",endDrag);
  els.card.addEventListener("pointerdown",startDrag);window.addEventListener("pointermove",moveDrag);window.addEventListener("pointerup",endDrag);
  els.interest.addEventListener("click",()=>decide("interested"));els.reject.addEventListener("click",()=>decide("rejected"));els.undo.addEventListener("click",undo);
  els.review.addEventListener("click",()=>{state.queue=[...state.all];showDeck();});
  window.addEventListener("keydown",e=>{if(e.key==="ArrowRight")decide("interested");if(e.key==="ArrowLeft")decide("rejected");if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="z")undo();});
  let resizeTimer;window.addEventListener("resize",()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(state.queue[0])drawAll(state.queue[0].series||[])},120);});
  init();
})();