// mc.worker.js (module)

function mulberry32(seed){
  let a = seed >>> 0;
  return function(){
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function boxMuller(rng){
  let u = 0, v = 0;
  while(u === 0) u = rng();
  while(v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function percentile(sortedArr, p){
  const n = sortedArr.length;
  if(n === 0) return NaN;
  const idx = (n - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if(lo === hi) return sortedArr[lo];
  const w = idx - lo;
  return sortedArr[lo] * (1 - w) + sortedArr[hi] * w;
}

function clamp(x, a, b){ return Math.max(a, Math.min(b, x)); }

self.onmessage = (e) => {
  const msg = e.data;
  if(msg.type !== "run") return;

  try{
    const p = msg.params;

    const years = Math.floor(p.years);
    const sims = Math.floor(p.sims);
    const steps = years * 12;
    const dt = 1/12;

    const muAnn = p.mu;
    const sigmaAnn = p.sigma;

    const muNetAnn = muAnn - p.ter - p.bollo;
    const drift = (muNetAnn - 0.5 * sigmaAnn * sigmaAnn) * dt;
    const vol = sigmaAnn * Math.sqrt(dt);

    const rng = mulberry32(p.seed || 1);

    function monthlyContribution(step){
      if(p.monthly <= 0) return 0;
      if(!p.adjInfl) return p.monthly;
      const yearIndex = Math.floor((step-1)/12);
      return p.monthly * Math.pow(1 + p.infl, yearIndex);
    }

    // end-of-year gross values (per sim)
    const endYearGross = Array.from({length: years}, () => new Array(sims));

    const finalNet = new Array(sims);
    const finalGross = new Array(sims);

    // Precompute total contributed (deterministico)
    let contributedTotal = p.initial;
    for(let t=1;t<=steps;t++){
      contributedTotal += monthlyContribution(t);
    }

    // Time-to-goal: primo anno in cui raggiunge obiettivo (0 = mai)
    const firstHitYear = new Array(sims).fill(0);

    for(let s=0; s<sims; s++){
      let w = p.initial;

      for(let t=1;t<=steps;t++){
        w += monthlyContribution(t);
        const z = boxMuller(rng);
        w = w * Math.exp(drift + vol * z);

        if(t % 12 === 0){
          const y = (t/12) - 1;
          endYearGross[y][s] = w;

          // primo anno di raggiungimento
          if(firstHitYear[s] === 0 && w >= p.goal){
            firstHitYear[s] = y + 1; // anni 1..years
          }
        }

        if(s === 0 && t % 240 === 0){
          const pct = Math.round((t/steps) * 12);
          self.postMessage({ type:"progress", pct: clamp(pct,0,95), text:`Simulazione… (${t}/${steps} mesi)` });
        }
      }

      finalGross[s] = w;
      const gain = Math.max(0, w - contributedTotal);
      finalNet[s] = w - p.tax * gain;

      if((s+1) % Math.max(1, Math.floor(sims/20)) === 0){
        const pct = Math.round(((s+1)/sims)*95);
        self.postMessage({ type:"progress", pct, text:`Simulazioni: ${s+1}/${sims}` });
      }
    }

    // final percentiles (net)
    const finalNetSorted = finalNet.slice().sort((a,b)=>a-b);
    const p10 = percentile(finalNetSorted, 0.10);
    const p50 = percentile(finalNetSorted, 0.50);
    const p90 = percentile(finalNetSorted, 0.90);

    // final gross p50 (per stima tasse su p50)
    const finalGrossSorted = finalGross.slice().sort((a,b)=>a-b);
    const p50Gross = percentile(finalGrossSorted, 0.50);
    const taxOnP50 = p.tax * Math.max(0, p50Gross - contributedTotal);

    // prob goal (final net)
    let hit = 0;
    for(const v of finalNet){ if(v >= p.goal) hit++; }
    const probGoal = hit / sims;

    // timeline percentiles (gross)
    const timelineYears = [];
    const timelineP10 = [];
    const timelineP50 = [];
    const timelineP90 = [];
    for(let y=0;y<years;y++){
      const arr = endYearGross[y].slice().sort((a,b)=>a-b);
      timelineYears.push(y+1);
      timelineP10.push(percentile(arr, 0.10));
      timelineP50.push(percentile(arr, 0.50));
      timelineP90.push(percentile(arr, 0.90));
    }

    // === 1) Probabilità obiettivo per anno (gross) ===
    const probGoalByYear = [];
    for(let y=0;y<years;y++){
      let c = 0;
      for(let s=0;s<sims;s++){
        if(endYearGross[y][s] >= p.goal) c++;
      }
      probGoalByYear.push(c / sims);
    }

    // === 2) Time-to-goal histogram (anni + "mai") ===
    const counts = new Array(years).fill(0);
    let never = 0;
    for(const y of firstHitYear){
      if(y === 0) never++;
      else counts[y-1] += 1;
    }
    const ttgLabels = [];
    const ttgCounts = [];
    for(let y=1;y<=years;y++){
      ttgLabels.push(String(y));
      ttgCounts.push(counts[y-1]);
    }
    ttgLabels.push("mai");
    ttgCounts.push(never);

    // stats ttg (solo quelli che raggiungono)
    const hits = firstHitYear.filter(x=>x>0).sort((a,b)=>a-b);
    const ttgMedian = hits.length ? percentile(hits, 0.50) : NaN;
    const neverPct = never / sims;

    const ttgMeta = hits.length
      ? `Mediana: ${Math.round(ttgMedian)} anni • Mai: ${(neverPct*100).toFixed(1)}%`
      : `Mai: ${(neverPct*100).toFixed(1)}%`;

    // histogram (net final)
    const p01 = percentile(finalNetSorted, 0.01);
    const p99 = percentile(finalNetSorted, 0.99);
    const binsCount = 25;
    const width = (p99 - p01) / binsCount || 1;
    const histBins = [];
    const histCounts = new Array(binsCount).fill(0);
    for(let i=0;i<binsCount;i++){
      histBins.push(p01 + i*width);
    }
    for(const v of finalNet){
      const idx = Math.floor((v - p01)/width);
      const j = clamp(idx, 0, binsCount-1);
      histCounts[j] += 1;
    }

    // CSV
    let csv = "sim,final_gross,final_net,contributed_total\n";
    for(let i=0;i<sims;i++){
      csv += `${i+1},${finalGross[i].toFixed(2)},${finalNet[i].toFixed(2)},${contributedTotal.toFixed(2)}\n`;
    }

    const result = {
      p10, p50, p90,
      probGoal,
      timelineYears, timelineP10, timelineP50, timelineP90,
      probGoalByYear,
      ttgLabels, ttgCounts, ttgMeta,
      contributedTotal, p50Gross, taxOnP50,
      histBins, histCounts,
      csv
    };

    self.postMessage({ type:"result", result });
  }catch(err){
    self.postMessage({ type:"error", message: err?.message || String(err), stack: err?.stack });
  }
};
