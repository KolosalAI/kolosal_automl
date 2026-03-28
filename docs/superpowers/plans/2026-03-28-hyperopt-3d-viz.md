# HyperOpt 3D Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an interactive Plotly.js `scatter3d` chart to the HyperOpt panel showing the optimization search space (two params on X/Y, score on Z) with the best trial highlighted as a gold star.

**Architecture:** All changes are in a single file (`src/server/handlers.rs`) which embeds the full HTML/JS UI as a string. The Rust backend fix corrects how trial params are serialized to JSON. The frontend additions load Plotly from CDN, inject a container div, and extend `eHoCharts()` to render the 3D scatter after optimization completes.

**Tech Stack:** Rust (serde_json), Plotly.js 2.27.0 (CDN), HTML5 Canvas (existing 2D charts unchanged)

---

### Task 1: Fix trial params serialization in the backend

**Files:**
- Modify: `src/server/handlers.rs:5355-5378`

The `params` field on both the `trials` array and `best_info` is currently a Rust debug string (`format!("{:?}", t.params)`). This task replaces both occurrences with proper JSON objects so the frontend can read param names and values.

`ParameterValue` is already re-exported from `crate::optimizer` (see `src/optimizer/mod.rs` line 21) and the `run_hyperopt` function already has `use crate::optimizer::{HyperOptX, OptimizationConfig, SamplerType, OptimizeDirection};` at its top — add `ParameterValue` to that import.

- [ ] **Step 1: Add a failing test for params serialization**

Add a `#[cfg(test)]` block at the very end of `src/server/handlers.rs` (after all existing code):

```rust
#[cfg(test)]
mod tests {
    use crate::optimizer::{ParameterValue, TrialParams};
    use std::collections::HashMap;

    fn params_to_json(params: &TrialParams) -> serde_json::Map<String, serde_json::Value> {
        params.iter()
            .filter_map(|(k, v)| {
                let json_val = match v {
                    ParameterValue::Float(f) => Some(serde_json::Value::from(*f)),
                    ParameterValue::Int(i)   => Some(serde_json::Value::from(*i)),
                    _                        => None,
                };
                json_val.map(|j| (k.clone(), j))
            })
            .collect()
    }

    #[test]
    fn test_params_to_json_numeric() {
        let mut p: TrialParams = HashMap::new();
        p.insert("learning_rate".to_string(), ParameterValue::Float(0.05));
        p.insert("max_depth".to_string(), ParameterValue::Int(6));
        let j = params_to_json(&p);
        assert_eq!(j["learning_rate"], serde_json::json!(0.05));
        assert_eq!(j["max_depth"], serde_json::json!(6i64));
    }

    #[test]
    fn test_params_to_json_skips_non_numeric() {
        let mut p: TrialParams = HashMap::new();
        p.insert("mode".to_string(), ParameterValue::String("auto".to_string()));
        p.insert("n_estimators".to_string(), ParameterValue::Int(100));
        let j = params_to_json(&p);
        assert!(!j.contains_key("mode"));
        assert_eq!(j["n_estimators"], serde_json::json!(100i64));
    }
}
```

- [ ] **Step 2: Run the test — expect compile failure (function not in scope yet)**

```bash
cargo test test_params_to_json -- --nocapture 2>&1 | head -30
```

Expected: compile error because `params_to_json` is a local test helper — it will compile and pass once we wire it in. Actually it should compile fine since it's self-contained. If it fails with "unresolved import", ensure `TrialParams` and `ParameterValue` are public (they are — `mod.rs` line 21 re-exports both).

Expected output: `test tests::test_params_to_json_numeric ... ok` and `test tests::test_params_to_json_skips_non_numeric ... ok`

- [ ] **Step 3: Fix the `trials` mapping (~line 5355)**

Find this block in `src/server/handlers.rs`:

```rust
                    let trials: Vec<serde_json::Value> = study.trials.iter().map(|t| {
                        serde_json::json!({
                            "trial_id": t.trial_id,
                            "params": format!("{:?}", t.params),
                            "value": t.value,
                            "duration_secs": t.duration_secs,
                            "pruned": t.pruned,
                        })
                    }).collect();
```

Also find `use crate::optimizer::{HyperOptX, OptimizationConfig, SamplerType, OptimizeDirection};` inside the `run_hyperopt` function body and add `ParameterValue` to it.

Replace both with:

```rust
                    use crate::optimizer::{HyperOptX, OptimizationConfig, SamplerType, OptimizeDirection, ParameterValue};
```

```rust
                    let trials: Vec<serde_json::Value> = study.trials.iter().map(|t| {
                        let params_json: serde_json::Map<String, serde_json::Value> = t.params.iter()
                            .filter_map(|(k, v)| {
                                let jv = match v {
                                    ParameterValue::Float(f) => Some(serde_json::Value::from(*f)),
                                    ParameterValue::Int(i)   => Some(serde_json::Value::from(*i)),
                                    _                        => None,
                                };
                                jv.map(|j| (k.clone(), j))
                            })
                            .collect();
                        serde_json::json!({
                            "trial_id": t.trial_id,
                            "params": params_json,
                            "value": t.value,
                            "duration_secs": t.duration_secs,
                            "pruned": t.pruned,
                        })
                    }).collect();
```

- [ ] **Step 4: Fix the `best_info` mapping (~line 5373)**

Find:

```rust
                let best_info = best_trial.map(|t| serde_json::json!({
                    "trial_id": t.trial_id,
                    "params": format!("{:?}", t.params),
                    "value": t.value,
                    "duration_secs": t.duration_secs,
                }));
```

Replace with:

```rust
                let best_info = best_trial.map(|t| {
                    let params_json: serde_json::Map<String, serde_json::Value> = t.params.iter()
                        .filter_map(|(k, v)| {
                            let jv = match v {
                                ParameterValue::Float(f) => Some(serde_json::Value::from(*f)),
                                ParameterValue::Int(i)   => Some(serde_json::Value::from(*i)),
                                _                        => None,
                            };
                            jv.map(|j| (k.clone(), j))
                        })
                        .collect();
                    serde_json::json!({
                        "trial_id": t.trial_id,
                        "params": params_json,
                        "value": t.value,
                        "duration_secs": t.duration_secs,
                    })
                });
```

- [ ] **Step 5: Verify tests still pass and the project compiles**

```bash
cargo test test_params_to_json -- --nocapture 2>&1 | tail -10
cargo build 2>&1 | tail -20
```

Expected: both tests pass, build succeeds with no errors.

- [ ] **Step 6: Commit**

```bash
git add src/server/handlers.rs
git commit -m "fix(hyperopt): serialize trial params as JSON object instead of debug string"
```

---

### Task 2: Load Plotly.js and add the 3D chart container

**Files:**
- Modify: `src/server/handlers.rs:1840` (before `</head>`)
- Modify: `src/server/handlers.rs:2123-2134` (`#e-ho-charts` div)

- [ ] **Step 1: Add Plotly CDN script tag**

Find the line in `handlers.rs` (around line 1840):

```html
    </style>
</head>
```

Replace with:

```html
    </style>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" defer></script>
</head>
```

- [ ] **Step 2: Add the 3D chart container div**

Find the closing tag of `#e-ho-charts`:

```html
                            <div id="e-ho-charts" style="display:none;margin-top:16px">
                                <div class="grid-2">
                                    <div class="card">
                                        <p style="font-size:14px;font-weight:500;margin-bottom:12px;color:#6a6f73">Optimization Convergence</p>
                                        <canvas id="e-ho-cvg" width="520" height="260" style="width:100%;border-radius:8px"></canvas>
                                    </div>
                                    <div class="card">
                                        <p style="font-size:14px;font-weight:500;margin-bottom:12px;color:#6a6f73">Best Score Progress</p>
                                        <canvas id="e-ho-best" width="520" height="260" style="width:100%;border-radius:8px"></canvas>
                                    </div>
                                </div>
                            </div>
```

Replace with:

```html
                            <div id="e-ho-charts" style="display:none;margin-top:16px">
                                <div class="grid-2">
                                    <div class="card">
                                        <p style="font-size:14px;font-weight:500;margin-bottom:12px;color:#6a6f73">Optimization Convergence</p>
                                        <canvas id="e-ho-cvg" width="520" height="260" style="width:100%;border-radius:8px"></canvas>
                                    </div>
                                    <div class="card">
                                        <p style="font-size:14px;font-weight:500;margin-bottom:12px;color:#6a6f73">Best Score Progress</p>
                                        <canvas id="e-ho-best" width="520" height="260" style="width:100%;border-radius:8px"></canvas>
                                    </div>
                                </div>
                                <div id="e-ho-3d-wrap" style="margin-top:16px">
                                    <div class="card">
                                        <p style="font-size:14px;font-weight:500;margin-bottom:4px;color:#6a6f73">Search Space <span id="e-ho-3d-label" style="font-weight:400;font-size:12px">(3D)</span></p>
                                        <div id="e-ho-3d" style="width:100%;height:380px"></div>
                                    </div>
                                </div>
                            </div>
```

- [ ] **Step 3: Build to confirm no syntax errors**

```bash
cargo build 2>&1 | tail -10
```

Expected: builds cleanly.

- [ ] **Step 4: Commit**

```bash
git add src/server/handlers.rs
git commit -m "feat(hyperopt): add Plotly CDN and 3D chart container div"
```

---

### Task 3: Implement the 3D scatter chart in `eHoCharts()`

**Files:**
- Modify: `src/server/handlers.rs:2505-2511` (`eHoCharts` function)

The existing function is:

```javascript
    function eHoCharts(trials){
        $('e-ho-charts').style.display='block';
        var pts=[],bestPts=[],bestVal=Infinity;
        for(var i=0;i<trials.length;i++){var v=trials[i].value;if(v===Infinity||v==null)continue;pts.push({x:trials[i].trial_id,y:v});if(v<bestVal)bestVal=v;bestPts.push({x:trials[i].trial_id,y:bestVal})}
        eDrawLine('e-ho-cvg',pts,{color:'#8b5cf6',label:'Trial #'});
        eDrawLine('e-ho-best',bestPts,{color:'#3abc3f',label:'Trial #'});
    }
```

This task replaces it with a version that also renders the Plotly 3D (or 2D fallback) chart.

- [ ] **Step 1: Replace `eHoCharts` with the extended version**

Find the exact function text above and replace with:

```javascript
    function eHoCharts(trials){
        $('e-ho-charts').style.display='block';
        var pts=[],bestPts=[],bestVal=Infinity;
        for(var i=0;i<trials.length;i++){var v=trials[i].value;if(v===Infinity||v==null)continue;pts.push({x:trials[i].trial_id,y:v});if(v<bestVal)bestVal=v;bestPts.push({x:trials[i].trial_id,y:bestVal})}
        eDrawLine('e-ho-cvg',pts,{color:'#8b5cf6',label:'Trial #'});
        eDrawLine('e-ho-best',bestPts,{color:'#3abc3f',label:'Trial #'});
        eHo3dChart(trials);
    }
    function eHo3dChart(trials){
        if(typeof Plotly==='undefined')return;
        var valid=trials.filter(function(t){return t.value!==Infinity&&t.value!=null&&t.params&&typeof t.params==='object'});
        if(!valid.length)return;
        var paramKeys=Object.keys(valid[0].params).filter(function(k){return typeof valid[0].params[k]==='number'});
        if(!paramKeys.length)return;
        var bestIdx=0;for(var i=1;i<valid.length;i++){if(valid[i].value<valid[bestIdx].value)bestIdx=i;}
        var regular=valid.filter(function(_,i){return i!==bestIdx});
        var best=valid[bestIdx];
        var layout={
            paper_bgcolor:'#0f0f1a',plot_bgcolor:'#0f0f1a',
            font:{color:'#c9cdd1',size:11},
            margin:{l:40,r:20,t:20,b:40},
            showlegend:false,
        };
        var container=$('e-ho-3d');
        if(paramKeys.length===1){
            // 1-param fallback: 2D scatter
            var xKey=paramKeys[0];
            $('e-ho-3d-label').textContent='(2D — only one param)';
            var trace={
                x:regular.map(function(t){return t.params[xKey]}),
                y:regular.map(function(t){return t.value}),
                mode:'markers',type:'scatter',
                marker:{color:regular.map(function(t){return t.value}),colorscale:'RdYlGn',reversescale:false,size:8,opacity:0.85,
                    colorbar:{title:{text:'score',font:{size:10}},thickness:12,len:0.6}},
                customdata:regular.map(function(t){return t.trial_id}),
                hovertemplate:'Trial #%{customdata}<br>'+xKey+': %{x:.4f}<br>Score: %{y:.4f}<extra></extra>',
            };
            var bestTrace={
                x:[best.params[xKey]],y:[best.value],mode:'markers+text',type:'scatter',
                marker:{symbol:'star',size:16,color:'#ffd700',line:{color:'#fff',width:1}},
                text:['BEST'],textposition:'top right',textfont:{size:10,color:'#ffd700'},
                customdata:[best.trial_id],
                hovertemplate:'Trial #%{customdata} — BEST<br>'+xKey+': %{x:.4f}<br>Score: %{y:.4f}<extra></extra>',
            };
            layout.xaxis={title:{text:xKey},gridcolor:'#2a2a3a',zerolinecolor:'#444'};
            layout.yaxis={title:{text:'score'},gridcolor:'#2a2a3a',zerolinecolor:'#444'};
            Plotly.react(container,[trace,bestTrace],layout,{responsive:true,displayModeBar:false});
        } else {
            // 2+ params: full 3D scatter
            var xKey=paramKeys[0],yKey=paramKeys[1];
            $('e-ho-3d-label').textContent='(X: '+xKey+' · Y: '+yKey+' · Z: score)';
            var trace={
                x:regular.map(function(t){return t.params[xKey]}),
                y:regular.map(function(t){return t.params[yKey]}),
                z:regular.map(function(t){return t.value}),
                mode:'markers',type:'scatter3d',
                marker:{color:regular.map(function(t){return t.value}),colorscale:'RdYlGn',reversescale:false,size:5,opacity:0.85,
                    colorbar:{title:{text:'score',font:{size:10}},thickness:12,len:0.5}},
                customdata:regular.map(function(t){return t.trial_id}),
                hovertemplate:'Trial #%{customdata}<br>'+xKey+': %{x:.4f}<br>'+yKey+': %{y:.4f}<br>Score: %{z:.4f}<extra></extra>',
            };
            var bestTrace={
                x:[best.params[xKey]],y:[best.params[yKey]],z:[best.value],
                mode:'markers+text',type:'scatter3d',
                marker:{symbol:'diamond',size:8,color:'#ffd700',line:{color:'#fff',width:1}},
                text:['BEST'],textposition:'top right',textfont:{size:10,color:'#ffd700'},
                customdata:[best.trial_id],
                hovertemplate:'Trial #%{customdata} — BEST<br>'+xKey+': %{x:.4f}<br>'+yKey+': %{y:.4f}<br>Score: %{z:.4f}<extra></extra>',
            };
            layout.scene={
                xaxis:{title:{text:xKey},gridcolor:'#2a2a3a',backgroundcolor:'#0f0f1a',zerolinecolor:'#444'},
                yaxis:{title:{text:yKey},gridcolor:'#2a2a3a',backgroundcolor:'#0f0f1a',zerolinecolor:'#444'},
                zaxis:{title:{text:'score'},gridcolor:'#2a2a3a',backgroundcolor:'#0f0f1a',zerolinecolor:'#444'},
                bgcolor:'#0f0f1a',
            };
            Plotly.react(container,[trace,bestTrace],layout,{responsive:true,displayModeBar:false});
        }
    }
```

- [ ] **Step 2: Build to confirm no Rust compile errors**

```bash
cargo build 2>&1 | tail -10
```

Expected: builds cleanly. (The JavaScript is just a string literal — Rust won't validate JS syntax, so any JS errors will only appear in the browser.)

- [ ] **Step 3: Manual smoke test in the browser**

Start the server, load the UI, run a HyperOpt job (e.g. Random Forest, 10 trials). After completion:

1. The two existing 2D canvas charts appear (convergence and best score progress) — unchanged.
2. A new "Search Space (3D)" card appears below with a dark-background Plotly scatter.
3. X axis = `n_estimators`, Y axis = `max_depth`, Z axis = `score` (for Random Forest which has 2 params).
4. Hovering a point shows the tooltip with trial #, param values, score.
5. Dragging rotates the scene; scroll zooms.
6. The best trial is shown as a larger gold diamond labeled "BEST".

Run with KNN (1 param `n_neighbors`) and verify the label reads "(2D — only one param)" and a 2D scatter renders instead.

```bash
cargo run 2>&1 &
# Open http://localhost:3000 (or whatever port is configured)
```

- [ ] **Step 4: Commit**

```bash
git add src/server/handlers.rs
git commit -m "feat(hyperopt): add Plotly 3D search space visualization with best trial highlight"
```
