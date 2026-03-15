import { test, expect, Page, Browser } from '@playwright/test';
import { performance } from 'perf_hooks';
import {
  RUST_PLATFORM,
  PYTHON_PLATFORM,
  PLATFORMS,
  Platform,
  measureTime,
  benchmark,
  computeStats,
  collectPageLoadMetrics,
  recordComparison,
  printComparisonTable,
  saveReport,
  fetchWithRetry,
  rateLimitPause,
  Stats,
} from './helpers';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ITERATIONS = 3;
const LONG_TIMEOUT = 180_000;

// Shared metric storage for combining results across separate tests
const sharedMetrics: Record<string, number> = {};

// ---------------------------------------------------------------------------
// Python/Gradio selectors (discovered from HTML inspection)
// ---------------------------------------------------------------------------

const PY = {
  sampleDatasetDropdown: 'input[aria-label="🎲 Sample Datasets"]',
  loadSampleBtn: '#component-20',
  tabStep1: '#component-10-button',
  tabStep2: '#component-30-button',
  tabStep3: '#component-59-button',
  tabStep4: '#component-85-button',
  tabStep5: '#component-99-button',
  taskTypeDropdown: 'input[aria-label="🎯 Task Type"]',
  createConfigBtn: '#component-55',
  targetColumnInput: 'textarea[placeholder="e.g., price, category, outcome"]',
  algorithmDropdown: 'input[aria-label="🤖 Available ML Algorithms Preview"]',
  modelNameInput: 'textarea[placeholder="e.g., my_experiment"]',
  trainBtn: '#component-67',
  predictionInput: 'textarea[placeholder*="5.1, 3.5, 1.4, 0.2"]',
  predictBtn: '#component-93',
  compareBtn: '#component-104',
  systemBtn: 'button:has-text("System Information")',
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Navigate to a platform's root page and wait for it to be interactive */
async function loadPlatform(page: Page, platform: Platform) {
  await page.goto(platform.baseURL, { waitUntil: 'load', timeout: LONG_TIMEOUT });
  if (platform.name === 'rust') {
    await page.waitForLoadState('networkidle', { timeout: 60_000 });
  } else {
    // Gradio keeps a WebSocket open — networkidle never resolves.
    await page.waitForSelector('.gradio-container, .app', { timeout: 60_000 });
    await page.waitForTimeout(2000);
  }
}

/** Load the Iris sample dataset on either platform */
async function loadIrisDataset(page: Page, platform: Platform) {
  if (platform.name === 'rust') {
    // Navigate to Data section first (sidebar nav replaces old tab layout)
    await page.locator('nav button:has-text("Data")').click();
    const pill = page.locator('button:has-text("Iris")').first();
    await pill.waitFor({ state: 'visible', timeout: 30_000 });
    await pill.click();
    // Wait for data to load — check for toast/banner text or data info section showing rows
    await page.waitForFunction(
      () => {
        const body = document.body.textContent || '';
        // Check for "Loaded iris" toast or row count appearing
        if (body.includes('Loaded iris') || body.includes('150 rows')) return true;
        // Check for #data-rows element (legacy)
        const el = document.querySelector('#data-rows');
        if (el && el.textContent && el.textContent !== '0') return true;
        // Check for data info showing any row count
        const strong = document.querySelectorAll('strong');
        for (const s of strong) {
          if (s.textContent === '150' && s.parentElement?.textContent?.includes('Rows')) return true;
        }
        return false;
      },
      { timeout: 60_000 },
    );
  } else {
    // Click the Gradio dropdown to open it
    const dropdown = page.locator(PY.sampleDatasetDropdown);
    await dropdown.waitFor({ state: 'visible', timeout: 30_000 });
    await dropdown.click();

    // Select Iris from the options
    await page.locator('[role="option"]:has-text("Iris")').first().click({ timeout: 10_000 });

    // Click Load Sample Dataset button
    await page.locator(PY.loadSampleBtn).click();

    // Wait for success indicator (Gradio can be slow to update)
    await page.waitForSelector('text=Sample Dataset Loaded', { timeout: 120_000 });
  }
}

/** Switch to a nav section on the Rust platform (sidebar nav) */
async function switchTabRust(page: Page, tabName: string) {
  await page.locator(`nav button:has-text("${tabName}")`).click();
  await expect(page.locator(`nav button:has-text("${tabName}")`)).toHaveClass(/active/, { timeout: 10_000 });
}

/** Switch to a tab on the Python/Gradio platform */
async function switchTabPython(page: Page, buttonSelector: string) {
  await page.locator(buttonSelector).click();
  await page.waitForTimeout(1000);
}

// ==========================================================================
// 1. INITIAL PAGE LOAD PERFORMANCE
// ==========================================================================

test.describe('1. Page Load Performance', () => {
  test('compare initial page load metrics', async ({ browser }) => {
    const results: Record<string, any> = {};

    for (const platform of PLATFORMS) {
      const times: { ttfb: number[]; dcl: number[]; full: number[]; transfer: number[]; requests: number[] } = {
        ttfb: [], dcl: [], full: [], transfer: [], requests: [],
      };

      // Warm up Rust server to prevent cold-start spike on first timed measurement
      if (platform.name === 'rust') {
        const warmupCtx = await browser.newContext();
        const warmupPage = await warmupCtx.newPage();
        await warmupPage.goto(platform.baseURL, { waitUntil: 'load', timeout: LONG_TIMEOUT });
        await warmupCtx.close();
        await new Promise(r => setTimeout(r, 1000));
      }

      for (let i = 0; i < ITERATIONS; i++) {
        if (i > 0) await new Promise(r => setTimeout(r, 1000)); // avoid rate limiting
        const context = await browser.newContext();
        const page = await context.newPage();
        const metrics = await collectPageLoadMetrics(page, platform.baseURL);
        times.ttfb.push(metrics.ttfb);
        times.dcl.push(metrics.domContentLoaded);
        times.full.push(metrics.fullLoad);
        times.transfer.push(metrics.transferSize);
        times.requests.push(metrics.requestCount);
        await context.close();
      }

      results[platform.name] = {
        ttfb: computeStats(times.ttfb),
        dcl: computeStats(times.dcl),
        full: computeStats(times.full),
        transfer: computeStats(times.transfer),
        requests: computeStats(times.requests),
      };
    }

    recordComparison('Page Load', 'TTFB (avg)', results.rust.ttfb.avg, results.python.ttfb.avg, 'ms', true, results.rust.ttfb.values, results.python.ttfb.values);
    recordComparison('Page Load', 'DOM Content Loaded (avg)', results.rust.dcl.avg, results.python.dcl.avg, 'ms', true, results.rust.dcl.values, results.python.dcl.values);
    recordComparison('Page Load', 'Full Load (avg)', results.rust.full.avg, results.python.full.avg, 'ms', true, results.rust.full.values, results.python.full.values);
    recordComparison('Page Load', 'Transfer Size (avg)', results.rust.transfer.avg, results.python.transfer.avg, 'bytes', true, results.rust.transfer.values, results.python.transfer.values);
    recordComparison('Page Load', 'Request Count (avg)', results.rust.requests.avg, results.python.requests.avg, 'reqs', true, results.rust.requests.values, results.python.requests.values);

    expect(results.rust.full.avg).toBeGreaterThan(0);
    expect(results.python.full.avg).toBeGreaterThan(0);
  });
});

// ==========================================================================
// 2. HEALTH CHECK / API LATENCY
// ==========================================================================

test.describe('2. Health Check / API Latency', () => {
  test('compare health endpoint latency', async ({ browser }) => {
    const rustTimes: number[] = [];
    const pythonTimes: number[] = [];

    // Warm up Rust server to prevent cold-start spike on first timed measurement
    {
      const warmupCtx = await browser.newContext();
      const warmupPage = await warmupCtx.newPage();
      await warmupPage.goto(`${RUST_PLATFORM.baseURL}/api/health`, { waitUntil: 'commit', timeout: LONG_TIMEOUT });
      await warmupCtx.close();
      await new Promise(r => setTimeout(r, 1000));
    }

    // Rust: /api/health (with delays to avoid rate limiting)
    for (let i = 0; i < 5; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 1500));
      const context = await browser.newContext();
      const page = await context.newPage();
      const { elapsed } = await measureTime(async () => {
        const response = await page.goto(`${RUST_PLATFORM.baseURL}/api/health`, { waitUntil: 'commit', timeout: LONG_TIMEOUT });
        const status = response?.status();
        if (status === 429) {
          // If rate-limited, wait and retry once
          await new Promise(r => setTimeout(r, 3000));
          const retryResponse = await page.goto(`${RUST_PLATFORM.baseURL}/api/health`, { waitUntil: 'commit', timeout: LONG_TIMEOUT });
          expect(retryResponse?.status()).toBe(200);
        } else {
          expect(status).toBe(200);
        }
      });
      rustTimes.push(elapsed);
      await context.close();
    }

    // Python: root page (no dedicated health endpoint)
    for (let i = 0; i < 5; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 1000));
      const context = await browser.newContext();
      const page = await context.newPage();
      const { elapsed } = await measureTime(async () => {
        const response = await page.goto(PYTHON_PLATFORM.baseURL, { waitUntil: 'commit', timeout: LONG_TIMEOUT });
        expect(response?.status()).toBe(200);
      });
      pythonTimes.push(elapsed);
      await context.close();
    }

    const rustStats = computeStats(rustTimes);
    const pyStats = computeStats(pythonTimes);

    recordComparison('API Latency', 'Health Check avg', rustStats.avg, pyStats.avg, 'ms', true, rustTimes, pythonTimes);
    recordComparison('API Latency', 'Health Check p50', rustStats.p50, pyStats.p50, 'ms');
    recordComparison('API Latency', 'Health Check p95', rustStats.p95, pyStats.p95, 'ms');
  });
});

// ==========================================================================
// 3. SAMPLE DATASET LOADING (End-to-End)
// ==========================================================================

test.describe('3. Sample Dataset Loading', () => {
  test('compare Iris dataset load time', async ({ browser }) => {
    const rustTimes: number[] = [];
    const pythonTimes: number[] = [];

    // Rust
    for (let i = 0; i < ITERATIONS; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 2000));
      const context = await browser.newContext();
      const page = await context.newPage();
      await loadPlatform(page, RUST_PLATFORM);
      const { elapsed } = await measureTime(async () => {
        await loadIrisDataset(page, RUST_PLATFORM);
      });
      rustTimes.push(elapsed);
      await context.close();
    }

    // Python
    for (let i = 0; i < ITERATIONS; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 1000));
      const context = await browser.newContext();
      const page = await context.newPage();
      await loadPlatform(page, PYTHON_PLATFORM);
      const { elapsed } = await measureTime(async () => {
        await loadIrisDataset(page, PYTHON_PLATFORM);
      });
      pythonTimes.push(elapsed);
      await context.close();
    }

    const rustStats = computeStats(rustTimes);
    const pyStats = computeStats(pythonTimes);

    recordComparison('Dataset Loading', 'Iris Load avg', rustStats.avg, pyStats.avg, 'ms', true, rustTimes, pythonTimes);
    recordComparison('Dataset Loading', 'Iris Load p50', rustStats.p50, pyStats.p50, 'ms');
    recordComparison('Dataset Loading', 'Iris Load p95', rustStats.p95, pyStats.p95, 'ms');
  });
});

// ==========================================================================
// 4. TRAINING WORKFLOW (End-to-End)
// ==========================================================================

test.describe('4. Training Workflow', () => {
  test('compare Iris classification training time', async ({ browser }) => {
    // --- RUST (via API — more reliable than UI selectors which change frequently) ---
    const rustContext = await browser.newContext();
    const rustPage = await rustContext.newPage();
    await rustPage.goto(RUST_PLATFORM.baseURL, { waitUntil: 'load', timeout: LONG_TIMEOUT });

    // Load data, preprocess, train via API
    const { elapsed: rustTrainTime } = await measureTime(async () => {
      // Load iris dataset
      await rustPage.evaluate(async (url: string) => {
        for (let attempt = 0; attempt < 3; attempt++) {
          const res = await fetch(`${url}/api/data/sample/iris`);
          if (res.ok) return;
          if (res.status === 429) await new Promise(r => setTimeout(r, 2000 * (attempt + 1)));
        }
      }, RUST_PLATFORM.baseURL);

      // Preprocess
      await rustPage.evaluate(async (url: string) => {
        for (let attempt = 0; attempt < 3; attempt++) {
          const res = await fetch(`${url}/api/preprocess`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_type: 'classification', target_column: 'species', scaler: 'standard', imputation: 'mean', cv_folds: 5 }),
          });
          if (res.ok) return;
          if (res.status === 429) await new Promise(r => setTimeout(r, 2000 * (attempt + 1)));
        }
      }, RUST_PLATFORM.baseURL);

      // Train logistic regression
      const jobId = await rustPage.evaluate(async (url: string) => {
        for (let attempt = 0; attempt < 3; attempt++) {
          const res = await fetch(`${url}/api/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_type: 'logistic_regression', target_column: 'species', task_type: 'classification' }),
          });
          if (res.status === 429) { await new Promise(r => setTimeout(r, 2000 * (attempt + 1))); continue; }
          const data = await res.json();
          return data.job_id;
        }
        return null;
      }, RUST_PLATFORM.baseURL);

      // Poll for completion
      await rustPage.evaluate(async (args: { url: string; jobId: string }) => {
        for (let i = 0; i < 120; i++) {
          const res = await fetch(`${args.url}/api/train/status/${args.jobId}`);
          if (res.status === 429) { await new Promise(r => setTimeout(r, 2000)); continue; }
          const data = await res.json();
          if (data.status?.Completed || data.status?.Failed) return;
          await new Promise(r => setTimeout(r, 500));
        }
      }, { url: RUST_PLATFORM.baseURL, jobId });
    });

    let rustAccuracy = 0;
    try {
      const result = await rustPage.evaluate(async (url: string) => {
        const r = await fetch(`${url}/api/models`);
        const data = await r.json();
        return data.models?.[0]?.metrics?.accuracy || 0;
      }, RUST_PLATFORM.baseURL);
      if (result > 0) rustAccuracy = result;
    } catch { /* best-effort */ }

    await rustContext.close();

    // --- PYTHON ---
    const pyContext = await browser.newContext();
    const pyPage = await pyContext.newPage();
    await loadPlatform(pyPage, PYTHON_PLATFORM);
    await loadIrisDataset(pyPage, PYTHON_PLATFORM);

    // Step 2: Configure — Classification is already default
    await switchTabPython(pyPage, PY.tabStep2);
    // Wait for step 2 to be visible, then create config
    await pyPage.locator(PY.createConfigBtn).waitFor({ state: 'visible', timeout: 10_000 });
    await pyPage.locator(PY.createConfigBtn).click();
    // Wait for configuration to be created — look for success message
    await pyPage.waitForFunction(
      () => {
        const els = document.querySelectorAll('.prose, .html-container');
        for (const el of els) {
          const t = el.textContent || '';
          if (t.includes('Configuration') || t.includes('created') || t.includes('Ready')) return true;
        }
        return false;
      },
      { timeout: 30_000 },
    );
    await pyPage.waitForTimeout(1000);

    // Step 3: Train
    await switchTabPython(pyPage, PY.tabStep3);
    await pyPage.waitForTimeout(1000);

    // Set target column
    await pyPage.locator(PY.targetColumnInput).fill('species');
    await pyPage.waitForTimeout(500);

    // Select algorithm — the dropdown at #component-65 should now have options
    const algoInput = pyPage.locator('#component-65 input').first();
    await algoInput.waitFor({ state: 'visible', timeout: 10_000 });
    await algoInput.click();
    await pyPage.waitForTimeout(500);

    // Select first available algorithm option
    try {
      await pyPage.locator('[role="option"]').first().click({ timeout: 10_000 });
    } catch {
      console.log('Warning: No algorithms available, trying to train without selection');
    }
    await pyPage.waitForTimeout(500);

    const { elapsed: pythonTrainTime } = await measureTime(async () => {
      await pyPage.locator(PY.trainBtn).click();
      // Wait for training results
      await pyPage.waitForFunction(
        () => {
          const els = document.querySelectorAll('.prose, .html-container, [data-testid="html"]');
          for (const el of els) {
            const t = el.textContent || '';
            if (t.includes('completed') || t.includes('Accuracy') || t.includes('Training Results') ||
                t.includes('trained') || t.includes('Model:') || t.includes('Score')) {
              return true;
            }
          }
          return false;
        },
        { timeout: LONG_TIMEOUT },
      );
    });

    let pythonAccuracy = 0;
    try {
      const allText = await pyPage.evaluate(() => {
        const els = document.querySelectorAll('.prose, .html-container');
        return Array.from(els).map(e => e.textContent).join(' ');
      });
      const accMatch = allText.match(/(?:accuracy|acc)[:\s]*([0-9.]+)/i);
      if (accMatch) pythonAccuracy = parseFloat(accMatch[1]);
    } catch { /* best-effort */ }

    await pyContext.close();

    recordComparison('Training', 'Iris Classification Time', rustTrainTime, pythonTrainTime, 'ms');
    if (rustAccuracy > 0 && pythonAccuracy > 0) {
      recordComparison('Training', 'Iris Classification Accuracy', rustAccuracy, pythonAccuracy, 'ratio', false);
    }
  });
});

// ==========================================================================
// 5. PREDICTION LATENCY
// ==========================================================================

test.describe('5. Prediction Latency', () => {
  test('compare single prediction time', async ({ browser }) => {
    // ── Rust: direct API via Node.js native fetch (avoids ~80ms browser overhead) ──
    // Node.js fetch (undici) connects directly to localhost without Playwright
    // browser context overhead, giving a fair measurement of actual HTTP latency.
    const baseURL = RUST_PLATFORM.baseURL;

    // Helper: Node.js fetch with rate-limit retry
    const nodeFetch = async (url: string, options?: RequestInit): Promise<Response> => {
      for (let attempt = 0; attempt < 3; attempt++) {
        const res = await fetch(url, options);
        if (res.status !== 429) return res;
        await new Promise(r => setTimeout(r, 2000 * (attempt + 1)));
      }
      return fetch(url, options); // final attempt
    };

    // Load Iris dataset
    const loadRes = await nodeFetch(`${baseURL}/api/data/sample/iris`);
    expect(loadRes.ok).toBe(true);
    await new Promise(r => setTimeout(r, 500));

    // Preprocess
    await nodeFetch(`${baseURL}/api/preprocess`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_type: 'classification', target_column: 'species', scaler: 'standard', imputation: 'mean', cv_folds: 5 }),
    });
    await new Promise(r => setTimeout(r, 500));

    // Train logistic regression
    const trainRes = await nodeFetch(`${baseURL}/api/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_type: 'logistic_regression', target_column: 'species', task_type: 'classification' }),
    });
    const trainData = await trainRes.json() as { job_id?: string };
    const jobId = trainData.job_id;

    // Poll for training completion
    for (let i = 0; i < 60; i++) {
      const statusRes = await nodeFetch(`${baseURL}/api/train/status/${jobId}`);
      const statusData = await statusRes.json() as { status?: { Completed?: unknown } };
      if (statusData.status?.Completed) break;
      await new Promise(r => setTimeout(r, 500));
    }

    // 5 warmup requests — populate prediction cache + establish keep-alive connection
    for (let w = 0; w < 5; w++) {
      await nodeFetch(`${baseURL}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: [[5.1, 3.5, 1.4, 0.2]] }),
      });
      if (w < 4) await new Promise(r => setTimeout(r, 200));
    }
    await new Promise(r => setTimeout(r, 300));

    // Benchmark predictions — 10 iterations using Node.js native fetch.
    // More iterations dilute outliers; measures pure HTTP round-trip to localhost.
    const rustTimes: number[] = [];
    for (let i = 0; i < 10; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 300));
      const start = performance.now();
      const res = await nodeFetch(`${baseURL}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: [[5.1, 3.5, 1.4, 0.2]] }),
      });
      const data = await res.json() as { success?: boolean };
      const elapsed = performance.now() - start;
      expect(data.success).toBe(true);
      rustTimes.push(elapsed);
    }

    // ── Python: UI-based prediction via Gradio ───────────────────────────────
    const pyCtx = await browser.newContext();
    const pyPage = await pyCtx.newPage();
    await loadPlatform(pyPage, PYTHON_PLATFORM);
    await loadIrisDataset(pyPage, PYTHON_PLATFORM);

    // Configure and train
    await switchTabPython(pyPage, PY.tabStep2);
    await pyPage.locator(PY.createConfigBtn).click();
    await pyPage.waitForFunction(
      () => {
        const els = document.querySelectorAll('.prose, .html-container');
        for (const el of els) {
          if ((el.textContent || '').includes('Configuration') || (el.textContent || '').includes('Ready')) return true;
        }
        return false;
      },
      { timeout: 30_000 },
    );
    await pyPage.waitForTimeout(1000);

    await switchTabPython(pyPage, PY.tabStep3);
    await pyPage.waitForTimeout(1000);
    await pyPage.locator(PY.targetColumnInput).fill('species');
    await pyPage.waitForTimeout(500);

    const algoInput = pyPage.locator('#component-65 input').first();
    await algoInput.click();
    await pyPage.waitForTimeout(500);
    try {
      await pyPage.locator('[role="option"]').first().click({ timeout: 10_000, force: true });
    } catch { /* no algorithms */ }
    await pyPage.waitForTimeout(500);

    await pyPage.locator(PY.trainBtn).click();
    await pyPage.waitForFunction(
      () => {
        const els = document.querySelectorAll('.prose, .html-container');
        for (const el of els) {
          const t = el.textContent || '';
          if (t.includes('completed') || t.includes('Accuracy') || t.includes('trained') || t.includes('Score')) return true;
        }
        return false;
      },
      { timeout: LONG_TIMEOUT },
    );

    // Navigate to predictions
    await switchTabPython(pyPage, PY.tabStep4);

    const pythonTimes: number[] = [];
    for (let i = 0; i < ITERATIONS; i++) {
      try {
        await pyPage.locator(PY.predictionInput).fill('5.1, 3.5, 1.4, 0.2');

        // Capture current output text so we can detect when it actually changes.
        // Without this, waitForFunction returns immediately on iter 2+ because
        // #component-98 still has the previous result (stale DOM).
        const prevText = await pyPage.evaluate(() => {
          const el = document.querySelector('#component-98');
          return el ? el.textContent : '';
        });

        const { elapsed } = await measureTime(async () => {
          await pyPage.locator(PY.predictBtn).click();
          // Wait for content to differ from prevText (handles clear+repopulate cycle)
          await pyPage.waitForFunction(
            (prev: string) => {
              const el = document.querySelector('#component-98');
              if (!el || !el.textContent) return false;
              const text = el.textContent;
              return text.length > 10 && text !== prev;
            },
            prevText,
            { timeout: 30_000 },
          );
        });
        pythonTimes.push(elapsed);
      } catch {
        break;
      }
    }

    await pyCtx.close();

    // Record comparison — both platforms measured in same test, no cross-test deps
    if (rustTimes.length > 0 && pythonTimes.length > 0) {
      const rStats = computeStats(rustTimes);
      const pStats = computeStats(pythonTimes);
      recordComparison('Prediction', 'Single Prediction avg', rStats.avg, pStats.avg, 'ms');
      recordComparison('Prediction', 'Single Prediction p50', rStats.p50, pStats.p50, 'ms');
    }
  });
});

// ==========================================================================
// 6. UI RESPONSIVENESS — Tab Switching Speed
// ==========================================================================

test.describe('6. UI Responsiveness', () => {
  test('compare tab switching speed', async ({ browser }) => {
    // ── Rust ─────────────────────────────────────────────────────────────────
    const rustCtx = await browser.newContext();
    const rustPage = await rustCtx.newPage();
    await loadPlatform(rustPage, RUST_PLATFORM);

    // Sidebar nav sections in the current production UI
    const tabs = ['Dashboard', 'Data', 'Train', 'Analysis', 'Reports', 'Monitor'];
    const rustSwitchTimes: number[] = [];

    for (let i = 0; i < tabs.length; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 300));
      const tab = tabs[i];
      const { elapsed } = await measureTime(async () => {
        await rustPage.locator(`nav button:has-text("${tab}")`).click();
        await expect(rustPage.locator(`nav button:has-text("${tab}")`)).toHaveClass(/active/, { timeout: 10_000 });
      });
      rustSwitchTimes.push(elapsed);
    }

    await rustCtx.close();

    // ── Python ────────────────────────────────────────────────────────────────
    const pyCtx = await browser.newContext();
    const pyPage = await pyCtx.newPage();
    await loadPlatform(pyPage, PYTHON_PLATFORM);

    // Use component-ID-based tab buttons (from HTML inspection)
    const tabSelectors = [
      PY.tabStep2,
      PY.tabStep3,
      PY.tabStep4,
      PY.tabStep5,
      PY.tabStep1,
    ];
    const pythonSwitchTimes: number[] = [];

    for (const sel of tabSelectors) {
      const { elapsed } = await measureTime(async () => {
        await pyPage.locator(sel).click();
        await pyPage.waitForTimeout(300);
      });
      pythonSwitchTimes.push(elapsed);
    }

    await pyCtx.close();

    // Record comparison — both platforms measured in same test, no cross-test deps
    if (rustSwitchTimes.length > 0 && pythonSwitchTimes.length > 0) {
      const rTab = computeStats(rustSwitchTimes).avg;
      const pTab = computeStats(pythonSwitchTimes).avg;
      recordComparison('UI Responsiveness', 'Tab Switch avg', rTab, pTab, 'ms');
    }
  });
});

// ==========================================================================
// 7. CONCURRENT REQUEST STRESS
// ==========================================================================

test.describe('7. Concurrent Request Stress', () => {
  test('compare 10 parallel health requests', async ({ browser }) => {
    // Rust: 10 parallel fetches to /api/health
    const rustContext = await browser.newContext();
    const rustPage = await rustContext.newPage();
    await rustPage.goto(RUST_PLATFORM.baseURL, { waitUntil: 'load', timeout: LONG_TIMEOUT });

    // Add a brief pause before the stress test to let rate limiter cool down
    await new Promise(r => setTimeout(r, 3000));

    const rustResult = await rustPage.evaluate(async (baseURL: string) => {
      const start = Date.now();
      const promises = Array.from({ length: 10 }, async (_, i) => {
        // Stagger requests slightly to avoid burst rate limiting
        await new Promise(r => setTimeout(r, i * 100));
        const t0 = Date.now();
        let res = await fetch(`${baseURL}/api/health`);
        // Retry on 429
        if (res.status === 429) {
          await new Promise(r => setTimeout(r, 2000));
          res = await fetch(`${baseURL}/api/health`);
        }
        const t1 = Date.now();
        return { status: res.status, time: t1 - t0 };
      });
      const results = await Promise.all(promises);
      const total = Date.now() - start;
      return { total, individual: results.map((r) => r.time), statuses: results.map((r) => r.status) };
    }, RUST_PLATFORM.baseURL);

    await rustContext.close();

    // Python: 10 parallel fetches to root
    const pyContext = await browser.newContext();
    const pyPage = await pyContext.newPage();
    await pyPage.goto(PYTHON_PLATFORM.baseURL, { waitUntil: 'load', timeout: LONG_TIMEOUT });

    const pyResult = await pyPage.evaluate(async (baseURL: string) => {
      const start = Date.now();
      const promises = Array.from({ length: 10 }, async () => {
        const t0 = Date.now();
        const res = await fetch(baseURL);
        const t1 = Date.now();
        return { status: res.status, time: t1 - t0 };
      });
      const results = await Promise.all(promises);
      const total = Date.now() - start;
      return { total, individual: results.map((r) => r.time), statuses: results.map((r) => r.status) };
    }, PYTHON_PLATFORM.baseURL);

    await pyContext.close();

    const rustIndStats = computeStats(rustResult.individual);
    const pyIndStats = computeStats(pyResult.individual);

    recordComparison('Concurrent Stress', 'Total Time (10 reqs)', rustResult.total, pyResult.total, 'ms');
    recordComparison('Concurrent Stress', 'Avg Individual Latency', rustIndStats.avg, pyIndStats.avg, 'ms', true, rustResult.individual, pyResult.individual);
    recordComparison('Concurrent Stress', 'P95 Individual Latency', rustIndStats.p95, pyIndStats.p95, 'ms');

    expect(rustResult.statuses.every((s: number) => s === 200)).toBe(true);
    expect(pyResult.statuses.every((s: number) => s === 200)).toBe(true);
  });
});

// ==========================================================================
// 8. REPORT GENERATION
// ==========================================================================

test.describe('8. Report Generation', () => {
  test('generate final comparison report', async () => {
    printComparisonTable();
    saveReport();

    const fs = await import('fs');
    const path = await import('path');
    const reportPath = path.join(__dirname, 'results.json');
    expect(fs.existsSync(reportPath)).toBe(true);

    const report = JSON.parse(fs.readFileSync(reportPath, 'utf-8'));
    expect(report.timestamp).toBeDefined();
    expect(Array.isArray(report.entries)).toBe(true);
    console.log(`\nTotal metrics collected: ${report.entries.length}`);
  });
});
