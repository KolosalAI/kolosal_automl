import { test, expect, Page, Browser } from '@playwright/test';
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
    const pill = page.locator('button.pill', { hasText: /^Iris$/i });
    await pill.waitFor({ state: 'visible', timeout: 30_000 });
    await pill.click();
    await expect(page.locator('#data-rows')).not.toHaveText('0', { timeout: 60_000 });
  } else {
    // Click the Gradio dropdown to open it
    const dropdown = page.locator(PY.sampleDatasetDropdown);
    await dropdown.waitFor({ state: 'visible', timeout: 30_000 });
    await dropdown.click();

    // Select Iris from the options
    await page.locator('[role="option"]:has-text("Iris")').first().click({ timeout: 10_000 });

    // Click Load Sample Dataset button
    await page.locator(PY.loadSampleBtn).click();

    // Wait for success indicator
    await page.waitForSelector('text=Sample Dataset Loaded', { timeout: 60_000 });
  }
}

/** Switch to a tab on the Rust platform */
async function switchTabRust(page: Page, tabName: string) {
  await page.locator(`button.nav-tab[data-tab="${tabName}"]`).click();
  await expect(page.locator(`#tab-${tabName}`)).toBeVisible({ timeout: 10_000 });
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

      for (let i = 0; i < ITERATIONS; i++) {
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

    // Rust: /api/health
    for (let i = 0; i < 5; i++) {
      const context = await browser.newContext();
      const page = await context.newPage();
      const { elapsed } = await measureTime(async () => {
        const response = await page.goto(`${RUST_PLATFORM.baseURL}/api/health`, { waitUntil: 'commit', timeout: LONG_TIMEOUT });
        expect(response?.status()).toBe(200);
      });
      rustTimes.push(elapsed);
      await context.close();
    }

    // Python: root page (no dedicated health endpoint)
    for (let i = 0; i < 5; i++) {
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
    // --- RUST ---
    const rustContext = await browser.newContext();
    const rustPage = await rustContext.newPage();
    await loadPlatform(rustPage, RUST_PLATFORM);
    await loadIrisDataset(rustPage, RUST_PLATFORM);

    await switchTabRust(rustPage, 'train');

    // Setup sub-tab: configure task type and target
    await rustPage.locator('#cfg-taskType').selectOption('classification');
    await rustPage.locator('#cfg-targetColumn').selectOption('species');

    // Click Apply Preprocessing
    await rustPage.locator('#btn-preprocess').click();
    await rustPage.waitForTimeout(2000);

    // Switch to Models sub-tab
    await rustPage.locator('button[data-subtab="models"]').click();
    await rustPage.waitForTimeout(1000);

    // Open the model multiselect dropdown by clicking its header
    await rustPage.locator('#model-checkboxes').click();
    await rustPage.waitForTimeout(500);

    // Click on "Logistic Regression" text to select it
    await rustPage.locator('#model-checkboxes').locator('text=Logistic Regression').click();
    await rustPage.waitForTimeout(300);

    // Close the dropdown by pressing Escape and clicking outside
    await rustPage.keyboard.press('Escape');
    await rustPage.waitForTimeout(300);
    // Click on the "Training Progress" heading to dismiss any overlay
    await rustPage.locator('text=Training Progress').first().click({ force: true });
    await rustPage.waitForTimeout(300);

    const { elapsed: rustTrainTime } = await measureTime(async () => {
      await rustPage.locator('#btn-train').click({ force: true });
      // Wait for training to complete — look for job results or metrics
      await rustPage.waitForFunction(
        () => {
          const page = document;
          // Check for completed training jobs
          const jobItems = page.querySelectorAll('.job-card, .job-item, [class*="job"]');
          for (const job of jobItems) {
            const text = job.textContent || '';
            if (text.includes('complete') || text.includes('done') || text.includes('accuracy') || text.includes('Accuracy')) {
              return true;
            }
          }
          // Check for any metrics appearing
          const text = page.body.textContent || '';
          if (text.includes('Training complete') || text.includes('Model trained')) return true;
          // Check if "No training jobs yet" has been replaced
          const empty = page.querySelector('#jobs-empty');
          if (empty && getComputedStyle(empty).display === 'none') return true;
          const jobs = page.querySelector('#jobs-list');
          if (jobs && jobs.children.length > 0) return true;
          return false;
        },
        { timeout: LONG_TIMEOUT },
      );
    });

    let rustAccuracy = 0;
    try {
      const allText = await rustPage.evaluate(() => {
        const el = document.querySelector('#jobs-list, #train-results, #sub-models');
        return el?.textContent || '';
      });
      const accMatch = allText.match(/(?:accuracy|acc)[:\s]*([0-9.]+)/i);
      if (accMatch) rustAccuracy = parseFloat(accMatch[1]);
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
  test('compare single prediction time (Rust)', async ({ browser }) => {
    // Use the Rust API directly for prediction benchmarking
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST_PLATFORM.baseURL, { waitUntil: 'load', timeout: LONG_TIMEOUT });

    const baseURL = RUST_PLATFORM.baseURL;

    // Load Iris dataset via API (GET)
    const loaded = await page.evaluate(async (url: string) => {
      const res = await fetch(`${url}/api/data/sample/iris`);
      return res.ok;
    }, baseURL);
    expect(loaded).toBe(true);

    // Preprocess
    await page.evaluate(async (url: string) => {
      await fetch(`${url}/api/preprocess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_type: 'classification', target_column: 'species', scaler: 'standard', imputation: 'mean', cv_folds: 5 }),
      });
    }, baseURL);

    // Train a logistic regression model and wait for completion
    const jobId = await page.evaluate(async (url: string) => {
      const res = await fetch(`${url}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_type: 'logistic_regression', target_column: 'species', task_type: 'classification' }),
      });
      const data = await res.json();
      return data.job_id;
    }, baseURL);

    // Poll for training completion
    await page.evaluate(async (args: { url: string; jobId: string }) => {
      for (let i = 0; i < 60; i++) {
        const res = await fetch(`${args.url}/api/train/status/${args.jobId}`);
        const data = await res.json();
        if (data.status && data.status.Completed) return;
        await new Promise(r => setTimeout(r, 500));
      }
    }, { url: baseURL, jobId });

    // Benchmark predictions via API (5 iterations)
    const predictionTimes: number[] = [];
    for (let i = 0; i < 5; i++) {
      const { elapsed } = await measureTime(async () => {
        const result = await page.evaluate(async (url: string) => {
          const res = await fetch(`${url}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: [[5.1, 3.5, 1.4, 0.2]] }),
          });
          return res.json();
        }, baseURL);
        expect(result.success).toBe(true);
      });
      predictionTimes.push(elapsed);
    }

    await context.close();

    if (predictionTimes.length > 0) {
      const stats = computeStats(predictionTimes);
      sharedMetrics['rust_predict_avg'] = stats.avg;
      sharedMetrics['rust_predict_p50'] = stats.p50;
    }
  });

  test('compare single prediction time (Python)', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await loadPlatform(page, PYTHON_PLATFORM);
    await loadIrisDataset(page, PYTHON_PLATFORM);

    // Configure and train
    await switchTabPython(page, PY.tabStep2);
    await page.locator(PY.createConfigBtn).click();
    await page.waitForFunction(
      () => {
        const els = document.querySelectorAll('.prose, .html-container');
        for (const el of els) {
          if ((el.textContent || '').includes('Configuration') || (el.textContent || '').includes('Ready')) return true;
        }
        return false;
      },
      { timeout: 30_000 },
    );
    await page.waitForTimeout(1000);

    await switchTabPython(page, PY.tabStep3);
    await page.waitForTimeout(1000);
    await page.locator(PY.targetColumnInput).fill('species');
    await page.waitForTimeout(500);

    const algoInput = page.locator('#component-65 input').first();
    await algoInput.click();
    await page.waitForTimeout(500);
    try {
      await page.locator('[role="option"]').first().click({ timeout: 10_000 });
    } catch { /* no algorithms */ }
    await page.waitForTimeout(500);

    await page.locator(PY.trainBtn).click();
    await page.waitForFunction(
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
    await switchTabPython(page, PY.tabStep4);

    const predictionTimes: number[] = [];
    for (let i = 0; i < ITERATIONS; i++) {
      try {
        await page.locator(PY.predictionInput).fill('5.1, 3.5, 1.4, 0.2');

        const { elapsed } = await measureTime(async () => {
          await page.locator(PY.predictBtn).click();
          await page.waitForFunction(
            () => {
              const el = document.querySelector('#component-98');
              return el && el.textContent && el.textContent.length > 10;
            },
            { timeout: 30_000 },
          );
        });
        predictionTimes.push(elapsed);
      } catch {
        break;
      }
    }

    await context.close();

    if (predictionTimes.length > 0) {
      const stats = computeStats(predictionTimes);
      sharedMetrics['python_predict_avg'] = stats.avg;
      sharedMetrics['python_predict_p50'] = stats.p50;
    }

    // Combine Rust + Python prediction metrics
    const rAvg = sharedMetrics['rust_predict_avg'] || 0;
    const pAvg = sharedMetrics['python_predict_avg'] || 0;
    const rP50 = sharedMetrics['rust_predict_p50'] || 0;
    const pP50 = sharedMetrics['python_predict_p50'] || 0;
    if (rAvg > 0 && pAvg > 0) {
      recordComparison('Prediction', 'Single Prediction avg', rAvg, pAvg, 'ms');
      recordComparison('Prediction', 'Single Prediction p50', rP50, pP50, 'ms');
    }
  });
});

// ==========================================================================
// 6. UI RESPONSIVENESS — Tab Switching Speed
// ==========================================================================

test.describe('6. UI Responsiveness', () => {
  test('compare tab switching speed (Rust)', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await loadPlatform(page, RUST_PLATFORM);

    const tabs = ['automl', 'train', 'predict', 'analyze', 'serve', 'system', 'data'];
    const switchTimes: number[] = [];

    for (const tab of tabs) {
      const { elapsed } = await measureTime(async () => {
        await page.locator(`button.nav-tab[data-tab="${tab}"]`).click();
        await expect(page.locator(`#tab-${tab}`)).toBeVisible({ timeout: 10_000 });
      });
      switchTimes.push(elapsed);
    }

    await context.close();
    const stats = computeStats(switchTimes);
    sharedMetrics['rust_tab_avg'] = stats.avg;
  });

  test('compare tab switching speed (Python)', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await loadPlatform(page, PYTHON_PLATFORM);

    // Use component-ID-based tab buttons (from HTML inspection)
    const tabSelectors = [
      PY.tabStep2,
      PY.tabStep3,
      PY.tabStep4,
      PY.tabStep5,
      PY.tabStep1,
    ];
    const switchTimes: number[] = [];

    for (const sel of tabSelectors) {
      const { elapsed } = await measureTime(async () => {
        await page.locator(sel).click();
        await page.waitForTimeout(300);
      });
      switchTimes.push(elapsed);
    }

    await context.close();
    const stats = computeStats(switchTimes);
    sharedMetrics['python_tab_avg'] = stats.avg;

    // Combine Rust + Python tab switching metrics
    const rTab = sharedMetrics['rust_tab_avg'] || 0;
    const pTab = sharedMetrics['python_tab_avg'] || 0;
    if (rTab > 0 && pTab > 0) {
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

    const rustResult = await rustPage.evaluate(async (baseURL: string) => {
      const start = Date.now();
      const promises = Array.from({ length: 10 }, async () => {
        const t0 = Date.now();
        const res = await fetch(`${baseURL}/api/health`);
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
