import { test, expect, Page } from '@playwright/test';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Wait for the server to be up and page ready. */
async function waitForServer(page: Page) {
  await page.goto('/');
  await page.waitForLoadState('networkidle');
  // Wait for the status badge to render (confirms JS has loaded)
  await page.locator('#status-badge').waitFor({ state: 'attached', timeout: 15_000 });
}

/** Load a built-in sample dataset by clicking its pill. */
async function loadSampleDataset(page: Page, name: string) {
  // Ensure the page is on the Data tab where pills are visible
  const pill = page.locator(`button.pill`, { hasText: new RegExp(`^${name}$`, 'i') });
  await pill.waitFor({ state: 'visible', timeout: 10_000 });
  await pill.click();

  // Wait for the data-info bar to reflect loaded rows (> 0)
  await expect(page.locator('#data-rows')).not.toHaveText('0', { timeout: 30_000 });
}

/** Switch to a top-level tab. */
async function switchTab(page: Page, tabName: string) {
  await page.locator(`button.nav-tab[data-tab="${tabName}"]`).click();
  await expect(page.locator(`#tab-${tabName}`)).toBeVisible();
}

/** Switch to a sub-tab within the Train panel. */
async function switchSubTab(page: Page, subtab: string) {
  await page.locator(`button.sub-tab[data-subtab="${subtab}"]`).click();
}

/** Wait for a UMAP API call to finish successfully. */
async function waitForUmapResponse(page: Page, timeout = 60_000) {
  return page.waitForResponse(
    (res) => res.url().includes('/api/visualization/umap') && res.status() === 200,
    { timeout },
  );
}

/** Clear the server-side dataset to reset state. */
async function clearData(page: Page) {
  await page.evaluate(async () => {
    await fetch('/api/data/clear', { method: 'DELETE' });
  });
}

/** Select a model from the custom multiselect widget. */
async function selectModelFromMultiselect(page: Page, modelText: string) {
  // Open the multiselect dropdown
  const trigger = page.locator('#model-checkboxes .multiselect-trigger');
  await trigger.click();
  // Wait for dropdown to be visible
  await page.locator('#model-checkboxes .multiselect-dropdown').waitFor({ state: 'visible', timeout: 5_000 });
  // Click the option
  const option = page.locator('#model-checkboxes .multiselect-option', { hasText: new RegExp(modelText, 'i') }).first();
  await option.click();
  // Close the dropdown by clicking outside it
  await page.locator('h2', { hasText: 'Select Models' }).click();
  await page.waitForTimeout(200);
}

// ---------------------------------------------------------------------------
// Tests — UMAP UI Presence
// ---------------------------------------------------------------------------

test.describe('UMAP UI Elements', () => {
  test.beforeEach(async ({ page }) => {
    await waitForServer(page);
  });

  test('UMAP card exists in Analyze tab', async ({ page }) => {
    await switchTab(page, 'analyze');
    await expect(page.locator('#btn-umap')).toBeVisible();
    await expect(page.locator('#umap-neighbors')).toBeVisible();
    await expect(page.locator('#umap-mindist')).toBeVisible();
    await expect(page.locator('#umap-colorby')).toBeVisible();
    await expect(page.locator('#umap-empty')).toBeVisible();
  });

  test('UMAP button is disabled before data load', async ({ page }) => {
    await switchTab(page, 'analyze');
    await expect(page.locator('#btn-umap')).toBeDisabled();
  });

  test('UMAP button enables after data load', async ({ page }) => {
    await loadSampleDataset(page, 'Iris');
    await switchTab(page, 'analyze');
    await expect(page.locator('#btn-umap')).toBeEnabled({ timeout: 20_000 });
  });

  test('Color-by dropdown populates with column names after data load', async ({ page }) => {
    await loadSampleDataset(page, 'Iris');
    await switchTab(page, 'analyze');

    const select = page.locator('#umap-colorby');
    const options = select.locator('option');
    const count = await options.count();
    // "None" + at least 1 column
    expect(count).toBeGreaterThan(1);
  });

  test('Live UMAP card exists in Train > Models sub-panel', async ({ page }) => {
    await switchTab(page, 'train');
    await switchSubTab(page, 'models');
    // Card should exist but hidden before training
    const card = page.locator('#train-umap-card');
    await expect(card).toBeHidden();
  });
});

// ---------------------------------------------------------------------------
// Tests — UMAP Auto-Run on Data Load
// ---------------------------------------------------------------------------

test.describe('UMAP Auto-Run on Data Load', () => {
  test('UMAP auto-triggers when Iris dataset is loaded', async ({ page }) => {
    await waitForServer(page);

    // Set up listener BEFORE loading data
    const umapPromise = waitForUmapResponse(page);
    await loadSampleDataset(page, 'Iris');

    // Wait for the auto-triggered UMAP call to complete
    const response = await umapPromise;
    const body = await response.json();
    expect(body.success).toBe(true);
    expect(body.points).toBeDefined();
    expect(body.points.length).toBeGreaterThan(0);
    expect(body.n_samples).toBeGreaterThan(0);
  });

  test('UMAP chart renders after auto-run', async ({ page }) => {
    await waitForServer(page);

    const umapPromise = waitForUmapResponse(page);
    await loadSampleDataset(page, 'Iris');
    await umapPromise;

    await switchTab(page, 'analyze');

    // Chart wrap should become visible
    await expect(page.locator('#umap-chart-wrap')).toBeVisible({ timeout: 15_000 });
    // Canvas should exist
    await expect(page.locator('#umapChart')).toBeVisible();
    // Empty state should be hidden
    await expect(page.locator('#umap-empty')).toBeHidden();
    // Results summary should show sample count
    await expect(page.locator('#umap-results')).toContainText('Samples');
  });
});

// ---------------------------------------------------------------------------
// Tests — UMAP Manual Trigger
// ---------------------------------------------------------------------------

test.describe('UMAP Manual Trigger', () => {
  test('clicking Run UMAP sends request and renders chart', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Wine');

    // Wait for auto-UMAP to finish first
    await waitForUmapResponse(page);

    await switchTab(page, 'analyze');
    await expect(page.locator('#btn-umap')).toBeEnabled({ timeout: 10_000 });

    // Click manually
    const umapPromise = waitForUmapResponse(page);
    await page.locator('#btn-umap').click();
    const response = await umapPromise;
    const body = await response.json();

    expect(body.success).toBe(true);
    expect(body.n_samples).toBeGreaterThan(0);
    await expect(page.locator('#umap-chart-wrap')).toBeVisible({ timeout: 15_000 });
  });

  test('UMAP respects custom n_neighbors parameter', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');
    await waitForUmapResponse(page);

    await switchTab(page, 'analyze');

    // Set custom neighbors
    await page.locator('#umap-neighbors').fill('5');

    const umapPromise = waitForUmapResponse(page);
    await page.locator('#btn-umap').click();
    const response = await umapPromise;
    const body = await response.json();

    expect(body.success).toBe(true);
    expect(body.params.n_neighbors).toBe(5);
  });

  test('UMAP respects custom min_dist parameter', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');
    await waitForUmapResponse(page);

    await switchTab(page, 'analyze');

    await page.locator('#umap-mindist').fill('0.5');

    const umapPromise = waitForUmapResponse(page);
    await page.locator('#btn-umap').click();
    const response = await umapPromise;
    const body = await response.json();

    expect(body.success).toBe(true);
    expect(body.params.min_dist).toBe(0.5);
  });
});

// ---------------------------------------------------------------------------
// Tests — Classification Mode (Iris - species column)
// ---------------------------------------------------------------------------

test.describe('UMAP Classification Visualization', () => {
  test('classification mode returns labels and boundaries', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');
    await waitForUmapResponse(page);

    await switchTab(page, 'analyze');

    // Select color_by = species (column in Iris)
    await page.locator('#umap-colorby').selectOption('species');

    const umapPromise = waitForUmapResponse(page);
    await page.locator('#btn-umap').click();
    const response = await umapPromise;
    const body = await response.json();

    expect(body.success).toBe(true);
    expect(body.color_mode).toBe('classification');
    expect(body.labels).toBeDefined();
    expect(body.labels.length).toBeGreaterThan(0);
    expect(body.label_names).toBeDefined();
    expect(body.label_names.length).toBeGreaterThanOrEqual(2);
    expect(body.color_values).toBeNull();
  });

  test('classification chart shows legend with class names', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');
    await waitForUmapResponse(page);

    await switchTab(page, 'analyze');
    await expect(page.locator('#btn-umap')).toBeEnabled({ timeout: 10_000 });
    await page.locator('#umap-colorby').selectOption('species');

    const umapPromise = waitForUmapResponse(page);
    await page.locator('#btn-umap').click();
    await umapPromise;

    // Wait for chart to render and results to populate
    await expect(page.locator('#umap-chart-wrap')).toBeVisible({ timeout: 15_000 });
    await page.waitForTimeout(500);

    // Results should show "Classification" mode and class info
    await expect(page.locator('#umap-results')).toContainText('Classification', { timeout: 5_000 });
    await expect(page.locator('#umap-results')).toContainText('species');
    await expect(page.locator('#umap-results')).toContainText('classes');
  });

  test('classification chart has multiple datasets (points + boundary lines)', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');
    await waitForUmapResponse(page);

    await switchTab(page, 'analyze');
    await expect(page.locator('#btn-umap')).toBeEnabled({ timeout: 10_000 });
    await page.locator('#umap-colorby').selectOption('species');

    const umapPromise = waitForUmapResponse(page);
    await page.locator('#btn-umap').click();
    await umapPromise;

    await expect(page.locator('#umap-chart-wrap')).toBeVisible({ timeout: 15_000 });

    // Wait for chart rendering to complete
    await page.waitForTimeout(1500);

    // Verify Chart.js instance has datasets via evaluate
    const datasetCount = await page.evaluate(() => {
      return (window as any).umapChartInstance?.data?.datasets?.length ?? 0;
    });
    // 3 classes × 2 (scatter + hull) = 6 datasets minimum
    expect(datasetCount).toBeGreaterThanOrEqual(4);
  });
});

// ---------------------------------------------------------------------------
// Tests — Regression Mode (Diabetes - target column)
// ---------------------------------------------------------------------------

test.describe('UMAP Regression Visualization', () => {
  test('regression mode returns color_values and trend', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Diabetes');
    await waitForUmapResponse(page);

    await switchTab(page, 'analyze');

    // Select color_by = target (continuous column)
    await page.locator('#umap-colorby').selectOption('target');

    const umapPromise = waitForUmapResponse(page);
    await page.locator('#btn-umap').click();
    const response = await umapPromise;
    const body = await response.json();

    expect(body.success).toBe(true);
    expect(body.color_mode).toBe('regression');
    expect(body.color_values).toBeDefined();
    expect(body.color_values.length).toBeGreaterThan(0);
    expect(body.labels).toBeNull();
    expect(body.label_names).toBeNull();
  });

  test('regression chart shows gradient mode label', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Diabetes');
    await waitForUmapResponse(page);

    await switchTab(page, 'analyze');
    await page.locator('#umap-colorby').selectOption('target');

    const umapPromise = waitForUmapResponse(page);
    await page.locator('#btn-umap').click();
    await umapPromise;

    await expect(page.locator('#umap-chart-wrap')).toBeVisible({ timeout: 15_000 });
    await expect(page.locator('#umap-results')).toContainText('Regression');
    await expect(page.locator('#umap-results')).toContainText('target');
  });

  test('regression chart has scatter + trend line datasets', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Diabetes');
    await waitForUmapResponse(page);

    await switchTab(page, 'analyze');
    await page.locator('#umap-colorby').selectOption('target');

    const umapPromise = waitForUmapResponse(page);
    await page.locator('#btn-umap').click();
    await umapPromise;

    await expect(page.locator('#umap-chart-wrap')).toBeVisible({ timeout: 15_000 });
    await page.waitForTimeout(1000);

    const datasetCount = await page.evaluate(() => {
      return (window as any).umapChartInstance?.data?.datasets?.length ?? 0;
    });
    // At least 2: scatter data + trend line
    expect(datasetCount).toBeGreaterThanOrEqual(2);

    // Check trend line exists
    const hasTrend = await page.evaluate(() => {
      const chart = (window as any).umapChartInstance;
      if (!chart) return false;
      return chart.data.datasets.some((ds: any) => ds.label === 'Trend');
    });
    expect(hasTrend).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Tests — Live UMAP During Training
// ---------------------------------------------------------------------------

test.describe('Live UMAP During Training', () => {
  test('UMAP card appears in training panel when training starts', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');
    await waitForUmapResponse(page); // wait for auto-UMAP

    // Configure training
    await switchTab(page, 'train');
    await switchSubTab(page, 'setup');
    await page.locator('#cfg-taskType').selectOption('classification');
    await page.locator('#cfg-targetColumn').selectOption('species');

    // Select a model via custom multiselect
    await switchSubTab(page, 'models');
    await selectModelFromMultiselect(page, 'Decision Tree');

    // Start training — this triggers launchTrainUmap
    const umapPromise = waitForUmapResponse(page, 90_000);
    await page.locator('#btn-train').click();

    // Wait for UMAP to complete
    await umapPromise;

    // The live UMAP card should be visible
    const card = page.locator('#train-umap-card');
    await expect(card).toBeVisible({ timeout: 15_000 });

    // Badge should show "Live"
    const badge = page.locator('#train-umap-badge');
    await expect(badge).toBeVisible();

    // Canvas should exist
    await expect(page.locator('#trainUmapChart')).toBeVisible();

    // Status should show sample count and mode
    await expect(page.locator('#train-umap-status')).toContainText('samples');
  });

  test('live UMAP renders classification boundaries during Iris training', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');
    await waitForUmapResponse(page);

    await switchTab(page, 'train');
    await switchSubTab(page, 'setup');
    await page.locator('#cfg-taskType').selectOption('classification');
    await page.locator('#cfg-targetColumn').selectOption('species');

    await switchSubTab(page, 'models');
    await selectModelFromMultiselect(page, 'Decision Tree');

    const umapPromise = waitForUmapResponse(page, 90_000);
    await page.locator('#btn-train').click();

    const response = await umapPromise;
    const body = await response.json();

    expect(body.success).toBe(true);
    expect(body.color_mode).toBe('classification');
    expect(body.labels).toBeDefined();
    expect(body.label_names).toBeDefined();

    // Verify the training canvas chart has classification datasets
    await page.waitForTimeout(2000); // let chart animate
    const datasetCount = await page.evaluate(() => {
      return (window as any).trainUmapChartInstance?.data?.datasets?.length ?? 0;
    });
    expect(datasetCount).toBeGreaterThanOrEqual(4); // scatter + hulls
  });

  test('live UMAP renders regression gradient during Diabetes training', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Diabetes');
    await waitForUmapResponse(page);

    await switchTab(page, 'train');
    await switchSubTab(page, 'setup');
    await page.locator('#cfg-taskType').selectOption('regression');
    await page.locator('#cfg-targetColumn').selectOption('target');

    await switchSubTab(page, 'models');
    await selectModelFromMultiselect(page, 'Decision Tree');

    const umapPromise = waitForUmapResponse(page, 90_000);
    await page.locator('#btn-train').click();

    const response = await umapPromise;
    const body = await response.json();

    expect(body.success).toBe(true);
    expect(body.color_mode).toBe('regression');
    expect(body.color_values).toBeDefined();

    await page.waitForTimeout(2000);
    const hasTrend = await page.evaluate(() => {
      const chart = (window as any).trainUmapChartInstance;
      if (!chart) return false;
      return chart.data.datasets.some((ds: any) => ds.label === 'Trend');
    });
    expect(hasTrend).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Tests — UMAP API Direct Validation
// ---------------------------------------------------------------------------

test.describe('UMAP API Validation', () => {
  test('UMAP returns 2D points with correct structure', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');
    await waitForUmapResponse(page);

    // Direct API call
    const response = await page.evaluate(async () => {
      const res = await fetch('/api/visualization/umap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_neighbors: 10, min_dist: 0.1 }),
      });
      return res.json();
    });

    expect(response.success).toBe(true);
    expect(response.points).toBeDefined();
    expect(Array.isArray(response.points)).toBe(true);
    expect(response.points.length).toBeGreaterThan(0);

    // Each point should be [x, y]
    const point = response.points[0];
    expect(Array.isArray(point)).toBe(true);
    expect(point.length).toBe(2);
    expect(typeof point[0]).toBe('number');
    expect(typeof point[1]).toBe('number');
    expect(Number.isFinite(point[0])).toBe(true);
    expect(Number.isFinite(point[1])).toBe(true);
  });

  test('UMAP rejects invalid n_neighbors', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');
    await waitForUmapResponse(page);

    const status = await page.evaluate(async () => {
      const res = await fetch('/api/visualization/umap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_neighbors: 999 }),
      });
      return res.status;
    });

    expect(status).toBe(400);
  });

  test('UMAP rejects invalid min_dist', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');
    await waitForUmapResponse(page);

    const status = await page.evaluate(async () => {
      const res = await fetch('/api/visualization/umap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ min_dist: -0.5 }),
      });
      return res.status;
    });

    expect(status).toBe(400);
  });

  test('UMAP fails gracefully with no data loaded', async ({ page }) => {
    await waitForServer(page);

    // Clear any previously loaded data
    await clearData(page);

    const status = await page.evaluate(async () => {
      const res = await fetch('/api/visualization/umap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      return res.status;
    });

    // Should return 404 (no data loaded)
    expect(status).toBe(404);
  });

  test('UMAP none mode when no color_by specified', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');
    await waitForUmapResponse(page);

    const response = await page.evaluate(async () => {
      const res = await fetch('/api/visualization/umap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_neighbors: 10 }),
      });
      return res.json();
    });

    expect(response.success).toBe(true);
    expect(response.color_mode).toBe('none');
    expect(response.labels).toBeNull();
    expect(response.color_values).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Tests — UMAP Chart Animations
// ---------------------------------------------------------------------------

test.describe('UMAP Chart Animation', () => {
  test('Chart.js scatter instance is created with animation config', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');
    await waitForUmapResponse(page);

    await switchTab(page, 'analyze');
    await expect(page.locator('#umap-chart-wrap')).toBeVisible({ timeout: 15_000 });

    const animConfig = await page.evaluate(() => {
      const chart = (window as any).umapChartInstance;
      if (!chart) return null;
      return {
        type: chart.config.type,
        duration: chart.config.options?.animation?.duration,
        easing: chart.config.options?.animation?.easing,
        hasDelay: typeof chart.config.options?.animation?.delay === 'function',
      };
    });

    expect(animConfig).not.toBeNull();
    expect(animConfig!.type).toBe('scatter');
    expect(animConfig!.duration).toBe(2000);
    expect(animConfig!.easing).toBe('easeOutQuart');
    expect(animConfig!.hasDelay).toBe(true);
  });

  test('training panel UMAP chart has longer animation for presentation', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');
    await waitForUmapResponse(page);

    await switchTab(page, 'train');
    await switchSubTab(page, 'setup');
    await page.locator('#cfg-taskType').selectOption('classification');
    await page.locator('#cfg-targetColumn').selectOption('species');
    await switchSubTab(page, 'models');

    await selectModelFromMultiselect(page, 'Decision Tree');

    const umapPromise = waitForUmapResponse(page, 90_000);
    await page.locator('#btn-train').click();
    await umapPromise;

    await page.waitForTimeout(1000);

    const animConfig = await page.evaluate(() => {
      const chart = (window as any).trainUmapChartInstance;
      if (!chart) return null;
      return {
        duration: chart.config.options?.animation?.duration,
        easing: chart.config.options?.animation?.easing,
      };
    });

    expect(animConfig).not.toBeNull();
    expect(animConfig!.duration).toBe(2500); // longer for presentation
    expect(animConfig!.easing).toBe('easeOutQuart');
  });
});

// ---------------------------------------------------------------------------
// Tests — Multiple Dataset Switching
// ---------------------------------------------------------------------------

test.describe('UMAP with Dataset Switching', () => {
  test('UMAP updates when switching from Iris to Diabetes', async ({ page }) => {
    await waitForServer(page);

    // Load Iris first
    await loadSampleDataset(page, 'Iris');
    const firstResponse = await waitForUmapResponse(page);
    const first = await firstResponse.json();
    expect(first.n_samples).toBeGreaterThan(0);
    const irisCount = first.n_samples;

    // Load Diabetes
    const umapPromise = waitForUmapResponse(page);
    await loadSampleDataset(page, 'Diabetes');
    const secondResponse = await umapPromise;
    const second = await secondResponse.json();

    expect(second.success).toBe(true);
    expect(second.n_samples).toBeGreaterThan(0);
    // Different dataset should produce different sample count
    expect(second.n_samples).not.toBe(irisCount);
  });
});
