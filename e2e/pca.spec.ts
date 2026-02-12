import { test, expect, Page } from '@playwright/test';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function waitForServer(page: Page) {
  await page.goto('/');
  await page.waitForLoadState('networkidle');
  await page.locator('#status-badge').waitFor({ state: 'attached', timeout: 15_000 });
}

async function loadSampleDataset(page: Page, name: string) {
  const pill = page.locator(`button.pill`, { hasText: new RegExp(`^${name}$`, 'i') });
  await pill.waitFor({ state: 'visible', timeout: 10_000 });
  await pill.click();
  await expect(page.locator('#data-rows')).not.toHaveText('0', { timeout: 30_000 });
}

async function switchTab(page: Page, tabName: string) {
  await page.locator(`button.nav-tab[data-tab="${tabName}"]`).click();
  await expect(page.locator(`#tab-${tabName}`)).toBeVisible();
}

async function waitForPcaResponse(page: Page, timeout = 60_000) {
  return page.waitForResponse(
    (res) => res.url().includes('/api/visualization/pca') && res.status() === 200,
    { timeout },
  );
}

async function clearData(page: Page) {
  await page.evaluate(async () => {
    await fetch('/api/data/clear', { method: 'DELETE' });
  });
}

// ---------------------------------------------------------------------------
// Tests — PCA UI Presence
// ---------------------------------------------------------------------------

test.describe('PCA UI Elements', () => {
  test.beforeEach(async ({ page }) => {
    await waitForServer(page);
  });

  test('PCA card exists in Analyze tab', async ({ page }) => {
    await switchTab(page, 'analyze');
    await expect(page.locator('#btn-pca')).toBeVisible();
    await expect(page.locator('#pca-scale')).toBeVisible();
    await expect(page.locator('#pca-colorby')).toBeVisible();
    await expect(page.locator('#pca-variance')).toBeVisible();
    await expect(page.locator('#pca-empty')).toBeVisible();
  });

  test('PCA button is disabled before data load', async ({ page }) => {
    await switchTab(page, 'analyze');
    await expect(page.locator('#btn-pca')).toBeDisabled();
  });

  test('PCA button enables after data load', async ({ page }) => {
    await loadSampleDataset(page, 'Iris');
    await switchTab(page, 'analyze');
    await expect(page.locator('#btn-pca')).toBeEnabled({ timeout: 20_000 });
  });

  test('PCA color-by dropdown populates with column names', async ({ page }) => {
    await loadSampleDataset(page, 'Iris');
    await switchTab(page, 'analyze');

    const select = page.locator('#pca-colorby');
    const options = select.locator('option');
    const count = await options.count();
    expect(count).toBeGreaterThan(1); // "None" + at least 1 column
  });
});

// ---------------------------------------------------------------------------
// Tests — PCA Manual Trigger
// ---------------------------------------------------------------------------

test.describe('PCA Manual Trigger', () => {
  test('clicking Run PCA sends request and renders chart', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');

    await switchTab(page, 'analyze');
    await expect(page.locator('#btn-pca')).toBeEnabled({ timeout: 20_000 });

    const pcaPromise = waitForPcaResponse(page);
    await page.locator('#btn-pca').click();
    const response = await pcaPromise;
    const body = await response.json();

    expect(body.success).toBe(true);
    expect(body.n_samples).toBeGreaterThan(0);
    expect(body.points).toBeDefined();
    expect(body.points.length).toBeGreaterThan(0);
    expect(body.explained_variance_ratio).toBeDefined();
    expect(body.explained_variance_ratio.length).toBe(2);

    await expect(page.locator('#pca-chart-wrap')).toBeVisible({ timeout: 15_000 });
  });

  test('PCA returns explained variance ratios', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');

    await switchTab(page, 'analyze');
    await expect(page.locator('#btn-pca')).toBeEnabled({ timeout: 20_000 });

    const pcaPromise = waitForPcaResponse(page);
    await page.locator('#btn-pca').click();
    const response = await pcaPromise;
    const body = await response.json();

    expect(body.success).toBe(true);
    const evr = body.explained_variance_ratio;
    expect(evr.length).toBe(2);
    // Each ratio should be between 0 and 1
    expect(evr[0]).toBeGreaterThan(0);
    expect(evr[0]).toBeLessThanOrEqual(1);
    expect(evr[1]).toBeGreaterThanOrEqual(0);
    expect(evr[1]).toBeLessThanOrEqual(1);
    // Total should be <= 1
    expect(evr[0] + evr[1]).toBeLessThanOrEqual(1.001);
  });

  test('PCA results show variance percentage', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');

    await switchTab(page, 'analyze');
    const pcaPromise = waitForPcaResponse(page);
    await page.locator('#btn-pca').click();
    await pcaPromise;

    // Variance display should update
    const variance = page.locator('#pca-variance');
    const text = await variance.textContent();
    expect(text).not.toBe('--');
    expect(text).toContain('%');

    // Results should show variance info
    await expect(page.locator('#pca-results')).toContainText('Variance Explained');
    await expect(page.locator('#pca-results')).toContainText('PC1');
    await expect(page.locator('#pca-results')).toContainText('PC2');
  });

  test('PCA respects scale parameter', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');

    await switchTab(page, 'analyze');
    await expect(page.locator('#btn-pca')).toBeEnabled({ timeout: 20_000 });

    // Select "No" for scaling
    await page.locator('#pca-scale').selectOption('false');

    const pcaPromise = waitForPcaResponse(page);
    await page.locator('#btn-pca').click();
    const response = await pcaPromise;
    const body = await response.json();

    expect(body.success).toBe(true);
    expect(body.params.scale).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Tests — PCA Classification Mode (Iris - species column)
// ---------------------------------------------------------------------------

test.describe('PCA Classification Visualization', () => {
  test('classification mode returns labels and boundaries', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');

    await switchTab(page, 'analyze');
    await expect(page.locator('#btn-pca')).toBeEnabled({ timeout: 20_000 });

    await page.locator('#pca-colorby').selectOption('species');

    const pcaPromise = waitForPcaResponse(page);
    await page.locator('#btn-pca').click();
    const response = await pcaPromise;
    const body = await response.json();

    expect(body.success).toBe(true);
    expect(body.color_mode).toBe('classification');
    expect(body.labels).toBeDefined();
    expect(body.labels.length).toBeGreaterThan(0);
    expect(body.label_names).toBeDefined();
    expect(body.label_names.length).toBeGreaterThanOrEqual(2);
    expect(body.color_values).toBeNull();
  });

  test('classification chart shows class info in results', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');

    await switchTab(page, 'analyze');
    await page.locator('#pca-colorby').selectOption('species');

    const pcaPromise = waitForPcaResponse(page);
    await page.locator('#btn-pca').click();
    await pcaPromise;

    await expect(page.locator('#pca-chart-wrap')).toBeVisible({ timeout: 15_000 });
    await expect(page.locator('#pca-results')).toContainText('Classification');
  });

  test('classification chart has multiple datasets', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');

    await switchTab(page, 'analyze');
    await page.locator('#pca-colorby').selectOption('species');

    const pcaPromise = waitForPcaResponse(page);
    await page.locator('#btn-pca').click();
    await pcaPromise;

    await expect(page.locator('#pca-chart-wrap')).toBeVisible({ timeout: 15_000 });

    const datasetCount = await page.evaluate(() => {
      const chart = (window as any).pcaChartInstance;
      return chart ? chart.data.datasets.length : 0;
    });
    // 3 classes × 2 (scatter + hull) = 6 datasets minimum
    expect(datasetCount).toBeGreaterThanOrEqual(4);
  });
});

// ---------------------------------------------------------------------------
// Tests — PCA Regression Mode (Diabetes - target column)
// ---------------------------------------------------------------------------

test.describe('PCA Regression Visualization', () => {
  test('regression mode returns color_values', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Diabetes');

    await switchTab(page, 'analyze');
    await expect(page.locator('#btn-pca')).toBeEnabled({ timeout: 20_000 });

    await page.locator('#pca-colorby').selectOption('target');

    const pcaPromise = waitForPcaResponse(page);
    await page.locator('#btn-pca').click();
    const response = await pcaPromise;
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

    await switchTab(page, 'analyze');
    await page.locator('#pca-colorby').selectOption('target');

    const pcaPromise = waitForPcaResponse(page);
    await page.locator('#btn-pca').click();
    await pcaPromise;

    await expect(page.locator('#pca-chart-wrap')).toBeVisible({ timeout: 15_000 });
    await expect(page.locator('#pca-results')).toContainText('Regression');
  });
});

// ---------------------------------------------------------------------------
// Tests — PCA API Direct Validation
// ---------------------------------------------------------------------------

test.describe('PCA API Validation', () => {
  test('PCA returns 2D points with correct structure', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');

    const response = await page.evaluate(async () => {
      const res = await fetch('/api/visualization/pca', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
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

    // Should have eigenvalues and explained_variance_ratio
    expect(response.eigenvalues).toBeDefined();
    expect(response.eigenvalues.length).toBe(2);
    expect(response.explained_variance_ratio).toBeDefined();
    expect(response.explained_variance_ratio.length).toBe(2);
  });

  test('PCA fails gracefully with no data loaded', async ({ page }) => {
    await waitForServer(page);

    // Clear any previously loaded data
    await clearData(page);

    const status = await page.evaluate(async () => {
      const res = await fetch('/api/visualization/pca', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      return res.status;
    });

    expect(status).toBe(404);
  });

  test('PCA none mode when no color_by specified', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');

    const response = await page.evaluate(async () => {
      const res = await fetch('/api/visualization/pca', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      return res.json();
    });

    expect(response.success).toBe(true);
    expect(response.color_mode).toBe('none');
    expect(response.labels).toBeNull();
    expect(response.color_values).toBeNull();
  });

  test('PCA with scale=false returns different results than scale=true', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');

    const scaled = await page.evaluate(async () => {
      const res = await fetch('/api/visualization/pca', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scale: true }),
      });
      return res.json();
    });

    const unscaled = await page.evaluate(async () => {
      const res = await fetch('/api/visualization/pca', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scale: false }),
      });
      return res.json();
    });

    expect(scaled.success).toBe(true);
    expect(unscaled.success).toBe(true);

    // Eigenvalues should differ when scaling vs not
    expect(scaled.eigenvalues[0]).not.toBeCloseTo(unscaled.eigenvalues[0], 2);
  });
});

// ---------------------------------------------------------------------------
// Tests — PCA Chart Animation
// ---------------------------------------------------------------------------

test.describe('PCA Chart Animation', () => {
  test('Chart.js instance is created with animation config', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');

    await switchTab(page, 'analyze');
    await expect(page.locator('#btn-pca')).toBeEnabled({ timeout: 20_000 });

    const pcaPromise = waitForPcaResponse(page);
    await page.locator('#btn-pca').click();
    await pcaPromise;

    await expect(page.locator('#pca-chart-wrap')).toBeVisible({ timeout: 15_000 });

    const animConfig = await page.evaluate(() => {
      const chart = (window as any).pcaChartInstance;
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
    expect(animConfig!.duration).toBe(1500);
    expect(animConfig!.easing).toBe('easeOutQuart');
    expect(animConfig!.hasDelay).toBe(true);
  });

  test('PCA axis labels include variance percentages', async ({ page }) => {
    await waitForServer(page);
    await loadSampleDataset(page, 'Iris');

    await switchTab(page, 'analyze');

    const pcaPromise = waitForPcaResponse(page);
    await page.locator('#btn-pca').click();
    await pcaPromise;

    await expect(page.locator('#pca-chart-wrap')).toBeVisible({ timeout: 15_000 });

    const axisLabels = await page.evaluate(() => {
      const chart = (window as any).pcaChartInstance;
      if (!chart) return null;
      return {
        x: chart.config.options?.scales?.x?.title?.text,
        y: chart.config.options?.scales?.y?.title?.text,
      };
    });

    expect(axisLabels).not.toBeNull();
    expect(axisLabels!.x).toContain('PC1');
    expect(axisLabels!.x).toContain('%');
    expect(axisLabels!.y).toContain('PC2');
    expect(axisLabels!.y).toContain('%');
  });
});

// ---------------------------------------------------------------------------
// Tests — PCA with Dataset Switching
// ---------------------------------------------------------------------------

test.describe('PCA with Dataset Switching', () => {
  test('PCA produces different results for different datasets', async ({ page }) => {
    await waitForServer(page);

    // Load Iris
    await loadSampleDataset(page, 'Iris');

    const irisResult = await page.evaluate(async () => {
      const res = await fetch('/api/visualization/pca', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      return res.json();
    });

    const irisCount = irisResult.n_samples;

    // Load Diabetes
    await loadSampleDataset(page, 'Diabetes');

    const diabetesResult = await page.evaluate(async () => {
      const res = await fetch('/api/visualization/pca', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      return res.json();
    });

    expect(irisResult.success).toBe(true);
    expect(diabetesResult.success).toBe(true);
    expect(diabetesResult.n_samples).not.toBe(irisCount);
  });
});
