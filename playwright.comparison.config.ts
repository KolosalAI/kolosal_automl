import { defineConfig } from '@playwright/test';

/**
 * Playwright config for remote performance comparison tests.
 * Compares Rust (Railway) vs Python/Gradio (Railway) — no local webServer needed.
 */
export default defineConfig({
  testDir: './e2e/comparison',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0, // No retries for benchmarks — we want clean timing data
  workers: 1, // Sequential execution for accurate benchmarks
  reporter: [
    ['list'],
    ['html', { outputFolder: 'playwright-report-comparison', open: 'never' }],
    ['json', { outputFile: 'playwright-report-comparison/results.json' }],
  ],
  timeout: 180_000, // 3 minutes per test — remote servers may be cold
  use: {
    // No baseURL — tests use absolute URLs for both platforms
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
    // Use a consistent viewport
    viewport: { width: 1280, height: 720 },
  },
  // No webServer — both targets are remote Railway deployments
});
