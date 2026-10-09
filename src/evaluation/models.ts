// System performance (spec §33, §53-54): latency percentiles, tokens and cost, per mode and per
// model. Engine latency is measured apart from the model provider's (spec §54).

export function percentile(xs: number[], p: number): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.max(0, Math.ceil((p / 100) * s.length) - 1));
  return s[i];
}

export interface RunCost {
  /** total wall time of the answer */
  took_ms: number;
  /** of which the model provider (0 when no model ran) */
  model_ms: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
}

export interface PerformanceSummary {
  n: number;
  /** engine time: the answer's time without the provider's */
  engine_ms: { p50: number; p95: number; p99: number; mean: number };
  model_ms: { p50: number; p95: number; p99: number; calls: number };
  tokens: { input: number; output: number; per_call_input: number; per_call_output: number };
  cost_usd: { total: number; per_question: number; per_1000: number };
}

export function summarizePerformance(runs: RunCost[]): PerformanceSummary {
  const engine = runs.map((r) => Math.max(0, r.took_ms - r.model_ms));
  const withModel = runs.filter((r) => r.model_ms > 0 || r.input_tokens > 0);
  const model = withModel.map((r) => r.model_ms);
  const input = runs.reduce((s, r) => s + r.input_tokens, 0);
  const output = runs.reduce((s, r) => s + r.output_tokens, 0);
  const cost = runs.reduce((s, r) => s + r.cost_usd, 0);
  const r1 = (x: number) => Math.round(x * 10) / 10;
  return {
    n: runs.length,
    engine_ms: { p50: percentile(engine, 50), p95: percentile(engine, 95), p99: percentile(engine, 99), mean: r1(engine.reduce((s, x) => s + x, 0) / Math.max(1, engine.length)) },
    model_ms: { p50: percentile(model, 50), p95: percentile(model, 95), p99: percentile(model, 99), calls: withModel.length },
    tokens: { input, output, per_call_input: withModel.length ? Math.round(input / withModel.length) : 0, per_call_output: withModel.length ? Math.round(output / withModel.length) : 0 },
    cost_usd: { total: cost, per_question: runs.length ? cost / runs.length : 0, per_1000: runs.length ? (1000 * cost) / runs.length : 0 },
  };
}
