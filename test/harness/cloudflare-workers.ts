// Stand-in for the "cloudflare:workers" module so Node tests can import the Durable Object and
// Workflow classes (and everything that imports them, such as the Hono app).
export class DurableObject<E = unknown> {
  ctx: any;
  env: E;
  constructor(ctx: any, env: E) {
    this.ctx = ctx;
    this.env = env;
  }
}
export class WorkflowEntrypoint<E = unknown, P = unknown> {
  ctx: any;
  env: E;
  constructor(ctx: any, env: E) {
    this.ctx = ctx;
    this.env = env;
  }
  async run(_event: { payload: P; instanceId: string }, _step: any): Promise<unknown> {
    return undefined;
  }
}
export type WorkflowEvent<P> = { payload: P; instanceId: string; timestamp: Date };
export type WorkflowStep = {
  do<T>(name: string, a: any, b?: any): Promise<T>;
};
