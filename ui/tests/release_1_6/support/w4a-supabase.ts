// Test double for "@/integrations/supabase/client" used by the 1.6 release
// suite. Every query chain (from().select().eq()...) is awaitable and resolves
// to the rows registered for its table, so components render their real
// branches without a network. Not a test file (no .test. in the name).
import { vi } from "vitest";

export function supabaseClientMock(tables: Record<string, unknown[]> = {}) {
  const chain = (table: string): unknown => {
    const rows = tables[table] ?? [];
    const result = { data: rows, error: null };
    const proxy: unknown = new Proxy(function noop() {}, {
      get(_target, prop) {
        if (prop === "then") {
          return (resolve: (v: unknown) => unknown, reject: (e: unknown) => unknown) =>
            Promise.resolve(result).then(resolve, reject);
        }
        if (prop === "maybeSingle" || prop === "single") {
          return () => Promise.resolve({ data: rows[0] ?? null, error: null });
        }
        return () => proxy;
      },
    });
    return proxy;
  };
  const channel = () => {
    const ch: Record<string, unknown> = {};
    ch.on = () => ch;
    ch.subscribe = () => ch;
    ch.unsubscribe = () => undefined;
    return ch;
  };
  return {
    supabase: {
      from: vi.fn((table: string) => chain(table)),
      rpc: vi.fn(async () => ({ data: null, error: null })),
      channel: vi.fn(channel),
      removeChannel: vi.fn(),
      auth: {
        getSession: vi.fn(async () => ({ data: { session: { access_token: "test-token" } }, error: null })),
        getUser: vi.fn(async () => ({ data: { user: { id: "user-1" } }, error: null })),
        onAuthStateChange: vi.fn(() => ({ data: { subscription: { unsubscribe: () => undefined } } })),
      },
      storage: { from: vi.fn(() => chain("storage")) },
    },
  };
}
