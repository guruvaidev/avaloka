/**
 * Error text the UI may show a user.
 *
 * Owner rule: the user sees a plain, code-written message; a traceback, a
 * stack frame or a raw exception (`ValueError('...')`, `KeyError: 'col'`)
 * stays in the server log. The server already sends plain words; this is the
 * last check before a string reaches the screen, so an older server or a new
 * leak shows the fallback instead of the raw text.
 */
export const RAW_ERROR_PATTERNS: readonly RegExp[] = [
  /Traceback \(most recent call last\)/,
  /File "[^"]*", line \d+/,
  /\b[A-Z]\w*(?:Error|Exception)\((?:'|")/,
  /\b[A-Z]\w*(?:Error|Exception)\s*:\s*\S/,
];

export const RUN_LOG_ERROR_LINE = "[error] The run stopped with an error. The technical details are in the server log.";

export function looksLikeRawError(text: unknown): boolean {
  if (text == null) return false;
  const s = typeof text === "string" ? text : JSON.stringify(text);
  return RAW_ERROR_PATTERNS.some((rx) => rx.test(s ?? ""));
}

/** `text` when it is plain words, else `fallback`. */
export function userFacingError(text: unknown, fallback: string): string {
  if (typeof text !== "string" || !text.trim() || looksLikeRawError(text)) return fallback;
  return text;
}

/** Run-log entries with each raw error entry replaced by one plain line. */
export function scrubLogLines(lines: string[]): string[] {
  const out: string[] = [];
  for (const line of lines) {
    if (looksLikeRawError(line)) {
      if (out[out.length - 1] !== RUN_LOG_ERROR_LINE) out.push(RUN_LOG_ERROR_LINE);
      continue;
    }
    out.push(line);
  }
  return out;
}
