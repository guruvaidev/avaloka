/** Display-only union; never modifies persisted messages or sends a request. */
export function normalizeFinding(text: string): string {
  return text.trim().toLowerCase().replace(/\s+/g, " ").replace(/[\p{P}]+$/gu, "").trim();
}

/**
 * Each finding once: exact repeats are dropped, and when one bullet's normalised
 * text starts with another bullet's full text only the longer one is kept (at
 * the position of the first copy).
 */
export function unionFindings(existing: string[], incoming: string[]): string[] {
  const out: { key: string; text: string }[] = [];
  for (const text of existing.concat(incoming)) {
    const key = normalizeFinding(text);
    if (!key) continue;
    const index = out.findIndex((item) => item.key.startsWith(key) || key.startsWith(item.key));
    if (index < 0) out.push({ key, text });
    else if (key.length > out[index].key.length) out[index] = { key, text };
  }
  return out.map((item) => item.text);
}

export function withKeyFindings(content: string, bullets: string[]): string {
  const lines = content.split("\n");
  const heading = lines.findIndex((line) => /^(?:#{1,6}\s+)?key findings\s*:?[ \t]*$/i.test(line.trim().replace(/\*\*/g, "")));
  if (heading < 0 && !bullets.length) return content;
  if (heading < 0) {
    const section = ["**Key findings**", ...unionFindings([], bullets).map((text) => `- ${text}`)].join("\n");
    const suggestions = lines.findIndex((line) => /^(?:#{1,6}\s+|\*\*)?try asking/i.test(line.trim()));
    if (suggestions < 0) return `${content.trimEnd()}\n\n${section}`;
    lines.splice(suggestions, 0, section, "");
    return lines.join("\n");
  }
  let end = heading + 1;
  const existing: string[] = [];
  while (end < lines.length) {
    const bullet = lines[end].match(/^\s*[-*+]\s+(.+)$/);
    if (bullet) existing.push(bullet[1]);
    else if (lines[end].trim()) break;
    end += 1;
  }
  const combined = unionFindings(existing, bullets);
  if (combined.length === existing.length && combined.every((text, i) => text === existing[i])) return content;
  lines.splice(heading + 1, end - heading - 1, ...combined.map((text) => `- ${text}`), "");
  return lines.join("\n");
}

export function messagesWithKeyFindings<T extends { role: string; content: string; thinking?: boolean; error?: boolean }>(messages: T[], bullets: string[]): T[] {
  const first = messages.findIndex((message) => message.role === "ai" && !message.thinking && !message.error);
  if (first < 0) return messages;
  const content = withKeyFindings(messages[first].content, bullets);
  if (content === messages[first].content) return messages;
  return messages.map((message, index) => index === first ? { ...message, content } : message);
}
/** Splits a message around its "Key findings" section (heading + bullets) so a structured list can replace it. */
export function splitAtKeyFindings(content: string): { before: string; after: string; found: boolean } {
  const lines = content.split("\n");
  const heading = lines.findIndex((line) => /^(?:#{1,6}\s+)?key findings\s*:?[ \t]*$/i.test(line.trim().replace(/\*\*/g, "")));
  if (heading < 0) {
    const suggestions = lines.findIndex((line) => /^(?:#{1,6}\s+|\*\*)?(watch out for|try asking)/i.test(line.trim()));
    if (suggestions < 0) return { before: content, after: "", found: false };
    return { before: lines.slice(0, suggestions).join("\n"), after: lines.slice(suggestions).join("\n"), found: false };
  }
  let end = heading + 1;
  while (end < lines.length && (/^\s*[-*+]\s+/.test(lines[end]) || !lines[end].trim())) end += 1;
  return { before: lines.slice(0, heading).join("\n"), after: lines.slice(end).join("\n"), found: true };
}

/** Bullets currently listed under a message's "Key findings" heading. */
export function keyFindingBullets(content: string): string[] {
  const lines = content.split("\n");
  const heading = lines.findIndex((line) => /^(?:#{1,6}\s+)?key findings\s*:?[ \t]*$/i.test(line.trim().replace(/\*\*/g, "")));
  if (heading < 0) return [];
  const out: string[] = [];
  for (let i = heading + 1; i < lines.length; i += 1) {
    const bullet = lines[i].match(/^\s*[-*+]\s+(.+)$/);
    if (bullet) out.push(bullet[1].trim());
    else if (lines[i].trim()) break;
  }
  return out;
}

/**
 * Single source for "Key findings": the deduplicated bullets of the first AI
 * message as displayed in chat. The card under the charts reads this same list.
 */
export function getKeyFindings<T extends { role: string; content: string; thinking?: boolean; error?: boolean }>(messages: T[]): string[] {
  const first = messages.find((message) => message.role === "ai" && !message.thinking && !message.error);
  return first ? unionFindings([], keyFindingBullets(first.content)) : [];
}
