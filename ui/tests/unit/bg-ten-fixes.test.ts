import { describe, expect, it } from "vitest";
import { getKeyFindings, messagesWithKeyFindings, unionFindings } from "../../src/lib/workspace-findings";
import {
  CHAT_WIDTH_MIN,
  CHAT_WIDTH_STORAGE_KEY,
  clampChatWidth,
  clearChatWidth,
  maxChatWidth,
  readChatWidth,
  writeChatWidth,
} from "../../src/lib/chat-width";
import { shouldStartReplyDrag } from "../../src/lib/reply-drag";
import { exactBytesLabel, formatBytes } from "../../src/lib/format-bytes";
import { storageConnectionFromSource } from "../../src/lib/connection-card";
import type { DataSource } from "../../src/lib/data-sources";

function memoryStorage() {
  const map = new Map<string, string>();
  return {
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => void map.set(k, v),
    removeItem: (k: string) => void map.delete(k),
    map,
  };
}

const headline = "A tiny fraction of channels generate the overwhelming majority of video views.";
const withNumbers = "A tiny fraction of channels generate the overwhelming majority of video views. The top 1% hold 38% of views.";

describe("key findings (issue 1)", () => {
  it("keeps only the longer bullet when one starts with another", () => {
    expect(unionFindings([headline], [withNumbers])).toEqual([withNumbers]);
    expect(unionFindings([withNumbers, "  a TINY fraction of channels generate the overwhelming majority of video views!"], [])).toEqual([withNumbers]);
  });

  it("chart card and chat read the same list in the same order, without duplicates", () => {
    const messages = [
      { role: "user", content: "hi" },
      { role: "ai", content: `Dataset\n\n**Key findings**\n- ${headline}\n- Views are skewed.\n\n**Try asking**\n- Why?` },
    ];
    const display = messagesWithKeyFindings(messages, [withNumbers, "Views are skewed"]);
    const chatBullets = display[1].content.split("\n").filter((l) => l.startsWith("- ")).slice(0, 2).map((l) => l.slice(2));
    const card = getKeyFindings(display);
    expect(card).toEqual([withNumbers, "Views are skewed."]);
    expect(chatBullets).toEqual(card);
    expect(display[1].content.match(/A tiny fraction/g)).toHaveLength(1);
  });

  it("deduplicates the chat even when there are no chart findings to merge", () => {
    const display = messagesWithKeyFindings([{ role: "ai", content: `**Key findings**\n- ${headline}\n- ${withNumbers}` }], []);
    expect(getKeyFindings(display)).toEqual([withNumbers]);
  });
});

describe("chat width (issue 2)", () => {
  it("clamps to 320px and min(720px, 55% of the window) while keeping 480px for data", () => {
    expect(clampChatWidth(100, 1920)).toBe(CHAT_WIDTH_MIN);
    expect(clampChatWidth(5000, 1920)).toBe(720);
    expect(maxChatWidth(1280)).toBe(Math.round(Math.min(720, 1280 * 0.55, 1280 - 112 - 480)));
    expect(clampChatWidth(5000, 1280)).toBe(maxChatWidth(1280));
  });

  it("persists, restores clamped, and resets", () => {
    const storage = memoryStorage();
    writeChatWidth(560, storage);
    expect(storage.map.get(CHAT_WIDTH_STORAGE_KEY)).toBe("560");
    expect(readChatWidth(1920, storage)).toBe(560);
    expect(readChatWidth(1100, storage)).toBe(maxChatWidth(1100));
    clearChatWidth(storage);
    expect(readChatWidth(1920, storage)).toBeNull();
  });

  it("survives broken storage", () => {
    const broken = { getItem: () => { throw new Error("x"); }, setItem: () => { throw new Error("x"); }, removeItem: () => { throw new Error("x"); } };
    expect(readChatWidth(1920, broken)).toBeNull();
    expect(() => writeChatWidth(400, broken)).not.toThrow();
  });
});

describe("drag to reply (issue 4)", () => {
  it("never starts from message text or during a selection", () => {
    expect(shouldStartReplyDrag({ fromHandle: false })).toBe(false);
    expect(shouldStartReplyDrag({ fromHandle: true, selectionText: "copied part" })).toBe(false);
    expect(shouldStartReplyDrag({ fromHandle: true, selectionText: "", button: 0 })).toBe(true);
  });
});

describe("formatBytes (issue 9)", () => {
  it("formats binary units with one decimal and drops .0", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(287)).toBe("287 B");
    expect(formatBytes(1037)).toBe("1 KB");
    expect(formatBytes(200279)).toBe("195.6 KB");
    expect(formatBytes(4.2 * 1024 * 1024)).toBe("4.2 MB");
    expect(formatBytes(1.3 * 1024 ** 3)).toBe("1.3 GB");
    expect(formatBytes("200279")).toBe("195.6 KB");
    expect(formatBytes(null)).toBe("—");
    expect(formatBytes(undefined)).toBe("—");
    expect(exactBytesLabel(200279)).toBe("200,279 bytes");
  });
});

describe("connection card (issue 8)", () => {
  it("targets that connection's own bucket, ignoring previously selected files", () => {
    const source = {
      id: "conn-1",
      name: "avaloka-test-user-filestore",
      provider: "gcs",
      connectionType: "storage",
      storageProvider: "gcp",
      status: "connected",
      config: { bucket_name: "avaloka-test-user-filestore", selected_files: ["yellow_tripdata.csv"] },
    } as unknown as DataSource;
    expect(storageConnectionFromSource(source)).toEqual({
      id: "conn-1",
      name: "avaloka-test-user-filestore",
      provider: "gcp",
      bucket_name: "avaloka-test-user-filestore",
    });
  });
});
