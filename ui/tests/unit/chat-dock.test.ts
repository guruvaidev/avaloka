import { describe, expect, it } from "vitest";
import { CHAT_DOCK_STORAGE_KEY, readChatDock, writeChatDock } from "../../src/lib/chat-dock";

function fakeStorage(initial: Record<string, string> = {}) {
  const map = new Map(Object.entries(initial));
  return {
    getItem: (key: string) => (map.has(key) ? map.get(key)! : null),
    setItem: (key: string, value: string) => void map.set(key, String(value)),
    map,
  };
}

describe("chat dock mode persistence", () => {
  it("defaults to right when nothing is stored", () => {
    expect(readChatDock(fakeStorage())).toBe("right");
  });

  it("restores a stored left choice", () => {
    expect(readChatDock(fakeStorage({ [CHAT_DOCK_STORAGE_KEY]: "left" }))).toBe("left");
  });

  it("restores a stored float choice", () => {
    expect(readChatDock(fakeStorage({ [CHAT_DOCK_STORAGE_KEY]: "float" }))).toBe("float");
  });

  it("falls back to right for an unknown stored value", () => {
    expect(readChatDock(fakeStorage({ [CHAT_DOCK_STORAGE_KEY]: "popout" }))).toBe("right");
    expect(readChatDock(fakeStorage({ [CHAT_DOCK_STORAGE_KEY]: "window" }))).toBe("right");
  });

  it("persists the choice and restores it", () => {
    const storage = fakeStorage();
    writeChatDock("left", storage);
    expect(storage.map.get(CHAT_DOCK_STORAGE_KEY)).toBe("left");
    expect(readChatDock(storage)).toBe("left");
    writeChatDock("float", storage);
    expect(readChatDock(storage)).toBe("float");
    writeChatDock("right", storage);
    expect(readChatDock(storage)).toBe("right");
  });

  it("survives a throwing storage", () => {
    const broken = {
      getItem: () => {
        throw new Error("denied");
      },
      setItem: () => {
        throw new Error("denied");
      },
    };
    expect(readChatDock(broken)).toBe("right");
    expect(() => writeChatDock("float", broken)).not.toThrow();
  });
});
