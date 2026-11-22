import { afterEach, describe, expect, it, vi } from "vitest";
import { clearPendingChatPrompt, deliverPendingChatPrompt, getPendingChatPrompt, setPendingChatPrompt } from "../../src/lib/pending-chat-prompt";

afterEach(() => clearPendingChatPrompt());

describe("pending chat prompt", () => {
  it("is cleared after sending; a re-mounted screen does not send again", async () => {
    setPendingChatPrompt("Correlate growth", "a1");
    const send = vi.fn(async () => true);
    const sentIds = new Set<string>();
    expect(await deliverPendingChatPrompt({ aid: "a1", isBusy: false, sentIds, send, onFailed: vi.fn() })).toBe("sent");
    expect(getPendingChatPrompt()).toBeNull();
    // fresh mount: new ref set, nothing pending
    expect(await deliverPendingChatPrompt({ aid: "a1", isBusy: false, sentIds: new Set(), send, onFailed: vi.fn() })).toBe("none");
    expect(send).toHaveBeenCalledTimes(1);
    expect(send).toHaveBeenCalledWith("Correlate growth");
  });

  it("waits while the chat is busy, then sends once idle", async () => {
    setPendingChatPrompt("Next step", "a1");
    const send = vi.fn(async () => true);
    const sentIds = new Set<string>();
    expect(await deliverPendingChatPrompt({ aid: "a1", isBusy: true, sentIds, send, onFailed: vi.fn() })).toBe("waiting");
    expect(send).not.toHaveBeenCalled();
    expect(getPendingChatPrompt()?.text).toBe("Next step");
    expect(await deliverPendingChatPrompt({ aid: "a1", isBusy: false, sentIds, send, onFailed: vi.fn() })).toBe("sent");
    expect(send).toHaveBeenCalledTimes(1);
  });

  it("a failed send puts the text back in the input and does not retry", async () => {
    setPendingChatPrompt("Impute missing values", "a1");
    const send = vi.fn(async () => false);
    const onFailed = vi.fn();
    const sentIds = new Set<string>();
    expect(await deliverPendingChatPrompt({ aid: "a1", isBusy: false, sentIds, send, onFailed })).toBe("failed");
    expect(onFailed).toHaveBeenCalledWith("Impute missing values");
    await deliverPendingChatPrompt({ aid: "a1", isBusy: false, sentIds, send, onFailed });
    expect(send).toHaveBeenCalledTimes(1);
  });

  it("opens the chat panel before sending", async () => {
    setPendingChatPrompt("Remove constant column", "a1");
    const order: string[] = [];
    await deliverPendingChatPrompt({ aid: "a1", isBusy: false, sentIds: new Set(), send: async () => { order.push("send"); return true; }, onBeforeSend: () => order.push("open"), onFailed: vi.fn() });
    expect(order).toEqual(["open", "send"]);
  });

  it("double effect runs with the same ref send only once", async () => {
    setPendingChatPrompt("Once", "a1");
    const send = vi.fn(async () => true);
    const sentIds = new Set<string>();
    await Promise.all([
      deliverPendingChatPrompt({ aid: "a1", isBusy: false, sentIds, send, onFailed: vi.fn() }),
      deliverPendingChatPrompt({ aid: "a1", isBusy: false, sentIds, send, onFailed: vi.fn() }),
    ]);
    expect(send).toHaveBeenCalledTimes(1);
  });
});
