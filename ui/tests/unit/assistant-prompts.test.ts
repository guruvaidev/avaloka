// Avaloka's per-step prompts, as the chat thread reads them from a response.
import { describe, expect, it } from "vitest";
import { promptsToMessages, readAssistantPrompts } from "@/lib/assistant-prompts";

const response = {
  messages: [{ role: "assistant", content: "The average is 25,400." }],
  assistant_prompts: [
    {
      step: "sampling",
      message: "I worked on 7,013 of 55,500 rows (Portfolio Samples) so you'd have an answer quickly.",
      question: "Want me to rerun on the entire dataset for the final number?",
      quick_replies: [{ label: "Use the entire dataset", send: "Use the entire dataset" }, "Keep the sample"],
    },
    {
      step: "swarm",
      message: "My team is back. Here's what they did:\n- Team size: 2 clones",
      question: "Want me to explain the verdict?",
      quick_replies: [],
      link: { label: "Watch it on the swarm page", href: "/swarm?mission=abc" },
    },
  ],
};

describe("readAssistantPrompts", () => {
  it("reads every prompt and normalises bare-string quick replies", () => {
    const prompts = readAssistantPrompts(response);
    expect(prompts.map((p) => p.step)).toEqual(["sampling", "swarm"]);
    expect(prompts[0]!.quick_replies).toEqual([
      { label: "Use the entire dataset", send: "Use the entire dataset" },
      { label: "Keep the sample", send: "Keep the sample" },
    ]);
    expect(prompts[1]!.link).toEqual({ label: "Watch it on the swarm page", href: "/swarm?mission=abc" });
  });

  it("yields nothing for an older API or a strange shape, and drops only the broken entry", () => {
    expect(readAssistantPrompts({ messages: [] })).toEqual([]);
    expect(readAssistantPrompts(null)).toEqual([]);
    expect(readAssistantPrompts({ assistant_prompts: "nope" })).toEqual([]);
    expect(readAssistantPrompts({ assistant_prompts: [null, { step: "x" }, { message: "kept" }] })).toHaveLength(1);
  });
});

describe("promptsToMessages", () => {
  it("makes one AI bubble per step with the question appended and the buttons attached", () => {
    const msgs = promptsToMessages(response, "p1", "9:41 AM");
    expect(msgs).toHaveLength(2);
    expect(msgs[0]!.role).toBe("ai");
    expect(msgs[0]!.content).toBe(
      "I worked on 7,013 of 55,500 rows (Portfolio Samples) so you'd have an answer quickly.\n\nWant me to rerun on the entire dataset for the final number?",
    );
    expect(msgs[0]!.quickReplies).toHaveLength(2);
    expect(msgs[1]!.quickReplies).toBeUndefined();
    expect(msgs[1]!.link?.href).toBe("/swarm?mission=abc");
    expect(new Set(msgs.map((m) => m.id)).size).toBe(2);
  });

  it("does not tell the swarm story twice when it is already the assistant message", () => {
    const story = "My team is back. Here's what they did:\n- Team size: 2 clones";
    const msgs = promptsToMessages(response, "p2", "9:41 AM", story);
    expect(msgs[1]!.content).toBe("Want me to explain the verdict?");
    expect(msgs[0]!.content).toContain("I worked on 7,013");
  });
});
