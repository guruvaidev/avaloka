import { describe, expect, it } from "vitest";
import {
  CHATBOT_INTRO,
  CHATBOT_RELEASE_LINE,
  chatIntroKey,
  chatbotIntro,
  isNewChatThread,
  readIntroDismissed,
  writeIntroDismissed,
} from "../../src/lib/chatbot-intro";

// Per-line allowlist. Every word of the intro is code-written, so it must be
// true for the line it ships on; docs/design/1.6-CHATBOT-INTRO.md holds the
// evidence for each claim. Changing the copy means changing this list on
// purpose, after re-checking the claims.
const ALLOWED: Record<string, string> = {
  "1.6":
    "Hi, I'm Avaloka, a data scientist you can talk to. Tell me your goal in plain English: " +
    "analyse this data, train a model on it, or look at a trend over time. I'll find the data " +
    "sources you've connected, suggest how your datasets join, plan the steps, move and analyse " +
    "the data, and test my code on a sample of your data before I run it. What would you like " +
    "to achieve?",
};

// Claims that are not true on 1.6: the swarm and the outcomes work are 2.x,
// learning from feedback is 1.8 and not built. The last three were in the
// owner's draft and failed the 1.6 truth check (design note, truth table).
const FORBIDDEN_ON_16 = [
  /swarm/i,
  /\bclon(e|es|ing)\b/i,
  /what[- ]if/i,
  /recommend/i,
  /knowledge graph/i,
  /\blearn/i,
  /get(ting)? better/i,
  /(better|improv\w*|learn\w*)[^.]*over time/i,
  /in parallel/i,
  /forecast/i,
  /secure/i,
  /check my numbers/i,
];

function storage() {
  const data = new Map<string, string>();
  return {
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => void data.set(k, v),
  };
}

describe("chatbot intro copy", () => {
  it("ships only lines that have an allowlisted copy", () => {
    expect(Object.keys(CHATBOT_INTRO).sort()).toEqual(Object.keys(ALLOWED).sort());
    expect(CHATBOT_RELEASE_LINE).toBe("1.6");
  });

  it.each(Object.keys(ALLOWED))("line %s matches its allowlisted copy exactly", (line) => {
    expect(CHATBOT_INTRO[line as keyof typeof CHATBOT_INTRO]).toBe(ALLOWED[line]);
  });

  it.each(FORBIDDEN_ON_16.map((re) => [re.source, re] as const))(
    "the 1.6 copy never claims %s",
    (_name, re) => {
      expect(chatbotIntro("1.6")).not.toMatch(re);
    },
  );
});

describe("intro dismissal state", () => {
  it("is per user and thread", () => {
    const s = storage();
    expect(readIntroDismissed("u1", "t1", s)).toBe(false);
    writeIntroDismissed("u1", "t1", s);
    expect(readIntroDismissed("u1", "t1", s)).toBe(true);
    expect(readIntroDismissed("u1", "t2", s)).toBe(false);
    expect(readIntroDismissed("u2", "t1", s)).toBe(false);
    expect(chatIntroKey(null, null)).toBe("avaloka.chatIntro.dismissed:anon:draft");
  });

  it("survives a storage that throws", () => {
    const broken = {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
    };
    expect(() => writeIntroDismissed("u", "t", broken)).not.toThrow();
    expect(readIntroDismissed("u", "t", broken)).toBe(false);
  });

  it("a thread stops being new at its first user message, not at the upload summary", () => {
    expect(isNewChatThread([])).toBe(true);
    expect(isNewChatThread([{ role: "ai" }])).toBe(true);
    expect(isNewChatThread([{ role: "ai" }, { role: "user" }])).toBe(false);
  });
});
