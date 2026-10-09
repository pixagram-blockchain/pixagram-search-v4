import { describe, expect, it } from "vitest";
import { contentTokens, editDistance, fnv1a, fold, guessLang, isStopword, singular, tokens, trigrams } from "../src/lib/text";
import { aliasesOf, ancestors, ancestorsWithDepth, extractArtworkConcepts, matchConcepts, STRUCTURAL } from "../src/concepts";

const ids = (s: string) => matchConcepts(s).map((m) => m.concept);

describe("text", () => {
  it("folds case, width and diacritics, keeps CJK", () => {
    expect(fold("Crème BRÛLÉE")).toBe("creme brulee");
    expect(fold("Ｃａｔ")).toBe("cat");
    expect(fold("Mädchen")).toBe("madchen");
    expect(fold("猫の王様")).toBe("猫の王様");
  });

  it("tokenises: apostrophes split, hyphenated words whole and in parts", () => {
    expect(tokens("L'avion d'Éric")).toEqual(["l", "avion", "d", "eric"]);
    expect(tokens("sponge-bob rocks")).toEqual(["sponge-bob", "sponge", "bob", "rocks"]);
    expect(tokens("sponge-bob", { keepHyphenated: false })).toEqual(["sponge", "bob"]);
    expect(contentTokens("the cat in the hat")).toEqual(["cat", "hat"]);
    expect(contentTokens("the the")).toEqual(["the", "the"]); // nothing but stopwords: keep them
    expect(isStopword("dans")).toBe(true);
    expect(isStopword("chat")).toBe(false);
  });

  it("singularises English/French plurals for matching only", () => {
    expect(singular("cats")).toBe("cat");
    expect(singular("butterflies")).toBe("butterfly");
    expect(singular("boxes")).toBe("box");
    expect(singular("bus")).toBe("bus");
    expect(singular("glass")).toBe("glass");
    expect(singular("gateaux")).toBe("gateau");
    expect(singular("sky")).toBe("sky");
  });

  it("edit distance counts adjacent transpositions once and stops early", () => {
    expect(editDistance("kitten", "sitting")).toBe(3);
    expect(editDistance("elodrado", "eldorado", 2)).toBe(1); // one adjacent transposition
    expect(editDistance("chta", "chat", 1)).toBe(1);
    expect(editDistance("teh", "the", 1)).toBe(1);
    expect(editDistance("abcdefgh", "zzzzzzzz", 2)).toBe(3); // capped at max + 1
    expect(editDistance("cat", "cat")).toBe(0);
  });

  it("trigrams are padded and unique", () => {
    expect(trigrams("cat")).toEqual(["^ca", "cat", "at$"]);
    expect(trigrams("aaaa")).toEqual(["^aa", "aaa", "aa$"]);
  });

  it("guesses the language of short questions", () => {
    expect(guessLang("Qui a posté le premier chat ?")).toBe("fr");
    expect(guessLang("Wer hat die erste Katze gepostet?")).toBe("de");
    expect(guessLang("who posted the first cat")).toBe("en");
    expect(guessLang("最初の猫")).toBe("ja");
    expect(guessLang("dragon")).toBe("en");
  });

  it("fnv1a is stable", () => {
    expect(fnv1a("")).toBe("811c9dc5");
    expect(fnv1a("pixagram")).toBe(fnv1a("pixagram"));
    expect(fnv1a("a")).not.toBe(fnv1a("b"));
  });
});

describe("concepts", () => {
  it("match the same concept in every language", () => {
    for (const q of ["cat", "cats", "chat", "Katze", "gatos", "gatto", "猫", "ねこ"]) expect(ids(q), q).toContain("cat");
    expect(ids("chien et chat")).toEqual(["dog", "cat"]);
  });

  it("prefer the longest alias and do not overlap", () => {
    expect(ids("mirror selfie")).toEqual(["mirror selfie"]);
    expect(ids("selfie")).toEqual(["selfie"]);
    expect(ids("coucher de soleil sur la plage")).toEqual(["sunset", "beach"]);
  });

  it("ignore one-word aliases that are function words elsewhere", () => {
    // French "thé" folds to "the": it must not tag every English sentence with tea
    expect(ids("the king of the cats")).toEqual(["king", "cat"]);
    expect(ids("the end")).not.toContain("drink");
    expect(ids("une tasse de thé")).toContain("drink"); // "tasse" still says it
  });

  it("match glued tags, hyphen parts and CJK runs", () => {
    expect(ids("greeneyes")).toContain("eyes");
    expect(ids("videocall")).toContain("video call");
    expect(ids("cat-girl")).toEqual(expect.arrayContaining(["cat", "woman"]));
    expect(ids("猫の王様")).toContain("cat");
  });

  it("expose parents with their depth", () => {
    expect(ancestors("cat")).toEqual(expect.arrayContaining(["pet", "animal"]));
    const d = ancestorsWithDepth("mirror selfie");
    expect(d.find((a) => a.id === "selfie")?.depth).toBe(1);
    expect(d.find((a) => a.id === "portrait")?.depth).toBe(2);
    expect(aliasesOf("cat")).toEqual(expect.arrayContaining(["cat", "chat", "katze", "猫"]));
  });

  it("extract artwork concepts with a confidence per source, parents decayed, no structural ids", () => {
    const c = extractArtworkConcepts({
      tags: ["kitty", "space_invader"],
      title: "Le roi des chats",
      description: "",
      ai: { subjects: ["robot"], objects: [], tags: [], caption: "A cat on a beach at sunset.", style: "pixel art" },
    });
    const by = new Map(c.map((x) => [x.concept, x]));
    expect(by.get("cat")).toMatchObject({ confidence: 0.95, source: "tag" }); // tag beats title and caption
    expect(by.get("robot")).toMatchObject({ confidence: 0.85, source: "vlm" });
    expect(by.get("beach")?.confidence).toBe(0.65);
    expect(by.get("pet")).toMatchObject({ confidence: 0.76, source: "parent" }); // 0.95 · 0.8
    for (const x of c) expect(STRUCTURAL.has(x.concept), x.concept).toBe(false);
    expect(c.map((x) => x.confidence)).toEqual([...c.map((x) => x.confidence)].sort((a, b) => b - a));
  });
});
