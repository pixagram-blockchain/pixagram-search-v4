// Evidence verification (spec §16, §24): before anything reasons over the evidence, every card is
// checked against the index itself, and evidence that disagrees is reported instead of resolved.
//
//   id            the post exists, under that author and permlink, deleted or not as the card says
//   author        a valid account name
//   timestamps    finite, inside the chain's era, in order (first seen ≤ image since ≤ now)
//   history       an exact history has chain operations behind it (else: a warning)
//   image         an exact first sighting in another post is a version of that post with the
//                 same bytes (else: a warning)
//   duplicates    cards showing the same image are grouped (several cards, one image)
//   conflicts     a chain operation showing the image earlier than the card's first sighting;
//                 two cards of the same image giving different first sightings
//
// Invalid evidence is not presented as authoritative: it stays out of the deterministic answer's
// support and is flagged for the model. A conflict becomes a C-card the model must report.

import type { Env } from "../env";
import { now } from "../env";
import { fmtDate } from "./answer-text";
import { CREATE_SKEW } from "../chain/versions";
import type { ArtworkCard, ConflictCard, EvidenceCard } from "./evidence";

export interface EvidenceVerification {
  /** cards that failed a critical check, with why: they support no answer */
  invalid: Map<string, string[]>;
  /** minor issues, reported to the model with the card */
  warnings: Map<string, string[]>;
  conflicts: ConflictCard[];
  /** cards showing the same image bytes, grouped */
  duplicates: string[][];
  checked: number;
  notes: string[];
}

const ACCOUNT = /^[a-z0-9][a-z0-9.-]{1,31}$/;
const PATH = /^\/@([a-z0-9][a-z0-9.-]{1,31})\/([^/\s]+)$/;
const HASH = /^[0-9a-f]{64}$/;
/** The earliest a post can be (STEEM's genesis: nothing on a fork of it predates it). */
const ERA_START = Date.UTC(2016, 2, 24) / 1000;

const toTs = (s?: string) => (s ? Math.floor(Date.parse(s) / 1000) : null);

/** Issues that make a card unfit to support an answer (an answer resting on it is not authoritative). */
export const CRITICAL = new Set([
  "path and author disagree",
  "invalid account name",
  "creation time out of range",
  "image shown before the post existed",
  "first sighting out of range",
  "first sighting after this post showed the image",
  "first sighting after this post was created",
  "no such post in the index",
  "the post's author or permlink differs from the index",
  "deletion state differs from the index",
]);

/** Checks of a card's own fields, without the database. Minor issues (a malformed image hash) are reported, not disqualifying. */
export function cardIssues(c: ArtworkCard, at = now()): string[] {
  const issues: string[] = [];
  const m = PATH.exec(c.path);
  if (!m || m[1] !== c.author) issues.push("path and author disagree");
  if (!ACCOUNT.test(c.author)) issues.push("invalid account name");
  const created = toTs(c.created_at);
  if (created === null || !Number.isFinite(created) || created < ERA_START || created > at + 86400) issues.push("creation time out of range");
  const since = toTs(c.image_since_at);
  const first = toTs(c.first_seen_at);
  if (since !== null && created !== null && since + CREATE_SKEW < created) issues.push("image shown before the post existed");
  if (first !== null && (first < ERA_START || first > at + 86400)) issues.push("first sighting out of range");
  if (first !== null && since !== null && first > since + CREATE_SKEW) issues.push("first sighting after this post showed the image");
  if (first !== null && since === null && created !== null && first > created + CREATE_SKEW && c.first_seen_match !== "self") issues.push("first sighting after this post was created");
  if (c.image && !HASH.test(c.image)) issues.push("malformed image hash");
  if (c.first_seen_match && !["exact", "near", "self"].includes(c.first_seen_match)) issues.push("unknown identity match");
  return issues;
}

export interface ConflictText {
  first_seen: string;
}

const CONFLICT_TEXT: Record<string, (x: { a: string; b: string; ea: string; eb: string }) => string> = {
  en: (x) => `The records disagree on when this image first appeared: ${x.a} (${x.ea}) or ${x.b} (${x.eb}).`,
  fr: (x) => `Les données divergent sur la première apparition de cette image : ${x.a} (${x.ea}) ou ${x.b} (${x.eb}).`,
  de: (x) => `Die Daten widersprechen sich, wann dieses Bild zuerst erschien: ${x.a} (${x.ea}) oder ${x.b} (${x.eb}).`,
  es: (x) => `Los registros no coinciden sobre cuándo apareció esta imagen por primera vez: ${x.a} (${x.ea}) o ${x.b} (${x.eb}).`,
  it: (x) => `I dati non concordano su quando questa immagine è apparsa per la prima volta: ${x.a} (${x.ea}) o ${x.b} (${x.eb}).`,
};

const CONFLICT_ONE: Record<string, (x: { a: string; b: string; p: string }) => string> = {
  en: (x) => `The records disagree on when this image first appeared: the index says ${x.a}, but a chain operation showed it on ${x.b}${x.p}.`,
  fr: (x) => `Les données divergent sur la première apparition de cette image : l'index indique le ${x.a}, mais une opération de la chaîne la montrait le ${x.b}${x.p}.`,
  de: (x) => `Die Daten widersprechen sich, wann dieses Bild zuerst erschien: Der Index nennt den ${x.a}, eine Operation der Chain zeigte es aber am ${x.b}${x.p}.`,
  es: (x) => `Los registros no coinciden sobre cuándo apareció esta imagen por primera vez: el índice dice ${x.a}, pero una operación de la cadena la mostró el ${x.b}${x.p}.`,
  it: (x) => `I dati non concordano su quando questa immagine è apparsa per la prima volta: l'indice indica il ${x.a}, ma un'operazione della catena la mostrava il ${x.b}${x.p}.`,
};

/**
 * The sentence that explains a conflict in the answer (spec §24: the uncertainty is said, not
 * resolved). Posts are named by their paths when known, never by internal evidence ids alone.
 */
export function conflictSentence(c: ConflictCard, lang: string, pathOf: (id: string) => string | undefined = () => undefined): string {
  const name = (id: string) => pathOf(id) ?? id;
  if (c.evidence.length < 2) {
    const p = pathOf(c.evidence[0]);
    return (CONFLICT_ONE[lang] ?? CONFLICT_ONE.en)({ a: c.values[0], b: c.values[1], p: p ? ` (${p})` : "" });
  }
  return (CONFLICT_TEXT[lang] ?? CONFLICT_TEXT.en)({ a: c.values[0], b: c.values[1], ea: name(c.evidence[0]), eb: name(c.evidence[1]) });
}

/**
 * Verify the artwork cards against the index: the posts exist as described, exact histories have
 * chain operations, recorded first sightings match a version showing the image, and cards that
 * describe the same image agree on when it first appeared.
 */
export async function verifyEvidence(env: Env, cards: EvidenceCard[]): Promise<EvidenceVerification> {
  const art = cards.filter((c): c is ArtworkCard => c.type === "artwork" || c.type === "post");
  const invalid = new Map<string, string[]>();
  const warnings = new Map<string, string[]>();
  const notes: string[] = [];
  const flag = (id: string, why: string) => {
    const m = CRITICAL.has(why) ? invalid : warnings;
    m.set(id, [...(m.get(id) ?? []), why]);
  };
  for (const c of art) for (const i of cardIssues(c)) flag(c.evidence_id, i);

  // one round trip: the posts, the earliest exact version of each card's image, and the versions of
  // the posts the cards are and name as first sightings
  const ids = art.map((c) => c.artwork_id).filter((x) => Number.isInteger(x));
  const hashes = [...new Set(art.map((c) => c.image).filter((h): h is string => !!h && HASH.test(h)))];
  const refs = [...new Set(art.flatMap((c) => [c.path, ...(c.first_seen_in && c.first_seen_in !== c.path ? [c.first_seen_in] : [])]))].map((p) => PATH.exec(p)).filter((m): m is RegExpExecArray => !!m).slice(0, 40);
  type VersionRow = { author: string; permlink: string; content_hash: string | null; source: string };
  const [posts, versions, ofPosts] = await Promise.all([
    ids.length ? env.DB.prepare(`SELECT id, author, permlink, deleted FROM posts WHERE id IN (${ids.map((i) => Math.trunc(i)).join(",")})`).all<{ id: number; author: string; permlink: string; deleted: number }>() : Promise.resolve({ results: [] as Array<{ id: number; author: string; permlink: string; deleted: number }> }),
    hashes.length
      ? env.DB.prepare(`SELECT content_hash, MIN(at) AS first_at, COUNT(*) AS n FROM post_versions WHERE content_hash IN (${hashes.map(() => "?").join(",")}) AND source != 'snapshot' GROUP BY content_hash`).bind(...hashes).all<{ content_hash: string; first_at: number; n: number }>()
      : Promise.resolve({ results: [] as Array<{ content_hash: string; first_at: number; n: number }> }),
    refs.length
      ? env.DB.prepare(`SELECT author, permlink, content_hash, source FROM post_versions WHERE ${refs.map(() => "(author = ? AND permlink = ?)").join(" OR ")} LIMIT 2000`).bind(...refs.flatMap((m) => [m[1], m[2]])).all<VersionRow>().catch(() => ({ results: [] as VersionRow[] }))
      : Promise.resolve({ results: [] as VersionRow[] }),
  ]);
  const postById = new Map((posts.results ?? []).map((p) => [p.id, p]));
  const firstVersion = new Map((versions.results ?? []).map((v) => [v.content_hash, v.first_at]));
  const versionsOf = new Map<string, VersionRow[]>();
  for (const v of ofPosts.results ?? []) versionsOf.set(`/@${v.author}/${v.permlink}`, [...(versionsOf.get(`/@${v.author}/${v.permlink}`) ?? []), v]);
  const versionsKnown = refs.length > 0 && (ofPosts.results ?? []).length > 0;
  for (const c of art) {
    const p = postById.get(c.artwork_id);
    if (!p) flag(c.evidence_id, "no such post in the index");
    else if (`/@${p.author}/${p.permlink}` !== c.path) flag(c.evidence_id, "the post's author or permlink differs from the index");
    else if ((p.deleted === 1) !== !!c.deleted) flag(c.evidence_id, "deletion state differs from the index");
    if (!versionsKnown) continue;
    // an exact history rests on chain operations; one read from snapshots only is inferred
    const own = versionsOf.get(c.path) ?? [];
    if (c.type === "artwork" && c.history_exact !== false && own.length && !own.some((v) => v.source !== "snapshot")) flag(c.evidence_id, "history marked exact without chain operations");
    // an exact first sighting elsewhere: that post showed these bytes in one of its versions
    if (c.image && c.first_seen_in && c.first_seen_in !== c.path && c.first_seen_match === "exact" && versionsOf.has(c.first_seen_in) && !versionsOf.get(c.first_seen_in)!.some((v) => v.content_hash === c.image)) flag(c.evidence_id, "first sighting in a post none of whose versions showed this image");
  }

  // conflicts: a card's first sighting against the earliest exact version of its image, and
  // cards with the same image against each other
  const conflicts: ConflictCard[] = [];
  const seenPair = new Set<string>();
  const add = (field: string, a: ArtworkCard, va: number, b: ArtworkCard | null, vb: number, note: string) => {
    const k = `${field}|${a.evidence_id}|${b?.evidence_id ?? "versions"}`;
    if (seenPair.has(k)) return;
    seenPair.add(k);
    conflicts.push({
      evidence_id: `C${conflicts.length + 1}`,
      type: "conflict",
      source: "verifier",
      status: "conflict",
      field,
      evidence: b ? [a.evidence_id, b.evidence_id] : [a.evidence_id],
      values: [fmtDate(va), fmtDate(vb)],
      note,
    });
  };
  for (const c of art) {
    if (!c.image || !c.first_seen_at || c.first_seen_match === "near") continue;
    const recorded = firstVersion.get(c.image);
    const first = toTs(c.first_seen_at)!;
    // only a chain operation that showed the image EARLIER than the card says contradicts it: a
    // first sighting may precede every recorded operation (a history read from a snapshot, or
    // operations not indexed), which is no conflict. The creating operation's block time trails
    // `created` by a block (CREATE_SKEW).
    if (recorded !== undefined && first - recorded > Math.max(CREATE_SKEW, 3600) && fmtDate(recorded) !== fmtDate(first)) {
      add("first_seen", c, first, null, recorded, "a chain operation showed this image before the card's first sighting");
    }
  }
  const byImage = new Map<string, ArtworkCard[]>();
  for (const c of art) if (c.image) byImage.set(c.image, [...(byImage.get(c.image) ?? []), c]);
  const duplicates: string[][] = [];
  for (const group of byImage.values()) {
    if (group.length < 2) continue;
    duplicates.push(group.map((c) => c.evidence_id));
    const dated = group.filter((c) => c.first_seen_at && c.first_seen_match !== "near");
    for (let i = 1; i < dated.length; i++) {
      const a = toTs(dated[0].first_seen_at)!;
      const b = toTs(dated[i].first_seen_at)!;
      if (Math.abs(a - b) > Math.max(CREATE_SKEW, 3600) && fmtDate(a) !== fmtDate(b)) add("first_seen", dated[0], a, dated[i], b, "two posts of the same image give different first sightings");
    }
  }
  if (duplicates.length) notes.push(`${duplicates.length} image(s) appear in several cards: one image, not independent evidence`);
  return { invalid, warnings, conflicts, duplicates, checked: art.length, notes };
}

/** Mark the cards with the verification's outcome (valid, issues) and append the conflict cards. */
export function applyVerification(cards: EvidenceCard[], v: EvidenceVerification): EvidenceCard[] {
  for (const c of cards) {
    if (c.type !== "artwork" && c.type !== "post") continue;
    const bad = v.invalid.get(c.evidence_id) ?? [];
    const minor = v.warnings.get(c.evidence_id) ?? [];
    c.valid = !bad.length;
    if (bad.length || minor.length) c.issues = [...bad, ...minor];
  }
  return [...cards, ...v.conflicts];
}
