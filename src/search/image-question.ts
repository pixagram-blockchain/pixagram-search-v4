// Questions about an image (spec §46-47): "who posted this first?", "is this on Pixagram?", "find
// artworks like this one", "what is it?" — asked with an uploaded image (POST /ask {question, image}).
//
//   uploaded image ─► sha-256 of its bytes, pHash, dHash, colours, SigLIP vector (when on), and in
//                     the deeper modes a vision model's description (caption, subjects, text in it)
//                  ─► the identity of every candidate, kept apart (spec §47):
//                       exact_identity        the same bytes, in any post or any version of one,
//                                             deleted posts included
//                       perceptual_identity   pHash ≤ NEAR (4) and the same colours (≥ 0.8): a
//                                             re-encoded or rescaled copy
//                       historical_identity   where the identical posts' image first appeared
//                       visual_similarity     SigLIP similarity: never identity, however high
//                  ─► the deterministic answer: the first poster and date, whether it is on
//                     Pixagram, or the artworks that only look like it
//
// The reasoning model gets structured visual evidence (the I1 card and the artwork cards with their
// identity), not the image itself (spec §46).

import type { Env } from "../env";
import { fold, type Lang } from "../lib/text";
import { hamming } from "../enrich/phash";
import { colourAgreement, NEAR, NEAR_COLOURS } from "../chain/versions";
import { matchConcepts } from "../concepts";
import { imageSimilarity, phashNeighbours } from "./visual";
import { hydrateRows } from "./service";
import { knn } from "./vectors";
import { emptyRequest, visibleUnder, type NsfwMode } from "./params";
import type { QueryPlan } from "./planner";
import type { ImageIdentity, QueryImageCard } from "./evidence";
import { afterColon, type OperatorResult, type Provenance, type Row, type Scope, type Verified } from "./operators";
import type { StepOutcome } from "./executor";
import type { Step } from "./query-planner";
import { answerLang, fmtDate, quote, say, yesNo, type AnswerLang } from "./answer-text";

export interface QueryImage {
  /** sha-256 of the uploaded bytes: an artwork's content_hash when it is the same file */
  sha256: string;
  phash: string;
  dhash: string;
  /** named-colour shares (enrich/stats.ts), as the index stores them */
  buckets: Array<{ name: string; weight: number }>;
  width: number;
  height: number;
  format: string;
  /** SigLIP image vector (EMBED_VIEWS), when the embedding endpoint is on */
  vector: number[] | null;
  /** a vision model's description of the image (deep and expert modes, or a "what is it" question) */
  description?: { caption: string; subjects: string[]; tags: string[]; text_in_image: string; model: string };
}

export type ImageTask = "origin" | "exists" | "similar" | "describe";

// on folded text
const EXISTS = /\b(is (this|it) (on|in|already)|already (on|posted|been)|(has|was) (this|it) (been )?(posted|uploaded|published)|does (this|it) exist|exist\w*|est (elle|il|ce) (sur|deja)|deja (sur|poste|publie)|existe\w*|gibt es|schon (auf|gepostet|veroffentlicht)|ist (das|dieses bild) (auf|schon)|ya (esta|existe|fue)|esta (en|ya)|esiste|e gia|gia (su|pubblicat))\b/;
const SIMILAR = /\b(similar|alike|like (this|it)|looks? like|resembl\w*|semblable\w*|similaire\w*|ressembl\w*|comme (celle|celui|ca)|ahnlich\w*|wie (dieses|das)|parecid\w*|como (esta|este)|simil\w*|come questa|come questo)\b/;
const DESCRIBE = /\b(what (is|does|do) (this|it|in|on)|what s (this|in|on)|describe|what (is|are) (shown|depicted)|qu est ce|que (represente|montre)|decri\w*|was (ist|zeigt) (das|dieses)|beschreib\w*|que (es|muestra|representa)|describ\w*|cos e|cosa (mostra|rappresenta)|descriv\w*)\b/;
const ORIGIN = /\b(first|originally|original\w*|origin|earliest|who (posted|made|created|drew|painted|published|uploaded)|whose|when|premier\w*|origine|qui (a|l a) (poste|publie|fait|cree|dessine)|quand|zuerst|erste\w*|ursprung\w*|wer hat|wann|primer\w*|origen|quien (publico|subio|hizo|creo|dibujo)|cuando|prim[oa]|chi (ha|l ha) (pubblicato|postato|fatto|creato|disegnato)|quando)\b/;

/** What the question wants to know about the image. */
export function imageTask(question: string): ImageTask {
  const f = ` ${fold(question).replace(/['’]/g, " ").replace(/[?!.,;:()"“”«»„]+/g, " ").replace(/\s+/g, " ")} `;
  if (SIMILAR.test(f)) return "similar";
  if (EXISTS.test(f)) return "exists";
  if (ORIGIN.test(f)) return "origin";
  if (DESCRIBE.test(f)) return "describe";
  return "origin";
}

export interface ImageMatch {
  row: Row;
  identity: ImageIdentity;
  /** exact or perceptual: the same artwork */
  same: boolean;
}

export interface ImageFindings {
  matches: ImageMatch[];
  /** where the image first appeared: the same bytes (any post, any version), or a near-identical upload */
  origin: { author: string; permlink: string; at: number; match: "exact" | "near"; deleted: boolean } | null;
  /** posts that have shown the same bytes, in any version (deleted ones included) */
  appearances: Array<{ author: string; permlink: string; at: number }>;
  /** posts with this image (the same bytes, a near-identical copy, a first sighting) the nsfw filter hides: left out of everything above */
  hidden: number;
  /** of those, posts that still show it */
  hidden_live: number;
  /** the image first appeared in a post the filter hides: `origin` is then the first post the caller may see */
  origin_hidden: boolean;
  legs: Record<string, number>;
  notes: string[];
}

const pathOf = (r: { author: string; permlink: string }) => `/@${r.author}/${r.permlink}`;
const r3 = (x: number) => Math.round(x * 1000) / 1000;

/** Everything the index knows about an image: identical posts, near-identical ones, look-alikes, and where it first appeared. */
export async function findImage(env: Env, qi: QueryImage, opts: { nsfw: NsfwMode; visualK?: number }): Promise<ImageFindings> {
  const notes: string[] = [];
  const buckets = JSON.stringify(qi.buckets);
  const [exactRows, versions, near] = await Promise.all([
    env.DB.prepare("SELECT post_id AS id FROM artworks WHERE content_hash = ? LIMIT 50").bind(qi.sha256).all<{ id: number }>(),
    env.DB.prepare("SELECT author, permlink, MIN(at) AS at FROM post_versions WHERE content_hash = ? AND source != 'snapshot' GROUP BY author, permlink ORDER BY at ASC LIMIT 50").bind(qi.sha256).all<{ author: string; permlink: string; at: number }>(),
    phashNeighbours(env, qi.phash, NEAR + 4, 60).catch(() => ({ hits: [] as Array<{ id: number; distance: number; dhash_distance: number | null }>, method: "bands" as const })),
  ]);
  const cos = new Map<number, number>();
  if (qi.vector && qi.vector.length) {
    try {
      for (const h of await knn(env, "image", qi.vector, emptyRequest({ nsfw: opts.nsfw }), opts.visualK ?? 24)) cos.set(h.id, h.score);
    } catch (e) {
      notes.push(`visual similarity unavailable: ${e instanceof Error ? e.message : String(e)}`);
    }
  } else notes.push("no image vector (embedding endpoint off): identity by hashes only, no visual similarity");
  const exactIds = new Set((exactRows.results ?? []).map((r) => r.id));
  const ids = [...new Set([...exactIds, ...near.hits.map((h) => h.id), ...cos.keys()])];
  const rows = await hydrateRows(env.DB, ids, null);
  const matches: ImageMatch[] = [];
  // posts with the image that the filter hides: never named, only counted (as /search leaves them out)
  const hiddenPaths = new Set<string>();
  const hiddenLive = new Set<string>();
  const nearBy = new Map(near.hits.map((h) => [h.id, h]));
  for (const id of ids) {
    const row = rows.get(id);
    if (!row || row.type !== "artwork") continue;
    const exact = exactIds.has(id);
    const nh = nearBy.get(id);
    const pd = row.phash ? hamming(qi.phash, row.phash) : (nh?.distance ?? null);
    const dd = row.dhash ? hamming(qi.dhash, row.dhash) : null;
    // pHash reads brightness only: a recoloured variant has the same pHash. Same shapes and same colours.
    const nearIdentical = !exact && pd !== null && pd <= NEAR && colourAgreement(row.buckets_json, buckets) >= NEAR_COLOURS;
    const c = cos.get(id);
    const identity: ImageIdentity = {
      exact,
      perceptual: pd === null && dd === null ? null : { phash_distance: pd, dhash_distance: dd, near_identical: exact || nearIdentical },
      visual: c === undefined ? null : r3(imageSimilarity(c)),
      historical: null,
    };
    const same = exact || nearIdentical;
    if (!visibleUnder(row, opts.nsfw)) {
      if (same) {
        hiddenPaths.add(pathOf(row as { author: string; permlink: string }));
        if (row.deleted !== 1) hiddenLive.add(pathOf(row as { author: string; permlink: string }));
      }
      continue;
    }
    // identical posts stay even when deleted (they are the image's history); look-alikes are live
    // posts the vector search found (a pHash neighbour that is neither identical nor similar is dropped)
    if (!same && (row.deleted === 1 || c === undefined)) continue;
    matches.push({ row, identity, same });
  }

  // where it first appeared: the same bytes in any version of any post, or an identical post's own first sighting
  type Origin = NonNullable<ImageFindings["origin"]>;
  const cands: Origin[] = [];
  const deletedPaths = new Set([...rows.values()].filter((r) => r.deleted === 1).map((r) => pathOf(r as { author: string; permlink: string })));
  for (const v of versions.results ?? []) cands.push({ author: v.author, permlink: v.permlink, at: v.at, match: "exact", deleted: false });
  for (const m of matches.filter((x) => x.same)) {
    const r = m.row;
    const own = typeof r.image_since === "number" ? r.image_since : r.created;
    cands.push({ author: r.author, permlink: r.permlink, at: own, match: m.identity.exact ? "exact" : "near", deleted: r.deleted === 1 });
    if (typeof r.first_seen === "number" && r.first_seen_author && r.first_seen_permlink) cands.push({ author: r.first_seen_author, permlink: r.first_seen_permlink, at: r.first_seen, match: m.identity.exact && r.first_seen_match !== "near" ? "exact" : "near", deleted: false });
  }
  cands.sort((a, b) => a.at - b.at || (a.match === "exact" ? -1 : 1) - (b.match === "exact" ? -1 : 1) || pathOf(a).localeCompare(pathOf(b)));
  // the state of every post named: deleted? hidden by the filter?
  const named = [...new Set(cands.map(pathOf))].slice(0, 60);
  const state = new Map<string, { deleted: number; nsfw: number; ai_nsfw: number | null }>();
  if (named.length) {
    const res = await env.DB.batch(named.map((p) => env.DB.prepare("SELECT p.deleted, p.nsfw, a.ai_nsfw FROM posts p LEFT JOIN artworks a ON a.post_id = p.id WHERE p.author = ? AND p.permlink = ?").bind(p.slice(2, p.indexOf("/", 2)), p.slice(p.indexOf("/", 2) + 1)))).catch(() => [] as D1Result[]);
    named.forEach((p, i) => {
      const r = res[i]?.results?.[0] as { deleted: number; nsfw: number; ai_nsfw: number | null } | undefined;
      if (r) state.set(p, r);
    });
  }
  const hiddenPost = (p: string) => hiddenPaths.has(p) || (!!state.get(p) && !visibleUnder(state.get(p)!, opts.nsfw));
  for (const c of cands) if (hiddenPost(pathOf(c))) (hiddenPaths.add(pathOf(c)), state.get(pathOf(c))?.deleted !== 1 && hiddenLive.add(pathOf(c)));
  const visible = cands.filter((c) => !hiddenPost(pathOf(c)));
  const originHidden = !!cands.length && hiddenPost(pathOf(cands[0])) && (!visible.length || cands[0].at < visible[0].at);
  let origin: Origin | null = visible[0] ?? null;
  if (origin) {
    // is the first post deleted? (a version row may name a post whose current row is gone)
    const p = state.get(pathOf(origin));
    origin = { ...origin, deleted: p ? p.deleted === 1 : deletedPaths.has(pathOf(origin)) };
    const originPath = pathOf(origin);
    for (const m of matches) {
      if (!m.same) continue;
      const theirs = m.row.first_seen_author ? `/@${m.row.first_seen_author}/${m.row.first_seen_permlink}` : pathOf(m.row as { author: string; permlink: string });
      m.identity.historical = theirs === originPath || pathOf(m.row as { author: string; permlink: string }) === originPath ? "same_origin" : "different_origin";
    }
  }
  matches.sort((a, b) => Number(b.identity.exact) - Number(a.identity.exact) || Number(b.same) - Number(a.same) || (b.identity.visual ?? 0) - (a.identity.visual ?? 0) || a.row.id - b.row.id);
  const appearances = (versions.results ?? []).filter((v) => !hiddenPost(pathOf(v))).map((v) => ({ author: v.author, permlink: v.permlink, at: v.at }));
  for (const m of matches) if (m.same && !appearances.some((a) => a.author === m.row.author && a.permlink === m.row.permlink)) appearances.push({ author: m.row.author, permlink: m.row.permlink, at: typeof m.row.image_since === "number" ? m.row.image_since : m.row.created });
  appearances.sort((a, b) => a.at - b.at);
  const hidden = hiddenPaths.size;
  if (hidden) notes.push(`${hidden} post(s) with this image are hidden by the nsfw filter`);
  return {
    matches,
    origin,
    appearances,
    hidden,
    hidden_live: hiddenLive.size,
    origin_hidden: originHidden,
    legs: { exact_identity: matches.filter((m) => m.identity.exact).length, perceptual_identity: matches.filter((m) => m.same && !m.identity.exact).length, visual_similarity: matches.filter((m) => m.identity.visual !== null).length, versions: appearances.length },
    notes,
  };
}

// ---- answers -------------------------------------------------------------------------------------

type Msg = (x: Record<string, string | number>) => string;
const T: Record<string, Record<AnswerLang, Msg>> = {
  origin: {
    en: (x) => `This image was first posted by @${x.author} on ${x.date}, in ${x.where}.`,
    fr: (x) => `Cette image a été postée pour la première fois par @${x.author} le ${x.date}, dans ${x.where}.`,
    de: (x) => `Dieses Bild wurde zuerst am ${x.date} von @${x.author} gepostet, in ${x.where}.`,
    es: (x) => `Esta imagen fue publicada por primera vez por @${x.author} el ${x.date}, en ${x.where}.`,
    it: (x) => `Questa immagine è stata pubblicata per la prima volta da @${x.author} il ${x.date}, in ${x.where}.`,
  },
  origin_near: {
    en: (x) => `A near-identical version of this image (same shapes and colours) was first posted by @${x.author} on ${x.date}, in ${x.where}.`,
    fr: (x) => `Une version quasi identique de cette image (mêmes formes et couleurs) a été postée pour la première fois par @${x.author} le ${x.date}, dans ${x.where}.`,
    de: (x) => `Eine nahezu identische Version dieses Bildes (gleiche Formen und Farben) wurde zuerst am ${x.date} von @${x.author} gepostet, in ${x.where}.`,
    es: (x) => `Una versión casi idéntica de esta imagen (mismas formas y colores) fue publicada por primera vez por @${x.author} el ${x.date}, en ${x.where}.`,
    it: (x) => `Una versione quasi identica di questa immagine (stesse forme e colori) è stata pubblicata per la prima volta da @${x.author} il ${x.date}, in ${x.where}.`,
  },
  also: {
    en: (x) => ` It appears in ${x.n} other post${Number(x.n) === 1 ? "" : "s"} too.`,
    fr: (x) => ` Elle apparaît aussi dans ${x.n} autre${Number(x.n) === 1 ? "" : "s"} post${Number(x.n) === 1 ? "" : "s"}.`,
    de: (x) => ` Es erscheint auch in ${x.n} weiteren Beiträgen.`,
    es: (x) => ` También aparece en ${x.n} publicación${Number(x.n) === 1 ? "" : "es"} más.`,
    it: (x) => ` Compare anche in altri ${x.n} post.`,
  },
  not_indexed: {
    en: (x) => `No indexed artwork shows this image${Number(x.n) ? `; the ${x.n} closest ones only look similar, which does not make them the same artwork` : ""}.`,
    fr: (x) => `Aucune œuvre indexée ne montre cette image${Number(x.n) ? ` ; les ${x.n} plus proches lui ressemblent seulement, ce qui n'en fait pas la même œuvre` : ""}.`,
    de: (x) => `Kein indexiertes Kunstwerk zeigt dieses Bild${Number(x.n) ? `; die ${x.n} nächsten sehen nur ähnlich aus, das macht sie nicht zum selben Kunstwerk` : ""}.`,
    es: (x) => `Ninguna obra indexada muestra esta imagen${Number(x.n) ? `; las ${x.n} más cercanas solo se parecen, lo que no las convierte en la misma obra` : ""}.`,
    it: (x) => `Nessuna opera indicizzata mostra questa immagine${Number(x.n) ? `; le ${x.n} più vicine sono solo simili, il che non le rende la stessa opera` : ""}.`,
  },
  gone: {
    en: () => "this image is not on Pixagram any more.",
    fr: () => "cette image n'est plus sur Pixagram.",
    de: () => "dieses Bild ist nicht mehr auf Pixagram.",
    es: () => "esta imagen ya no está en Pixagram.",
    it: () => "questa immagine non è più su Pixagram.",
  },
  hidden_only: {
    en: () => "no post you can see shows this image.",
    fr: () => "aucun post visible ne montre cette image.",
    de: () => "kein sichtbarer Beitrag zeigt dieses Bild.",
    es: () => "ninguna publicación visible muestra esta imagen.",
    it: () => "nessun post visibile mostra questa immagine.",
  },
  origin_hidden: {
    en: () => "This image first appeared in a post the content filter hides.",
    fr: () => "Cette image est apparue pour la première fois dans un post masqué par le filtre de contenu.",
    de: () => "Dieses Bild erschien zuerst in einem Beitrag, den der Inhaltsfilter ausblendet.",
    es: () => "Esta imagen apareció por primera vez en una publicación que el filtro de contenido oculta.",
    it: () => "Questa immagine è apparsa per la prima volta in un post nascosto dal filtro dei contenuti.",
  },
  earliest_visible: {
    en: (x) => ` The earliest post you can see with it is ${x.where} by @${x.author} (${x.date}).`,
    fr: (x) => ` Le premier post visible qui la montre est ${x.where} de @${x.author} (${x.date}).`,
    de: (x) => ` Der früheste sichtbare Beitrag damit ist ${x.where} von @${x.author} (${x.date}).`,
    es: (x) => ` La primera publicación visible con ella es ${x.where} de @${x.author} (${x.date}).`,
    it: (x) => ` Il primo post visibile che la mostra è ${x.where} di @${x.author} (${x.date}).`,
  },
  exists_yes: {
    en: () => "this image is on Pixagram.",
    fr: () => "cette image est sur Pixagram.",
    de: () => "dieses Bild ist auf Pixagram.",
    es: () => "esta imagen está en Pixagram.",
    it: () => "questa immagine è su Pixagram.",
  },
  similar: {
    en: (x) => `${x.n} indexed artwork${Number(x.n) === 1 ? " looks" : "s look"} similar to this image (similar, not the same artwork).`,
    fr: (x) => `${x.n} œuvre${Number(x.n) === 1 ? "" : "s"} indexée${Number(x.n) === 1 ? " ressemble" : "s ressemblent"} à cette image (semblable${Number(x.n) === 1 ? "" : "s"}, pas la même œuvre).`,
    de: (x) => `${x.n} indexierte${Number(x.n) === 1 ? "s Kunstwerk sieht" : " Kunstwerke sehen"} diesem Bild ähnlich (ähnlich, nicht dasselbe Kunstwerk).`,
    es: (x) => `${x.n} obra${Number(x.n) === 1 ? " indexada se parece" : "s indexadas se parecen"} a esta imagen (parecida${Number(x.n) === 1 ? "" : "s"}, no la misma obra).`,
    it: (x) => `${x.n} opera${Number(x.n) === 1 ? " indicizzata somiglia" : " indicizzate somigliano"} a questa immagine (simili, non la stessa opera).`,
  },
  is_post: {
    en: (x) => `This image is ${x.where} by @${x.author} (${x.date}).`,
    fr: (x) => `Cette image est ${x.where} de @${x.author} (${x.date}).`,
    de: (x) => `Dieses Bild ist ${x.where} von @${x.author} (${x.date}).`,
    es: (x) => `Esta imagen es ${x.where} de @${x.author} (${x.date}).`,
    it: (x) => `Questa immagine è ${x.where} di @${x.author} (${x.date}).`,
  },
  is_near: {
    en: (x) => `This image is a near-identical version of ${x.where} by @${x.author} (${x.date}).`,
    fr: (x) => `Cette image est une version quasi identique de ${x.where} de @${x.author} (${x.date}).`,
    de: (x) => `Dieses Bild ist eine nahezu identische Version von ${x.where} von @${x.author} (${x.date}).`,
    es: (x) => `Esta imagen es una versión casi idéntica de ${x.where} de @${x.author} (${x.date}).`,
    it: (x) => `Questa immagine è una versione quasi identica di ${x.where} di @${x.author} (${x.date}).`,
  },
  neighbours_show: {
    en: (x) => ` The most similar artworks are about ${x.list}.`,
    fr: (x) => ` Les œuvres les plus proches portent sur ${x.list}.`,
    de: (x) => ` Die ähnlichsten Kunstwerke zeigen ${x.list}.`,
    es: (x) => ` Las obras más parecidas tratan de ${x.list}.`,
    it: (x) => ` Le opere più simili riguardano ${x.list}.`,
  },
};
const DELETED: Record<AnswerLang, string> = { en: "since deleted", fr: "supprimé depuis", de: "inzwischen gelöscht", es: "eliminada después", it: "poi eliminato" };

function where(author: string, permlink: string, title: string | null | undefined, deleted: boolean, l: AnswerLang): string {
  const base = title ? quote(title, l) : `/@${author}/${permlink}`;
  return deleted ? `${base} (${DELETED[l]})` : base;
}

const verifiedOf = (m: ImageMatch): Verified => ({
  row: m.row,
  v: {
    score: m.identity.exact ? 1 : m.same ? 0.9 : r3(m.identity.visual ?? 0),
    lexical: 0,
    semantic: m.identity.visual,
    text: null,
    signals: [m.identity.exact ? "exact image (same bytes)" : m.same ? `near-identical image (pHash distance ${m.identity.perceptual?.phash_distance}, same colours)` : `visually similar (${m.identity.visual})`],
  },
});

/** The concepts the closest look-alikes share (what the most similar artworks are about). */
function neighbourConcepts(matches: ImageMatch[]): string[] {
  const n = new Map<string, number>();
  for (const m of matches.filter((x) => !x.same).slice(0, 8)) {
    const tags: string[] = (() => {
      try {
        return JSON.parse(m.row.tags_json || "[]");
      } catch {
        return [];
      }
    })();
    for (const c of new Set(tags.slice(0, 8).flatMap((t) => matchConcepts(String(t)).map((x) => x.concept)))) n.set(c, (n.get(c) ?? 0) + 1);
  }
  return [...n.entries()].filter(([, k]) => k >= 2).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 3).map(([c]) => c);
}

/** The deterministic outcome of an image question (one program step, executor-shaped). */
export async function imageOutcome(env: Env, step: Step, qi: QueryImage, opts: { lang: Lang; nsfw: NsfwMode; plan: QueryPlan; limit: number }): Promise<StepOutcome & { identities: Map<number, ImageIdentity>; findings: ImageFindings; task: ImageTask }> {
  const t0 = Date.now();
  const l = answerLang(opts.lang);
  const task = imageTask(step.text ?? "");
  const f = await findImage(env, qi, { nsfw: opts.nsfw });
  const same = f.matches.filter((m) => m.same);
  const looks = f.matches.filter((m) => !m.same && (m.identity.visual ?? 0) >= 0.35);
  const identities = new Map(f.matches.map((m) => [m.row.id as number, m.identity]));
  const originRow = f.origin ? (f.matches.find((m) => m.row.author === f.origin!.author && m.row.permlink === f.origin!.permlink)?.row ?? null) : null;
  const originTitle = originRow?.title ?? (f.origin ? await env.DB.prepare("SELECT title FROM posts WHERE author = ? AND permlink = ?").bind(f.origin.author, f.origin.permlink).first<{ title: string }>().then((r) => r?.title ?? null).catch(() => null) : null);
  const others = f.origin ? new Set(f.appearances.map((a) => `/@${a.author}/${a.permlink}`).filter((p) => p !== `/@${f.origin!.author}/${f.origin!.permlink}`)).size : 0;
  const hiddenNote = f.hidden ? say("hidden_posts", l, { n: f.hidden }) : "";
  const originText = (): string => {
    const o = f.origin!;
    const x = { author: o.author, date: fmtDate(o.at), where: where(o.author, o.permlink, originTitle, o.deleted, l) };
    // the first post is one the filter hides: who posted it is not said, only the first post the caller may see
    if (f.origin_hidden) return `${T.origin_hidden[l]({})}${T.earliest_visible[l](x)}`;
    return `${T[o.match === "near" ? "origin_near" : "origin"][l](x)}${others ? T.also[l]({ n: others }) : ""}`;
  };
  const evidence = [...same.map(verifiedOf), ...(same.length ? [] : looks.slice(0, 6).map(verifiedOf))];
  const items = [...same, ...looks].slice(0, opts.limit).map(verifiedOf);
  const details: Record<string, unknown> = {
    task,
    identity: same.length ? (same.some((m) => m.identity.exact) ? "exact_identity" : "perceptual_identity") : looks.length ? "visual_similarity" : "none",
    identical_posts: same.map((m) => `/@${m.row.author}/${m.row.permlink}`).slice(0, 12),
    appearances: f.appearances.length,
    ...(f.origin ? { first_author: f.origin.author, post: `/@${f.origin.author}/${f.origin.permlink}`, time: f.origin.at, first_seen: fmtDate(f.origin.at), match: f.origin.match, ...(f.origin.deleted ? { origin_deleted: true } : {}) } : {}),
    ...(looks.length ? { similar: looks.slice(0, 8).map((m) => ({ post: `/@${m.row.author}/${m.row.permlink}`, visual: m.identity.visual })) } : {}),
  };
  let result: OperatorResult;
  if (task === "exists") {
    // on Pixagram: a live post shows it; posted once, but only in posts since deleted: no longer;
    // shown only by posts the filter hides: none the caller can see (and it is said that some are hidden)
    const yes = same.some((m) => m.row.deleted !== 1);
    const gone = !yes && !f.hidden_live && (same.length > 0 || !!f.origin);
    result = {
      op: "identify",
      answer: yes,
      answerType: "boolean",
      text: `${yes ? `${yesNo(true, l)}${T.exists_yes[l]({})} ${originText()}` : gone ? `${yesNo(false, l)}${T.gone[l]({})} ${originText()}` : f.hidden_live ? `${yesNo(false, l)}${T.hidden_only[l]({})}` : `${yesNo(false, l)}${afterColon(T.not_indexed[l]({ n: Math.min(looks.length, 5) }))}`}${hiddenNote}`,
      confidence: yes || gone ? (same.some((m) => m.identity.exact) || f.origin?.match === "exact" ? 0.98 : 0.85) : looks.length ? 0.7 : 0.8,
      evidence,
      items,
      exact: true,
      details,
      notes: [],
    };
  } else if (task === "similar") {
    result = {
      op: "similar",
      answer: looks.length,
      answerType: "count",
      text: `${T.similar[l]({ n: looks.length })}${same.length && f.origin ? ` ${originText()}` : ""}`,
      confidence: looks.length ? r3(Math.min(0.9, 0.5 + 0.4 * (looks[0].identity.visual ?? 0))) : 0.5,
      evidence: [...same.map(verifiedOf), ...looks.slice(0, 8).map(verifiedOf)],
      items,
      details,
      notes: [],
    };
  } else if (task === "describe" && same.length) {
    const m = same.find((x) => x.row.author === f.origin?.author && x.row.permlink === f.origin?.permlink) ?? same[0];
    const x = { author: m.row.author, date: fmtDate(m.row.created), where: where(m.row.author, m.row.permlink, m.row.title, m.row.deleted === 1, l) };
    result = { op: "identify", answer: `/@${m.row.author}/${m.row.permlink}`, answerType: "post", text: `${T[m.identity.exact ? "is_post" : "is_near"][l](x)}${f.origin && (f.origin.author !== m.row.author || f.origin.permlink !== m.row.permlink) ? ` ${originText()}` : ""}`, confidence: m.identity.exact ? 0.98 : 0.85, evidence, items, exact: true, details, notes: [] };
  } else if (f.origin || f.origin_hidden) {
    // a first poster the filter hides is not named: no author is the answer then
    result = f.origin_hidden
      ? { op: "identify", answer: null, answerType: "none", text: `${f.origin ? originText() : T.origin_hidden[l]({})}${hiddenNote}`, confidence: 0.6, evidence, items, exact: true, details, notes: [] }
      : { op: "identify", answer: f.origin!.author, answerType: "author", text: `${originText()}${hiddenNote}`, confidence: f.origin!.match === "exact" ? 0.97 : 0.85, evidence, items, exact: true, details, notes: [] };
  } else {
    const concepts = neighbourConcepts(looks);
    result = {
      op: "identify",
      answer: null,
      answerType: task === "describe" ? "post" : "author",
      text: `${T.not_indexed[l]({ n: Math.min(looks.length, 5) })}${task === "describe" && concepts.length ? T.neighbours_show[l]({ list: concepts.join(", ") }) : ""}`,
      confidence: looks.length ? 0.6 : 0.8,
      evidence,
      items,
      details: { ...details, ...(concepts.length ? { neighbour_concepts: concepts } : {}) },
      notes: [],
    };
  }
  const verified = f.matches.map(verifiedOf);
  const provenance = new Map<number, Provenance>(f.matches.map((m) => [m.row.id as number, { sources: [m.identity.exact ? "exact_identity" : m.same ? "perceptual_identity" : "visual_similarity"], scores: { ...(m.identity.visual !== null ? { visual: m.identity.visual } : {}), ...(m.identity.perceptual?.phash_distance !== null && m.identity.perceptual ? { phash_distance: m.identity.perceptual.phash_distance } : {}) } }]));
  const scope: Scope = { kind: "subject", plan: opts.plan, req: emptyRequest({ nsfw: opts.nsfw }), all: verified, verified, truncated: false, provenance, legs: f.legs, notes: f.notes };
  const answered = (result.answer !== null && !(result.answerType === "count" && result.answer === 0 && !same.length)) || f.origin_hidden;
  return {
    step,
    status: answered || result.answerType === "boolean" ? "ok" : "no_match",
    result,
    plan: opts.plan,
    scope,
    values: { author: f.origin?.author, post: f.origin ? `/@${f.origin.author}/${f.origin.permlink}` : undefined, time: f.origin?.at, title: originTitle ?? undefined, row: originRow ?? same[0]?.row },
    ms: Date.now() - t0,
    identities,
    findings: f,
    task,
  };
}

/** The I1 card: the uploaded image as the model may know it (no pixels, no bytes). */
export function queryImageCard(qi: QueryImage, f: ImageFindings | null): QueryImageCard {
  const card: QueryImageCard = {
    evidence_id: "I1",
    type: "query_image",
    source: "query-image",
    width: qi.width,
    height: qi.height,
    phash: qi.phash,
    sha256: qi.sha256,
    colors: qi.buckets.filter((b) => b.weight >= 0.08).sort((a, b) => b.weight - a.weight).slice(0, 4).map((b) => b.name),
  };
  if (f) card.identity = f.matches.some((m) => m.identity.exact) ? "exact_identity" : f.matches.some((m) => m.same) ? "perceptual_identity" : f.matches.length ? "visual_similarity" : "none";
  if (qi.description) {
    card.description = {
      caption: qi.description.caption.slice(0, 400),
      ...(qi.description.subjects.length ? { subjects: qi.description.subjects.slice(0, 8) } : {}),
      ...(qi.description.tags.length ? { tags: qi.description.tags.slice(0, 12) } : {}),
      ...(qi.description.text_in_image ? { text_in_image: qi.description.text_in_image.slice(0, 160) } : {}),
      by: "a vision model, not the index",
    };
    const concepts = [...new Set([...qi.description.tags, ...qi.description.subjects].flatMap((t) => matchConcepts(t).map((m) => m.concept)))].slice(0, 8);
    if (concepts.length) card.concepts = concepts;
  }
  return card;
}
