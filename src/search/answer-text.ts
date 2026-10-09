// The deterministic answers' sentences, in English, French, German, Spanish and Italian (spec §45:
// the answer is normally in the user's language). v3 wrote English, French and German, and
// English for everything else; mode=v3 keeps exactly that (answerLang(lang, true)).
//
// Every sentence is built from computed values only: accounts, dates, titles, counts. Titles are
// quoted the way each language quotes: “…” (en), « … » (fr), „…“ (de), «…» (es, it).

import { singular, type Lang } from "../lib/text";
import type { QueryPlan } from "./planner";

export type AnswerLang = "en" | "fr" | "de" | "es" | "it";

/** The language answers are written in. v3 wrote en, fr and de only (English otherwise). */
export function answerLang(lang: Lang | string, v3 = false): AnswerLang {
  if (lang === "fr" || lang === "de") return lang;
  if (!v3 && (lang === "es" || lang === "it")) return lang;
  return "en";
}

export const fmtDate = (t: number) => new Date(t * 1000).toISOString().slice(0, 10);

type L<T> = Record<AnswerLang, T>;
type Msg = (x: Record<string, string | number>) => string;

const Q: L<[string, string]> = { en: ["“", "”"], fr: ["« ", " »"], de: ["„", "“"], es: ["«", "»"], it: ["«", "»"] };
/** A title or subject in the language's quotation marks. */
export const quote = (s: string, l: AnswerLang) => `${Q[l][0]}${s}${Q[l][1]}`;

const YES: L<string> = { en: "Yes", fr: "Oui", de: "Ja", es: "Sí", it: "Sì" };
const NO: L<string> = { en: "No", fr: "Non", de: "Nein", es: "No", it: "No" };
/** "Yes: " / "Oui : " … */
export const yesNo = (b: boolean, l: AnswerLang) => `${b ? YES[l] : NO[l]}${l === "fr" ? " : " : ": "}`;

// ---- first / latest (v3, plus Spanish and Italian) ---------------------------------------------------

/**
 * "The first …" / "the latest …" sentences, built around what the question asks for:
 *   author    "who posted the first cat?"            The first cat artwork was posted by @a on D: “T”.
 *   date      "when was the first cat posted?"       The first cat artwork appeared on D (“T” by @a).
 *   post      "what is the first cat?"               The first cat artwork is “T”, posted by @a on D.
 *   post_by   "what is the first artwork from @a?"   The first artwork posted by @a is “T” (D).
 * French agrees with "œuvre" (feminine), Spanish with "obra", Italian with "opera".
 */
export type FirstLastKind = "author" | "date" | "post" | "post_by";

export function firstLastText(kind: FirstLastKind, first: boolean, lang: Lang | AnswerLang, x: { subject: string; author: string; date: string; title: string }, v3 = false): string {
  const l = answerLang(lang, v3);
  const { subject: s, author: a, date: d, title: t } = x;
  if (l === "fr") {
    const fem = s === "œuvre";
    const e = fem ? "e" : "";
    const head = fem ? (first ? "La première œuvre" : "La dernière œuvre") : s === "post" ? (first ? "Le premier post" : "Le dernier post") : `${first ? "Le premier" : "Le dernier"} « ${s} »`;
    switch (kind) {
      case "author": return `${head} a été posté${e} par @${a} le ${d} : « ${t} ».`;
      case "date": return `${head} est apparu${e} le ${d} (« ${t} » par @${a}).`;
      case "post": return `${head} est « ${t} », posté${e} par @${a} le ${d}.`;
      case "post_by": return `${head} posté${e} par @${a} est « ${t} » (${d}).`;
    }
  }
  if (l === "de") {
    const head = `Das ${first ? "erste" : "neueste"} „${s}“`;
    switch (kind) {
      case "author": return `${head} hat @${a} am ${d} gepostet: „${t}“.`;
      case "date": return `${head} erschien am ${d} („${t}“ von @${a}).`;
      case "post": return `${head} ist „${t}“, gepostet von @${a} am ${d}.`;
      case "post_by": return `${head} von @${a} ist „${t}“ (${d}).`;
    }
  }
  if (l === "es") {
    const fem = s === "obra" || s === "publicación";
    const o = fem ? "a" : "o";
    const head = fem ? `${first ? "La primera" : "La última"} ${s}` : `${first ? "El primer" : "El último"} «${s}»`;
    switch (kind) {
      case "author": return `${head} fue publicad${o} por @${a} el ${d}: «${t}».`;
      case "date": return `${head} apareció el ${d} («${t}» de @${a}).`;
      case "post": return `${head} es «${t}», publicad${o} por @${a} el ${d}.`;
      case "post_by": return `${head} publicad${o} por @${a} es «${t}» (${d}).`;
    }
  }
  if (l === "it") {
    const fem = s === "opera";
    const o = fem ? "a" : "o";
    const head = fem ? (first ? "La prima opera" : "L'ultima opera") : s === "post" ? (first ? "Il primo post" : "L'ultimo post") : `${first ? "Il primo" : "L'ultimo"} «${s}»`;
    switch (kind) {
      case "author": return `${head} è stat${o} pubblicat${o} da @${a} il ${d}: «${t}».`;
      case "date": return `${head} è appars${o} il ${d} («${t}» di @${a}).`;
      case "post": return `${head} è «${t}», pubblicat${o} da @${a} il ${d}.`;
      case "post_by": return `${head} pubblicat${o} da @${a} è «${t}» (${d}).`;
    }
  }
  const head = `The ${first ? "first" : "latest"} ${s}`;
  switch (kind) {
    case "author": return `${head} was posted by @${a} on ${d}: “${t}”.`;
    case "date": return `${head} appeared on ${d} (“${t}” by @${a}).`;
    case "post": return `${head} is “${t}”, posted by @${a} on ${d}.`;
    case "post_by": return `${head} posted by @${a} is “${t}” (${d}).`;
  }
}

/** "cat artwork" → "cat artworks" (the English labels end with the object word). */
export const pluralEn = (s: string) => (/(artwork|post)$/.test(s) ? `${s}s` : s);

const T: Record<string, L<Msg>> = {
  post: {
    en: (x) => `“${x.title}” by @${x.author} (${x.date}).`,
    fr: (x) => `« ${x.title} » par @${x.author} (${x.date}).`,
    de: (x) => `„${x.title}“ von @${x.author} (${x.date}).`,
    es: (x) => `«${x.title}» de @${x.author} (${x.date}).`,
    it: (x) => `«${x.title}» di @${x.author} (${x.date}).`,
  },
  count: {
    en: (x) => `${x.n} ${Number(x.n) === 1 ? x.subject : pluralEn(String(x.subject))} found${x.atLeast ? " (at least)" : ""}.`,
    fr: (x) => `${x.n} « ${x.subject} » trouvé(s)${x.atLeast ? " (au moins)" : ""}.`,
    de: (x) => `${x.n} „${x.subject}“ gefunden${x.atLeast ? " (mindestens)" : ""}.`,
    es: (x) => `${x.n} «${x.subject}» encontrado(s)${x.atLeast ? " (como mínimo)" : ""}.`,
    it: (x) => `${x.n} «${x.subject}» trovat${Number(x.n) === 1 ? "o" : "i"}${x.atLeast ? " (almeno)" : ""}.`,
  },
  count_authors: {
    en: (x) => `${x.n} artist${Number(x.n) === 1 ? "" : "s"} posted ${pluralEn(String(x.subject))}${x.atLeast ? " (at least)" : ""}.`,
    fr: (x) => `${x.n} artiste(s) ont posté « ${x.subject} »${x.atLeast ? " (au moins)" : ""}.`,
    de: (x) => `${x.n} Künstler haben „${x.subject}“ gepostet${x.atLeast ? " (mindestens)" : ""}.`,
    es: (x) => `${x.n} artista(s) publicaron «${x.subject}»${x.atLeast ? " (como mínimo)" : ""}.`,
    it: (x) => `${x.n} artist${Number(x.n) === 1 ? "a ha" : "i hanno"} pubblicato «${x.subject}»${x.atLeast ? " (almeno)" : ""}.`,
  },
  top: {
    en: (x) => `The most ${x.metric} ${x.subject} is “${x.title}” by @${x.author} (${x.value}).`,
    fr: (x) => `Le « ${x.subject} » le plus ${x.metric} est « ${x.title} » par @${x.author} (${x.value}).`,
    de: (x) => `Das „${x.subject}“ mit den meisten ${x.metric} ist „${x.title}“ von @${x.author} (${x.value}).`,
    es: (x) => `El «${x.subject}» más ${x.metric} es «${x.title}» de @${x.author} (${x.value}).`,
    it: (x) => `Il «${x.subject}» più ${x.metric} è «${x.title}» di @${x.author} (${x.value}).`,
  },
  /** v4: the top artwork of all, without a subject ("L'œuvre la plus aimée", not « œuvre ») */
  top_any: {
    en: (x) => `The most ${x.metric} artwork is “${x.title}” by @${x.author} (${x.value}).`,
    fr: (x) => `L'œuvre la plus ${x.metric}e est « ${x.title} » par @${x.author} (${x.value}).`,
    de: (x) => `Das Kunstwerk mit den meisten ${x.metric} ist „${x.title}“ von @${x.author} (${x.value}).`,
    es: (x) => `La obra más ${String(x.metric).replace(/o$/, "a")} es «${x.title}» de @${x.author} (${x.value}).`,
    it: (x) => `L'opera più ${String(x.metric).replace(/o$/, "a")} è «${x.title}» di @${x.author} (${x.value}).`,
  },
  compare: {
    en: (x) => `@${x.author} posted the most ${pluralEn(String(x.subject))} (${x.n}).`,
    fr: (x) => `@${x.author} a posté le plus de « ${x.subject} » (${x.n}).`,
    de: (x) => `@${x.author} hat die meisten „${x.subject}“ gepostet (${x.n}).`,
    es: (x) => `@${x.author} publicó la mayor cantidad de «${x.subject}» (${x.n}).`,
    it: (x) => `@${x.author} ha pubblicato più «${x.subject}» di tutti (${x.n}).`,
  },
  // several artists share the first place: all of them, not the first in alphabetical order
  compare_tie: {
    en: (x) => `${x.authors} posted the most ${pluralEn(String(x.subject))} (${x.n} each).`,
    fr: (x) => `${x.authors} ont posté le plus de « ${x.subject} » (${x.n} chacun).`,
    de: (x) => `${x.authors} haben die meisten „${x.subject}“ gepostet (je ${x.n}).`,
    es: (x) => `${x.authors} publicaron la mayor cantidad de «${x.subject}» (${x.n} cada uno).`,
    it: (x) => `${x.authors} hanno pubblicato più «${x.subject}» (${x.n} ciascuno).`,
  },
  none: {
    en: (x) => `No ${x.subject} found.`,
    fr: (x) => `Aucun « ${x.subject} » trouvé.`,
    de: (x) => `Kein „${x.subject}“ gefunden.`,
    es: (x) => `No se encontró ningún «${x.subject}».`,
    it: (x) => `Nessun «${x.subject}» trovato.`,
  },
  repost: {
    en: (x) => ` The same image first appeared on ${x.date} in ${x.post}.`,
    fr: (x) => ` La même image est apparue pour la première fois le ${x.date} dans ${x.post}.`,
    de: (x) => ` Dasselbe Bild erschien zuerst am ${x.date} in ${x.post}.`,
    es: (x) => ` La misma imagen apareció por primera vez el ${x.date} en ${x.post}.`,
    it: (x) => ` La stessa immagine è apparsa per la prima volta il ${x.date} in ${x.post}.`,
  },
  repost_near: {
    en: (x) => ` A near-identical version (same shapes and colours) first appeared on ${x.date} in ${x.post}.`,
    fr: (x) => ` Une version presque identique (mêmes formes et couleurs) est apparue le ${x.date} dans ${x.post}.`,
    de: (x) => ` Eine nahezu identische Fassung (gleiche Formen und Farben) erschien zuerst am ${x.date} in ${x.post}.`,
    es: (x) => ` Una versión casi idéntica (mismas formas y colores) apareció por primera vez el ${x.date} en ${x.post}.`,
    it: (x) => ` Una versione quasi identica (stesse forme e colori) è apparsa per la prima volta il ${x.date} in ${x.post}.`,
  },

  // ---- v4 -----------------------------------------------------------------------------------------
  similar: {
    en: (x) => `${x.n} ${x.dup ? "near-duplicate" : "similar"} artworks of #${x.id}.`,
    fr: (x) => (x.dup ? `${x.n} quasi-doublon(s) de #${x.id}.` : `${x.n} œuvre(s) similaire(s) à #${x.id}.`),
    de: (x) => (x.dup ? `${x.n} Beinahe-Duplikate von #${x.id}.` : `${x.n} ähnliche Kunstwerke zu #${x.id}.`),
    es: (x) => (x.dup ? `${x.n} casi duplicado(s) de #${x.id}.` : `${x.n} obra(s) similar(es) a #${x.id}.`),
    it: (x) => (x.dup ? `${x.n} quasi duplicati di #${x.id}.` : `${x.n} opere simili a #${x.id}.`),
  },
  clarify_reference: {
    en: () => "Which artwork do you mean? Name it (its title in quotes, or its id), or ask with the image.",
    fr: () => "De quelle œuvre parlez-vous ? Donnez son titre entre guillemets ou son numéro, ou posez la question avec l'image.",
    de: () => "Welches Kunstwerk meinen Sie? Nennen Sie seinen Titel in Anführungszeichen oder seine Nummer, oder fragen Sie mit dem Bild.",
    es: () => "¿A qué obra se refiere? Indique su título entre comillas o su número, o pregunte con la imagen.",
    it: () => "A quale opera si riferisce? Indichi il titolo tra virgolette o il numero, oppure chieda con l'immagine.",
  },
  insufficient: {
    en: () => "There is insufficient evidence to determine this.",
    fr: () => "Les éléments disponibles ne permettent pas de le déterminer.",
    de: () => "Die vorhandenen Belege reichen nicht aus, um das zu bestimmen.",
    es: () => "No hay evidencia suficiente para determinarlo.",
    it: () => "Non ci sono prove sufficienti per stabilirlo.",
  },
  name_artwork: {
    en: () => "Name the artwork: “similar to 123” (its id).",
    fr: () => "Indiquez l'œuvre : « similaire à 123 » (son numéro).",
    de: () => "Nennen Sie das Kunstwerk: „ähnlich wie 123“ (seine Nummer).",
    es: () => "Indique la obra: «parecida a 123» (su número).",
    it: () => "Indica l'opera: «simile a 123» (il suo numero).",
  },
  author_of: {
    en: (x) => `“${x.title}” was posted by @${x.author} on ${x.date}.`,
    fr: (x) => `« ${x.title} » a été publié par @${x.author} le ${x.date}.`,
    de: (x) => `„${x.title}“ wurde am ${x.date} von @${x.author} gepostet.`,
    es: (x) => `«${x.title}» fue publicado por @${x.author} el ${x.date}.`,
    it: (x) => `«${x.title}» è stato pubblicato da @${x.author} il ${x.date}.`,
  },
  author_not: {
    en: (x) => `“${x.title}” was posted by @${x.author} on ${x.date}, not by @${x.claimed}.`,
    fr: (x) => `« ${x.title} » a été publié par @${x.author} le ${x.date}, pas par @${x.claimed}.`,
    de: (x) => `„${x.title}“ wurde am ${x.date} von @${x.author} gepostet, nicht von @${x.claimed}.`,
    es: (x) => `«${x.title}» fue publicado por @${x.author} el ${x.date}, no por @${x.claimed}.`,
    it: (x) => `«${x.title}» è stato pubblicato da @${x.author} il ${x.date}, non da @${x.claimed}.`,
  },
  identify: {
    en: (x) => `“${x.title}” by @${x.author} is post ${x.id}: ${x.path}.`,
    fr: (x) => `« ${x.title} » de @${x.author} est le post ${x.id} : ${x.path}.`,
    de: (x) => `„${x.title}“ von @${x.author} ist der Beitrag ${x.id}: ${x.path}.`,
    es: (x) => `«${x.title}» de @${x.author} es el post ${x.id}: ${x.path}.`,
    it: (x) => `«${x.title}» di @${x.author} è il post ${x.id}: ${x.path}.`,
  },
  not_found_title: {
    en: (x) => `No post titled “${x.title}” was found.`,
    fr: (x) => `Aucun post intitulé « ${x.title} » n'a été trouvé.`,
    de: (x) => `Kein Beitrag mit dem Titel „${x.title}“ gefunden.`,
    es: (x) => `No se encontró ningún post titulado «${x.title}».`,
    it: (x) => `Nessun post intitolato «${x.title}» trovato.`,
  },
  clarify_title: {
    en: (x) => `Several posts are titled “${x.title}”: ${x.list}. Which one do you mean?`,
    fr: (x) => `Plusieurs posts s'intitulent « ${x.title} » : ${x.list}. Lequel voulez-vous dire ?`,
    de: (x) => `Mehrere Beiträge heißen „${x.title}“: ${x.list}. Welchen meinen Sie?`,
    es: (x) => `Varios posts se titulan «${x.title}»: ${x.list}. ¿Cuál quiere decir?`,
    it: (x) => `Diversi post si intitolano «${x.title}»: ${x.list}. Quale intende?`,
  },
  repost_yes: {
    en: (x) => `the same image was posted again on ${x.date} in ${x.post}${x.near ? " (a near-identical re-upload)" : ""}.`,
    fr: (x) => `la même image a été publiée à nouveau le ${x.date} dans ${x.post}${x.near ? " (une version presque identique)" : ""}.`,
    de: (x) => `dasselbe Bild wurde am ${x.date} erneut in ${x.post} gepostet${x.near ? " (eine nahezu identische Fassung)" : ""}.`,
    es: (x) => `la misma imagen se publicó de nuevo el ${x.date} en ${x.post}${x.near ? " (una versión casi idéntica)" : ""}.`,
    it: (x) => `la stessa immagine è stata pubblicata di nuovo il ${x.date} in ${x.post}${x.near ? " (una versione quasi identica)" : ""}.`,
  },
  repost_no: {
    en: () => "the image has not been posted again since.",
    fr: () => "l'image n'a pas été publiée à nouveau depuis.",
    de: () => "das Bild wurde seitdem nicht erneut gepostet.",
    es: () => "la imagen no se ha vuelto a publicar desde entonces.",
    it: () => "l'immagine non è stata più pubblicata da allora.",
  },
  repost_by_other: {
    en: (x) => `@${x.author} did not post it again, but @${x.other} posted the same image on ${x.date} (${x.post}).`,
    fr: (x) => `@${x.author} ne l'a pas republiée, mais @${x.other} a publié la même image le ${x.date} (${x.post}).`,
    de: (x) => `@${x.author} hat es nicht erneut gepostet, aber @${x.other} hat dasselbe Bild am ${x.date} gepostet (${x.post}).`,
    es: (x) => `@${x.author} no la volvió a publicar, pero @${x.other} publicó la misma imagen el ${x.date} (${x.post}).`,
    it: (x) => `@${x.author} non l'ha ripubblicata, ma @${x.other} ha pubblicato la stessa immagine il ${x.date} (${x.post}).`,
  },
  edited: {
    en: (x) => `${x.title} was edited ${x.n} time${Number(x.n) === 1 ? "" : "s"}, last on ${x.date}.`,
    fr: (x) => `${x.title} a été modifié ${x.n} fois, la dernière le ${x.date}.`,
    de: (x) => `${x.title} wurde ${x.n}-mal bearbeitet, zuletzt am ${x.date}.`,
    es: (x) => `${x.title} se editó ${x.n} ${Number(x.n) === 1 ? "vez" : "veces"}, la última el ${x.date}.`,
    it: (x) => `${x.title} è stato modificato ${x.n} ${Number(x.n) === 1 ? "volta" : "volte"}, l'ultima il ${x.date}.`,
  },
  not_edited: {
    en: (x) => `${x.title} has not been edited since it was posted on ${x.date}.`,
    fr: (x) => `${x.title} n'a pas été modifié depuis sa publication le ${x.date}.`,
    de: (x) => `${x.title} wurde seit der Veröffentlichung am ${x.date} nicht bearbeitet.`,
    es: (x) => `${x.title} no se ha editado desde su publicación el ${x.date}.`,
    it: (x) => `${x.title} non è stato modificato dalla pubblicazione del ${x.date}.`,
  },
  deleted: {
    en: (x) => `${x.title} by @${x.author} was deleted on ${x.date}.`,
    fr: (x) => `${x.title} de @${x.author} a été supprimé le ${x.date}.`,
    de: (x) => `${x.title} von @${x.author} wurde am ${x.date} gelöscht.`,
    es: (x) => `${x.title} de @${x.author} se eliminó el ${x.date}.`,
    it: (x) => `${x.title} di @${x.author} è stato eliminato il ${x.date}.`,
  },
  not_deleted: {
    en: (x) => `${x.title} has not been deleted.`,
    fr: (x) => `${x.title} n'a pas été supprimé.`,
    de: (x) => `${x.title} wurde nicht gelöscht.`,
    es: (x) => `${x.title} no se ha eliminado.`,
    it: (x) => `${x.title} non è stato eliminato.`,
  },
  first_seen: {
    en: (x) => `Its image first appeared on ${x.date} in ${x.post}.`,
    fr: (x) => `Son image est apparue pour la première fois le ${x.date} dans ${x.post}.`,
    de: (x) => `Sein Bild erschien zuerst am ${x.date} in ${x.post}.`,
    es: (x) => `Su imagen apareció por primera vez el ${x.date} en ${x.post}.`,
    it: (x) => `La sua immagine è apparsa per la prima volta il ${x.date} in ${x.post}.`,
  },
  edit_by_other: {
    en: (x) => `${x.title} was posted by @${x.author}, and only its author edits it: @${x.actor} has not edited it.`,
    fr: (x) => `${x.title} a été publié par @${x.author}, et seul son auteur le modifie : @${x.actor} ne l'a pas modifié.`,
    de: (x) => `${x.title} wurde von @${x.author} gepostet, und nur der Autor bearbeitet es: @${x.actor} hat es nicht bearbeitet.`,
    es: (x) => `${x.title} fue publicado por @${x.author}, y solo su autor lo edita: @${x.actor} no lo ha editado.`,
    it: (x) => `${x.title} è stato pubblicato da @${x.author}, e solo il suo autore lo modifica: @${x.actor} non l'ha modificato.`,
  },
  delete_by_other: {
    en: (x) => `${x.title} was posted by @${x.author}, and only its author deletes it: @${x.actor} has not deleted it.`,
    fr: (x) => `${x.title} a été publié par @${x.author}, et seul son auteur le supprime : @${x.actor} ne l'a pas supprimé.`,
    de: (x) => `${x.title} wurde von @${x.author} gepostet, und nur der Autor löscht es: @${x.actor} hat es nicht gelöscht.`,
    es: (x) => `${x.title} fue publicado por @${x.author}, y solo su autor lo elimina: @${x.actor} no lo ha eliminado.`,
    it: (x) => `${x.title} è stato pubblicato da @${x.author}, e solo il suo autore lo elimina: @${x.actor} non l'ha eliminato.`,
  },
  spelling_read: {
    en: (x) => `Nothing matches “${x.from}”; read as “${x.to}”:`,
    fr: (x) => `Rien ne correspond à « ${x.from} » ; lu comme « ${x.to} » :`,
    de: (x) => `Nichts passt zu „${x.from}“; gelesen als „${x.to}“:`,
    es: (x) => `Nada coincide con «${x.from}»; leído como «${x.to}»:`,
    it: (x) => `Nessun risultato per «${x.from}»; letto come «${x.to}»:`,
  },
  hidden_posts: {
    en: (x) => ` (${x.n} post${Number(x.n) === 1 ? "" : "s"} with this image ${Number(x.n) === 1 ? "is" : "are"} hidden by the content filter.)`,
    fr: (x) => ` (${x.n} post${Number(x.n) === 1 ? "" : "s"} avec cette image ${Number(x.n) === 1 ? "est masqué" : "sont masqués"} par le filtre de contenu.)`,
    de: (x) => ` (${x.n} ${Number(x.n) === 1 ? "Beitrag" : "Beiträge"} mit diesem Bild ${Number(x.n) === 1 ? "wird" : "werden"} vom Inhaltsfilter ausgeblendet.)`,
    es: (x) => ` (${x.n} publicaci${Number(x.n) === 1 ? "ón" : "ones"} con esta imagen ${Number(x.n) === 1 ? "está oculta" : "están ocultas"} por el filtro de contenido.)`,
    it: (x) => ` (${x.n} post con questa immagine ${Number(x.n) === 1 ? "è nascosto" : "sono nascosti"} dal filtro dei contenuti.)`,
  },
  still_shown: {
    en: (x) => `Its image is still shown by ${x.post}.`,
    fr: (x) => `Son image est toujours visible dans ${x.post}.`,
    de: (x) => `Sein Bild ist weiterhin in ${x.post} zu sehen.`,
    es: (x) => `Su imagen sigue visible en ${x.post}.`,
    it: (x) => `La sua immagine è ancora visibile in ${x.post}.`,
  },
  last_seen: {
    en: (x) => `Its image was last shown until ${x.date}, in ${x.post}.`,
    fr: (x) => `Son image a été visible pour la dernière fois jusqu'au ${x.date}, dans ${x.post}.`,
    de: (x) => `Sein Bild war zuletzt bis zum ${x.date} zu sehen, in ${x.post}.`,
    es: (x) => `Su imagen estuvo visible por última vez hasta el ${x.date}, en ${x.post}.`,
    it: (x) => `La sua immagine è stata visibile l'ultima volta fino al ${x.date}, in ${x.post}.`,
  },
  more: {
    en: (x) => `@${x.w} posted more ${x.s} than @${x.l} (${x.nw} vs ${x.nl}).`,
    fr: (x) => `@${x.w} a posté plus de « ${x.s} » que @${x.l} (${x.nw} contre ${x.nl}).`,
    de: (x) => `@${x.w} hat mehr „${x.s}“ gepostet als @${x.l} (${x.nw} zu ${x.nl}).`,
    es: (x) => `@${x.w} publicó más «${x.s}» que @${x.l} (${x.nw} frente a ${x.nl}).`,
    it: (x) => `@${x.w} ha pubblicato più «${x.s}» di @${x.l} (${x.nw} contro ${x.nl}).`,
  },
  group_author_total: {
    en: (x) => `@${x.key}'s artworks have the most ${x.unit} in total (${x.n}).`,
    fr: (x) => `Les œuvres de @${x.key} ont le plus de ${x.unit} au total (${x.n}).`,
    de: (x) => `Die Kunstwerke von @${x.key} haben insgesamt die meisten ${x.unit} (${x.n}).`,
    es: (x) => `Las obras de @${x.key} tienen más ${x.unit} en total (${x.n}).`,
    it: (x) => `Le opere di @${x.key} hanno più ${x.unit} in totale (${x.n}).`,
  },
  neither_count: {
    en: (x) => `Neither @${x.a} nor @${x.b} has posted any ${x.s}.`,
    fr: (x) => `Ni @${x.a} ni @${x.b} n'ont posté de « ${x.s} ».`,
    de: (x) => `Weder @${x.a} noch @${x.b} hat „${x.s}“ gepostet.`,
    es: (x) => `Ni @${x.a} ni @${x.b} han publicado «${x.s}».`,
    it: (x) => `Né @${x.a} né @${x.b} hanno pubblicato «${x.s}».`,
  },
  more_total: {
    en: (x) => `@${x.w}'s artworks have more ${x.unit} in total than @${x.l}'s (${x.vw} vs ${x.vl}).`,
    fr: (x) => `Les œuvres de @${x.w} ont plus de ${x.unit} au total que celles de @${x.l} (${x.vw} contre ${x.vl}).`,
    de: (x) => `Die Kunstwerke von @${x.w} haben insgesamt mehr ${x.unit} als die von @${x.l} (${x.vw} zu ${x.vl}).`,
    es: (x) => `Las obras de @${x.w} tienen más ${x.unit} en total que las de @${x.l} (${x.vw} frente a ${x.vl}).`,
    it: (x) => `Le opere di @${x.w} hanno più ${x.unit} in totale di quelle di @${x.l} (${x.vw} contro ${x.vl}).`,
  },
  same_total: {
    en: (x) => `@${x.a}'s and @${x.b}'s artworks have the same number of ${x.unit} in total (${x.v} each).`,
    fr: (x) => `Les œuvres de @${x.a} et celles de @${x.b} ont autant de ${x.unit} au total (${x.v} chacune).`,
    de: (x) => `Die Kunstwerke von @${x.a} und die von @${x.b} haben insgesamt gleich viele ${x.unit} (je ${x.v}).`,
    es: (x) => `Las obras de @${x.a} y las de @${x.b} tienen los mismos ${x.unit} en total (${x.v} cada una).`,
    it: (x) => `Le opere di @${x.a} e quelle di @${x.b} hanno gli stessi ${x.unit} in totale (${x.v} ciascuna).`,
  },
  same_count: {
    en: (x) => `@${x.a} and @${x.b} posted the same number of ${x.s} (${x.n} each).`,
    fr: (x) => `@${x.a} et @${x.b} ont posté autant de « ${x.s} » (${x.n} chacun).`,
    de: (x) => `@${x.a} und @${x.b} haben gleich viele „${x.s}“ gepostet (je ${x.n}).`,
    es: (x) => `@${x.a} y @${x.b} publicaron la misma cantidad de «${x.s}» (${x.n} cada uno).`,
    it: (x) => `@${x.a} e @${x.b} hanno pubblicato lo stesso numero di «${x.s}» (${x.n} ciascuno).`,
  },
  more_metric: {
    en: (x) => `${x.w} has more ${x.unit} than ${x.l} (${x.vw} vs ${x.vl}).`,
    fr: (x) => `${x.w} a plus de ${x.unit} que ${x.l} (${x.vw} contre ${x.vl}).`,
    de: (x) => `${x.w} hat mehr ${x.unit} als ${x.l} (${x.vw} zu ${x.vl}).`,
    es: (x) => `${x.w} tiene más ${x.unit} que ${x.l} (${x.vw} frente a ${x.vl}).`,
    it: (x) => `${x.w} ha più ${x.unit} di ${x.l} (${x.vw} contro ${x.vl}).`,
  },
  same_metric: {
    en: (x) => `${x.a} and ${x.b} have the same number of ${x.unit} (${x.v}).`,
    fr: (x) => `${x.a} et ${x.b} ont le même nombre de ${x.unit} (${x.v}).`,
    de: (x) => `${x.a} und ${x.b} haben gleich viele ${x.unit} (${x.v}).`,
    es: (x) => `${x.a} y ${x.b} tienen el mismo número de ${x.unit} (${x.v}).`,
    it: (x) => `${x.a} e ${x.b} hanno lo stesso numero di ${x.unit} (${x.v}).`,
  },
  before: {
    en: (x) => `${x.a} (${x.da}) came before ${x.b} (${x.db}).`,
    fr: (x) => `${x.a} (${x.da}) est antérieur à ${x.b} (${x.db}).`,
    de: (x) => `${x.a} (${x.da}) kam vor ${x.b} (${x.db}).`,
    es: (x) => `${x.a} (${x.da}) fue anterior a ${x.b} (${x.db}).`,
    it: (x) => `${x.a} (${x.da}) è venuto prima di ${x.b} (${x.db}).`,
  },
  same_time: {
    en: (x) => `${x.a} and ${x.b} appeared at the same time (${x.d}).`,
    fr: (x) => `${x.a} et ${x.b} sont apparus au même moment (${x.d}).`,
    de: (x) => `${x.a} und ${x.b} erschienen gleichzeitig (${x.d}).`,
    es: (x) => `${x.a} y ${x.b} aparecieron al mismo tiempo (${x.d}).`,
    it: (x) => `${x.a} e ${x.b} sono apparsi nello stesso momento (${x.d}).`,
  },
  duration: {
    en: (x) => `${x.dur} passed between ${x.a} (${x.da}) and ${x.b} (${x.db}).`,
    fr: (x) => `${x.dur} séparent ${x.a} (${x.da}) et ${x.b} (${x.db}).`,
    de: (x) => `Zwischen ${x.a} (${x.da}) und ${x.b} (${x.db}) liegen ${x.dur}.`,
    es: (x) => `Entre ${x.a} (${x.da}) y ${x.b} (${x.db}) pasaron ${x.dur}.`,
    it: (x) => `Tra ${x.a} (${x.da}) e ${x.b} (${x.db}) sono passati ${x.dur}.`,
  },
  aggregate: {
    en: (x) => `${x.scope}: ${x.v} ${x.unit} ${x.agg} (${x.n} artwork${Number(x.n) === 1 ? "" : "s"}).`,
    fr: (x) => `${x.scope} : ${x.v} ${x.unit} ${x.agg} (${x.n} œuvre(s)).`,
    de: (x) => `${x.scope}: ${x.v} ${x.unit} ${x.agg} (${x.n} Kunstwerk${Number(x.n) === 1 ? "" : "e"}).`,
    es: (x) => `${x.scope}: ${x.v} ${x.unit} ${x.agg} (${x.n} obra(s)).`,
    it: (x) => `${x.scope}: ${x.v} ${x.unit} ${x.agg} (${x.n} oper${Number(x.n) === 1 ? "a" : "e"}).`,
  },
  group_month: {
    en: (x) => `${x.scope}: the most in ${x.key} (${x.n} of ${x.total}).`,
    fr: (x) => `${x.scope} : le plus en ${x.key} (${x.n} sur ${x.total}).`,
    de: (x) => `${x.scope}: die meisten im ${x.key} (${x.n} von ${x.total}).`,
    es: (x) => `${x.scope}: la mayoría en ${x.key} (${x.n} de ${x.total}).`,
    it: (x) => `${x.scope}: la maggior parte nel ${x.key} (${x.n} su ${x.total}).`,
  },
  group_author: {
    en: (x) => `${x.scope}: ${x.key} posted the most (${x.n}).`,
    fr: (x) => `${x.scope} : ${x.key} en a posté le plus (${x.n}).`,
    de: (x) => `${x.scope}: ${x.key} hat die meisten gepostet (${x.n}).`,
    es: (x) => `${x.scope}: ${x.key} publicó la mayoría (${x.n}).`,
    it: (x) => `${x.scope}: ${x.key} ne ha pubblicati di più (${x.n}).`,
  },
  group_tags: {
    en: (x) => `${x.scope} are mostly tagged ${x.list}.`,
    fr: (x) => `${x.scope} portent surtout les tags ${x.list}.`,
    de: (x) => `${x.scope} tragen vor allem die Tags ${x.list}.`,
    es: (x) => `${x.scope} llevan sobre todo las etiquetas ${x.list}.`,
    it: (x) => `${x.scope} hanno soprattutto i tag ${x.list}.`,
  },
  premise_false: {
    en: (x) => `@${x.claimed} did not post the first ${x.s}: @${x.actual} did, on ${x.date} (“${x.title}”).`,
    fr: (x) => `@${x.claimed} n'a pas posté le premier « ${x.s} » : c'est @${x.actual}, le ${x.date} (« ${x.title} »).`,
    de: (x) => `@${x.claimed} hat nicht das erste „${x.s}“ gepostet, sondern @${x.actual}, am ${x.date} („${x.title}“).`,
    es: (x) => `@${x.claimed} no publicó el primer «${x.s}»: lo hizo @${x.actual}, el ${x.date} («${x.title}»).`,
    it: (x) => `@${x.claimed} non ha pubblicato il primo «${x.s}»: l'ha fatto @${x.actual}, il ${x.date} («${x.title}»).`,
  },
  exists_yes: {
    en: (x) => `@${x.author} posted ${x.n} ${x.s}.`,
    fr: (x) => `@${x.author} a posté ${x.n} « ${x.s} ».`,
    de: (x) => `@${x.author} hat ${x.n} „${x.s}“ gepostet.`,
    es: (x) => `@${x.author} publicó ${x.n} «${x.s}».`,
    it: (x) => `@${x.author} ha pubblicato ${x.n} «${x.s}».`,
  },
  exists_no: {
    en: (x) => `@${x.author} has not posted any ${x.s}.`,
    fr: (x) => `@${x.author} n'a posté aucun « ${x.s} ».`,
    de: (x) => `@${x.author} hat kein „${x.s}“ gepostet.`,
    es: (x) => `@${x.author} no ha publicado ningún «${x.s}».`,
    it: (x) => `@${x.author} non ha pubblicato nessun «${x.s}».`,
  },
};

export type MessageKey = keyof typeof T;

export function say(key: MessageKey, lang: Lang | AnswerLang, x: Record<string, string | number> = {}, v3 = false): string {
  return T[key][answerLang(lang, v3)](x);
}

/** A sentence part that starts a sentence: the first letter in capitals. */
export const capitalize = (s: string) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

/**
 * "@a, @b and @c" in the answer's language (four names at most); past four, "@a, @b, @c and 2
 * others". `total`: how many there are when `authors` is only the start of the list.
 */
export function authorList(authors: string[], lang: Lang | AnswerLang, total = authors.length, v3 = false): string {
  const l = answerLang(lang, v3);
  const and = { en: "and", fr: "et", de: "und", es: "y", it: "e" }[l];
  const shown = (total <= 4 ? authors.slice(0, 4) : authors.slice(0, 3)).map((a) => `@${a}`);
  const more = total - shown.length;
  if (more > 0) {
    const others = { en: more === 1 ? "other" : "others", fr: more === 1 ? "autre" : "autres", de: "weitere", es: more === 1 ? "otro" : "otros", it: more === 1 ? "altro" : "altri" }[l];
    return `${shown.join(", ")} ${and} ${more} ${others}`;
  }
  return shown.length <= 1 ? shown.join("") : `${shown.slice(0, -1).join(", ")} ${and} ${shown[shown.length - 1]}`;
}

const round = (x: number) => Math.round(x * 1000) / 1000;

/**
 * The "who posted the most" sentence and its confidence: a tie names everyone in it. `tiedTotal`:
 * how many share the first place when `ranked` is only the start of the ranking.
 */
export function compareAnswer(ranked: Array<{ author: string; n: number }>, lang: Lang | AnswerLang, subject: string, confidence: number, tiedTotal?: number, v3 = false): { answer_text: string; confidence: number } {
  const top = ranked[0];
  const tied = ranked.filter((r) => r.n === top.n).map((r) => r.author);
  const total = Math.max(tied.length, tiedTotal ?? 0);
  if (total > 1) return { answer_text: say("compare_tie", lang, { authors: authorList(tied, lang, total, v3), n: top.n, subject }, v3), confidence: round(0.5 * confidence) };
  return { answer_text: say("compare", lang, { author: top.author, n: top.n, subject }, v3), confidence: round(confidence) };
}

const OBJ: L<{ artwork: string; blog: string }> = {
  en: { artwork: "artwork", blog: "post" },
  fr: { artwork: "œuvre", blog: "post" },
  de: { artwork: "Kunstwerk", blog: "Beitrag" },
  es: { artwork: "obra", blog: "publicación" },
  it: { artwork: "opera", blog: "post" },
};

/** What the answer calls the subject: "cat artwork", « chat », „Katze“, "œuvre" when there is none. */
export function subjectLabel(plan: Pick<QueryPlan, "object" | "residual" | "filters">, lang: Lang | AnswerLang, v3 = false): string {
  const l = answerLang(lang, v3);
  const obj = OBJ[l][plan.object === "blog" ? "blog" : "artwork"];
  const s = plan.residual || [...(plan.filters.colors ?? []), ...(plan.filters.tones ?? [])].join(" ");
  if (!s) return obj;
  // English: "cats" → "cat artwork" (one word only: "good vibes" is a name)
  if (l === "en") return `${s.includes(" ") ? s : singular(s)} ${OBJ.en[plan.object === "blog" ? "blog" : "artwork"]}`;
  // German: a noun is written with a capital ("katze" → „Katze“; one word only)
  if (l === "de" && !s.includes(" ")) return s.charAt(0).toUpperCase() + s.slice(1);
  return s;
}

/** The plural phrase of a subject for comparisons ("cat artworks", « chat »…). */
export function subjectPlural(plan: Pick<QueryPlan, "object" | "residual" | "filters">, lang: Lang | AnswerLang): string {
  const l = answerLang(lang);
  const s = subjectLabel(plan, l);
  return l === "en" ? pluralEn(s) : s;
}

/** "the first cat artwork", "la première œuvre « chat »"… : an event as a noun phrase. */
export function firstLabel(subject: string, first: boolean, lang: Lang | AnswerLang): string {
  const l = answerLang(lang);
  switch (l) {
    case "fr":
      return subject === "œuvre" ? (first ? "la première œuvre" : "la dernière œuvre") : `${first ? "le premier" : "le dernier"} « ${subject} »`;
    case "de":
      return `das ${first ? "erste" : "neueste"} „${subject}“`;
    case "es":
      return subject === "obra" ? (first ? "la primera obra" : "la última obra") : `${first ? "el primer" : "el último"} «${subject}»`;
    case "it":
      return subject === "opera" ? (first ? "la prima opera" : "l'ultima opera") : `${first ? "il primo" : "l'ultimo"} «${subject}»`;
    default:
      return `the ${first ? "first" : "latest"} ${subject}`;
  }
}

const UNITS: L<{ day: [string, string]; hour: [string, string]; week: [string, string]; votes: string; payout: string; lessHour: string }> = {
  en: { day: ["day", "days"], hour: ["hour", "hours"], week: ["week", "weeks"], votes: "votes", payout: "PXS", lessHour: "less than an hour" },
  fr: { day: ["jour", "jours"], hour: ["heure", "heures"], week: ["semaine", "semaines"], votes: "votes", payout: "PXS", lessHour: "moins d'une heure" },
  de: { day: ["Tag", "Tage"], hour: ["Stunde", "Stunden"], week: ["Woche", "Wochen"], votes: "Stimmen", payout: "PXS", lessHour: "weniger als eine Stunde" },
  es: { day: ["día", "días"], hour: ["hora", "horas"], week: ["semana", "semanas"], votes: "votos", payout: "PXS", lessHour: "menos de una hora" },
  it: { day: ["giorno", "giorni"], hour: ["ora", "ore"], week: ["settimana", "settimane"], votes: "voti", payout: "PXS", lessHour: "meno di un'ora" },
};

/** A duration in the unit a question asks for ("how many hours …"): the number, and how it is written. */
export function durationIn(seconds: number, unit: "hours" | "days" | "weeks", lang: Lang | AnswerLang): { n: number; text: string } {
  const u = UNITS[answerLang(lang)];
  const s = Math.abs(seconds);
  if (unit === "hours") {
    const h = Math.round(s / 3600);
    return { n: h, text: h ? `${h} ${u.hour[h === 1 ? 0 : 1]}` : u.lessHour };
  }
  if (unit === "weeks") {
    const w = Math.round((s / 604800) * 10) / 10;
    return { n: w, text: `${answerLang(lang) === "en" ? w : String(w).replace(".", ",")} ${u.week[w === 1 ? 0 : 1]}` };
  }
  const d = Math.round(s / 86400);
  return { n: d, text: d ? `${d} ${u.day[d === 1 ? 0 : 1]}` : fmtDuration(s, lang) };
}

/** "12 days", "5 hours", "less than an hour" (German capitalised at the start of a sentence by the caller). */
export function fmtDuration(seconds: number, lang: Lang | AnswerLang): string {
  const u = UNITS[answerLang(lang)];
  const s = Math.abs(seconds);
  if (s >= 2 * 86400) {
    const d = Math.round(s / 86400);
    return `${d} ${u.day[d === 1 ? 0 : 1]}`;
  }
  if (s >= 3600) {
    const h = Math.round(s / 3600);
    return `${h} ${u.hour[h === 1 ? 0 : 1]}`;
  }
  return u.lessHour;
}

/** The unit of a metric in the answer's language: votes, PXS. */
export function metricUnit(metric: "net_votes" | "payout", lang: Lang | AnswerLang): string {
  return UNITS[answerLang(lang)][metric === "payout" ? "payout" : "votes"];
}

/** "1.234 PXS" or "7 votes" (v3 always wrote "votes"; v4 writes the language's word). */
export function metricValue(metric: "net_votes" | "payout", value: number, lang: Lang | AnswerLang, v3 = false): string {
  if (metric === "payout") return `${Number(value).toFixed(3)} PXS`;
  return `${value} ${v3 ? "votes" : metricUnit("net_votes", lang)}`;
}

/** The word for "liked" / "rewarded" in the most-liked sentence. */
export function metricWord(metric: "net_votes" | "payout", lang: Lang | AnswerLang, v3 = false): string {
  const l = answerLang(lang, v3);
  const liked: L<string> = { en: "liked", fr: "aimé", de: "Stimmen", es: "votado", it: "votato" };
  const paid: L<string> = { en: "rewarded", fr: "rémunéré", de: "Belohnungen", es: "recompensado", it: "premiato" };
  return (metric === "payout" ? paid : liked)[l];
}
