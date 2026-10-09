// The single search box: one text field, three destinations.
//
//   search  a list of artworks and posts: names, subjects, colours ("red dragon", "first snow")
//   ask     a question about the artworks and their history, answered from the index
//           ("who posted the first cat?", "how many dragons?", "combien d'artistes ?")
//   help    a question about Pixagram itself, answered from the documentation repository
//           ("how do I mint?", "what are the fees?", "qu'est-ce que PXS ?")
//
// Rules only, no model call, in this order:
//   1. "similar to 42", "duplicates of 42"                                   → ask
//   2. not phrased as a question ("top hat", "the last samurai" are titles)  → search
//      A question mark makes a question; so does a question word at the start ("who", "how",
//      "combien", "wie"…), unless the text is the title of a post ("Come and fly with me",
//      "How to train your dragon" stay searches).
//   3. an explicit question about posts ("first", "how many", "most liked"… about an artwork
//      subject, with a posting verb or "post", or with no subject at all), and no platform word
//      left (see below); advice ("the best time to post") is not one                → ask
//   4. platform words (fees, wallet, PXS, android…), how-to or permission phrasing,
//      or the platform's name without an artwork subject                     → help
//   5. "what is …" / "who is …": the documentation knows the words → help; otherwise → search
//   6. other questions: an artwork subject → ask; the documentation knows the words → help;
//      otherwise → search
// A platform word is the subject of a question about posts, not its topic, only when the question
// says so: a posting verb, or a post noun it qualifies ("who posted the first wallet?", "how many
// android artworks?", "artworks that show a token", "œuvres de jetons"), or the people counted
// ("how many accounts are there?"). "Fees per post" and "tokens for a post" stay platform questions. Never in a how-to or
// permission question ("how many accounts can I create?"), and never a policy word ("is AI art
// allowed?"). A word the question ranks by ("the best earning artwork") is part of its intent.
// The title and documentation checks (full text only, no AI call) run only when a rule needs them.

import { matchConcepts } from "../concepts";
import { fold, singular } from "../lib/text";
import { OBJECT_WORDS } from "./lexicon";
import type { QueryPlan } from "./planner";

export type Route = "search" | "ask" | "help";

export interface RouteSignals {
  question: boolean;
  /** a question mark (a question even when the words are a title) */
  questionMark: boolean;
  /** intent words present ("first", "how many", "most liked") */
  explicitIntent: boolean;
  /** concepts, colours, tones, orientation, #tags, @authors or object words ("artwork") */
  artSubject: boolean;
  /** the only subject is an author ("who is laura?") */
  authorOnly: boolean;
  /** platform words and phrases in the question */
  platform: string[];
  /** those that are not its topic: the subject of a question about posts, or what it ranks by (see above) */
  platformAsSubject: string[];
  platformName: boolean;
  /** an artwork word, a posting verb or "post" as a noun */
  posts: boolean;
  /** asks for advice: "the best time to post", "le meilleur moment pour poster" */
  advice: boolean;
  howto: boolean;
  definitional: boolean;
  /** best lexical match in the documentation, when it was needed */
  docs: number | null;
}

export interface RouteDecision {
  route: Route;
  reason: string;
  signals: RouteSignals;
}

// Words that only make sense about the platform, and its policies ("allowed"). Kept out: words of
// artwork questions ("posted", "uploaded", "payout", "votes"), common subjects of pixel art
// ("coin", "key", "market", "bridge", "avatar"), and "forbidden" ("the forbidden fruit", "la forêt
// interdite"). Some listed words are subjects too ("wallet", "android"): see platformAsSubject.
const PLATFORM_WORDS = new Set(
  `account accounts wallet wallets password passwords login logon signin signup register registration
   fee fees cost costs price prices pricing pay paying payment payments withdraw withdrawal withdrawals deposit deposits transfer transfers exchange swap
   royalty royalties license licence licenses licences licensing copyright
   mint minting minted marketplace sell selling sale sales buy buying purchase purchases auction auctions bid bids offer offers listing listings
   stake staking unstake delegate delegation reward rewards earn earns earned earning earnings curation curator curators witness witnesses node nodes validator validators
   token tokens tokenomics pxs airdrop whitepaper roadmap
   founder founders founded cofounder company foundation team contact support faq rules terms privacy policy policies guideline guidelines moderation moderator report ban banned verification kyc
   allowed allow permitted
   profile username followers notification notifications settings app application website extension keychain blockchain transaction transactions
   android ios iphone ipad mobile
   compte comptes portefeuille connexion inscription frais cout couts prix payer paiement paiements retrait retirer depot transfert echange echanger
   redevance redevances licence droits minter vendre vente ventes acheter achat achats enchere encheres offre offres recompense recompenses gagner jeton jetons
   fondateur fondateurs fonde fondee fondation societe entreprise equipe aide regles conditions confidentialite politique signaler profil pseudo abonnes
   autorise autorisee autorises autorisees
   konto konten passwort anmelden anmeldung registrieren registrierung gebuhr gebuhren kosten preis preise bezahlen zahlung auszahlung einzahlung uberweisung tauschen
   lizenz lizenzen urheberrecht minten marktplatz verkaufen verkauf kaufen kauf auktion angebot belohnung belohnungen verdienen grunder gegrundet firma unternehmen kontakt hilfe regeln bedingungen datenschutz benutzername
   erlaubt
   cuenta cuentas cartera billetera contrasena registro comision comisiones costo coste precio precios pagar pago retiro retirar transferencia licencia derechos mintear vender venta comprar compra subasta recompensa recompensas ganar ganancias fundador fundadores fundada empresa equipo contacto ayuda reglas terminos privacidad perfil
   permitido permitida
   portafoglio accesso registrazione commissione commissioni prezzo pagare pagamento prelievo trasferimento licenza diritti vendere vendita comprare acquisto asta ricompensa ricompense guadagnare guadagni fondatore fondatori fondata azienda squadra contatto aiuto regole termini profilo
   consentito consentita`
    .split(/\s+/)
    .filter(Boolean),
);

/** Policies: "is AI art allowed?" asks about the rules whatever the subject. */
const POLICY_WORDS = new Set("allowed allow permitted autorise autorisee autorises autorisees erlaubt permitido permitida consentito consentita".split(" "));

const PLATFORM_PHRASES = [
  "private key", "posting key", "active key", "owner key", "master key", "memo key", "log in", "sign in", "sign up", "power up", "power down",
  "resource credits", "terms of service", "terms of use", "place de marche", "mot de passe", "cle privee", "droits d auteur", "conditions d utilisation",
];

const PLATFORM_NAMES = new Set(["pixagram", "pixa"]);

/**
 * Question words that start a question even without a question mark. Auxiliaries ("is", "can",
 * "do") and words that are also ordinary words elsewhere ("come", "que", "y") need the mark.
 */
export const QUESTION_START =
  /^(who|whom|whose|what|whats|which|when|where|why|how|qui|quoi|quel|quelle|quels|quelles|quand|comment|combien|pourquoi|qu est-ce|qu est ce|est-ce que|est ce que|c est quoi|y a-t-il|y a t il|wer|welche|welcher|welches|wann|wo|wie|warum|wieso|gibt es|quien|quienes|cual|cuales|cuando|donde|cuanto|cuantos|cuantas|chi|quale|quali|quando|quanto|quanti|quante|perche|explain|define|tell me|explique|erklare|explica|spiega)\b/;

/** How-to, permission and why questions: about using the platform. */
const HOWTO_START =
  /^(how (do|does|did|can|could|should|would|to|is|are)|why|what happens|where (can|do|should|is|are) (i|my)|comment|pourquoi|que se passe|ou (puis|peut|peux|est|sont)|wie (kann|konnen|funktioniert|funktionieren|geht|mache|bekomme|erhalte|lange)|wie \S+ ich|warum|wieso|was passiert|wo (kann|finde)|como (puedo|se|funciona|funcionan|hago)|por que|que pasa|donde (puedo|esta|estan)|come (posso|si|funziona|funzionano|faccio)|perche|cosa succede|dove (posso|trovo))\b/;
const PERMISSION = /\b(can i|could i|may i|do i (need|have)|is it possible|puis-je|puis je|je peux|on peut|peut-on|peut on|est-il possible|est il possible|dois-je|dois je|je dois|faut-il|faut il|kann ich|darf ich|muss ich|ist es moglich|puedo|se puede|debo|posso|si puo|devo)\b/;

const DEFINITIONAL_START =
  /^(what is|what are|whats|what s|what does|who is|who are|who s|tell me about|explain|define|qu est-ce|qu est ce|c est quoi|que veut dire|que signifie|qui est|qui sont|was ist|was sind|was bedeutet|wer ist|wer sind|que es|que son|quien es|quienes son|que significa|cos e|che cos e|cosa e|chi e|chi sono|cosa significa)\b/;

/** People counted by "how many …" (accounts, users): not the platform's "account" pages. */
const COUNTED = /^(accounts?|users?|members?|comptes?|utilisateurs?|membres?|konten|konto|nutzer|mitglieder|cuentas?|usuarios?|miembros?|utenti|utente|membri)$/;

const POSTED = /\b(posted|uploaded|published|shared|drawn|drew|made|created|poste|postee|postes|publie|publiee|partage|dessine|cree|creee|gepostet|veroffentlicht|hochgeladen|gezeichnet|erstellt|publicado|publicada|subido|dibujado|creado|pubblicato|pubblicata|caricato|disegnato|creato)\b/;

/** "post" as a noun ("the latest post", "Beiträge"); "to post", "can I post" are the verb. */
const POST_NOUN =
  /\b(?:posts|beitrag|beitrage|publication|publications|publicacion|publicaciones|pubblicazione|pubblicazioni)\b|(?<!\b(?:to|i|you|we|they|can|could|will|would|should|must|do|does|did|not|never|cannot|cant|and|or) )\bpost\b/;

/** Advice, not a question about posts: "what is the best time to post?", "meilleur moment pour poster". */
const ADVICE =
  /\b(?:(?:best|good|right|ideal|optimal) (?:time|times|way|ways|moment|day|days|hour|hours|size|format|resolution)|(?:meilleur|meilleure|bon|bonne) (?:moment|heure|jour|facon|maniere|taille|format)|(?:moment|heure|jour|facon|maniere) (?:ideal|ideale)|(?:beste|besten|gute|guten|richtige|richtigen) (?:zeit|tageszeit|uhrzeit|weg|tag|moment|art|grosse|format)|(?:mejor|buen|buena) (?:momento|hora|dia|manera|forma|tamano|formato)|(?:momento|hora|dia|manera|forma) (?:ideal|mejor)|(?:migliore|miglior|buon|buona) (?:momento|ora|giorno|modo|maniera|dimensione|formato)|(?:momento|ora|giorno|modo) (?:migliore|ideale))\b/;

const DATA_INTENTS = new Set<QueryPlan["intent"]>(["find_first", "find_last", "count", "top", "compare", "similar", "duplicate"]);

/** Folded text without punctuation (apostrophes become spaces, hyphens stay). */
export function routeText(q: string): string {
  return fold(q)
    .replace(/['’?!.,;:()¿¡"？]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Nouns for posts: artwork words and "post" ("wallet artworks", "a PXS post", "œuvres de jetons"). */
const POST_NOUNS = new Set("post posts beitrag beitrage publication publications publicacion publicaciones pubblicazione pubblicazioni".split(" "));
const isPostNoun = (w: string | undefined) => !!w && (OBJECT_WORDS[w] !== undefined || POST_NOUNS.has(w));
/** Between a post noun and what it shows: "artworks of wallets", "artworks that show a token", "dessins de jetons", "Bilder mit Wallet". */
const SHOWING = new Set(
  `of with about featuring feature features showing show shows depicting depict depicts containing contain contains mentioning mention mentions
   de des du d avec sur montrant montrent representant representent mit von uber zeigt zeigen con sobre mostrando muestra muestran
   di del della dei delle mostra mostrano raffigurante raffiguranti`
    .split(/\s+/)
    .filter(Boolean),
);
const RELATIVE = new Set("that which qui que die der das che".split(" "));
/** Platform words that are also things a picture shows, and money matters when they follow an artwork word ("artwork fees", "artwork tokens"). */
const MONEY = new Set("fee fees token tokens pxs wallet wallets witness witnesses blockchain jeton jetons".split(" "));
/** "my wallet", "mon téléphone", "meinem iPhone": the user's own thing, not what a post shows. */
const POSSESSIVE = new Set(
  `my your our mon ma mes ton ta tes notre nos votre vos mein meine meinem meinen meiner meines dein deine deinem deinen deiner unser unsere unserem unseren
   mi mis tu tus nuestro nuestra nuestros nuestras mio mia miei mie tuo tua tuoi tue nostro nostra`
    .split(/\s+/)
    .filter(Boolean),
);
/** "mobile version", "android app": the platform's product, not a picture. */
const PRODUCT = new Set("version versions app apps application applications appli website site client anwendung webseite aplicacion sitio versione applicazione sito".split(" "));

/**
 * The word at i names what the posts show:
 *   before a post noun, maybe through other subject words: "wallet artworks", "android robot artworks";
 *   after "<post noun> (that) of / with / showing", up to two words between: "artworks of red
 *   wallets", "first drawing of an old iphone", "artworks that show an android";
 *   right after an artwork word, unless it is money: "pixel art android", "œuvre android".
 * Not "fees per post", "tokens for a post", "wallet for my posts", nor the user's own thing ("artworks
 * with my wallet", "Bilder von meinem iPhone") or the platform's product ("pixel art mobile version").
 */
function depicted(tokens: string[], i: number, subject: (w: string) => boolean): boolean {
  if (POSSESSIVE.has(tokens[i - 1]) || PRODUCT.has(tokens[i + 1])) return false;
  for (let k = i + 1; k <= i + 3 && k < tokens.length; k++) {
    if (isPostNoun(tokens[k])) return true;
    if (!subject(tokens[k])) break;
  }
  for (let j = i - 1; j >= Math.max(0, i - 3); j--) {
    if (POSSESSIVE.has(tokens[j])) break;
    if (!SHOWING.has(tokens[j])) continue;
    const k = RELATIVE.has(tokens[j - 1]) ? j - 2 : j - 1;
    if (isPostNoun(tokens[k])) return true;
  }
  if (!MONEY.has(tokens[i])) {
    for (let k = i - 1; k >= Math.max(0, i - 2); k--) {
      if (OBJECT_WORDS[tokens[k]] !== undefined) return true;
      if (!subject(tokens[k])) break;
    }
  }
  return false;
}

export function routeSignals(raw: string, plan: QueryPlan): Omit<RouteSignals, "docs"> {
  const f = routeText(raw);
  const words = f.split(" ").filter(Boolean);
  // hyphenated words count by their parts too ("Android-App", "Wallet-Bilder") and joined ("sign-up")
  const tokens = words.flatMap((w) => (w.includes("-") ? w.split("-").filter(Boolean) : [w]));
  const joined = words.filter((w) => w.includes("-")).map((w) => w.replace(/-/g, ""));
  const platform = [...new Set([...[...tokens, ...joined].filter((w) => PLATFORM_WORDS.has(w)), ...PLATFORM_PHRASES.filter((p) => ` ${f} `.includes(` ${p} `))])];
  const objectWord = [...words, ...tokens].some((w) => OBJECT_WORDS[w] !== undefined) || /\bpixel art\b/.test(f);
  const posted = POSTED.test(f);
  const posts = objectWord || posted || POST_NOUN.test(f);
  const howto = HOWTO_START.test(f) || PERMISSION.test(f);
  // A platform word is not the question's topic when it is what the posts show (a concept named
  // with a posting verb or as what a post shows: "who posted the first wallet?", "wallet
  // artworks"; an unknown word with a posting verb), the people it counts ("how many accounts"),
  // or what it ranks by ("the best earning artwork"). "Fees per post", "tokens for a post" are
  // about the platform.
  const aliases = new Set(plan.conceptMatches.flatMap((m) => m.alias.split(" ")));
  // (the plan keeps one alias per concept: "android robot" records "android" only). "Fee" is the
  // platform's word; the fairy only with its French accent ("fée") or as a German noun inside a
  // sentence ("Bilder mit einer Fee").
  const fairy = /fées?\b/i.test(raw.normalize("NFC")) || /\S\s+Feen?\b/.test(raw);
  const isAlias = (w: string) => ((w !== "fee" && w !== "fees") || fairy) && (aliases.has(w) || aliases.has(singular(w)) || matchConcepts(w).length > 0);
  const residual = new Set(plan.residual.split(" ").filter(Boolean));
  const intent = new Set(plan.intentWords.flatMap((p) => p.split(" ")));
  const shown = (w: string) => tokens.some((t, i) => t === w && depicted(tokens, i, isAlias));
  const platformAsSubject = platform.filter(
    (w) =>
      !POLICY_WORDS.has(w) &&
      (intent.has(w) || (!howto && ((isAlias(w) && (posted || shown(w))) || (residual.has(w) && posted) || (plan.countOf === "authors" && COUNTED.test(w))))),
  );
  const conceptual = plan.concepts.length > 0 || !!plan.filters.colors?.length || !!plan.filters.background?.length || !!plan.filters.tones?.length || !!plan.filters.orientation?.length || !!plan.filters.tags?.length;
  const authors = !!plan.filters.authors?.length;
  const people = plan.countOf === "authors" || plan.intent === "compare";
  const questionMark = /[?？]/.test(raw);
  return {
    question: questionMark || QUESTION_START.test(f),
    questionMark,
    explicitIntent: plan.intentWords.length > 0 || plan.intent === "similar" || plan.intent === "duplicate",
    artSubject: conceptual || authors || objectWord || people,
    authorOnly: authors && !conceptual && !objectWord && !people,
    platform,
    platformAsSubject,
    platformName: tokens.some((w) => PLATFORM_NAMES.has(w)),
    posts,
    advice: ADVICE.test(f),
    howto,
    definitional: DEFINITIONAL_START.test(f),
  };
}

/** Lexical match of the documentation above which it is taken to know a question's words. */
export const DOCS_KNOWS = 0.6;

export interface RouteChecks {
  /** best lexical match of the question in the documentation, [0, 1] */
  docsScore: () => Promise<number>;
  /** whether the text is (part of) the title of a live post */
  isTitle: () => Promise<boolean>;
}

export async function routeQuery(raw: string, plan: QueryPlan, checks: RouteChecks): Promise<RouteDecision> {
  const s: RouteSignals = { ...routeSignals(raw, plan), docs: null };
  const docs = async () => (s.docs ??= await checks.docsScore().catch(() => 0));
  const done = (route: Route, reason: string): RouteDecision => ({ route, reason, signals: s });

  if ((plan.intent === "similar" || plan.intent === "duplicate") && plan.similarTo?.id && !(await checks.isTitle().catch(() => false))) return done("ask", `${plan.intent} to an artwork id`);
  if (!s.question) return done("search", "not a question");
  if (!s.questionMark && (await checks.isTitle().catch(() => false))) {
    s.question = false;
    return done("search", "the title of a post");
  }
  const dataIntent = DATA_INTENTS.has(plan.intent);
  const platformTopic = s.platform.filter((w) => !s.platformAsSubject.includes(w));
  // about posts: an artwork subject, a posting verb or "post", or no subject at all ("what's the latest?")
  const aboutPosts = !s.advice && (s.artSubject || s.posts || !plan.residual);
  if (s.explicitIntent && dataIntent && aboutPosts && !platformTopic.length) return done("ask", "a question about artworks or artists");
  if (platformTopic.length) return done("help", `platform words: ${platformTopic.join(", ")}`);
  if (s.howto) return done("help", "how-to or permission question");
  if (s.platformName && !s.artSubject) return done("help", "about Pixagram itself");
  if (s.definitional) {
    // "what is a sprite?" wants a definition: the documentation's, or results, never a made-up answer
    if ((await docs()) >= DOCS_KNOWS) return done("help", "the documentation knows these words");
    return done("search", "a what-is question the documentation does not cover");
  }
  if (s.artSubject && (dataIntent || !s.authorOnly)) return done("ask", "a question about artworks");
  if ((await docs()) >= DOCS_KNOWS) return done("help", "the documentation knows these words");
  return done("search", "a question with nothing to answer it from");
}
