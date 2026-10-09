// Multilingual words the planner turns into structure: colours, tones, orientation, intents,
// time. Everything is matched on folded text (lowercase, no accents), whole words only.

import { COLOR_NAMES } from "../enrich/color";

/** Colour words → named palette buckets (src/enrich/color.ts). Several names when the word is broad. */
export const COLOR_WORDS: Record<string, string[]> = (() => {
  const m: Record<string, string[]> = {};
  const add = (names: string[], words: string) => words.split(",").forEach((w) => (m[w.trim()] = names));
  for (const c of COLOR_NAMES) m[c] = [c];
  add(["black"], "noir,noire,noirs,noires,schwarz,schwarze,schwarzer,negro,negra,negros,negras,nero,nera,neri,nere,preto,preta,黒,黑");
  add(["white"], "blanc,blanche,blancs,blanches,weiss,weisse,weisser,blanco,blanca,blancos,bianco,bianca,bianchi,branco,branca,白");
  add(["gray"], "grey,gris,grise,grau,graue,grigio,grigia,cinza,灰色,灰");
  add(["red", "maroon"], "rouge,rouges,rot,rote,roter,rojo,roja,rojos,rosso,rossa,rossi,vermelho,vermelha,赤,红,紅");
  add(["orange"], "orangen,naranja,arancione,laranja,オレンジ,橙");
  add(["yellow"], "jaune,jaunes,gelb,gelbe,amarillo,amarilla,giallo,gialla,amarelo,amarela,黄,黄色");
  add(["green", "lime", "olive"], "vert,verte,verts,vertes,grun,grune,gruner,verde,verdi,緑,绿");
  add(["teal", "cyan"], "turquoise,turquesa,turchese,turkis");
  add(["blue", "navy", "sky"], "bleu,bleue,bleus,bleues,blau,blaue,blauer,azul,azules,blu,青,蓝,藍");
  add(["navy"], "dark blue,bleu marine,dunkelblau,azul marino,blu scuro");
  add(["sky"], "light blue,bleu clair,hellblau,azul claro,celeste,azzurro,azzurra");
  add(["purple", "magenta"], "violet,violette,violett,lila,morado,morada,viola,roxo,roxa,purpura,紫");
  add(["pink", "magenta"], "rosa,rosado,rosada,pinke,ピンク,粉红");
  add(["brown", "tan"], "marron,brun,brune,braun,braune,marron,castano,marrone,castanho,茶色,棕色");
  add(["tan"], "beige,khaki");
  add(["olive"], "kaki");
  add(["maroon"], "bordeaux,burgundy,weinrot,granate,bordo");
  add(["yellow", "orange", "tan"], "gold,golden,dore,doree,gelbgold,dorado,dorata,dourado");
  return m;
})();

export type Tone = "dark" | "light" | "greyscale" | "monochrome" | "colorful" | "pastel" | "high_contrast" | "minimal";

export const TONE_WORDS: Record<string, Tone> = (() => {
  const m: Record<string, Tone> = {};
  const add = (t: Tone, words: string) => words.split(",").forEach((w) => (m[w.trim()] = t));
  add("dark", "dark,darker,darkness,night-time,sombre,sombres,fonce,foncee,obscur,obscure,dunkel,dunkle,oscuro,oscura,scuro,scura,escuro,暗い,暗");
  add("light", "bright,brighter,lumineux,lumineuse,luminoso,luminosa,明るい,明亮");
  add("greyscale", "greyscale,grayscale,black and white,black & white,b&w,noir et blanc,noir & blanc,schwarz-weiss,schwarzweiss,blanco y negro,bianco e nero,preto e branco,白黒,黑白");
  add("monochrome", "monochrome,monochromatic,sepia,duotone,monochromatique,einfarbig");
  add("colorful", "colorful,colourful,colourfull,vibrant,vivid,multicolor,multicolour,rainbow,colore,coloree,colores,bunt,bunte,farbenfroh,colorido,colorida,colorato,colorata,カラフル,彩色");
  add("pastel", "pastel,pastels,pastell,soft colors,soft colours,couleurs douces");
  add("high_contrast", "high contrast,contrasted,contraste,contrastreich,alto contraste");
  add("minimal", "minimal,minimalist,minimalistic,minimaliste,minimalistisch,minimalista");
  return m;
})();

export const ORIENTATION_WORDS: Record<string, "portrait" | "landscape" | "square"> = {
  vertical: "portrait", hochformat: "portrait", "portrait format": "portrait", "format portrait": "portrait",
  horizontal: "landscape", wide: "landscape", widescreen: "landscape", panorama: "landscape", panoramic: "landscape", querformat: "landscape", "format paysage": "landscape",
  square: "square", carre: "square", carree: "square", quadratisch: "square", cuadrado: "square", quadrato: "square",
};

/** Words that only say "artwork" or "blog post" (removed from the residual text of questions). */
export const OBJECT_WORDS: Record<string, "artwork" | "blog"> = (() => {
  const m: Record<string, "artwork" | "blog"> = {};
  for (const w of "image,images,picture,pictures,pic,pics,photo,photos,artwork,artworks,art,drawing,drawings,painting,paintings,pixel art,pixelart,pixel-art,piece,pieces,illustration,illustrations,oeuvre,oeuvres,œuvre,œuvres,dessin,dessins,tableau,tableaux,bild,bilder,kunstwerk,kunstwerke,zeichnung,zeichnungen,imagen,imagenes,obra,obras,dibujo,dibujos,immagine,immagini,opera,opere,disegno,imagem,imagens,絵,画像,作品,图片,作品".split(","))
    m[w] = "artwork";
  // Not "post"/"posts" (also the verb: "who was the first to post a dragon?", and Pixagram users
  // call artworks posts) nor "text" ("artworks with text in them").
  for (const w of "blog,blogs,article,articles,story,stories,beitrag,beitrage,artikel,articulo,articulos,articolo,articoli,billet,billets".split(",")) m[w] = "blog";
  return m;
})();

export type Intent = "search" | "similar" | "duplicate" | "find_first" | "find_last" | "count" | "top" | "compare" | "browse";
export type Output = "results" | "author" | "date" | "count" | "post" | "summary";

/** Phrase patterns on folded text. Order matters: the first match per family wins. */
export const INTENT_PATTERNS: Array<{ re: RegExp; intent?: Intent; output?: Output; sort?: "votes" | "payout" }> = [
  // first / oldest
  { re: /\b(first|earliest|oldest|original|very first|premiere?s?|plus anciens?|plus ancienne?s?|erste[nrs]?|alteste[nrs]?|fruheste[nrs]?|primer[oa]?s?|mas antigu[oa]s?|prim[oa]|piu vecchi[oa]|piu antic[oa]|最初|一番古い|最早|第一)\b/u, intent: "find_first" },
  // last / newest
  { re: /\b(last|latest|newest|most recent|dernier|derniere|derniers|dernieres|plus recente?s?|letzte[nrs]?|neueste[nrs]?|ultim[oa]s?|mas recientes?|piu recenti?|最新|最後|最近)\b/u, intent: "find_last" },
  // how many
  { re: /\b(how many|number of|count of|combien|nombre de|wie viele|wieviele|anzahl|cuantos|cuantas|numero de|quanti|quante|いくつ|何枚|多少|几)\b/u, intent: "count", output: "count" },
  // which artist has the most … / who posted the most … / the most active artist
  { re: /\b(which|what) (artist|author|user|creator|account|person)s? (has|have|posted|made|drew|created) (the )?most\b|\b(quel|quelle) (artiste|auteur|utilisateur|createur) a (le plus|poste le plus)\b|\b(welche[rs]?) (kunstler|autor|nutzer) hat (die meisten|am meisten)\b|\bquien (tiene|publico) (mas|la mayoria)\b/u, intent: "compare", output: "author" },
  { re: /\b(who|qui|wer|quien|chi) (has |have |a |hat |ha )?(posted|post|published|made|drew|created|uploaded|shared|poste|publie|cree|dessine|gepostet|veroffentlicht|erstellt|publicado|publico|creado|pubblicato|creato|disegnato) (the )?(most|le plus|am meisten|die meisten|mas|piu|di piu)\b|\bwer hat (die meisten|am meisten)\b|\b(who|which (artist|author|user|account)) (has|have) (the )?most\b/u, intent: "compare", output: "author" },
  { re: /\b(most (active|prolific)|le plus (actif|prolifique)|la plus (active|prolifique)|aktivste[nrs]?|produktivste[nrs]?|mas (activo|activa|prolifico|prolifica)|piu (attivo|attiva|prolifico|prolifica))\b/u, intent: "compare", output: "author" },
  // best paid / most rewarded (before "best" and "top" below, which would otherwise take "best paid")
  { re: /\b(most (paid|rewarded|earning|lucrative)|(best|top|highest) (paid|earning|rewarded)|highest (payout|reward)s?|mieux payee?s?|plus rentables?|bestbezahlte[nrs]?)\b/u, intent: "top", sort: "payout" },
  // most liked / popular / best
  { re: /\b(most (liked|popular|voted|upvoted|loved|viewed)|(the )?most (likes|votes|upvotes|hearts|views)|le plus de (likes|votes|j aime)|die meisten (likes|stimmen|votes)|mas (likes|votos|me gusta)|piu (like|voti|mi piace)|best|top|highest rated|plus (aime|aimee|populaire|vote)s?|meilleure?s?|beliebteste[nrs]?|meistgeliked|beste[nrs]?|mas (popular|votad[oa]|gustad[oa])|mejor(es)?|piu (popolare|votat[oa]|amat[oa])|miglior[ei]?|人気|一番いい|最受欢迎)\b/u, intent: "top", sort: "votes" },
  // similar / duplicates
  { re: /\b(similar to|like this|looks like|resembl\w*|semblable a|ressemble a|ahnlich wie|parecid[oa] a|simile a)\b/u, intent: "similar" },
  { re: /\b(duplicates?|copies|copy of|reposts?|doublons?|kopien?|duplicad[oa]s?|duplicat[oi])\b/u, intent: "duplicate" },
];

/** Output words: who → author, when → date, … (folded). */
export const OUTPUT_PATTERNS: Array<{ re: RegExp; output: Output }> = [
  { re: /^\s*(who|whose|by whom|qui|de qui|par qui|wer|von wem|quien|quienes|de quien|chi|di chi|quem|誰|谁)\b/u, output: "author" },
  { re: /\b(which|what) (artist|author|user|creator|account)\b|\b(quel|quelle) (artiste|auteur)\b|\bwelche[rs]? (kunstler|autor)\b/u, output: "author" },
  { re: /^\s*(when|what date|what time|quand|a quelle date|wann|an welchem tag|cuando|en que fecha|quando|いつ|什么时候)\b/u, output: "date" },
  { re: /^\s*(what|which|show me|find|quel|quelle|quelles|quels|montre|montre-moi|trouve|welche[rs]?|zeig|zeige|finde|cual|cuales|muestrame|busca|quale|quali|mostrami|trova)\b/u, output: "post" },
];

export const QUESTION_FILLER = new Set(
  (
    // English
    "who,whose,whom,what,which,when,where,how,many,much,is,are,was,were,did,do,does,has,have,had,the,a,an,of,on,in,at,to,for,by,from,with,ever,posted,post,uploaded,upload,published,publish,made,make,drew,draw,drawn,created,create,shared,share,submitted,showing,show,shows,depicting,depicts,with,about,there,that,this,these,those,me,find,give,list,tell,please,any,some,all,one,ones,someone,anyone,image,images,picture,pictures,artwork,artworks,art,first,earliest,oldest,original,last,latest,newest,recent,most,liked,popular,voted,best,top,number,count,artist,artists,author,authors,user,users,creator,creators,account,accounts,person,date,time,day," +
    // French
    "qui,quoi,quel,quelle,quels,quelles,quand,combien,de,du,des,le,la,les,un,une,a,au,aux,est,sont,etait,ont,a-t-il,a-t-elle,poste,postee,postes,publie,publiee,publies,partage,dessine,premier,premiere,premiers,dernier,derniere,plus,ancien,ancienne,recent,recente,image,images,oeuvre,oeuvres,œuvre,œuvres,dessin,dessins,artiste,artistes,auteur,montre,moi,trouve,jamais,sur,il,elle,y,en,nombre," +
    // German
    "wer,was,welche,welcher,welches,wann,wie,viele,der,die,das,den,dem,des,ein,eine,einen,hat,haben,ist,sind,war,gepostet,veroffentlicht,hochgeladen,erste,ersten,erstes,letzte,letzten,neueste,alteste,bild,bilder,kunstwerk,kunstler,autor,von,zeig,mir,je,auf," +
    // Spanish / Italian
    "quien,quienes,que,cual,cuales,cuando,cuantos,cuantas,el,los,las,una,uno,es,fue,ha,han,publico,primero,primera,ultimo,ultima,imagen,imagenes,obra,artista,chi,che,quale,quali,quanti,quante,il,lo,gli,ha,hanno,pubblicato,primo,prima,ultimo,immagine,immagini"
  ).split(","),
);

/**
 * v4's /ask: the demonstratives too ("qui a posté ça ?", "¿quién publicó esto?", "chi ha
 * pubblicato questo?" name no subject: the router asks back). Not in v3's set, so mode=v3 plans
 * exactly as v3 did.
 */
export const QUESTION_FILLER_V4 = new Set([
  ...QUESTION_FILLER,
  ..."ca,cela,ceci,celle,celui,esto,eso,esta,este,aquello,questo,questa,quello,quella,dies,dieses,diese,dieser,das".split(","),
  // "I'm looking for …", "je cherche …", "ich suche …", "busco …", "cerco …": a search, not its subject
  ..."looking,searching,search,cherche,cherchons,recherche,suche,suchen,busco,buscando,buscar,cerco,cercando,cercare".split(","),
  // participles of the verbs of posting, in every gender and number ("combien d'œuvres a-t-elle publiées ?")
  ..."publiees,postees,creees,crees,dessinees,dessines,partagees,partages,publicadas,publicados,subidas,subidos,creadas,creados,pubblicate,pubblicati,postate,postati,create,creati,disegnate,disegnati,gepostete,veroffentlichte".split(","),
  ..."postato,postata,caricato,caricata,caricati,caricate,subido,subida,hochgeladen,mis,ligne".split(","),
  ..."erstellt,gemacht,gezeichnet,gemalt,veroffentlicht,geteilt,wurde,wurden".split(","),
  // "published during October 2026": the month is the filter, "during" no subject
  ..."during,throughout,pendant,durant,wahrend,durante".split(","),
  // "works", "travaux", "lavori": artworks, in a question about how many
  ..."works,travaux,lavori,lavoro,trabajos,arbeiten".split(","),
  // "¿qué se publicó …?": the reflexive is no subject
  "se",
]);

/** v4's questions said the way v3's patterns read them ("most recently" → "latest", "count the" → "how many"). */
export const REWRITES_V4: Array<[RegExp, string]> = [
  [/\bmost recently\b/g, "latest"],
  // "qui a posté le plus récemment ?", "chi ha pubblicato più di recente?": the latest, not the most
  [/\ble plus recemment\b|\bpiu di recente\b|\bpiu recentemente\b|\bmas recientemente\b|\bam neuesten\b|\bam kurzlichsten\b/g, "latest"],
  [/\b(?:start|started|begin|began|starts|begins) (?:posting|publishing|uploading)\b/g, "post the first artwork"],
  [/\b(?:came|come|comes) out\b|\bwent up\b|\bgot posted\b|\bwere released\b|\bwas released\b/g, "posted"],
  // "count the artworks of @x", "count @x's posts" ("Count Dracula artworks" is a search)
  [/^(?:count|tally) (?:(?:the|all|every) |(?=@|s |my |his |her |their |artworks?\b|posts?\b|pieces?\b|works?\b|images?\b|pictures?\b|drawings?\b))/g, "how many "],
  // "von wem ist das erste Kunstwerk?": who posted it
  [/^von wem (?:ist|stammt|kommt) (?=(?:das|der|die|den|dem) )/g, "wer hat "],
  // "answer only for @alice: …", "tell me only about @alice: …": the account stays, the phrasing goes
  [/^(?:please )?(?:answer|reply|respond|tell me|talk) only (?:for|about|regarding|on) /g, ""],
  // "ever since September" is since September
  [/\bever since\b/g, "since"],
  // "was wurde zuletzt gepostet?", "wer hat zuerst eine Katze gepostet?"
  [/\b(?:zuletzt|als letztes)\b/g, "latest"],
  [/\b(?:zuerst|als erstes)\b/g, "first"],
  [/\btotal number of\b/g, "number of"],
];

/** Phrases of time that say nothing about the subject ("so far", "to date", "bisher"), removed from v4's questions. */
export const TIME_FILLER_V4 = /\b(?:so far|to date|until now|up to now|till now|jusqu ici|jusqu a present|jusqu a maintenant|bisher|bis jetzt|hasta ahora|hasta la fecha|finora|fino ad ora|fino a ora)\b/g;

export const MONTHS: Record<string, number> = (() => {
  const m: Record<string, number> = {};
  const names = [
    "january,janvier,januar,enero,gennaio,jan,janv",
    "february,fevrier,februar,febrero,febbraio,feb,fev,fevr",
    "march,mars,marz,marzo,mar",
    "april,avril,abril,aprile,apr,avr",
    "may,mai,mayo,maggio",
    "june,juin,juni,junio,giugno,jun",
    "july,juillet,juli,julio,luglio,jul,juil",
    "august,aout,agosto,aug",
    "september,septembre,septiembre,settembre,sept,sep",
    "october,octobre,oktober,octubre,ottobre,oct,okt",
    "november,novembre,noviembre,nov",
    "december,decembre,dezember,diciembre,dicembre,dec,dez",
  ];
  names.forEach((line, i) => line.split(",").forEach((w) => (m[w] = i)));
  return m;
})();
