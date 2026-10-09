/**
 * Split normalized letters into sentence-sized chunks so long texts start playing early and attention stays short.
 * Splits after . ! ? (then ; : , if a piece is still too long); never inside a word.
 */
const MAX_LETTERS = 360;

function splitAfter(text: string, marks: RegExp): string[] {
  const parts: string[] = [];
  let start = 0;
  for (const m of text.matchAll(marks)) {
    const end = m.index + m[0].length;
    parts.push(text.slice(start, end).trim());
    start = end;
  }
  parts.push(text.slice(start).trim());
  return parts.filter((p) => p.length > 0);
}

function pack(pieces: string[], limit: number): string[] {
  const out: string[] = [];
  for (const piece of pieces) {
    const last = out.at(-1);
    if (last !== undefined && last.length + 1 + piece.length <= limit && last.length < 40) {
      out[out.length - 1] = `${last} ${piece}`; // glue very short sentences to the previous one
    } else {
      out.push(piece);
    }
  }
  return out;
}

export function chunk(letters: string, limit = MAX_LETTERS): string[] {
  const out: string[] = [];
  for (const sentence of pack(splitAfter(letters, /[.!?]+ /g), limit)) {
    if (sentence.length <= limit) {
      out.push(sentence);
      continue;
    }
    for (const clause of pack(splitAfter(sentence, /[;:,] /g), limit)) {
      if (clause.length <= limit) {
        out.push(clause);
        continue;
      }
      let rest = clause; // a run-on clause: cut at the last space before the limit
      while (rest.length > limit) {
        const cut = rest.lastIndexOf(" ", limit);
        const at = cut > 0 ? cut : limit;
        out.push(rest.slice(0, at).trim());
        rest = rest.slice(at).trim();
      }
      if (rest) out.push(rest);
    }
  }
  return out;
}
