/**
 * The word–letter timeline (okur.model.timeline): which word, and where inside it, every letter and output frame
 * sits. A word owns its letters and the spaces and punctuation that follow it.
 */

export interface Words {
  cw: Int32Array; // word of each letter
  wstart: Int32Array; // first letter of that word
  nWords: number;
}

export function words(letters: string): Words {
  const chars = Array.from(letters);
  const starts: number[] = [];
  chars.forEach((ch, i) => {
    if (ch !== " " && (i === 0 || chars[i - 1] === " ")) starts.push(i);
  });
  const bounds = [0, ...starts.slice(1), chars.length];
  const cw = new Int32Array(chars.length);
  const wstart = new Int32Array(chars.length);
  for (let w = 0; w + 1 < bounds.length; w++) {
    const a = bounds[w]!;
    const b = bounds[w + 1]!;
    cw.fill(w, a, b);
    wstart.fill(a, a, b);
  }
  return { cw, wstart, nWords: Math.max(1, bounds.length - 1) };
}

/** torch.round: halves go to the even neighbour. */
function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x - Math.trunc(x)) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Whole frames per word, rounded on the running total so the sum never drifts from the sum of durations. */
export function wordFrames(dur: Float32Array, cw: Int32Array, nWords: number): Int32Array {
  const perWord = new Float64Array(nWords);
  cw.forEach((w, i) => {
    perWord[w]! += dur[i]!;
  });
  const counts = new Int32Array(nWords);
  let total = 0;
  let previous = 0;
  for (let w = 0; w < nWords; w++) {
    total += perWord[w]!;
    const end = roundHalfEven(total);
    counts[w] = Math.max(0, end - previous);
    previous = end;
  }
  return counts;
}

/** Word of each frame and its position inside that word, in [0, 1). */
export function frameTimeline(counts: Int32Array): { fw: Int32Array; fp: Float32Array } {
  const frames = counts.reduce((a, b) => a + b, 0);
  const fw = new Int32Array(frames);
  const fp = new Float32Array(frames);
  let f = 0;
  counts.forEach((count, w) => {
    for (let i = 0; i < count; i++, f++) {
      fw[f] = w;
      fp[f] = i / count;
    }
  });
  return { fw, fp };
}
