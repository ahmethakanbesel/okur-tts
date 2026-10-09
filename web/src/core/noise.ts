/** Seeded standard normals, identical to okur.export_web.gaussian_noise: mulberry32 + Box–Muller. */
export function gaussianNoise(seed: number, count: number): Float32Array {
  let state = seed >>> 0;
  const uniform = (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 15), z | 1);
    z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 2) {
    const u1 = 1 - uniform();
    const u2 = uniform();
    const r = Math.sqrt(-2 * Math.log(u1));
    out[i] = r * Math.cos(2 * Math.PI * u2);
    if (i + 1 < count) out[i + 1] = r * Math.sin(2 * Math.PI * u2);
  }
  return out;
}
