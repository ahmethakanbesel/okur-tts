/** Save generated speech as WAV (48 kHz) or MP3 at 48, 24 or 16 kHz. The MP3 encoder (LAME, LGPL-3.0) loads only
 *  when someone exports MP3. */
import { wav } from "./wav";

export type ExportFormat = "wav" | "mp3-48" | "mp3-24" | "mp3-16";

const MP3: Record<Exclude<ExportFormat, "wav">, { rate: number; kbps: number }> = {
  "mp3-48": { rate: 48000, kbps: 128 },
  "mp3-24": { rate: 24000, kbps: 96 },
  "mp3-16": { rate: 16000, kbps: 64 },
};

async function resample(samples: Float32Array<ArrayBuffer>, from: number, to: number): Promise<Float32Array> {
  if (from === to) return samples;
  const ctx = new OfflineAudioContext(1, Math.ceil((samples.length * to) / from), to);
  const buffer = ctx.createBuffer(1, samples.length, from);
  buffer.copyToChannel(samples, 0);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.start();
  return (await ctx.startRendering()).getChannelData(0);
}

export async function exportAudio(samples: Float32Array<ArrayBuffer>, rate: number, format: ExportFormat): Promise<Blob> {
  if (format === "wav") return wav(samples, rate);
  const { rate: target, kbps } = MP3[format];
  const { Mp3Encoder } = await import("@breezystack/lamejs");
  const audio = await resample(samples, rate, target);
  const pcm = Int16Array.from(audio, (x) => Math.max(-1, Math.min(1, x)) * 0x7fff);
  const encoder = new Mp3Encoder(1, target, kbps);
  const parts: Uint8Array<ArrayBuffer>[] = [];
  for (let i = 0; i < pcm.length; i += 1152 * 16) {
    parts.push(Uint8Array.from(encoder.encodeBuffer(pcm.subarray(i, i + 1152 * 16))));
  }
  parts.push(Uint8Array.from(encoder.flush()));
  return new Blob(parts, { type: "audio/mpeg" });
}

export const extension = (format: ExportFormat): string => (format === "wav" ? "wav" : "mp3");

/** "Bu konuyla alâkalı, şirketin…" → "bu-konuyla-alakali-sirketin": a whole sentence as a safe ASCII file name. */
export function asciiName(text: string, max = 120): string {
  const turkish: Record<string, string> = { ç: "c", ğ: "g", ı: "i", ö: "o", ş: "s", ü: "u", â: "a", î: "i", û: "u",
    Ç: "C", Ğ: "G", İ: "I", Ö: "O", Ş: "S", Ü: "U", Â: "A", Î: "I", Û: "U" };
  const plain = Array.from(text, (c) => turkish[c] ?? c).join("").normalize("NFKD").replace(/[̀-ͯ]/g, "");
  const name = plain.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return (name.slice(0, max).replace(/-+$/, "") || "okur");
}
