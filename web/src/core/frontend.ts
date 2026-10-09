/**
 * The text frontend, compiled from Rust (frontend-rs) to WebAssembly: written Turkish in, the model's letters out.
 * Identical to the Python frontend used in training (tests/test_frontend_rs.py checks it on ~4k texts).
 */

interface Exports {
  memory: WebAssembly.Memory;
  alloc(len: number): number;
  dealloc(ptr: number, capacity: number): void;
  letters(ptr: number, len: number): number;
}

export class Frontend {
  private readonly encoder = new TextEncoder();
  private readonly decoder = new TextDecoder();

  private constructor(private readonly wasm: Exports) {}

  static async load(source: BufferSource | Response | Promise<Response>): Promise<Frontend> {
    const { instance } = source instanceof Response || source instanceof Promise
      ? await WebAssembly.instantiateStreaming(source, {})
      : await WebAssembly.instantiate(source, {});
    return new Frontend(instance.exports as unknown as Exports);
  }

  letters(text: string): string {
    const { memory, alloc, dealloc, letters } = this.wasm;
    const bytes = this.encoder.encode(text);
    const input = alloc(bytes.length);
    new Uint8Array(memory.buffer, input, bytes.length).set(bytes);
    const output = letters(input, bytes.length);
    dealloc(input, bytes.length);
    const length = new DataView(memory.buffer).getUint32(output, true);
    const result = this.decoder.decode(new Uint8Array(memory.buffer, output + 4, length));
    dealloc(output, length + 4);
    return result;
  }
}
