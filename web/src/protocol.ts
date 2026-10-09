/** Messages between the page and the synthesis worker. */
export type Request =
  | { type: "load"; threads?: number }
  | { type: "speak"; id: number; text: string; speed: number; seed: number; pause: number; temperature: number }
  | { type: "letters"; id: number; text: string }
  | { type: "cancel" };

export type Reply =
  | { type: "progress"; loaded: number; total: number }
  | { type: "stage"; message: string }
  | { type: "ready"; threads: number; isolated: boolean; loadMs: number }
  | { type: "letters"; id: number; letters: string }
  | { type: "chunk"; id: number; index: number; count: number; letters: string; dur: Float32Array<ArrayBuffer>;
      audio: Float32Array<ArrayBuffer>; seconds: number;
      computeMs: number }
  | { type: "done"; id: number }
  | { type: "error"; id?: number; message: string };
