"""Paper-style architecture figures for the README and the demo page, in English and Turkish.

    uv run python scripts/figures.py

Writes docs/figures/{pipeline,dit}.svg (English, for GitHub) and web/src/figures/{pipeline,dit}.{en,tr}.svg (inlined by
the page, so they use its typefaces). Shapes and counts follow okur.model: d = 224, 64-dim latents at 25 Hz.
"""

import re
from html import escape
from pathlib import Path

SANS = "Archivo, 'Helvetica Neue', Arial, sans-serif"
MATH = "'STIX Two Text', 'Times New Roman', Times, serif"
MONO = "'Courier Prime', 'Courier New', monospace"

STYLE = f"""
.fig text {{ font-family: {SANS}; fill: #151515; }}
.fig .t {{ font-size: 13.5px; font-weight: 650; }}
.fig .s {{ font-size: 11.5px; fill: #3c403e; }}
.fig .p {{ font-family: {MONO}; font-size: 11px; fill: #5b615e; }}
.fig .m {{ font-family: {MATH}; font-style: italic; font-size: 14px; }}
.fig .mono {{ font-family: {MONO}; font-size: 12.5px; }}
.fig .cap {{ font-size: 11px; font-weight: 700; letter-spacing: .12em; fill: #5b615e; }}
.fig .box {{ stroke: #151515; stroke-width: 1.1; }}
.fig .model {{ fill: #e3ebe7; }}
.fig .dec {{ fill: #ece8df; }}
.fig .rules {{ fill: #ffffff; }}
.fig .io {{ fill: #ffffff; stroke-dasharray: 0; }}
.fig .host {{ fill: #ffffff; stroke-dasharray: 5 4; }}
.fig .op {{ fill: #ffffff; }}
.fig .arrow {{ stroke: #151515; stroke-width: 1.2; fill: none; marker-end: url(#ah); }}
.fig .dash {{ stroke: #151515; stroke-width: 1.1; fill: none; stroke-dasharray: 4 3; marker-end: url(#ah); }}
.fig .axis {{ stroke: #151515; stroke-width: 1.2; }}
"""

T = {
    "en": {
        "text": "Text",
        "frontend": "Text frontend",
        "fe1": "normalizer-tr · ordinals",
        "fe2": "circumflexes · Rust → Wasm",
        "letters": "Letters",
        "encoder": "Text encoder",
        "enc1": "Emb(x) + g · 4× ConvNeXt",
        "enc2": "2× MHSA (RoPE)",
        "duration": "Duration predictor",
        "dur1": "2× Conv1d",
        "dur2": "frames per letter",
        "timeline": "Word timeline",
        "tl1": "host code",
        "aligner": "Aligner",
        "al1": "windowed Gaussian",
        "al2": "cross-attention, ±1 word",
        "dit": "DiT generator",
        "dit1": "6 blocks · RoPE · shared adaLN · SwiGLU",
        "dit2": "condition: t and speaker g",
        "steps": "× 4 steps",
        "noise": "Noise",
        "decoder": "Decoder",
        "dec1": "HiFi-GAN generator",
        "dec2": "upsampling 8·6·5·2·2·2 = 1920",
        "audio": "Audio",
        "leg_model": "acoustic model · 6.8 M",
        "leg_dec": "decoder · 3.0 M",
        "leg_rules": "rules (Rust → Wasm)",
        "leg_host": "host code (TS / Python)",
        "block": "DiT BLOCK  × 6",
        "adaln": "adaLN-ZERO MODULATION (SHARED)",
        "temb": "timestep emb.",
        "shared": "Linear (shared)",
        "offset": "+ block offset",
        "toall": "to every block",
        "sampling": "4-STEP SAMPLING (DMD2 STUDENT)",
        "pred": "predict",
        "renoise": "re-noise",
        "speaker": "speaker g",
        "out": "after 6 blocks: LN, adaLN, W_out",
    },
    "tr": {
        "text": "Metin",
        "frontend": "Metin ön işleyici",
        "fe1": "normalizer-tr · sıra sayıları",
        "fe2": "şapkalar · Rust → Wasm",
        "letters": "Harfler",
        "encoder": "Metin kodlayıcı",
        "enc1": "Emb(x) + g · 4× ConvNeXt",
        "enc2": "2× MHSA (RoPE)",
        "duration": "Süre tahmini",
        "dur1": "2× Conv1d",
        "dur2": "harf başına kare",
        "timeline": "Kelime zaman çizelgesi",
        "tl1": "ana makine kodu",
        "aligner": "Hizalayıcı",
        "al1": "Gauss pencereli",
        "al2": "çapraz dikkat, ±1 kelime",
        "dit": "DiT üretici",
        "dit1": "6 blok · RoPE · ortak adaLN · SwiGLU",
        "dit2": "koşul: t ve konuşmacı g",
        "steps": "× 4 adım",
        "noise": "Gürültü",
        "decoder": "Kod çözücü",
        "dec1": "HiFi-GAN üreteci",
        "dec2": "büyütme 8·6·5·2·2·2 = 1920",
        "audio": "Ses",
        "leg_model": "akustik model · 6,8 M",
        "leg_dec": "kod çözücü · 3,0 M",
        "leg_rules": "kurallar (Rust → Wasm)",
        "leg_host": "ana makine kodu (TS / Python)",
        "block": "DiT BLOĞU  × 6",
        "adaln": "adaLN-ZERO MODÜLASYONU (ORTAK)",
        "temb": "zaman gömmesi",
        "shared": "Linear (ortak)",
        "offset": "+ blok ofseti",
        "toall": "her bloğa",
        "sampling": "4 ADIMLI ÖRNEKLEME (DMD2 ÖĞRENCİSİ)",
        "pred": "tahmin",
        "renoise": "yeniden gürültü",
        "speaker": "konuşmacı g",
        "out": "6 bloktan sonra: LN, adaLN, W_out",
    },
}


def math(expr: str) -> str:
    """'x ∈ ℝ^{L×224}' → SVG tspans with super/subscripts."""
    out, i = [], 0
    for m in re.finditer(r"([\^_])\{([^}]*)\}", expr):
        out.append(escape(expr[i : m.start()]))
        shift = "super" if m.group(1) == "^" else "sub"
        out.append(f'<tspan baseline-shift="{shift}" font-size="70%">{escape(m.group(2))}</tspan>')
        i = m.end()
    out.append(escape(expr[i:]))
    return "".join(out)


class Fig:
    def __init__(self, w: int, h: int, title: str) -> None:
        self.w, self.h, self.title, self.parts = w, h, title, []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x: float, y: float, s: str, cls: str, anchor: str = "start", raw: bool = False) -> None:
        body = s if raw else escape(s)
        self.add(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{body}</text>')

    def box(self, x: float, y: float, w: float, h: float, kind: str, r: int = 4) -> None:
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" class="box {kind}"/>')

    def node(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        kind: str,
        title: str,
        lines: list[str],
        params: str = "",
        out: str = "",
    ) -> None:
        self.box(x, y, w, h, kind)
        self.text(x + 12, y + 21, title, "t")
        for k, line in enumerate(lines):
            self.text(x + 12, y + 39 + 15 * k, line, "s")
        if params:
            self.text(x + w - 10, y + 21, params, "p", "end")
        if out:
            self.text(x + 12, y + h - 10, math(out), "m", raw=True)

    def line(self, pts: list[tuple[float, float]], cls: str = "arrow") -> None:
        d = "M" + " L".join(f"{x} {y}" for x, y in pts)
        self.add(f'<path d="{d}" class="{cls}"/>')

    def curve(self, d: str, cls: str = "arrow") -> None:
        self.add(f'<path d="{d}" class="{cls}"/>')

    def svg(self) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
            f'width="{self.w}" height="{self.h}" class="fig" role="img" '
            f'aria-label="{escape(self.title)}"><title>{escape(self.title)}</title><style>{STYLE}</style>'
            '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path d="M0 1 L9 5 L0 9 z" fill="#151515"/></marker></defs>'
            f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>' + "".join(self.parts) + "</svg>"
        )


def pipeline(t: dict[str, str]) -> str:
    f = Fig(1040, 452, f"{t['frontend']} → {t['encoder']} → {t['aligner']} → {t['dit']} → {t['decoder']}")
    # row 1: text to durations
    f.node(16, 30, 116, 84, "io", t["text"], [], out="")
    f.text(28, 76, "karı hala 3.", "mono")
    f.node(160, 30, 214, 84, "rules", t["frontend"], [t["fe1"], t["fe2"]])
    f.node(402, 30, 152, 84, "io", t["letters"], [], out="x ∈ Σ^{L},  |Σ| = 52")
    f.text(414, 76, "kârı hâlâ üçüncü", "mono")
    f.node(582, 30, 210, 84, "model", t["encoder"], [t["enc1"], t["enc2"]], "2,0 M", "h ∈ ℝ^{L×224}")
    f.node(820, 30, 204, 84, "model", t["duration"], [t["dur1"], t["dur2"]], "0,4 M", "d ∈ ℝ^{L},  d > 0")
    for x0, x1 in ((132, 160), (374, 402), (554, 582), (792, 820)):
        f.line([(x0, 72), (x1 - 2, 72)])
    # row 2: timeline, aligner, generator
    f.node(820, 190, 204, 84, "host", t["timeline"], [t["tl1"]], out="T = Σ_{l} d_{l} ;  (w_{t}, p_{t})")
    f.node(552, 190, 240, 84, "model", t["aligner"], [t["al1"], t["al2"]], "0,2 M", "c ∈ ℝ^{T×224}")
    f.node(196, 190, 328, 84, "model", t["dit"], [t["dit1"], t["dit2"]], "4,2 M", "x̂ ∈ ℝ^{T×64}  (25 Hz)")
    f.node(16, 198, 150, 68, "io", t["noise"], [], out="z ∼ N(0, I) ∈ ℝ^{T×64}")
    f.line([(922, 114), (922, 188)])
    f.line([(820, 232), (794, 232)])
    f.line([(687, 114), (687, 188)])
    f.text(694, 156, math("h"), "m", raw=True)
    f.line([(552, 232), (526, 232)])
    f.text(530, 224, math("c"), "m", raw=True)
    f.line([(166, 232), (194, 232)])
    f.curve("M 470 190 C 470 160, 520 160, 520 186", "arrow")
    f.text(526, 168, t["steps"], "s")
    # row 3: decoder and audio
    f.node(196, 340, 328, 84, "dec", t["decoder"], [t["dec1"], t["dec2"]], "3,0 M", "y ∈ [−1, 1]^{1920 T}")
    f.line([(360, 274), (360, 338)])
    f.text(368, 312, math("x̂"), "m", raw=True)
    f.box(552, 340, 240, 84, "io")
    f.text(564, 361, t["audio"] + " · 48 kHz", "t")
    wave = " ".join(
        f"L {564 + i * 4} {392 + (9 if i % 2 else -9) * abs(((i * 37) % 11) - 5) / 5:.1f}" for i in range(52)
    )
    f.curve(f"M 564 392 {wave}", "axis")
    f.line([(524, 382), (550, 382)])
    # legend
    for k, (kind, label) in enumerate(
        (("model", t["leg_model"]), ("dec", t["leg_dec"]), ("rules", t["leg_rules"]), ("host", t["leg_host"]))
    ):
        y = 334 + 24 * k
        f.box(822, y, 26, 16, kind, 2)
        f.text(858, y + 12.5, label, "s")
    return f.svg()


def dit(t: dict[str, str]) -> str:
    f = Fig(1040, 372, f"{t['block']}, {t['sampling']}")
    f.text(16, 22, t["block"], "cap")

    # sub-layer 1 (row y=72) and sub-layer 2 (row y=190); boxes 40 tall
    def op(x: float, y: float, w: float, label: str, math_label: bool = False) -> None:
        f.box(x, y - 20, w, 40, "op", 3)
        f.text(
            x + w / 2, y + 5, math(label) if math_label else label, "m" if math_label else "t", "middle", raw=math_label
        )

    def plus(cx: float, cy: float) -> None:
        f.add(f'<circle cx="{cx}" cy="{cy}" r="12" class="box op"/>')
        f.add(f'<path d="M{cx - 6} {cy} H{cx + 6} M{cx} {cy - 6} V{cy + 6}" stroke="#151515" stroke-width="1.3"/>')

    op(16, 72, 112, "x_{t}W_{in} + c", True)
    for y, mod, layer, gate in (
        (72, "(1+γ_{1})·+β_{1}", "MHSA · RoPE", "α_{1} ⊙"),
        (190, "(1+γ_{2})·+β_{2}", "SwiGLU", "α_{2} ⊙"),
    ):
        op(156, y, 50, "LN")
        op(232, y, 112, mod, True)
        op(370, y, 112, layer)
        op(508, y, 56, gate, True)
        plus(600, y)
        for x0, x1 in ((206, 232), (344, 370), (482, 508), (564, 588)):
            f.line([(x0, y), (x1 - 1, y)])
    f.line([(128, 72), (155, 72)])
    f.line([(142, 72), (142, 34), (600, 34), (600, 59)])  # residual 1
    f.line([(600, 84), (600, 128), (142, 128), (142, 190), (155, 190)])
    f.line([(142, 190), (142, 236), (600, 236), (600, 203)])  # residual 2
    f.line([(612, 190), (650, 190)])
    f.text(656, 195, math("v_{θ}"), "m", raw=True)
    f.text(156, 270, t["out"], "s")
    # adaLN branch
    f.text(16, 302, t["adaln"], "cap")
    op(16, 336, 40, "t", True)
    op(78, 336, 116, t["temb"])
    plus(220, 336)
    f.text(220, 366, t["speaker"], "s", "middle")
    op(246, 336, 120, t["shared"])
    op(390, 336, 110, t["offset"])
    for x0, x1 in ((56, 78), (194, 208), (232, 246), (366, 390)):
        f.line([(x0, 336), (x1 - 1, 336)])
    f.line([(500, 336), (528, 336)])
    f.text(534, 341, math("(β_{1}, γ_{1}, α_{1}, β_{2}, γ_{2}, α_{2})"), "m", raw=True)
    f.line([(560, 322), (560, 262)], "dash")
    f.text(568, 296, t["toall"], "s")
    # sampling schedule
    x0, x1, ay = 724, 1004, 196
    f.text(704, 22, t["sampling"], "cap")
    f.add(f'<line x1="{x0}" y1="{ay}" x2="{x1}" y2="{ay}" class="axis"/>')
    ticks = [(0, "0"), (0.25, "¼"), (0.5, "½"), (0.75, "¾"), (1, "1")]
    xs = {v: x0 + v * (x1 - x0) for v, _ in ticks}
    for v, label in ticks:
        f.add(f'<line x1="{xs[v]}" y1="{ay - 5}" x2="{xs[v]}" y2="{ay + 5}" class="axis"/>')
        f.text(xs[v], ay + 22, math("t = " + label) if v in (0, 1) else label, "m", "middle", raw=True)
    for k, v in enumerate((0, 0.25, 0.5, 0.75)):
        h = 40 + 26 * (3 - k)
        f.curve(f"M {xs[v]} {ay - 7} C {xs[v]} {ay - h}, {xs[1]} {ay - h}, {xs[1]} {ay - 9}")
    for v in (0.25, 0.5, 0.75):
        h = 26 + 40 * (1 - v)
        f.curve(f"M {xs[1]} {ay + 34} C {xs[1]} {ay + 34 + h}, {xs[v]} {ay + 34 + h}, {xs[v]} {ay + 36}", "dash")
    f.text(704, 60, t["pred"] + ":", "s")
    f.text(704, 78, math("x̂_{1} = x_{t} + (1 − t)·v_{θ}(x_{t}, c, t)"), "m", raw=True)
    f.text(704, 318, t["renoise"] + ":", "s")
    f.text(704, 338, math("x_{t′} = (1 − t′)·ε + t′·x̂_{1},   ε ∼ N(0, I)"), "m", raw=True)
    return f.svg()


def main() -> None:
    docs, web = Path("docs/figures"), Path("web/src/figures")
    docs.mkdir(parents=True, exist_ok=True)
    web.mkdir(parents=True, exist_ok=True)
    for name, make in (("pipeline", pipeline), ("dit", dit)):
        for lang, labels in T.items():
            svg = make(labels)
            if lang == "en":  # decimal point in English parameter counts
                svg = re.sub(r"(\d),(\d) M", r"\1.\2 M", svg)
            (web / f"{name}.{lang}.svg").write_text(svg, encoding="utf-8")
            if lang == "en":
                (docs / f"{name}.svg").write_text(svg, encoding="utf-8")
    print("figures written")


if __name__ == "__main__":
    main()
