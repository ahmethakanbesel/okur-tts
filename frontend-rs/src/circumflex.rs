//! Put back circumflexes that everyday Turkish leaves out (alakalı → alâkalı, kar → kâr by context, hala → hâlâ).
//! A port of okur.frontend.circumflex; see that module for the rules' rationale.

const STEMS: &[(&str, &str)] = &[
    ("alaka", "alâka"), ("dükkan", "dükkân"), ("hikaye", "hikâye"), ("kağıt", "kâğıt"), ("kağıd", "kâğıd"),
    ("rüzgar", "rüzgâr"), ("imkan", "imkân"), ("kase", "kâse"), ("katip", "kâtip"), ("kafi", "kâfi"),
    ("kainat", "kâinat"), ("kabus", "kâbus"), ("kahya", "kâhya"), ("tezgah", "tezgâh"), ("dergah", "dergâh"),
    ("nikah", "nikâh"), ("yegane", "yegâne"), ("mekan", "mekân"),
];
const EXCEPTION_PREFIXES: &[&str] = &["mekani", "kaset", "kafile", "kafiye"];
const KAR_SUFFIXES: &[&str] = &["", "ı", "ın", "ını", "ına", "ında", "ından", "lı", "lılık", "lılığı", "sız", "dan", "da",
    "a", "lar", "ları", "ların"];
const PROFIT: &[&str] = &["şirket", "net", "brüt", "zarar", "marj", "pay", "satış", "ciro", "gelir", "vergi", "hisse",
    "yatırım", "milyon", "milyar", "lira", "dolar", "avro", "euro", "yüzde", "faaliyet", "bilanço", "elde", "amaç",
    "ticar", "ekonomi", "banka", "kazanç", "çeyrek", "dönem"];
const SNOW: &[&str] = &["yağ", "beyaz", "soğuk", "kış", "buz", "tipi", "dağ", "kalınlı", "santim", "fırtına", "lapa",
    "erim", "erid", "kayak", "örtü"];
const KIN: &[&str] = &["kızı", "oğlu", "teyze", "amca", "dayı", "yenge", "enişte", "kuzen", "anneanne", "babaanne"];
const WINDOW: usize = 4;

/// First code point (digit zero) of every run of Unicode decimal digits (category Nd), Unicode 15.1.0: Python's `\d`.
const DIGIT_ZEROS: &[u32] = &[
    0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66, 0xde6, 0xe50,
    0xed0, 0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90, 0x1b50, 0x1bb0, 0x1c40,
    0x1c50, 0xa620, 0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0, 0xff10, 0x104a0, 0x10d30, 0x11066,
    0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650, 0x116c0, 0x11730, 0x118e0, 0x11950,
    0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16a60, 0x16ac0, 0x16b50, 0x1d7ce, 0x1d7d8, 0x1d7e2, 0x1d7ec,
    0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950, 0x1fbf0
];

/// Python's `\d`: any Unicode decimal digit (0-9, ٠-٩, ०-९, ...).
pub fn is_decimal(c: char) -> bool {
    let c = c as u32;
    DIGIT_ZEROS.iter().any(|&z| (z..z + 10).contains(&c))
}

/// Python's `[^\W\d_]`: a letter or a non-digit numeral (½, ²), not a decimal digit, underscore or combining mark.
pub fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() && !is_decimal(c) && c != '_' && unicode_normalization::char::canonical_combining_class(c) == 0
}

fn has_circumflex(word: &str) -> bool {
    word.contains(['â', 'î', 'û'])
}

fn restore_stem(word: &str) -> String {
    if EXCEPTION_PREFIXES.iter().any(|p| word.starts_with(p)) {
        return word.to_string();
    }
    for (plain, marked) in STEMS {
        if let Some(rest) = word.strip_prefix(plain) {
            return format!("{marked}{rest}");
        }
    }
    word.to_string()
}

fn context<'a>(words: &'a [&'a str], i: usize) -> impl Iterator<Item = &'a str> {
    words[i.saturating_sub(WINDOW)..i].iter().chain(&words[(i + 1).min(words.len())..(i + 1 + WINDOW).min(words.len())]).copied()
}

fn restore_kar(words: &[&str], i: usize) -> String {
    let word = words[i];
    let is_kar = word.strip_prefix("kar").is_some_and(|rest| KAR_SUFFIXES.contains(&rest));
    if !is_kar {
        return word.to_string();
    }
    let count = |stems: &[&str]| context(words, i).filter(|w| stems.iter().any(|s| w.starts_with(s))).count();
    if count(PROFIT) > count(SNOW) { format!("kâ{}", &word[2..]) } else { word.to_string() }
}

fn restore_hala(words: &[&str], i: usize) -> String {
    if words[i] != "hala" {
        return words[i].to_string();
    }
    let kin = [i.checked_sub(2), i.checked_sub(1), Some(i + 1)].into_iter().flatten()
        .filter(|&j| j < words.len()).any(|j| KIN.contains(&words[j]));
    if kin { "hala".to_string() } else { "hâlâ".to_string() }
}

/// Return lowercase `text` with circumflexes restored where pronunciation needs them.
pub fn restore(text: &str) -> String {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        match (is_word_char(c), start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                spans.push((s, i));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        spans.push((s, text.len()));
    }
    let words: Vec<&str> = spans.iter().map(|&(a, b)| &text[a..b]).collect();
    let mut out = String::with_capacity(text.len() + 8);
    let mut last = 0;
    for (i, &(a, b)) in spans.iter().enumerate() {
        let original = words[i];
        let mut word = original.to_string();
        if !has_circumflex(original) {
            word = restore_hala(&words, i);
            if word == original {
                word = restore_kar(&words, i);
            }
            if word == original {
                word = restore_stem(original);
            }
        }
        out.push_str(&text[last..a]);
        out.push_str(&word);
        last = b;
    }
    out.push_str(&text[last..]);
    out
}
