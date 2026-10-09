//! The model's alphabet: Turkish letters with â, î and û kept, plus a few punctuation marks.

use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::canonical_combining_class;

pub const PAD: &str = "<pad>";
pub const UNK: &str = "<unk>";
pub const PUNCTUATION: &str = " !\"%&'(),-./:;?";
pub const LETTERS: &str = "abcdefghijklmnopqrstuvwxyzçğıöşüâîû";
pub const UNK_ID: i64 = 1;

const KEEP_ACCENT: &str = "çğıöşüâîûÇĞİÖŞÜÂÎÛ";

/// Id of a symbol: 0 is padding, 1 unknown, then punctuation, then letters.
pub fn symbol_id(c: char) -> Option<i64> {
    let offset = |s: &str, base: usize| s.chars().position(|x| x == c).map(|i| (base + i) as i64);
    offset(PUNCTUATION, 2).or_else(|| offset(LETTERS, 2 + PUNCTUATION.chars().count()))
}

/// All symbols in id order (for host code that builds its own tables).
pub fn symbols() -> Vec<String> {
    [PAD.to_string(), UNK.to_string()].into_iter().chain(PUNCTUATION.chars().chain(LETTERS.chars()).map(String::from)).collect()
}

fn typography(c: char) -> Option<&'static str> {
    Some(match c {
        '’' | '‘' | 'ʼ' | '´' | '`' => "'",
        '“' | '”' | '„' | '«' | '»' => "\"",
        '–' | '—' | '−' => "-",
        '…' => "...",
        _ => return None,
    })
}

fn is_unsafe(c: char) -> bool {
    matches!(c, '\u{00}'..='\u{08}' | '\u{0b}'..='\u{1f}' | '\u{7f}'..='\u{9f}' | '\u{061c}' | '\u{200e}' | '\u{200f}'
        | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
}

/// Compose (NFC) and drop control and bidirectional characters.
pub fn clean(text: &str) -> String {
    text.nfc().map(|c| if is_unsafe(c) { ' ' } else { c }).collect()
}

pub fn turkish_lower(text: &str) -> String {
    text.replace('İ', "i").replace('I', "ı").to_lowercase()
}

/// Lowercase the Turkish way, strip accents the alphabet lacks (é → e), drop anything else unreadable.
pub fn to_alphabet(text: &str) -> String {
    let mut typographic = String::with_capacity(text.len());
    for c in text.chars() {
        match typography(c) {
            Some(s) => typographic.push_str(s),
            None => typographic.push(c),
        }
    }
    let mut out = String::with_capacity(text.len());
    for c in turkish_lower(&typographic).chars() {
        let plain: String = if KEEP_ACCENT.contains(c) {
            c.to_string()
        } else {
            std::iter::once(c).nfkd().filter(|&x| canonical_combining_class(x) == 0).collect()
        };
        if !plain.is_empty() && plain.chars().all(|x| symbol_id(x).is_some()) {
            out.push_str(&plain);
        } else {
            out.push(' ');
        }
    }
    let mut collapsed = String::with_capacity(out.len());
    for c in out.chars() {
        if !(c == ' ' && collapsed.ends_with(' ')) {
            collapsed.push(c);
        }
    }
    collapsed.trim_matches(' ').to_string()
}

pub fn encode(letters: &str) -> Vec<i64> {
    letters.chars().map(|c| symbol_id(c).unwrap_or(UNK_ID)).collect()
}
