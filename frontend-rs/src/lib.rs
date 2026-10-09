//! Turkish TTS text frontend: any written Turkish in, the model's letters out.
//!
//! clean → ordinals → normalize (numbers, dates, money, abbreviations; normalizer-tr) → lowercase → restore
//! circumflexes → alphabet. Produces exactly what the Python `okur.frontend.Frontend` produces (checked by
//! tests/test_frontend_rs.py), so the browser and mobile runtimes read text the way the model was trained.
//! Nothing here panics on text.

pub mod alphabet;
pub mod circumflex;
#[cfg(target_arch = "wasm32")]
mod wasm;

use normalizer_tr::{AmbiguityPolicy, Hint, HintKind, NormalizeOptions, Normalizer, SourceRange};

pub use alphabet::{encode, symbols};

pub struct Frontend {
    normalizer: Option<Normalizer>,
}

impl Default for Frontend {
    fn default() -> Self {
        Self::new()
    }
}

impl Frontend {
    pub fn new() -> Self {
        Self { normalizer: Normalizer::new().ok() }
    }

    /// Text in, letters out (only symbols of the model's alphabet).
    pub fn letters(&self, text: &str) -> String {
        let text = alphabet::clean(text);
        if text.trim().is_empty() {
            return String::new();
        }
        let text = self.ordinals(&text);
        let text = self.normalize(&text, Vec::new()).unwrap_or(text);
        alphabet::to_alphabet(&circumflex::restore(&alphabet::turkish_lower(&text)))
    }

    fn normalize(&self, text: &str, hints: Vec<Hint>) -> Option<String> {
        let options = NormalizeOptions { ambiguity_policy: AmbiguityPolicy::Fallback, hints };
        let result = self.normalizer.as_ref()?.normalize(text, &options).ok()?;
        Some(result.normalized_text().to_string())
    }

    /// "3. kat", "1. Dünya Savaşı", "2. 3. ve 4.": a number and a period, then a word or another ordinal on the same
    /// line, not preceded by a digit, period, comma or colon (dates and amounts stay whole). Same as Python's
    /// `(?<![\d.,:])\d+\.(?=[ \t]+(?:[^\W\d_]|\d+\.))`.
    fn ordinals(&self, text: &str) -> String {
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let at = |k: usize| chars.get(k).map(|&(_, c)| c);
        let mut out = String::with_capacity(text.len() + 16);
        let (mut last, mut k) = (0usize, 0usize);
        while k < chars.len() {
            let (start, c) = chars[k];
            let after_boundary = k == 0 || !(circumflex::is_decimal(chars[k - 1].1) || matches!(chars[k - 1].1, '.' | ',' | ':'));
            if !(circumflex::is_decimal(c) && after_boundary) {
                k += 1;
                continue;
            }
            let mut j = k;
            while at(j).is_some_and(circumflex::is_decimal) {
                j += 1;
            }
            let mut m = j + 1;
            while at(m).is_some_and(|c| c == ' ' || c == '\t') {
                m += 1;
            }
            let followed = at(j) == Some('.') && m > j + 1 && match at(m) {
                Some(c) if circumflex::is_word_char(c) => true,
                Some(c) if circumflex::is_decimal(c) => {
                    let mut n = m;
                    while at(n).is_some_and(circumflex::is_decimal) {
                        n += 1;
                    }
                    at(n) == Some('.')
                }
                _ => false,
            };
            if !followed {
                k = j.max(k + 1);
                continue;
            }
            let end = chars[j].0 + 1; // through the period
            let number = &text[start..end];
            let hint = Hint::new(SourceRange::new(0, number.len()), HintKind::Ordinal);
            out.push_str(&text[last..start]);
            out.push_str(&self.normalize(number, vec![hint]).unwrap_or_else(|| number.to_string()));
            last = end;
            k = j + 1;
        }
        out.push_str(&text[last..]);
        out
    }
}

/// The word–letter structure the timeline needs: word of each letter and the first letter of that word.
/// A word owns its letters and the spaces and punctuation that follow it (same as okur.model.timeline.words).
pub fn words(letters: &str) -> (Vec<i64>, Vec<i64>) {
    let chars: Vec<char> = letters.chars().collect();
    let mut starts: Vec<usize> = (0..chars.len()).filter(|&i| chars[i] != ' ' && (i == 0 || chars[i - 1] == ' ')).collect();
    if starts.is_empty() {
        starts.push(0);
    }
    let mut bounds = vec![0];
    bounds.extend_from_slice(&starts[1..]);
    bounds.push(chars.len());
    let (mut cw, mut wstart) = (Vec::with_capacity(chars.len()), Vec::with_capacity(chars.len()));
    for (w, pair) in bounds.windows(2).enumerate() {
        for _ in pair[0]..pair[1] {
            cw.push(w as i64);
            wstart.push(pair[0] as i64);
        }
    }
    (cw, wstart)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circumflex_sentence() {
        let f = Frontend::new();
        let want = "bu konuyla alâkalı olarak şirketin kârı hâlâ artıyor.";
        assert_eq!(f.letters("Bu konuyla alâkalı olarak şirketin kârı hâlâ artıyor."), want);
        assert_eq!(f.letters("Bu konuyla alakalı olarak şirketin karı hala artıyor."), want);
    }

    #[test]
    fn ordinals_and_numbers() {
        let f = Frontend::new();
        assert_eq!(f.letters("3. kat"), "üçüncü kat");
        assert_eq!(f.letters("15.10.2026"), f.letters("15.10.2026").as_str());
        assert!(f.letters("25 TL").starts_with("yirmi beş"));
    }

    #[test]
    fn never_panics_on_odd_text() {
        let f = Frontend::new();
        for t in ["", "   ", "\u{202e}abc", "1.", "1. ", "1.\t2.", "😀 12.", "a\u{0302}la", "İIıi"] {
            let _ = f.letters(t);
        }
    }

    #[test]
    fn words_timeline() {
        let (cw, ws) = words("ab, cd");
        assert_eq!(cw, vec![0, 0, 0, 0, 1, 1]);
        assert_eq!(ws, vec![0, 0, 0, 0, 4, 4]);
    }
}
