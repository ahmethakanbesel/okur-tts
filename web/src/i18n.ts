/** Page text in Turkish (default) and English. Keys are used by data-i18n attributes in index.html and by main.ts. */

export type Lang = "tr" | "en";

const tr = {
  "meta.title": "Okur · Türkçeyi yazıldığı gibi okur",
  "meta.description": "Tarayıcında çalışan, açık kaynaklı Türkçe seslendirme modeli. Sayıları, sıra sayılarını ve şapkalı harfleri doğru okur.",
  "nav.try": "Dene",
  "nav.hard": "Örnek cümleler",
  "nav.numbers": "Ölçümler",
  "nav.how": "Nasıl çalışır",
  "nav.code": "Kod ve model",
  "lang.switch": "Dili değiştir: English",
  "hero.title": "Okur, Türkçeyi yazıldığı gibi okur.",
  "input.label": "Senin yazdığın",
  "input.default": "Bu konuyla alakalı olarak şirketin karı hala artıyor; yeni ofis 3. katta açılacak.",
  "examples.label": "Örnekler:",
  "reading.label": "Okur’un okuduğu",
  "reading.empty": "OKU’ya bastığında, okunan metin sesle birlikte buraya yazılır.",
  "reading.legend": "Kırmızıyla işaretli yerleri Okur yazıldığından farklı okur: şapkalı harfler, sayılar, kısaltmalar.",
  "key.read": "Oku",
  "key.stop": "Dur",
  "key.slow": "Yavaş",
  "key.normal": "Normal",
  "key.fast": "Hızlı",
  "key.take": "Bir daha oku",
  "key.take.title": "Aynı metni başka bir tonlamayla oku",
  "key.download": "WAV indir",
  "speed.label": "Hız",
  "adv.title": "Gelişmiş ayarlar",
  "adv.speed": "Hız",
  "adv.pause": "Duraklama uzunluğu",
  "adv.pause.hint": "Yalnız boşluk ve noktalama; konuşma hızı aynı kalır.",
  "adv.gap": "Cümle arası sessizlik",
  "adv.gap.hint": "Uzun metinlerde cümlelerin arasına eklenir.",
  "adv.temp": "İfade gücü",
  "adv.temp.hint": "Gürültü sıcaklığı. Düşük değer daha düz ama daha kararlı; model 1,0 ile eğitildi.",
  "adv.seed": "Tohum",
  "adv.seed.hint": "Aynı tohum ve ayarlar aynı okumayı verir; paylaşılabilir.",
  "adv.format": "Dışa aktarma",
  "adv.dict": "Söyleniş sözlüğü",
  "adv.dict.hint": "Her satıra bir kural: yazılış = okunuş. Ön işleyiciden önce uygulanır; adlar, markalar ve kısaltmalar için.",
  "adv.dict.placeholder": "NATO = nato\nMcDonald’s = mekdonılds",
  "adv.reset": "Varsayılanlar",
  "adv.seconds": "{v} sn",
  "key.download.wav": "WAV indir",
  "key.download.mp3": "MP3 indir",
  "facts.params": "parametre",
  "facts.steps": "adım",
  "facts.model": "model",
  "facts.open": "açık kaynak",
  "status.loading": "Model indiriliyor… {a} / {b} MB",
  "status.first": "Model indiriliyor (24 MB, yalnızca ilk ziyarette)…",
  "status.preparing": "Model hazırlanıyor…",
  "status.ready": "Hazır · WebAssembly, {n} iş parçacığı · {s} sn’de yüklendi",
  "status.reading": "Okunuyor…",
  "status.result": "{a} sn’lik ses {b} sn’de üretildi · {r}× gerçek zamanlı",
  "status.part": " · cümle {i}/{n}",
  "status.error": "Bir sorun oldu: {m}",
  "status.worker": "Ses motoru başlatılamadı: {m}",
  "hard.title": "Örnek cümleler",
  "hard.lede": "Aynı cümleleri beş açık modelden, ses seviyeleri eşitlenmiş olarak dinle. Özellikle şapkalı harflere, sıra sayılarına, para tutarlarına ve tarihlere kulak ver. CANLI OKU, cümleyi yukarıdaki kutuya taşır ve bu tarayıcıda okutur.",
  "hard.live": "Canlı oku",
  "hard.download": "MP3 indir",
  "hard.more": "16 cümlenin tamamını göster",
  "hard.less": "Daha az göster",
  "hard.loading": "Örnekler yükleniyor…",
  "numbers.title": "Ölçümler",
  "numbers.lede": "Freya-TR-Eval’in 495 Türkçe cümlesini her sistem birer kez okudu. Anlaşılırlık: Whisper-large-v3’ün çıkardığı metindeki kelime ve harf hata oranı. Doğallık: UTMOS, yani 1–5 arası tahmini dinleyici puanı. “Temiz” sütunları, Common Voice’ta da geçen cümleleri hesaba katmaz. Hız: aynı sunucu işlemcisinin aynı 8 çekirdeğinde, bir saniyede kaç saniyelik ses üretildiği.",
  "numbers.system": "Sistem",
  "numbers.size": "Boyut",
  "numbers.wer": "WER",
  "numbers.werClean": "Temiz WER",
  "numbers.cerClean": "Temiz CER",
  "numbers.speed": "Hız",
  "numbers.license": "Lisans",
  "numbers.ours": "bu model",
  "numbers.reportedHead": "Yazarlarının yayımladığı değerler · farklı protokol, burada ölçülmedi",
  "numbers.reported": "FreyaTTS raporundan (Temmuz 2026)",
  "numbers.reportedMos": "MOS 3,68 †",
  "numbers.reportedNote": "† FreyaTTS-small değerleri, yazarlarının teknik raporundan alındı; burada ölçülmedi. Raporda WER, ses 8 kHz’e indirildikten sonra Whisper’la hesaplanıyor (aynı Piper sesi o protokolde %4,4, bizim ölçümümüzde %3,14 alıyor). 3,68 bir UTMOS değeri değil, 7 kişinin verdiği insan dinleyici puanı (MOS). Hız işlemcide değil, RTX 4090 GPU’da ölçülmüş (gerçek zaman çarpanı ≈ 0,11); raporun dizüstü bilgisayar işlemcisi için verdiği değer ≈ 1,4×.",
  "numbers.honest": "Okur, bu dört sistem içinde doğallıkta sonuncu, anlaşılırlıkta da EMA Lightning’in gerisinde. Öte yandan tarayıcıda çalışan, şapkaları geri koyan ve sıra sayılarını doğru okuyan tek sistem. İşlemcide EMA Lightning’den 1,3 kat hızlı (PyTorch ile 28×, EMA 21×). Apple M4’te bu tarayıcıda gerçek zamanın yaklaşık 24 katı, aynı Mac’in GPU’sunda MLX ile 67 katı hızla çalışıyor. MMS’in metin normalleştiricisi olmadığı için rakamları atlıyor. Piper’ın dfki sesi ve MMS ticari kullanıma kapalı lisanslarla dağıtılıyor.",
  "how.title": "Nasıl çalışır",
  "how.lede": "Okur sıfırdan eğitildi ve cümlenin tamamını tek seferde üretiyor (otoregresif değil). Genel tasarımı, <a href=\"https://huggingface.co/canberkkkkkk/ema-lightning\">EMA Lightning</a>’in yayımlanmış mimarisine dayanıyor. Yapı taşları şu çalışmalardan geliyor: akış eşleme (<a href=\"https://arxiv.org/abs/2210.02747\">Lipman ve ark., 2023</a>), Diffusion Transformer (<a href=\"https://arxiv.org/abs/2212.09748\">Peebles ve Xie, 2023</a>), ConvNeXt (<a href=\"https://arxiv.org/abs/2201.03545\">Liu ve ark., 2022</a>), RoPE (<a href=\"https://arxiv.org/abs/2104.09864\">Su ve ark., 2021</a>), süreye dayalı Gauss hizalama (<a href=\"https://arxiv.org/abs/2010.04301\">Non-Attentive Tacotron, Shen ve ark., 2020</a>), DMD2 damıtma (<a href=\"https://arxiv.org/abs/2405.14867\">Yin ve ark., 2024</a>) ve HiFi-GAN kod çözücü (<a href=\"https://arxiv.org/abs/2010.05646\">Kong ve ark., 2020</a>). Her aşama, telefonda çalışabilecek kadar küçük.",
  "how.fig1": "<b>Şekil 1.</b> Çıkarım hattı. Metin önce Rust ile yazılıp WebAssembly’ye derlenmiş ön işleyiciden geçer: sayılar, tarihler ve para tutarları <a href=\"https://github.com/erdemtuna/normalizer-tr\">normalizer-tr</a> ile yazıya dökülür, “3.” “üçüncü” diye okunur, günlük yazıda düşen şapkalar geri eklenir. Kodlayıcı her harf için bir vektör ve bir süre (40 ms’lik kare sayısı) üretir; kelime zaman çizelgesi ve ±1 kelimelik Gauss pencereli hizalayıcı bunları her kare için bir koşula dönüştürür. Bu sayede hiçbir kelime atlanamaz ya da tekrarlanamaz. DiT üretici, gürültüden 4 adımda 25 Hz’lik gizli temsiller üretir; kod çözücü bunları 48 kHz’lik sese çevirir. <i>L</i>: harf sayısı, <i>T</i>: kare sayısı.",
  "how.fig2": "<b>Şekil 2.</b> Solda DiT bloğu: dikkat ve SwiGLU alt katmanları, zaman adımından ve konuşmacıdan gelen ortak bir adaLN modülasyonuyla ölçeklenir ve kapılanır. Sağda 4 adımlı örnekleme: her adımda temiz gizli temsil <i>x̂</i>₁ tahmin edilir, ardından bir sonraki zaman adımı için yeniden gürültü eklenir. Öğrenci model, 16 adımlı bir öğretmen modelden (yönlendirme ölçeği 3) DMD2 ile damıtıldı.",
  "footer.normalizer": "Metin normalleştirme: <a href=\"https://github.com/erdemtuna/normalizer-tr\">normalizer-tr</a> (Erdem Tuna, Apache-2.0). Sayıları, tarihleri ve para tutarlarını Türkçe yazıya döken bu Rust kütüphanesi hem Okur’un eğitiminde hem de bu sayfada kullanılıyor. Emeği için teşekkürler.",
  "code.title": "Kod ve model",
  "code.lede": "Veri, eğitim kodu, değerlendirme ve ağırlıkların hepsi açık. Model PyTorch, Apple MLX ve ONNX Runtime ile, bu sayfada da WebAssembly ile çalışıyor.",
  "code.install": "Kur ve konuştur",
  "code.data": "Veri",
  "code.data.d": "298 saatlik kayıt, 44 konuşmacı; hepsi açık lisanslı: Mozilla Common Voice (CC0), ISSAI Türkçe Konuşma Derlemi (MIT), MediaSpeech ve Google FLEURS (CC BY 4.0). Duyduğun ses, anonim bir Common Voice gönüllüsüne ait; model bu gönüllünün yaklaşık 5 saatlik kaydıyla ince ayarlandı. Eğitimin tamamı tek bir RTX 5090’da yaklaşık 21 GPU saati sürdü ve 12 dolar civarına mal oldu.",
  "code.license": "Lisans",
  "code.license.d": "Kod Apache-2.0, ağırlıklar CC BY 4.0 lisanslı.",
  "code.limits": "Sınırlar",
  "code.limits.d": "Tek ses var. Doğallıkta, stüdyo kayıtlarıyla eğitilmiş modellerin gerisinde. Şapkalar kurallarla geri ekleniyor: cümlede ticari bir bağlam yoksa “kar” olduğu gibi kalır. Yabancı adlar ve kısaltmalar yazıldığı gibi, Türkçe harf sesleriyle okunur. Lütfen kimsenin sesini taklit etmek için kullanma.",
  "code.cite": "Atıf",
  "code.links": "Bağlantılar",
  "footer.credits": "Karşılaştırma örnekleri: EMA Lightning (Apache-2.0), FreyaTTS-small (Apache-2.0; kendi hattı ve ses tohumuyla bu bilgisayarda üretildi), Piper dfki sesi (CC BY-NC-SA 4.0), Meta MMS-TTS (CC BY-NC 4.0). Bu örnekler, lisanslarına uygun şekilde, araştırma amaçlı karşılaştırma için kullanılıyor. Yazı tipleri: Courier Prime ve Archivo (SIL OFL). Sayfa ONNX Runtime Web ile çalışıyor.",
} as const;

export type Key = keyof typeof tr;

const en: Record<Key, string> = {
  "meta.title": "Okur · Turkish read the way it is written",
  "meta.description": "An open-source Turkish text-to-speech model that runs in your browser. It reads numbers, ordinals and circumflexed letters correctly.",
  "nav.try": "Try",
  "nav.hard": "Example sentences",
  "nav.numbers": "Numbers",
  "nav.how": "How it works",
  "nav.code": "Code & model",
  "lang.switch": "Switch language: Türkçe",
  "hero.title": "Okur reads Turkish the way it is written.",
  "input.label": "What you typed",
  "input.default": "Bu konuyla alakalı olarak şirketin karı hala artıyor; yeni ofis 3. katta açılacak.",
  "examples.label": "Examples:",
  "reading.label": "What Okur reads",
  "reading.empty": "Press READ and the text is typed here as it is spoken.",
  "reading.legend": "Okur reads the places marked in red differently from how they are written: circumflexed letters, numbers, abbreviations.",
  "key.read": "Read",
  "key.stop": "Stop",
  "key.slow": "Slow",
  "key.normal": "Normal",
  "key.fast": "Fast",
  "key.take": "Read again",
  "key.take.title": "Read the same text with a different intonation",
  "key.download": "Download WAV",
  "speed.label": "Speed",
  "adv.title": "Advanced options",
  "adv.speed": "Speed",
  "adv.pause": "Pause length",
  "adv.pause.hint": "Spaces and punctuation only; the speaking rate stays the same.",
  "adv.gap": "Silence between sentences",
  "adv.gap.hint": "Added between sentences of longer texts.",
  "adv.temp": "Expressiveness",
  "adv.temp.hint": "Noise temperature. Lower values sound flatter but steadier; the model was trained at 1.0.",
  "adv.seed": "Seed",
  "adv.seed.hint": "The same seed and settings give the same take; shareable.",
  "adv.format": "Export",
  "adv.dict": "Pronunciation dictionary",
  "adv.dict.hint": "One rule per line: written = spoken. Applied before the frontend; for names, brands and abbreviations.",
  "adv.dict.placeholder": "NATO = nato\nMcDonald’s = mekdonılds",
  "adv.reset": "Defaults",
  "adv.seconds": "{v} s",
  "key.download.wav": "Download WAV",
  "key.download.mp3": "Download MP3",
  "facts.params": "parameters",
  "facts.steps": "steps",
  "facts.model": "model",
  "facts.open": "open source",
  "status.loading": "Downloading the model… {a} / {b} MB",
  "status.first": "Downloading the model (24 MB, only on the first visit)…",
  "status.preparing": "Preparing the model…",
  "status.ready": "Ready · WebAssembly, {n} threads · loaded in {s} s",
  "status.reading": "Reading…",
  "status.result": "{a} s of audio generated in {b} s · {r}× real time",
  "status.part": " · sentence {i}/{n}",
  "status.error": "Something went wrong: {m}",
  "status.worker": "The speech engine failed to start: {m}",
  "hard.title": "Example sentences",
  "hard.lede": "Listen to the same sentences from five open models, loudness-matched. Pay particular attention to circumflexed letters, ordinals, amounts of money and dates. READ LIVE moves the sentence into the box above and reads it in this browser.",
  "hard.live": "Read live",
  "hard.download": "Download MP3",
  "hard.more": "Show all 16 sentences",
  "hard.less": "Show fewer",
  "hard.loading": "Loading samples…",
  "numbers.title": "Numbers",
  "numbers.lede": "Each system read the 495 Turkish sentences of Freya-TR-Eval once. Intelligibility: the word and character error rate of the text Whisper-large-v3 transcribes. Naturalness: UTMOS, a predicted listener score from 1 to 5. The “clean” columns leave out sentences that also appear in Common Voice. Speed: how many seconds of audio are produced per second on the same 8 cores of one server CPU.",
  "numbers.system": "System",
  "numbers.size": "Size",
  "numbers.wer": "WER",
  "numbers.werClean": "Clean WER",
  "numbers.cerClean": "Clean CER",
  "numbers.speed": "Speed",
  "numbers.license": "License",
  "numbers.ours": "this model",
  "numbers.reportedHead": "Published by its authors · different protocol, not measured here",
  "numbers.reported": "from the FreyaTTS report (July 2026)",
  "numbers.reportedMos": "MOS 3.68 †",
  "numbers.reportedNote": "† The FreyaTTS-small figures are taken from its authors’ technical report; they were not measured here. In the report, WER is computed with Whisper after the audio is downsampled to 8 kHz (the same Piper voice scores 4.4% under that protocol and 3.14% in our measurement). 3.68 is not a UTMOS value but a human listener score (MOS) from 7 raters. Speed was measured on an RTX 4090 GPU, not on a CPU (real-time factor ≈ 0.11); for a laptop CPU the report gives ≈ 1.4×.",
  "numbers.honest": "Among these four systems, Okur ranks last in naturalness and trails EMA Lightning in intelligibility. On the other hand, it is the only one that runs in a browser, restores circumflexes and reads ordinals correctly. On CPU it is 1.3× faster than EMA Lightning (28× with PyTorch, EMA 21×). On an Apple M4 it runs about 24× real time in this browser, and 67× with MLX on the same Mac’s GPU. MMS has no text normalizer, so it skips digits. Piper’s dfki voice and MMS are distributed under licenses that do not allow commercial use.",
  "how.title": "How it works",
  "how.lede": "Okur was trained from scratch and generates the whole sentence in one pass (it is not autoregressive). Its overall design is based on the published architecture of <a href=\"https://huggingface.co/canberkkkkkk/ema-lightning\">EMA Lightning</a>. Its building blocks come from these works: flow matching (<a href=\"https://arxiv.org/abs/2210.02747\">Lipman et al., 2023</a>), the Diffusion Transformer (<a href=\"https://arxiv.org/abs/2212.09748\">Peebles & Xie, 2023</a>), ConvNeXt (<a href=\"https://arxiv.org/abs/2201.03545\">Liu et al., 2022</a>), RoPE (<a href=\"https://arxiv.org/abs/2104.09864\">Su et al., 2021</a>), duration-based Gaussian alignment (<a href=\"https://arxiv.org/abs/2010.04301\">Non-Attentive Tacotron, Shen et al., 2020</a>), DMD2 distillation (<a href=\"https://arxiv.org/abs/2405.14867\">Yin et al., 2024</a>) and the HiFi-GAN decoder (<a href=\"https://arxiv.org/abs/2010.05646\">Kong et al., 2020</a>). Every stage is small enough to run on a phone.",
  "how.fig1": "<b>Figure 1.</b> Inference pipeline. Text first goes through a frontend written in Rust and compiled to WebAssembly: numbers, dates and amounts of money are spelled out with <a href=\"https://github.com/erdemtuna/normalizer-tr\">normalizer-tr</a>, “3.” is read as “üçüncü”, and circumflexes dropped in everyday writing are added back. The encoder produces a vector and a duration (a number of 40 ms frames) for each letter; the word timeline and the aligner with a ±1-word Gaussian window turn these into a condition for every frame. This way no word can be skipped or repeated. The DiT generator produces 25 Hz latent representations from noise in 4 steps; the decoder turns them into 48 kHz audio. <i>L</i>: number of letters, <i>T</i>: number of frames.",
  "how.fig2": "<b>Figure 2.</b> Left, the DiT block: the attention and SwiGLU sub-layers are scaled and gated by a shared adaLN modulation from the timestep and the speaker. Right, 4-step sampling: each step predicts the clean latent <i>x̂</i>₁, then adds noise again for the next timestep. The student model was distilled with DMD2 from a 16-step teacher model (guidance scale 3).",
  "footer.normalizer": "Text normalization: <a href=\"https://github.com/erdemtuna/normalizer-tr\">normalizer-tr</a> (Erdem Tuna, Apache-2.0). This Rust library, which spells out Turkish numbers, dates and amounts of money, is used both in Okur’s training and on this page. Thanks for the work.",
  "code.title": "Code & model",
  "code.lede": "Data, training code, evaluation and weights are all open. The model runs with PyTorch, Apple MLX and ONNX Runtime, and on this page with WebAssembly.",
  "code.install": "Install and speak",
  "code.data": "Data",
  "code.data.d": "298 hours of recordings from 44 speakers, all openly licensed: Mozilla Common Voice (CC0), ISSAI Turkish Speech Corpus (MIT), MediaSpeech and Google FLEURS (CC BY 4.0). The voice you hear belongs to an anonymous Common Voice volunteer; the model was fine-tuned on about 5 hours of their recordings. All training took about 21 GPU hours on a single RTX 5090 and cost around $12.",
  "code.license": "License",
  "code.license.d": "Code is licensed Apache-2.0, weights CC BY 4.0.",
  "code.limits": "Limits",
  "code.limits.d": "There is one voice. It is less natural than models trained on studio recordings. Circumflexes are added back by rules: if the sentence has no business context, “kar” stays as written. Foreign names and abbreviations are read as written, with Turkish letter sounds. Please don’t use it to imitate anyone’s voice.",
  "code.cite": "Cite",
  "code.links": "Links",
  "footer.credits": "Comparison samples: EMA Lightning (Apache-2.0), FreyaTTS-small (Apache-2.0; generated on this computer with its own pipeline and voice seed), Piper dfki voice (CC BY-NC-SA 4.0), Meta MMS-TTS (CC BY-NC 4.0). These samples are used for research comparison in line with their licenses. Typefaces: Courier Prime and Archivo (SIL OFL). The page runs on ONNX Runtime Web.",
};

/** Category names of the hard sentences (demo/hard_sentences.json ids). */
export const CATEGORIES: Record<Lang, Record<string, string>> = {
  tr: {
    "circumflex-marked": "Şapkalı yazım", "circumflex-plain": "Şapkasız yazım", "kar-snow-profit": "Eş yazım: kar / kâr",
    "hala-aunt-still": "Eş yazım: hala / hâlâ", ordinals: "Sıra sayıları", "numbers-money": "Sayı, para, yüzde",
    "date-time": "Tarih ve saat", agglutination: "Çok ekli uzun kelime", "tongue-twister": "Tekerleme",
    question: "Soru tonlaması", "proper-names": "Özel adlar ve kesme işareti", "soft-g": "Yumuşak g, ünlü uyumu",
    loanwords: "Alıntı kelimeler", long: "Uzun cümle", dialogue: "Ünlem ve tırnak", "mekan-mekanik": "İstisnalar: mekân / mekanik",
  },
  en: {
    "circumflex-marked": "Circumflex", "circumflex-plain": "Circumflex typed without ^", "kar-snow-profit": "Homograph: kar / kâr",
    "hala-aunt-still": "Homograph: hala / hâlâ", ordinals: "Ordinals", "numbers-money": "Numbers, money, percent",
    "date-time": "Date and time", agglutination: "Long agglutinative word", "tongue-twister": "Tongue twister",
    question: "Question intonation", "proper-names": "Proper names and apostrophes", "soft-g": "Soft g and vowel harmony",
    loanwords: "Loanwords", long: "Long sentence", dialogue: "Exclamation and quotes", "mekan-mekanik": "Exceptions: mekân / mekanik",
  },
};

export const EXAMPLES: { tr: string; en: string; text: string }[] = [
  { tr: "Şapkalı harf", en: "Circumflex", text: "Bu konuyla alakalı olarak şirketin karı hala artıyor." },
  { tr: "Sıra sayısı", en: "Ordinals", text: "Toplantı 3. katta, 2. koridorun sonundaki 15. odada yapılacak." },
  { tr: "Para", en: "Money", text: "Enflasyon yüzde 3,5 artınca fiyat 1.250,75 TL'ye çıktı." },
  { tr: "Tarih ve saat", en: "Date & time", text: "Sınav 14 Mart 2026 Cumartesi günü saat 09:30'da başlayacak." },
  { tr: "Tekerleme", en: "Tongue twister", text: "Şu köşe yaz köşesi, şu köşe kış köşesi, ortada su şişesi." },
  { tr: "Uzun metin", en: "Long text", text: "Bilim insanları, iklim değişikliğinin önümüzdeki on yıllarda tarım, su kaynakları ve kıyı kentleri üzerinde ciddi etkiler yaratacağını vurguluyor. Uzmanlara göre bu etkileri azaltmak için hemen harekete geçmek gerekiyor. Aksi halde 2050 yılına kadar pek çok bölgede su sıkıntısı yaşanabilir." },
];

const dictionaries: Record<Lang, Record<Key, string>> = { tr, en };

export function t(lang: Lang, key: Key, vars: Record<string, string | number> = {}): string {
  return dictionaries[lang][key].replace(/\{(\w+)\}/g, (_, name: string) => String(vars[name] ?? ""));
}

export function initialLang(): Lang {
  const param = new URLSearchParams(location.search).get("lang");
  if (param === "en" || param === "tr") return param;
  try {
    const stored = localStorage.getItem("okur.lang");
    if (stored === "en" || stored === "tr") return stored;
  } catch {
    // storage unavailable (private mode): fall back to the browser's languages
  }
  for (const preferred of navigator.languages ?? [navigator.language]) {
    const code = preferred.toLowerCase().slice(0, 2);
    if (code === "tr" || code === "en") return code;
  }
  return "en";
}

export function rememberLang(lang: Lang): void {
  try {
    localStorage.setItem("okur.lang", lang);
  } catch {
    // not persisted; the toggle still works for this visit
  }
}

/** Apply translations to every [data-i18n] (text) and [data-i18n-attr] ("attr:key;attr:key") element. */
export function apply(lang: Lang, root: ParentNode = document): void {
  document.documentElement.lang = lang;
  document.title = t(lang, "meta.title");
  document.querySelector('meta[name="description"]')?.setAttribute("content", t(lang, "meta.description"));
  for (const el of root.querySelectorAll<HTMLElement>("[data-i18n]")) {
    el.textContent = t(lang, el.dataset.i18n as Key);
  }
  for (const el of root.querySelectorAll<HTMLElement>("[data-i18n-html]")) {
    el.innerHTML = t(lang, el.dataset.i18nHtml as Key); // our own dictionary, never user input
  }
  for (const el of root.querySelectorAll<HTMLElement>("[data-i18n-attr]")) {
    for (const pair of (el.dataset.i18nAttr ?? "").split(";")) {
      const [attr, key] = pair.split(":");
      if (attr && key) el.setAttribute(attr, t(lang, key as Key));
    }
  }
}

export function number(lang: Lang, x: number, digits: number): string {
  return new Intl.NumberFormat(lang === "tr" ? "tr-TR" : "en-US", { minimumFractionDigits: digits,
    maximumFractionDigits: digits }).format(x);
}

export function percent(lang: Lang, x: number, digits = 2): string {
  return new Intl.NumberFormat(lang === "tr" ? "tr-TR" : "en-US", { style: "percent", minimumFractionDigits: digits,
    maximumFractionDigits: digits }).format(x / 100);
}
