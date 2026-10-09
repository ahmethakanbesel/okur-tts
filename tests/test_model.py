import pytest
import torch

from okur.frontend import encode
from okur.model.acoustic import Acoustic
from okur.model.config import AcousticConfig, DecoderConfig
from okur.model.decoder import Decoder
from okur.model.timeline import frame_timeline, word_frames, words


def _item(letters: str, frames_per_letter: float = 2.0):
    w = words(letters)
    dur = torch.full((len(letters),), frames_per_letter)
    fw, fp = frame_timeline(word_frames(dur, w.cw, w.n_words))
    return torch.tensor(encode(letters)), w, dur, fw, fp


def _batch(items: list[tuple]):
    n_letters = max(len(i[0]) for i in items)
    n_frames = max(len(i[3]) for i in items)

    def pad(rows: list[torch.Tensor], size: int, fill: float) -> torch.Tensor:
        out = torch.full((len(rows), size), fill, dtype=rows[0].dtype)
        for j, r in enumerate(rows):
            out[j, :len(r)] = r
        return out

    ids = pad([i[0] for i in items], n_letters, 0)
    return dict(ids=ids, mask=ids != 0, cw=pad([i[1].cw for i in items], n_letters, -1),
                wstart=pad([i[1].wstart for i in items], n_letters, 0), dur=pad([i[2] for i in items], n_letters, 0.0),
                fw=pad([i[3] for i in items], n_frames, -1), fp=pad([i[4] for i in items], n_frames, 0.0),
                fmask=pad([torch.ones(len(i[3]), dtype=torch.bool) for i in items], n_frames, False))


@pytest.fixture(scope="module")
def model() -> Acoustic:
    torch.manual_seed(0)
    m = Acoustic(AcousticConfig(n_speakers=3)).eval()
    for p in m.parameters():  # leave adaLN-zero init so outputs are not trivially zero
        p.data.add_(torch.randn_like(p) * 0.02)
    return m


def test_size(model: Acoustic) -> None:
    n = sum(p.numel() for p in model.parameters()) / 1e6
    assert 5.5 < n < 7.5, n
    assert 2.9 < sum(p.numel() for p in Decoder(DecoderConfig()).parameters()) / 1e6 < 3.1


def test_padding_does_not_change_outputs(model: Acoustic) -> None:
    """A sentence alone and the same sentence padded in a batch with a longer one give the same result."""
    short, long = _item("merhaba dünya."), _item("bu konuyla alâkalı olarak şirketin kârı hâlâ artıyor.")
    alone, batch = _batch([short]), _batch([short, long])
    spk = torch.tensor([1, 2])
    q = torch.zeros(2, dtype=torch.long)

    def run(b: dict, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        text = model.text_stage(b["ids"], b["mask"], spk[:n], q[:n])
        cond = model.condition(text.h, b["dur"], b["mask"], b["cw"], b["wstart"], b["fw"], b["fp"])
        x = torch.randn(1, b["fw"].shape[1], 64, generator=torch.Generator().manual_seed(0)).expand(n, -1, -1)
        v = model.velocity(x.contiguous(), cond, torch.full((n,), 0.3), text.g, b["fmask"])
        return text.log_dur, v

    d1, v1 = run(alone, 1)
    d2, v2 = run(batch, 2)
    letters, frames = alone["ids"].shape[1], alone["fw"].shape[1]
    torch.testing.assert_close(d1[0], d2[0, :letters], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(v1[0], v2[0, :frames], atol=1e-5, rtol=1e-4)


def test_loss_backward(model: Acoustic) -> None:
    b = _batch([_item("merhaba dünya."), _item("kâr arttı.")])
    model.train()
    latents = torch.randn(2, b["fw"].shape[1], 64)
    losses = model.loss(**b, latents=latents, speaker=torch.tensor([0, 1]), quality=torch.tensor([0, 0]))
    losses.total.backward()
    assert torch.isfinite(losses.total)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()


def test_word_frames_never_drift() -> None:
    w = words("a bb ccc dddd")
    dur = torch.tensor([1.3] * 13)
    counts = word_frames(dur, w.cw, w.n_words)
    assert counts.sum().item() == round(1.3 * 13)


@pytest.mark.slow
def test_matches_ema_lightning() -> None:
    """Load EMA Lightning's released weights into our implementation and check it computes exactly what EMA does."""
    from ema_lightning import EMA

    ema = EMA(device="cpu")
    ref = ema._engine.model
    ours = Acoustic(AcousticConfig(text_attn=0)).eval()
    sd = {k: v for k, v in ref.state_dict().items() if not k.startswith(("chardur.", "text.emb."))}
    sd = {k.replace("chardur.", "duration."): v for k, v in sd.items()}
    net = ref.chardur.net
    sd |= {"duration.c1.weight": net[0].weight, "duration.c1.bias": net[0].bias, "duration.n1.weight": net[2].weight,
           "duration.n1.bias": net[2].bias, "duration.c2.weight": net[4].weight, "duration.c2.bias": net[4].bias,
           "duration.n2.weight": net[6].weight, "duration.n2.bias": net[6].bias,
           "duration.out.weight": ref.chardur.out.weight, "duration.out.bias": ref.chardur.out.bias}
    emb = torch.zeros_like(ours.text.emb.weight)
    for i, sym in enumerate(ref.vocab):  # same letters, different order: copy by symbol
        if sym in ("<pad>", "<unk>") or sym in "abcdefghijklmnopqrstuvwxyzçğıöşü !\"%&'(),-./:;?":
            from okur.frontend.alphabet import STOI
            emb[STOI[sym]] = ref.text.emb.weight[i]
    sd["text.emb.weight"] = emb
    missing, unexpected = ours.load_state_dict(sd, strict=False)
    assert set(missing) == {"speaker.weight", "quality.weight", "latent_mean", "latent_std"}, missing
    assert not unexpected
    torch.nn.init.zeros_(ours.speaker.weight)
    torch.nn.init.zeros_(ours.quality.weight)

    letters = "bu konuyla alakalı olarak şirketin karı hala artıyor."
    ids_ref = torch.tensor([[ref.stoi[c] for c in letters]])
    ids_ours = torch.tensor([encode(letters)])
    mask = torch.ones_like(ids_ref, dtype=torch.bool)
    with torch.no_grad():
        h_ref, dur_ref = ref.text_stage(ids_ref, mask)
        text = ours.text_stage(ids_ours, mask, torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))
        torch.testing.assert_close(text.h, h_ref)
        torch.testing.assert_close(ours.frames_from_log(text.log_dur, mask), dur_ref)

        w = words(letters)
        fw, fp = frame_timeline(word_frames(dur_ref[0], w.cw, w.n_words))
        cw, wstart, fw, fp = w.cw[None], w.wstart[None], fw[None], fp[None]
        noise = torch.randn(1, len(ref.times), fw.shape[1], 64, generator=torch.Generator().manual_seed(0))
        z_ref = ref.sound_stage(h_ref, dur_ref, mask, cw, wstart, fw, fp, fw >= 0, noise)

        cond = ours.condition(text.h, dur_ref, mask, cw, wstart, fw, fp)
        x = noise[:, 0]
        for k, t in enumerate(ref.times):  # EMA's 4-step student schedule, through our network
            v = ours.velocity(x, cond, torch.full((1,), t), text.g, fw >= 0)
            x1 = x + (1 - t) * v
            if k + 1 < len(ref.times):
                x = (1 - ref.times[k + 1]) * noise[:, k + 1] + ref.times[k + 1] * x1
        torch.testing.assert_close(x1, z_ref, atol=1e-4, rtol=1e-4)
