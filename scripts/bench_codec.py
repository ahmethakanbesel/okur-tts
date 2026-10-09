"""How fast can the ACE-Step encoder go on this GPU, and what does each speed-up cost in accuracy?

Variants: fp32, TF32, bf16 autocast; cuDNN autotuning; torch.compile; batch of N equal-length clips.
Accuracy: mean and max absolute latent difference vs fp32, relative to the latents' std, and the VAE's own posterior
std (the noise level the latents already carry).
"""

import sys
import time

import torch

from okur.data.codec import HOP, Codec


def main(seconds: float = 4.0, batches: tuple[int, ...] = (8, 32, 64)) -> None:
    dev = "cuda"
    codec = Codec(dev)
    vae = codec.vae
    torch.manual_seed(0)
    frames = int(seconds * 25)
    x = (0.1 * torch.randn(max(batches), 2, frames * HOP)).to(dev)  # speech-level noise; speed is input-independent

    def enc(xb: torch.Tensor) -> torch.Tensor:
        return vae.encode(xb).latent_dist.mean

    with torch.inference_mode():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        ref = enc(x[:8]).float()
        post_std = vae.encode(x[:8]).latent_dist.std.float().mean().item()
    print(f"latent std {ref.std().item():.3f}, VAE posterior std (noise already in the latents) {post_std:.3f}")

    def run(name: str, *, tf32: bool, bf16: bool, bench: bool, compile_: bool) -> None:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = bench
        fn = torch.compile(enc) if compile_ else enc
        with torch.inference_mode():
            for b in batches:
                xb = x[:b]
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                        for _ in range(2):
                            fn(xb)
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        for _ in range(3):
                            z = fn(xb)
                        torch.cuda.synchronize()
                    el = (time.perf_counter() - t0) / 3
                    d = (z[:8].float() - ref).abs()
                    print(f"{name:26s} batch {b:3d}: {b * seconds / el:7.0f}x realtime | diff mean {d.mean():.4f} "
                          f"max {d.max():.4f} | peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
                except torch.OutOfMemoryError:
                    print(f"{name:26s} batch {b:3d}: out of memory", flush=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

    run("fp32", tf32=False, bf16=False, bench=False, compile_=False)
    run("tf32", tf32=True, bf16=False, bench=False, compile_=False)
    run("tf32 + cudnn.benchmark", tf32=True, bf16=False, bench=True, compile_=False)
    run("bf16 + cudnn.benchmark", tf32=True, bf16=True, bench=True, compile_=False)
    run("bf16 + benchmark + compile", tf32=True, bf16=True, bench=True, compile_=True)


if __name__ == "__main__":
    main(*(float(a) for a in sys.argv[1:2]))
