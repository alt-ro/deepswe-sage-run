#!/usr/bin/env python3
"""Write the router + attention-q abliteration deltas directly into a copy
of gbuzhf's Sage-A1.5-15G-ICEv2.gguf.

Why: llama.cpp (b12250) applies LoRA adapters to the router (ffn_gate_inp)
and attn_q tensors of this model incorrectly — the same deltas verified
numerically exact garble the output when loaded via --lora, but work when
added to the weights directly. The remaining classes (experts, ssm_out,
attn_k/v, attn_output) apply correctly via --lora and ship separately as
adapter-clean.gguf.

Usage:
    pip install gguf numpy
    python3 patch_ice.py Sage-A1.5-15G-ICEv2.gguf
        [--routers adapter-routers.gguf] [--attnq adapter-attnq.gguf]
        [--out Sage-A1.5-15G-ICEv2-abliterated.gguf]

Then run:
    llama-completion -m Sage-A1.5-15G-ICEv2-abliterated.gguf \
        --lora adapter-clean.gguf -p "..." 
"""
import argparse, shutil, sys
import numpy as np
import gguf
from gguf import quants, GGMLQuantizationType


def load_deltas(path, deltas):
    reader = gguf.GGUFReader(path)
    tensors = {t.name: np.array(t.data, dtype=np.float32) for t in reader.tensors}
    for key in sorted(n[:-7] for n in tensors if n.endswith(".lora_a")):
        a = tensors[key + ".lora_a"]
        b = tensors[key + ".lora_b"]
        deltas[key] = (b @ a) / b.shape[1]  # b.shape[1] == rank (llama.cpp convention)
    return deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to Sage-A1.5-15G-ICEv2.gguf (left untouched)")
    ap.add_argument("--routers", default="adapter-routers.gguf")
    ap.add_argument("--attnq", default="adapter-attnq.gguf")
    ap.add_argument("--out", default="Sage-A1.5-15G-ICEv2-abliterated.gguf")
    args = ap.parse_args()

    deltas = {}
    load_deltas(args.routers, deltas)
    load_deltas(args.attnq, deltas)
    print(f"loaded {len(deltas)} delta tensors")

    shutil.copyfile(args.model, args.out)
    reader = gguf.GGUFReader(args.out)
    by_name = {t.name: t for t in reader.tensors}

    done, skipped = 0, 0
    with open(args.out, "r+b") as f:
        for name, delta in deltas.items():
            t = by_name.get(name)
            if t is None:
                print(f"skip (not in model): {name}")
                skipped += 1
                continue
            if int(t.tensor_type) == 0:  # F32 (routers)
                base = np.array(t.data, dtype=np.float32)
                f.seek(t.data_offset)
                f.write((base + delta).astype(np.float32).tobytes())
            elif int(t.tensor_type) == 8:  # Q8_0 (attn_q)
                base = quants.dequantize(np.array(t.data), t.tensor_type).astype(np.float32)
                base = base.reshape(delta.shape)
                f.seek(t.data_offset)
                f.write(quants.quantize((base + delta).astype(np.float32),
                                        GGMLQuantizationType.Q8_0).tobytes())
            else:
                sys.exit(f"unexpected tensor type {t.tensor_type} for {name}")
            done += 1
    print(f"patched {done} tensors, skipped {skipped} -> {args.out}")


if __name__ == "__main__":
    main()
