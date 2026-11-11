# Abliterated Sage-A1.5 on Sangreal ICE 2.0 — working kit

The abliteration (stock vs Ablitrex BF16 weight delta, 99% energy trim,
extracted by streaming — see repo history) split by how llama.cpp handles
each tensor class:

- **Routers (ffn_gate_inp) and attention q_proj**: llama.cpp b12250 applies
  these LoRA deltas wrongly (output garbles even though the adapter numbers
  verify exact against Hugging Face weights). Written **directly into a
  copy of the model file** by `patch_ice.py` instead — verified coherent.
- **Everything else** (experts, ssm_out, attn_k, attn_v, attn_output):
  applies correctly as a normal LoRA — `adapter-clean.gguf`.

## Files you need

1. Base model (gbuzhf's genuine ICE 2.0 build, untouched):
   `Sage-A1.5-15G-ICEv2.gguf` from
   https://huggingface.co/gbuzhf/Sage-A1.5-Sangreal-ICE
   sha256 `b67e38a41fa2b1b473364b3c75b1aba45b17c74524203d4fe6caddcb7f37e292`
2. Adapters — stored in this repo under `bisect/` as 16 MB chunks:
   ```
   cat bisect/clean-a?   > adapter-clean.gguf    # sha256 89ae6cc820c46e86af0c2c056c74ad1f506d2af439fd3891f5a7b4b6c3aef35d
   cat bisect/attnq-a?   > adapter-attnq.gguf    # sha256 234d5d3f82041e92571fdcacf7940a68c9d0b0bba1172188eec442eb54487ed0
   cp  bisect/routers-aa   adapter-routers.gguf  # sha256 dac8b79448c38bd10db8e9857e1eca07656e1d3370a0ca0c344cffae28f1d105
   ```
3. `patch_ice.py` from this folder.

## Build and run

```
pip install gguf numpy
python3 patch_ice.py Sage-A1.5-15G-ICEv2.gguf
llama-completion -m Sage-A1.5-15G-ICEv2-abliterated.gguf \
    --lora adapter-clean.gguf -p "The capital of France is" -n 64
```

Tested with llama.cpp b12250 on GitHub Actions: the patched model is
coherent with and without the LoRA attached; the LoRA carries the
remaining abliteration classes.
