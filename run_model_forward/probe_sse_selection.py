"""Is SSE's sparse partition selection balanced and input-dependent on a trained checkpoint?

SSE picks ``num_writer`` of ``num_sparse_partition`` state partitions per token
(``sse_swa.py``: ``topk_value, topk_expert = torch.topk(e, k=K, dim=2)``), and a Switch-style
load-balance auxiliary loss pushes that selection towards uniform usage during training::

    p = torch.mean(eta.float(), dim=(0, 1))
    f = torch.mean(mask.float(), dim=(0, 1))
    aux_loss = torch.sum(p * f) * self.num_sparse_partition / self.num_writer

That loss is applied under ``self.training`` only, so nothing guarantees the property survives
into the released weights. This script measures whether it did. Two failure modes are worth
distinguishing, and the reported numbers separate them:

* **Usage collapse** -- some partitions are chosen far more than others. The expanded state is
  then effectively smaller than ``num_sparse_partition``, so the parameter and memory overhead
  is paid without the capacity it was meant to buy.
* **Input-independence** -- usage is balanced in aggregate, but every token picks the *same*
  set. That is worse than it looks in a usage histogram: the selection carries no information,
  so the mechanism reduces to a fixed sub-state rather than content-dependent routing.

Both are standard top-k routing failure modes; neither is visible in perplexity or downstream
benchmark scores, which is why this is worth measuring separately.

**This is a diagnostic, not a criticism.** Balanced and input-dependent is a good result and the
most likely one — it says the auxiliary loss did its job and the mechanism works as designed.

Method: loads the model, then rebinds the module-level selection helper on whichever module the
loaded model actually came from (``trust_remote_code`` imports the checkpoint's own copy), runs a
few batches of real text under ``no_grad``, and accumulates per-layer statistics. The model's
weights and structure are untouched, and the wrapper forwards to the original implementation.

The statistics are gathered at several context lengths in one pass, because balance that holds at
short context but decays as the context grows is the case that would actually matter — and because
loading the checkpoint costs far more than the forward passes do.

Run::

    python run_model_forward/probe_sse_selection.py --model-path /path/to/SpikingBrain-2.0-base-8k
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

def _find_repo_root() -> Path:
    """Walk up to the directory holding ``spb2/``, so this file runs from anywhere in the tree."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "spb2" / "sse_swa.py").is_file():
            return parent
    raise RuntimeError(f"could not locate the repository root (spb2/sse_swa.py) above {here}")


_REPO = _find_repo_root()
for _p in (str(_REPO), str(_REPO / "spb2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Accumulated by the patched selector: layer index -> stats.
_CAPTURE: dict[int, dict] = {}
_LAYER_COUNTER = {"n": 0}


def _record(selection: torch.Tensor, scores: torch.Tensor) -> None:
    """``selection``: (..., K) chosen partition ids. ``scores``: (..., N) post-softmax weights."""
    sel = selection.detach().reshape(-1, selection.shape[-1]).cpu()
    sc = scores.detach().float().reshape(-1, scores.shape[-1]).cpu()
    # Only SSE layers call this (MoBA attention layers do not), and each calls exactly once per
    # forward pass, so this is an index into the SSE layers in call order -- NOT a model layer id.
    idx = _LAYER_COUNTER["n"]
    _LAYER_COUNTER["n"] += 1

    st = _CAPTURE.setdefault(
        idx,
        {"usage": torch.zeros(sc.shape[-1]), "tokens": 0, "sets": Counter(), "score_sum": torch.zeros(sc.shape[-1])},
    )
    st["usage"] += torch.bincount(sel.reshape(-1), minlength=sc.shape[-1]).float()
    st["score_sum"] += sc.sum(dim=0)
    st["tokens"] += sel.shape[0]
    # Selection *sets* (order-insensitive), to separate balance from input-dependence.
    for row in sel.tolist():
        st["sets"][tuple(sorted(row))] += 1


def patch_selection() -> int:
    """Wrap every live copy of the SSE selection helper so it records what it chose.

    Call this *after* the model is built. ``trust_remote_code=True`` imports the modelling code
    bundled in the checkpoint directory as ``transformers_modules.<...>.sse_swa``, which is a
    different module object from the repository's ``spb2.sse_swa`` -- patching only the latter
    would leave the running model untouched and silently capture nothing. Both call sites invoke
    ``sort_along_l`` as a bare global, so rebinding it on whichever module actually defines the
    layer is enough, and patching every candidate covers either loading path.

    Returns the number of modules patched.
    """
    import torch.nn.functional as F

    def wrap(original):
        def patched(q, k, v, gk, beta, e, cu_seqlens, K, emulq, emulk):  # noqa: ANN001
            with torch.no_grad():
                probs = F.softmax(e.detach(), dim=-1, dtype=torch.float)
                _, chosen = torch.topk(probs, k=K, dim=2)
                _record(chosen, probs)
            return original(q, k, v, gk, beta, e, cu_seqlens, K, emulq, emulk)

        return patched

    patched_names = []
    for name, mod in list(sys.modules.items()):
        fn = getattr(mod, "sort_along_l", None)
        if callable(fn) and getattr(fn, "__name__", "") == "sort_along_l":
            setattr(mod, "sort_along_l", wrap(fn))
            patched_names.append(name)
    for name in patched_names:
        print(f"patched {name}.sort_along_l for selection capture")
    if not patched_names:
        print("WARNING: no module exposing sort_along_l was found -- nothing will be captured")
    return len(patched_names)


def reset_capture() -> None:
    """Clear accumulated statistics so the next context length starts from scratch."""
    _CAPTURE.clear()
    _LAYER_COUNTER["n"] = 0


def summarise(n_partitions: int, k: int) -> list[dict]:
    rows = []
    for layer in sorted(_CAPTURE):
        st = _CAPTURE[layer]
        picks = st["usage"].sum().clamp(min=1)
        usage = (st["usage"] / picks).tolist()  # fraction of picks per partition
        # Normalised entropy of usage: 1.0 = perfectly balanced, 0.0 = one partition takes all.
        ent = -sum(u * math.log(u) for u in usage if u > 0)
        norm_ent = ent / math.log(n_partitions) if n_partitions > 1 else 1.0
        total_sets = sum(st["sets"].values()) or 1
        top_set, top_count = st["sets"].most_common(1)[0]
        n_possible = math.comb(n_partitions, k)
        rows.append(
            {
                "sse_layer": layer,
                "tokens": st["tokens"],
                "usage": [round(u, 4) for u in usage],
                "usage_entropy_norm": round(norm_ent, 4),
                "distinct_sets_seen": len(st["sets"]),
                "possible_sets": n_possible,
                "modal_set": list(top_set),
                "modal_set_share": round(top_count / total_sets, 4),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True)
    ap.add_argument(
        "--seq-lens", default="1024,4096,8192",
        help="comma-separated context lengths to sweep. Selection balance is only interesting if "
             "it holds at the lengths the model is actually used at, so this defaults to a sweep "
             "up to the 8k base checkpoint's full trained range rather than one short length.",
    )
    ap.add_argument(
        "--tokens-per-length", type=int, default=8192,
        help="approximate token-selections gathered per layer at EACH length, so the statistics "
             "are comparable across lengths instead of scaling with the context.",
    )
    ap.add_argument("--out", default="sse_selection.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    seq_lens = [int(s) for s in args.seq_lens.split(",") if s.strip()]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.model_path} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:  # base checkpoints often ship none, and padding would then raise
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(args.device).eval()
    # AFTER the load: trust_remote_code only imports the checkpoint's own modelling module here,
    # so patching earlier would have missed the module the model actually calls into.
    if patch_selection() == 0:
        raise SystemExit("no sort_along_l found after loading the model -- nothing to measure")
    cfg = model.config
    n_part = int(getattr(cfg, "num_sparse_partition", 4))
    k = int(getattr(cfg, "num_writer", 2))
    print(f"num_sparse_partition={n_part} num_writer={k} num_reader={getattr(cfg,'num_reader',None)}")
    print(f"layers={cfg.num_hidden_layers} attn_layers={(cfg.attn or {}).get('layers')}")

    ctx_limit = int(getattr(cfg, "max_position_embeddings", 0) or 0)
    if ctx_limit:
        over = [n for n in seq_lens if n > ctx_limit]
        if over:
            print(f"dropping {over}: beyond this checkpoint's max_position_embeddings={ctx_limit}")
            seq_lens = [n for n in seq_lens if n <= ctx_limit]
    if not seq_lens:
        raise SystemExit("no requested context length fits this checkpoint")

    # Real text rather than random ids: selection is content-dependent by construction, and
    # random token ids would understate how much it actually varies.
    prompts = [
        "The development of brain-inspired computing has focused on reducing the energy cost of "
        "large language models while preserving their ability to model long-range dependencies. ",
        "In a linear attention layer, the recurrent state acts as an associative memory: keys are "
        "written as outer products and read back by queries, which keeps the cost linear in "
        "sequence length. ",
        "小米公司是一家总部位于北京的消费电子及智能制造公司，其产品线涵盖智能手机、智能家居与电动汽车。",
        "Sparse mixture-of-experts models route each token to a small subset of parameters, so the "
        "compute per token stays roughly constant as total capacity grows. ",
    ]

    # One long tokenised stream of mixed English/Chinese prose, sliced into non-overlapping
    # windows. Repeating a single prompt to fill the context would make the input artificially
    # self-similar and could flatter the selection's apparent stability; slicing a varied stream
    # keeps content diverse at every length. Batch size is 1 throughout, so no padding is involved
    # and nothing has to be masked out of the statistics.
    stream = tok(" ".join(prompts) * 400, return_tensors="pt")["input_ids"][0]
    print(f"text stream: {stream.numel()} tokens", flush=True)

    per_length: dict[str, list[dict]] = {}
    with torch.no_grad():
        for n in seq_lens:
            windows = max(1, args.tokens_per_length // n)
            if stream.numel() < windows * n:
                windows = max(1, stream.numel() // n)
            reset_capture()
            print(f"\n=== context {n} tokens, {windows} window(s) ===", flush=True)
            for w in range(windows):
                ids = stream[w * n:(w + 1) * n].unsqueeze(0).to(args.device)
                _LAYER_COUNTER["n"] = 0  # layer calls restart each forward
                model(input_ids=ids)
                print(f"  window {w + 1}/{windows} done "
                      f"(sse layers seen: {_LAYER_COUNTER['n']})", flush=True)
            rows = summarise(n_part, k)
            if not rows:
                print("NO SSE SELECTION CAPTURED -- the patched helper was never called. Either "
                      "this checkpoint routes every layer through MoBA attention, or the call "
                      "site changed.")
                return
            per_length[str(n)] = rows
            print(f"\n{'sse_layer':>9} {'usage per partition':>34} {'entropy':>9} "
                  f"{'sets seen':>10} {'modal set':>12} {'modal share':>12}")
            print("-" * 95)
            for r in rows:
                sets_seen = "{}/{}".format(r["distinct_sets_seen"], r["possible_sets"])
                print(f"{r['sse_layer']:>9} {str(r['usage']):>34} {r['usage_entropy_norm']:>9.4f} "
                      f"{sets_seen:>10} {str(r['modal_set']):>12} {r['modal_set_share']:>12.3f}")

    uniform_share = 1 / math.comb(n_part, k)
    print(f"\n=== summary across context lengths "
          f"(uniform-over-sets baseline = {uniform_share:.3f}) ===")
    print(f"{'context':>8} {'layers':>7} {'tok/layer':>10} {'entropy mean':>13} {'entropy min':>12} "
          f"{'modal mean':>11} {'modal max':>10} {'all sets':>9}")
    print("-" * 88)
    for n in seq_lens:
        rows = per_length[str(n)]
        ent = [r["usage_entropy_norm"] for r in rows]
        share = [r["modal_set_share"] for r in rows]
        allsets = all(r["distinct_sets_seen"] == r["possible_sets"] for r in rows)
        print(f"{n:>8} {len(rows):>7} {rows[0]['tokens']:>10} {sum(ent)/len(ent):>13.4f} "
              f"{min(ent):>12.4f} {sum(share)/len(share):>11.4f} {max(share):>10.4f} "
              f"{('yes' if allsets else 'NO'):>9}")

    print("\n# Balanced usage AND a low modal-set share = the mechanism works as designed.")
    print("# Low entropy = usage collapse: expanded state effectively smaller than the config says.")
    print("# High modal share with balanced usage = input-independent selection: the choice")
    print("# carries no content information even though the histogram looks healthy.")
    print("# Watch the trend across lengths, not just the absolute values: balance that holds at")
    print("# short context but decays as the context grows would matter most for long-context use.")

    Path(args.out).write_text(json.dumps({
        "config": {"num_sparse_partition": n_part, "num_writer": k,
                   "uniform_set_share": uniform_share},
        "by_context_length": per_length,
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
