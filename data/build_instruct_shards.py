import os, json, math, random, numpy as np
from datasets import load_dataset, DatasetDict
from transformers import GPT2TokenizerFast

random.seed(1234)

EOS = 50256
TOKENS_PER_SHARD = 50_000_000
MAGIC = 20240520
VERSION = 1

tok = GPT2TokenizerFast.from_pretrained("gpt2")
assert tok.eos_token_id == EOS

def _fmt(instr, resp):
    if instr is None or resp is None: return None
    instr = instr.strip(); resp = resp.strip()
    if not instr or not resp: return None
    return f"### Instruction:\n{instr}\n\n### Response:\n{resp}\n"

def _pairs_oasst1():
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    id2row = {r["message_id"]: r for r in ds}
    for r in ds:
        if r.get("role") == "assistant":
            p = id2row.get(r.get("parent_id"))
            if p and p.get("role") == "prompter" and (r.get("lang") or "en").startswith("en"):
                y = _fmt(p.get("text"), r.get("text"))
                if y: yield y

def _pairs_dolly15k():
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    for r in ds:
        ctx = r.get("context") or ""
        instr = r.get("instruction") or ""
        if ctx: instr = instr + "\n\n" + ctx
        y = _fmt(instr, r.get("response"))
        if y: yield y

def _pairs_openhermes():
    ds = load_dataset("HuggingFaceTB/OpenHermes-2.5-H4", split="train")
    for r in ds:
        msgs = r.get("messages") or r.get("conversations")
        if not msgs: continue
        u, a = None, None
        for m in msgs:
            if m.get("role") in ("user","human") and u is None:
                u = m.get("content")
            elif m.get("role") in ("assistant","gpt") and u is not None:
                a = m.get("content"); break
        y = _fmt(u, a)
        if y: yield y

def _pairs_openorca():
    ds = load_dataset("Open-Orca/OpenOrca", split="train")
    for r in ds:
        instr = r.get("system_prompt") or ""
        q = r.get("question") or r.get("input") or ""
        resp = r.get("response") or r.get("output")
        if q: instr = (instr + "\n\n" + q).strip()
        y = _fmt(instr, resp)
        if y: yield y

def _pairs_norobots():
    ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    for r in ds:
        # prefer messages if present
        msgs = r.get("messages")
        if msgs:
            u = next((m["content"] for m in msgs if m.get("role")=="user"), None)
            a = next((m["content"] for m in msgs if m.get("role")=="assistant"), None)
            y = _fmt(u, a)
        else:
            y = _fmt(r.get("prompt"), r.get("response") or r.get("output"))
        if y: yield y

def _pairs_codealpaca():
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    for r in ds:
        instr = r.get("instruction")
        inp = r.get("input") or ""
        if inp: instr = instr + "\n\n" + inp
        y = _fmt(instr, r.get("output"))
        if y: yield y

def _pairs_metamath_subset(max_rows=30000):
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    n = min(len(ds), max_rows)
    for i in range(n):
        r = ds[i]
        q = r.get("query") or r.get("problem") or r.get("instruction")
        a = r.get("response") or r.get("answer") or r.get("output")
        y = _fmt(q, a)
        if y: yield y

SOURCES = [
    ("oasst1", _pairs_oasst1, 1.5),
    ("dolly15k", _pairs_dolly15k, 1.0),
    ("openhermes", _pairs_openhermes, 1.0),
    ("openorca", _pairs_openorca, 0.7),
    ("no_robots", _pairs_norobots, 1.5),
    ("codealpaca", _pairs_codealpaca, 0.3),
    ("metamath", _pairs_metamath_subset, 0.3),
]

def iter_mixture():
    pools = []
    for name, fn, w in SOURCES:
        pools.append((name, list(fn()), w))
    lens = [len(p[1]) for p in pools]
    weights = [p[2]*math.sqrt(len_) for p, len_ in zip(pools, lens)]
    total = sum(weights)
    probs = [w/total for w in weights]
    iters = [iter(random.sample(p[1], len(p[1]))) for p in pools]
    while True:
        k = random.choices(range(len(pools)), probs)[0]
        try:
            yield next(iters[k])
        except StopIteration:
            return

def write_shards(out_dir, target_tokens=200_000_000, max_tokens_per_shard=TOKENS_PER_SHARD):
    os.makedirs(out_dir, exist_ok=True)
    acc = []
    total = 0
    shard_id = 0
    for text in iter_mixture():
        ids = tok.encode(text) + [EOS]
        acc.extend(ids)
        if len(acc) >= max_tokens_per_shard or (target_tokens and total+len(acc) >= target_tokens):
            arr = np.array(acc, dtype=np.uint16)
            assert (arr < 65536).all()
            header = np.zeros(256, dtype=np.int32)
            header[0] = MAGIC; header[1] = VERSION; header[2] = len(arr)
            path = os.path.join(out_dir, f"instruct_train_{shard_id:02d}.bin")
            with open(path, "wb") as f:
                f.write(header.tobytes()); f.write(arr.tobytes())
            total += len(arr); shard_id += 1; acc = []
            if target_tokens and total >= target_tokens: break
    # small validation shard
    if not os.path.exists(os.path.join(out_dir, "instruct_val_00.bin")):
        val_ids = tok.encode("### Instruction:\nSay hello.\n\n### Response:\nHello!\n") + [EOS]
        arr = np.array(val_ids*1000, dtype=np.uint16)
        header = np.zeros(256, dtype=np.int32); header[0]=MAGIC; header[1]=VERSION; header[2]=len(arr)
        with open(os.path.join(out_dir,"instruct_val_00.bin"),"wb") as f:
            f.write(header.tobytes()); f.write(arr.tobytes())

if __name__ == "__main__":
    write_shards("data/instruct_mix", target_tokens=200_000_000, max_tokens_per_shard=50_000_000)
