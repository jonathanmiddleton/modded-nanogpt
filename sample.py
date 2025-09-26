import torch
from torch import nn, load, tensor
import tiktoken
from model import GPT, Hyperparameters as args
from contextlib import nullcontext

device = 'cpu'
checkpoint = load("/Users/jonathanmiddleton/models/my_checkpoints/state_step005960_350M.pt", map_location=device)
model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=16, num_heads=8, model_dim=1024,
                       max_seq_len=args.train_seq_len)
state_dict = checkpoint['model']

unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
if device != 'cpu':
    model.to(device)
    model = torch.compile(model)
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

start_ids = encode("Q: The most popular movie of all time is Avatar. What is your favorite movie?\nA:")
x = tensor(start_ids, dtype=torch.long, device=device)[None, ...]

ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16) #if device != 'cpu' else nullcontext()
with torch.no_grad():
    with ctx:
        y = model.generate(x, max_new_tokens=50, temperature=0.3, top_k=50)
        print(decode(y[0].tolist()))