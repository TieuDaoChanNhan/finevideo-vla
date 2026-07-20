"""Standalone diagnostic: prints the FULL traceback for the Seed2Tokenizer
load failure that pipeline.py's `except Exception as e: print(f"...{e}")`
swallows down to just the exception message. Not part of Step A itself —
throwaway debug script, safe to delete after root cause is found."""
import os
import sys
import traceback

PROTOTYPE_DIR = "/e/project1/reformo/nguyen38/prototype"
sys.path.insert(0, PROTOTYPE_DIR)
os.chdir(PROTOTYPE_DIR)

import transformers.modeling_utils as _modeling_utils
import transformers.pytorch_utils as _pytorch_utils
for _name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
    if not hasattr(_modeling_utils, _name):
        setattr(_modeling_utils, _name, getattr(_pytorch_utils, _name))

sys.path.append("./seed2")
import torch
import seed2_tokenizer
from seed2_tokenizer import Seed2Tokenizer as LocalSeed2Tokenizer

# seed2_tokenizer.py deliberately sets Qformer.cls = None (encode-only path
# never needs the MLM head) — fine under the transformers version this was
# written for, but transformers==4.57.6's from_pretrained() now calls
# tie_weights() unconditionally on every submodule, and get_output_embeddings()
# crashes on self.cls being None instead of tolerating it. Patch both classes
# that share this pattern to return None (== "no output embeddings") instead
# of crashing, matching the original intent.
def _safe_get_output_embeddings(self):
    return None if self.cls is None else self.cls.predictions.decoder


def _safe_set_output_embeddings(self, new_embeddings):
    if self.cls is not None:
        self.cls.predictions.decoder = new_embeddings


for _cls in (seed2_tokenizer.BertLMHeadModel, seed2_tokenizer.BertForMaskedLM):
    _cls.get_output_embeddings = _safe_get_output_embeddings
    _cls.set_output_embeddings = _safe_set_output_embeddings

try:
    tok = LocalSeed2Tokenizer.from_pretrained("./seed2", torch_dtype=torch.float16)
    print("LOADED OK:", type(tok))
except Exception:
    print("=== FULL TRACEBACK ===")
    traceback.print_exc()
