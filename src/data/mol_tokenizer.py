"""QM9 tokenizer — the paper's published HuggingFace tokenizer (vocab 40).

Schiff et al. (2024) / kuleshov-group/discrete-diffusion-guidance tokenize QM9
SMILES with `yairschiff/qm9-tokenizer`: 5 special tokens (<bos> <eos> <mask>
<pad> <unk>) plus chemically-meaningful SMILES tokens such as `[NH2+]`, `[cH-]`
(charged/aromatic atoms are single tokens, not characters). A molecule is
encoded as <bos> ... <eos> then padded with <pad> to a fixed length (the paper
uses 32). Using their exact tokenizer makes our bits-per-token / perplexity
directly comparable to the paper's QM9 numbers.
"""
import transformers

TOKENIZER_NAME = "yairschiff/qm9-tokenizer"


class SmilesTokenizer:
    def __init__(self):
        self.tok = transformers.AutoTokenizer.from_pretrained(
            TOKENIZER_NAME, trust_remote_code=True)
        self.pad_id = self.tok.pad_token_id
        self.vocab_size = len(self.tok)            # 40
        self._special = set(self.tok.all_special_tokens)

    def encode(self, smiles: str, length: int):
        """SMILES -> fixed-length token ids (<bos> ... <eos>, then <pad>)."""
        return self.tok(smiles, max_length=length, padding="max_length",
                        truncation=True, add_special_tokens=True)["input_ids"]

    def decode(self, ids) -> str:
        """Token ids -> SMILES string (dropping special/pad tokens)."""
        toks = self.tok.convert_ids_to_tokens([int(i) for i in ids])
        return "".join(t for t in toks if t not in self._special)
