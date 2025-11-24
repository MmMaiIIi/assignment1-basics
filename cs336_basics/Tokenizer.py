import json
from typing import Iterable, Iterator
import regex as re # type: ignore

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None,
    ) -> None:
        '''
        Construct a tokenizer from a given
        vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept
        replacement character.
        '''

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = list(special_tokens) if special_tokens is not None else []
        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        
    # ------------PUBLIC API--------------
                
    def encode(self, text: str) -> list[int]:
        '''
        Encode an input text into a sequence of token IDs.
        '''
        special_pat = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
        parts = re.split(special_pat, text)
        # print("parts", parts)
        token_IDs = []
        
        cur_part = ""
        for part in parts:
            if part in self.special_tokens:
                
                for m in re.finditer(PAT, cur_part):
                    token = m.group()
                    for token_id in self._pre_token_to_integer_sequence(token):
                        token_IDs.append(token_id)
                                        
                token_IDs.append(self._pre_token_to_integer_sequence(part))
                cur_part = ""
                continue
            
            cur_part += part
            
        for m in re.finditer(PAT, cur_part):
            token = m.group()
            for token_id in self._pre_token_to_integer_sequence(token):
                token_IDs.append(token_id)

        return token_IDs
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        Given an iterable of
        strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory.
        '''
        token_IDs = []
        
        for string in iterable:
            for token_id in self.encode(string):
                token_IDs.append(token_id)

        return token_IDs
        
    def decode(self, ids: list[int]) -> str:
        '''
        Decode a sequence of token IDs into text.
        '''
        whole_bytes = b''.join(
            self.vocab[id]
            for id in ids
        )
        return whole_bytes.decode("utf-8", errors='replace') 
        
    # ---------INTERNAL API----------
    def _pre_token_to_integer_sequence(
        self,
        pre_token: str,
    ) -> list[int]:
        '''
        ' cat' -> [b' c', b'a', b't'] -> [7, 1, 5]
        ' ate' -> [b' at', b'e'] -> [10, 3]
        '''
        if pre_token in self.special_tokens:
            for tok_id, tok_bytes in self.vocab.items():
                if tok_bytes == pre_token.encode("utf-8"):
                    return tok_id 
        
        ### Pre-tokenize
        word_encoding = tuple(bytes([b]) for b in pre_token.encode("utf-8"))
        old_key = word_encoding
        new_key = ()
        
        ### Apply the merges
        for merge in self.merges:
            i=0
            while i < len(old_key)-1:
                pair_of_bytes = (old_key[i], old_key[i+1])
                
                if(pair_of_bytes==merge):
                    new_key += (old_key[i]+old_key[i+1],)
                    i+=2
                else:
                    new_key += (old_key[i],)
                    i+=1
            if i < len(old_key):
                new_key += (old_key[i],)  
            
            old_key = new_key
            new_key = ()  
        
        # print("old_key", old_key)
        
        ### trans to integer sequence
        integer_list = []
        for token in old_key:
            key = next((k for k, v in self.vocab.items() if v == token), None)
            integer_list.append(key)
        
        return integer_list