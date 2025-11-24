import regex as re # type: ignore
import time
from datetime import datetime
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Process, Queue

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _now():
    # Nice readable timestamp
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _pre_tokenize(input_path, start, end, q, special_tokens):
        
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
        # 🔧 关键：把所有 Windows 换行统一成 Linux 风格
        chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
                
        special_pat = "|".join(re.escape(tok) for tok in special_tokens)
        parts = re.split(special_pat, chunk)
        
        text_freq: dict[str, int] = {}
        for part in parts:
            
            if part in special_tokens:
                continue
            
            for m in re.finditer(PAT, part):
                text = m.group()
                text_freq[text] = text_freq.get(text, 0)+1
                
        q.put(text_freq)

try:
    from cs336_basics import fast_bpe
    HAS_FAST_BPE = True
except ImportError:
    fast_bpe = None
    HAS_FAST_BPE = False



class BPETokenizer:
    
    def __init__(self, vocab_size, special_tokens):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = [] # type: ignore
        
    # -------PUBLIC API------------
    
    def train_from_file(self, input_path):
        print(f"[{_now()}] Start train_from_file")

        t0 = time.perf_counter()
        print(f"[{_now()}] Start _pre_tokenize")
        pre_tokens = self._multiprocessing_pre_tokenize(input_path)
        t1 = time.perf_counter()
        print(f"[{_now()}] Finished _pre_tokenize, took {t1 - t0:.2f} s")

        print(f"[{_now()}] Start _run_bpe_merge")
        self._run_bpe_merge(pre_tokens)
        t2 = time.perf_counter()
        print(f"[{_now()}] Finished _run_bpe_merge, took {t2 - t1:.2f} s")

        print(f"[{_now()}] Start _build_vocab_from_merge")
        self._build_vocab_from_merge()
        t3 = time.perf_counter()
        print(f"[{_now()}] Finished _build_vocab_from_merge, took {t3 - t2:.2f} s")

        print(f"[{_now()}] Total train_from_file time: {t3 - t0:.2f} s")
    
    # -------INTERNAL DATA PIPELINE-----------
    
    def _multiprocessing_pre_tokenize(self, input_path):

        with open(input_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            
            processes = []
            q = Queue()
            
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                p = Process(target=_pre_tokenize, args=(input_path, start, end, q, self.special_tokens))
                processes.append(p)
                p.start()
                
            whole_text_freq: dict[str, int] = {}
            
            for _ in processes:
                tf = q.get()
                for token, count in tf.items():
                    whole_text_freq[token] = whole_text_freq.get(token, 0)+count
            
            for p in processes:
                p.join()
                
            words_encoding: dict[tuple[bytes, ...], int] = { # type: ignore
            tuple(bytes([b]) for b in token.encode("utf-8")): count
            for token, count in whole_text_freq.items()
        }
        
        return words_encoding
    
    def _run_bpe_merge(self, words_encoding):
        
        if HAS_FAST_BPE:
            print(">>> Using RUST fast_bpe backend")
            py_words = [
                ([chunk for chunk in word], int(count))
                for word, count in words_encoding.items()
            ]

            merges = fast_bpe.run_bpe_merge(
                py_words,
                int(self.vocab_size),
                int(len(self.special_tokens)),
            )
            # 把 Rust 返回的 merges 塞回 self.merges，就和原来 Python 版本一致了
            self.merges = merges
            return
        
        merges_iter_number = self.vocab_size - 256 - len(self.special_tokens)
        
        pair_freq: dict[tuple[bytes, bytes], int] = {}
        pair_to_words: dict[tuple[bytes, bytes], set(tuple[bytes, ...])] = {} # type: ignore
        
        for word, count in words_encoding.items():
            for i in range(len(word)-1):
                pair_of_bytes = (word[i], word[i+1])
                
                pair_freq[pair_of_bytes] = pair_freq.get(pair_of_bytes, 0)+count
                if pair_of_bytes not in pair_to_words:
                    pair_to_words[pair_of_bytes] = set()
                pair_to_words[pair_of_bytes].add(word)
            
        for iter_number in range(merges_iter_number):
            if iter_number % 100 == 0:
                print(f"[{_now()}] Merge step {iter_number}/{merges_iter_number}")
            
            max_key = max(
                pair_freq,
                key=lambda pair: (pair_freq[pair], pair)
            )
            self.merges.append(max_key)
            
            affecting_words = pair_to_words.pop(max_key)
            for word in affecting_words:
                count = words_encoding.pop(word)
                
                for i in range(len(word)-1):
                    pair_of_bytes = (word[i], word[i+1])
                    pair_freq[pair_of_bytes] -= count
                    
                    words = pair_to_words.get(pair_of_bytes)
                    if words is not None:
                        words.discard(word)
                        if not words:
                            del pair_to_words[pair_of_bytes]
                            
                new_key = ()
                i = 0
                while i < len(word)-1:
                    pair_of_bytes = (word[i], word[i+1])
                    
                    if(pair_of_bytes==max_key):
                        new_key += (word[i]+word[i+1],)
                        i += 2
                    else:
                        new_key += (word[i],)
                        i += 1
                if i < len(word):
                    new_key += (word[i],)
                
                for i in range(len(new_key)-1):
                    pair_of_bytes = (new_key[i], new_key[i+1])
                    
                    if pair_of_bytes not in pair_to_words:
                        pair_to_words[pair_of_bytes] = set()
                    pair_to_words[pair_of_bytes].add(new_key)
                    
                    pair_freq[pair_of_bytes] = pair_freq.get(pair_of_bytes, 0)+count
                    
                words_encoding[new_key] = words_encoding.get(new_key, 0)+count
                    
    def _build_vocab_from_merge(self):
        
        next_index = 0
        for tok in self.special_tokens:
            self.vocab[next_index] = tok.encode("utf-8", errors="ignore")
            next_index += 1
        start = len(self.vocab)
        self.vocab.update({i+start: bytes([i]) for i in range(256)})
        
        next_index = max(self.vocab.keys())+1
        for t1, t2 in self.merges:
            self.vocab[next_index] = t1+t2
            next_index += 1