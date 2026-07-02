import os
import time
import regex as re
import multiprocessing
from collections import Counter
from typing import BinaryIO, Iterable
from itertools import islice


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenization(input_tuple: tuple[str, int, int, str]) -> Iterable[re.Match[str]]:
    input_path, start, end, special_tokens = input_tuple
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    pre_tokens = re.finditer(PAT, re.sub(r'\s+'.join(special_tokens), '', chunk))
    counts = Counter()
    for token in pre_tokens:
        counts[token.group()] += 1    
    return counts

class BPETokenizer:

    def __init__(self, input_path, vocab_size, special_tokens, num_processes=4):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_processes = num_processes

        self.vocab = {} # token to id (integer)
        self.id_to_token = {} # id to token (string)
        # Keep track of the pairs we have merged
        self.merge_rank = {} # pair to merge rank (integer)
        self.merges = []
        self._pre_tokenizer = None
    
    def _reset(self) -> None:
        self.vocab = {}
        self.id_to_token = {}
        self.merge_rank = {}
        self.merges = []
        # Every byte is a vocab, to intialize 
        for byte in range(256):
            self.vocab[chr(byte)] = byte
            self.id_to_token[byte] = chr(byte)
        # Add special tokens to the vocab
        for special_token in self.special_tokens:
            special_token_id = len(self.vocab)
            self.vocab[special_token] = special_token_id
            self.id_to_token[special_token_id] = special_token
        

    def _train_pre_tokenizer(self):
        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, b"<|endoftext|>")
            tasks = [
                (self.input_path, start, end, self.special_tokens)
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ]

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        # 1. Pre-tokenization is done in parallel 
        with multiprocessing.Pool() as pool:
            counters = pool.map(pre_tokenization, tasks)
        
        self._pre_tokenizer = Counter()
        for counter in counters:
            self._pre_tokenizer.update(counter)

    def _read_corpus(self, input_path: str |  os.PathLike) -> str:
        with open(input_path, 'r') as f:
            return f.read()

    def train_bpe(self, vocab_size: int) -> None:
        self._reset()
        self._train_pre_tokenizer()
        self._train_bpe(vocab_size)

    def _train_bpe(self, vocab_size: int) -> None:
        _counter = self._initialize_counter()
        while len(self.vocab) < vocab_size:
            result = self._get_most_frequent_pair(_counter)
            if result is None:
                break
            pairs, cnt = result
            print(f"Most frequent pair: {pairs}")
            self._merge_pair(pairs, _counter)

    def _initialize_counter(self) -> Counter:
        counter = Counter()
        for token in self._pre_tokenizer:
            token_bytes = token.encode('utf-8')
            cnt = self._pre_tokenizer[token]
            for i in range(len(token_bytes) - 1):
                counter[(token_bytes[i], token_bytes[i + 1])] += cnt
        return counter

    def _get_most_frequent_pair(self, counter: Counter) -> tuple[int, int]:
        if counter.most_common(1):
            return counter.most_common(1)[0]
        else:
            return None

    def _merge_pair(self, pair: tuple[int, int], _counter: Counter) -> None:
        merged_bytes = self.id_to_token[pair[0]] + self.id_to_token[pair[1]]
        merged_id = len(self.vocab)

        self.vocab[merged_bytes] = merged_id
        self.id_to_token[merged_id] = merged_bytes
        self.merge_rank[pair] = len(self.merge_rank)
        self.merges.append((self.id_to_token[pair[0]], self.id_to_token[pair[1]]))

        # update the counter 
        del _counter[pair]

        for token, cnt in self._pre_tokenizer.items():
            token_bytes = token.encode('utf-8')
            for i in range(len(token_bytes) - len(merged_bytes) + 1):
                # Find all pair occurences and update the counter 
                if token_bytes[i:i+len(merged_bytes)] == merged_bytes:
                    _counter[(merged_id, self.vocab[token_bytes[i+len(merged_bytes)]])] += cnt

    def encode(self, text: str) -> list[int]:
        result = []
        # use the current tokenizer to tokenize the text
        for char in text:
            if char not in self.vocab:
                result.append(-1) # unknown token
                continue
            # find the longest prefix of the char in the vocab
            result.append(self.vocab[char])
        return result


if __name__ == "__main__":
    start_time = time.time()
    special_tokens = ["<|endoftext|>"]
    max_vocab_size = 10000
    
    input_path = "data/ashe.txt"
    input_path = "data/TinyStoriesV2-GPT4-train.txt"

    # 2. Train the BPE model 
    encoder = BPETokenizer(input_path, 500, special_tokens)
    start_time = time.time()
    encoder.train_bpe(max_vocab_size)
    print(f"Time taken for training BPE: {time.time() - start_time} seconds")
    print(f"Vocab: {encoder.vocab}")
    print(f"Id to token: {encoder.id_to_token}")
    print(f"Merge rank: {encoder.merge_rank}")