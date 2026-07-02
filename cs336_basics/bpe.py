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

    special_pattern = "|".join(re.escape(token) for token in special_tokens)
    if special_pattern:
        chunk = re.sub(special_pattern, "", chunk)
    pre_tokens = re.finditer(PAT, chunk)

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
        import time 
        start_time = time.time()
        self._reset()
        print(f"[{time.time() - start_time}] Training pre-tokenizer...")
        self._train_pre_tokenizer()
        print(f"[{time.time() - start_time}] Pre-tokenizer trained")
        pre_token_words = Counter()
        for token, count in self._pre_tokenizer.items():
            token_ids = tuple(token.encode('utf-8'))
            pre_token_words[token_ids] += count
        
        print(f"[{time.time() - start_time}] Training BPE...")
        self._train_bpe(vocab_size, pre_token_words, start_time)
        print(f"[{time.time() - start_time}] BPE trained")

    def _train_bpe(self, vocab_size: int, pre_token_words: Counter, start_time: float) -> None:
        _counter = self._initialize_counter(pre_token_words)
        while len(self.vocab) < vocab_size:
            if len(self.vocab) % 100 == 0:
                print(f"[{time.time() - start_time}] Training BPE... {len(self.vocab)} / {vocab_size}")
            result = self._get_most_frequent_pair(_counter)
            if result is None:
                break
            pairs, cnt = result
            # print(f"Len vocab: {len(self.vocab)}, Most frequent pair: {pairs}")
            self._merge_pair(pairs, pre_token_words)

            _counter = self._initialize_counter(pre_token_words)

    def _initialize_counter(self, pre_token_words: Counter) -> Counter:
        counter = Counter()
        
        for token_ids, count in pre_token_words.items():
            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                counter[pair] += count
        return counter

    def _get_most_frequent_pair(self, counter: Counter) -> tuple[int, int]:
        if not counter:
            return None
        return max(
            counter.items(),
            key=lambda item: (
                item[1],
                self.id_to_token[item[0][0]],
                self.id_to_token[item[0][1]],
            ),
        )

    def _merge_pair(self, pair: tuple[int, int], pre_token_words: Counter) -> None:
        left_token, right_token = self.id_to_token[pair[0]], self.id_to_token[pair[1]]
        merged_token = left_token + right_token

        merged_id = len(self.vocab)
        self.vocab[merged_token] = merged_id
        self.id_to_token[merged_id] = merged_token
        self.merge_rank[pair] = len(self.merge_rank)
        self.merges.append((left_token, right_token))

        for token_ids, count in list(pre_token_words.items()):
            new_token_ids = []
            merged = False
            i = 0
            while i < len(token_ids):
                if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair:
                    new_token_ids.append(merged_id)
                    merged = True
                    i += 2
                else:
                    new_token_ids.append(token_ids[i])
                    i += 1

            if merged:
                pre_token_words[token_ids] -= count
                if pre_token_words[token_ids] == 0:
                    del pre_token_words[token_ids]
                pre_token_words[tuple(new_token_ids)] += count

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