import os
import time
import regex as re
import multiprocessing
from collections import Counter
from typing import BinaryIO, Iterable
from heapq import heappush, heappop

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

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def pre_tokenization(input_tuple):
    input_path, start, end, special_tokens = input_tuple
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # split on special tokens (don't delete-then-concat), keep segments separate
    if special_tokens:
        split_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        segments = re.split(split_pattern, chunk)
    else:
        segments = [chunk]

    counts = Counter()
    for seg in segments:
        for m in re.finditer(PAT, seg):
            counts[m.group()] += 1
    return counts


class RevPair:
    """Wrap the (left_str, right_str) key so the min-heap pops the pair with the
    largest count and, on ties, the lexicographically greatest pair."""
    __slots__ = ("pair",)

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair  # reversed: bigger string sorts "smaller"


class BPETokenizer:

    def __init__(self, input_path, vocab_size, special_tokens, num_processes=4):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_processes = num_processes

        self.vocab = {}         # token (str) -> id (int)
        self.id_to_token = {}   # id (int) -> token (str)
        self.merge_rank = {}    # pair -> merge rank
        self.merges = []
        self._pre_tokenizer = None

    def _reset(self) -> None:
        self.vocab = {}
        self.id_to_token = {}
        self.merge_rank = {}
        self.merges = []
        for byte in range(256):
            self.vocab[bytes([byte])] = byte
            self.id_to_token[byte] = bytes([byte])
        for special_token in self.special_tokens:
            sid = len(self.vocab)
            tok = special_token.encode("utf-8")
            self.vocab[tok] = sid
            self.id_to_token[sid] = tok

    def _train_pre_tokenizer(self):
        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, b"<|endoftext|>")
            tasks = [
                (self.input_path, start, end, self.special_tokens)
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ]
        with multiprocessing.Pool() as pool:
            counters = pool.map(pre_tokenization, tasks)

        self._pre_tokenizer = Counter()
        for counter in counters:
            self._pre_tokenizer.update(counter)

    def train_bpe(self, vocab_size: int) -> None:
        start_time = time.time()
        self._reset()
        print(f"[{time.time() - start_time:.2f}] Training pre-tokenizer...")
        self._train_pre_tokenizer()
        print(f"[{time.time() - start_time:.2f}] Pre-tokenizer trained")

        pre_token_words = Counter()
        for token, count in self._pre_tokenizer.items():
            pre_token_words[tuple(token.encode("utf-8"))] += count

        print(f"[{time.time() - start_time:.2f}] Training BPE...")
        self._train_bpe(vocab_size, pre_token_words, start_time)
        print(f"[{time.time() - start_time:.2f}] BPE trained")

    def _train_bpe(self, vocab_size, pre_token_words, start_time) -> None:
        pair_counter, heap, pair_to_words = self._initialize_counter(pre_token_words)
        while len(self.vocab) < vocab_size:
            if len(self.vocab) % 500 == 0:
                print(f"[{time.time() - start_time:.2f}] {len(self.vocab)} / {vocab_size}")
            result = self._pop_most_frequent_pair(pair_counter, heap)
            if result is None:
                break
            _count, pair = result
            self._merge_pair(pair, pair_counter, pair_to_words, pre_token_words, heap)

    def _initialize_counter(self, pre_token_words):
        pair_counter = Counter()
        pair_to_words = {}      # pair(id, id) -> set of words (current form)
        for word, count in pre_token_words.items():
            for a, b in zip(word, word[1:]):
                pair_counter[(a, b)] += count
                pair_to_words.setdefault((a, b), set()).add(word)

        heap = []
        for pair, count in pair_counter.items():
            key = (self.id_to_token[pair[0]], self.id_to_token[pair[1]])
            heappush(heap, (-count, RevPair(key), pair))
        return pair_counter, heap, pair_to_words

    def _pop_most_frequent_pair(self, pair_counter, heap):
        # Lazy deletion: skip entries whose stored count no longer matches reality.
        while heap:
            neg_count, _rev, pair = heappop(heap)
            if pair_counter.get(pair, 0) == -neg_count:
                return -neg_count, pair
        return None

    def _merge_pair(self, pair, pair_counter, pair_to_words, pre_token_words, heap) -> None:
        left_id, right_id = pair
        merged_token = self.id_to_token[left_id] + self.id_to_token[right_id]
        merged_id = len(self.vocab)
        self.vocab[merged_token] = merged_id
        self.id_to_token[merged_id] = merged_token
        self.merge_rank[pair] = len(self.merge_rank)
        self.merges.append((self.id_to_token[left_id], self.id_to_token[right_id]))

        affected = pair_to_words.pop(pair, set())
        delta = Counter()   # net change to each pair's global count this merge

        for word in affected:
            count = pre_token_words.get(word, 0)
            if count == 0:
                continue

            # subtract every adjacent pair of the old word
            for a, b in zip(word, word[1:]):
                delta[(a, b)] -= count
                s = pair_to_words.get((a, b))
                if s is not None:
                    s.discard(word)

            # build the merged word (greedy, left to right)
            new_word = []
            i, n = 0, len(word)
            while i < n:
                if i < n - 1 and word[i] == left_id and word[i + 1] == right_id:
                    new_word.append(merged_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)

            # add every adjacent pair of the new word
            for a, b in zip(new_word, new_word[1:]):
                delta[(a, b)] += count
                pair_to_words.setdefault((a, b), set()).add(new_word)

            # update word frequencies (two old words may collapse into one)
            del pre_token_words[word]
            pre_token_words[new_word] = pre_token_words.get(new_word, 0) + count

        # apply net changes; only push pairs that actually moved
        for p, d in delta.items():
            if d == 0:
                continue
            pair_counter[p] += d
            c = pair_counter[p]
            if c <= 0:
                pair_counter.pop(p, None)
                pair_to_words.pop(p, None)
            else:
                key = (self.id_to_token[p[0]], self.id_to_token[p[1]])
                heappush(heap, (-c, RevPair(key), p))

    def encode(self, text: str) -> list[int]:
        # NOTE: placeholder — single-char lookup, does not yet apply learned merges.
        result = []
        for char in text:
            result.append(self.vocab.get(char, -1))
        return result


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    max_vocab_size = 10000

    input_path = "data/TinyStoriesV2-GPT4-train.txt"

    encoder = BPETokenizer(input_path, max_vocab_size, special_tokens)
    start_time = time.time()
    encoder.train_bpe(max_vocab_size)
    print(f"Time taken for training BPE: {time.time() - start_time:.2f} seconds")