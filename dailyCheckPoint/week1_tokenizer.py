text = open("data/shakespeare.txt").read()
total_chars = len(text)
chars = sorted(set(text))
vocab_size = len(chars)

print(f"\nTotal karakter (panjang teks) : {total_chars:,}")
print(f"Jumlah karakter unik (vocab)  : {vocab_size}")

# 20 karakter paling sering muncul
from collections import Counter
char_count = Counter(text)
top20 = char_count.most_common(20)

print("\nTop 20 Karakter Paling Sering Muncul:")
for ch, count in top20:
    print(repr(ch), "->", count)

#TOKENIZER 
stoi = {c: i for i, c in enumerate(chars)}  # huruf → angka
itos = {i: c for c, i in stoi.items()}      # angka → huruf

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join([itos[i] for i in ids])


test_kata = "Hello World"
hasil_encode = encode(test_kata)
hasil_decode = decode(hasil_encode)
print(f"Teks asli      : '{test_kata}'")
print(f"Setelah encode : {hasil_encode}")
print(f"Setelah decode : '{hasil_decode}'")


# ── 4. BIGRAM GENERATOR ──────────────────────────────────────
import random
print("         BIGRAM GENERATOR")
# Hitung semua frekuensi bigram dari teks Shakespeare
bigram_count = Counter()
for i in range(len(text) - 1):
    bigram = (text[i], text[i+1])
    bigram_count[bigram] += 1

# Buat tabel probabilitas: untuk setiap karakter,
# karakter apa yang paling sering muncul berikutnya?
# Contoh: setelah "t", paling sering muncul "h" (karena "th")
from collections import defaultdict

next_char_prob = defaultdict(dict)
for (ch1, ch2), count in bigram_count.items():
    next_char_prob[ch1][ch2] = count

# Fungsi generate teks dengan bigram
def generate_text_bigram(start_char=None, length=200, seed=42):
    random.seed(seed)

    # Pilih karakter awal
    if start_char is None or start_char not in next_char_prob:
        start_char = random.choice(list(next_char_prob.keys()))

    result = [start_char]
    current = start_char

    for _ in range(length - 1):
        if current not in next_char_prob:
            break

        # mengambil semua kemungkinan karakter berikutnya dan valuenya
        choices = list(next_char_prob[current].keys())
        weights = list(next_char_prob[current].values())

        # Pilih secara acak berbobot (karakter yang lebih sering = lebih besar peluangnya)
        next_c = random.choices(choices, weights=weights, k=1)[0]
        result.append(next_c)
        current = next_c

    return "".join(result)

# Generate beberapa contoh
print("\nContoh teks yang di-generate oleh Bigram Model:")
print("-" * 55)
for i, mulai in enumerate(["F", "T", "W"], 1):
    generated = generate_text_bigram(start_char=mulai, length=150, seed=i*7)
    print(f"\n[Sample {i} — mulai dari '{mulai}']")
    print(generated)

# RUN SUMMARY
print("\n" + "=" * 55)
print("         RUN SUMMARY — WEEK 1")
print("=" * 55)
print(f"File dataset     : data/shakespeare.txt")
print(f"Total karakter   : {total_chars:,}")
print(f"Vocab size       : {vocab_size} karakter unik")
print(f"Total bigram unik: {len(bigram_count):,}")
print(f"Tokenizer        : Character-level (stoi/itos)")
print(f"Generator        : Bigram frequency sampling")
print("Status           : SELESAI ✓")
print("=" * 55)
