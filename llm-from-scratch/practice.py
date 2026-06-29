text = open("data/shakespeare.txt").read()
chars = sorted(set(text))
vocab_size = len(chars)

print(f"Total karakter unik: {vocab_size}")
print(f"Karakter unik: {chars}")

stoi = {c: i for i, c in enumerate(chars)}  # huruf ke angka
itos = {i: c for c, i in stoi.items()}      # angka ke huruf

def encode(s):
    return [stoi[c] for c in s] 

def decode(ids):
    return "".join([itos[i] for i in ids])


pesan = "Hello"
pesan_encode = encode(pesan)
print(f"\nKata asli: '{pesan}'")
print(f"Setelah di-encode (jadi angka): {pesan_encode}")

pesan_kembali = decode(pesan_encode)
print(f"Setelah di-decode (kembali ke huruf): '{pesan_kembali}'")
