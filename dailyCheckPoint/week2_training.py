import torch
import torch.nn as nn

# =============================================================
# SETUP: Baca teks dan buat kamus huruf <-> angka
# =============================================================
text = open("data/shakespeare.txt").read()

# Kumpulkan semua karakter unik yang ada di teks
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}  # huruf ke angka
itos = {i: c for c, i in stoi.items()}       # angka ke huruf

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join([itos[i] for i in ids])

# Gunakan GPU kalau ada, kalau tidak pakai CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Menggunakan device:", device)

# =============================================================
# PENGATURAN (Hyperparameter)
# =============================================================
BLOCK_SIZE    = 8     # model melihat 8 karakter ke belakang
BATCH_SIZE    = 32    # belajar dari 32 contoh sekaligus
MAX_STEPS     = 500   # total langkah training
EVAL_EVERY    = 25    # catat log setiap 25 langkah
LEARNING_RATE = 1e-2
EMBED_DIM     = 32    # ukuran vektor tiap karakter

# =============================================================
# BUILD 1: TRAIN / VALIDATION SPLIT
# 90% data untuk belajar, 10% untuk evaluasi
# =============================================================
data = torch.tensor(encode(text), dtype=torch.long)

n          = int(0.9 * len(data))
train_data = data[:n]   # bagian belajar
val_data   = data[n:]   # bagian evaluasi

print("Ukuran data:")
print("  Total :", len(data))
print("  Train :", len(train_data))
print("  Val   :", len(val_data))

# =============================================================
# BUILD 2: FUNGSI GET_BATCH
# Ambil potongan teks acak sebagai soal (x) dan jawaban (y)
# Contoh: x = [H,e,l,l]  →  y = [e,l,l,o]
# =============================================================
def get_batch(split):
    # Pilih data train atau val
    if split == "train":
        d = train_data
    else:
        d = val_data

    # Pilih BATCH_SIZE titik awal secara acak
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))

    # Kumpulkan x dan y dari setiap titik awal
    x_list = []
    y_list = []
    for i in ix:
        x_list.append(d[i : i + BLOCK_SIZE])
        y_list.append(d[i + 1 : i + BLOCK_SIZE + 1])

    x = torch.stack(x_list)
    y = torch.stack(y_list)
    return x.to(device), y.to(device)

# Cek bentuk tensor — ini untuk Submit: Tensor shape notes
xb, yb = get_batch("train")
print("\nBentuk tensor:")
print("  x (input) :", xb.shape)
print("  y (target):", yb.shape)

# =============================================================
# BUILD 3: MODEL BAHASA SEDERHANA
# Embedding: ubah angka → vektor
# Linear   : ubah vektor → skor untuk tiap karakter
# =============================================================
class SimpleLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_DIM)
        self.head  = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x, targets=None):
        # Ubah token angka jadi vektor → [B, T, EMBED_DIM]
        out = self.embed(x)

        # Ubah vektor jadi skor tiap karakter → [B, T, vocab_size]
        logits = self.head(out)

        # Hitung loss kalau ada target yang diberikan
        loss = None
        if targets is not None:
            # BUILD 4: CROSS-ENTROPY LOSS
            # Ratakan tensor agar bisa dihitung
            B = logits.shape[0]
            T = logits.shape[1]
            C = logits.shape[2]
            logits_flat  = logits.view(B * T, C)  # [256, 65]
            targets_flat = targets.view(B * T)     # [256]
            # Semakin kecil loss = semakin pintar model
            loss = nn.functional.cross_entropy(logits_flat, targets_flat)

        return logits, loss

# Buat model
model = SimpleLM().to(device)
total_params = sum(p.numel() for p in model.parameters())
print("\nModel berhasil dibuat! Total parameter:", total_params)

# =============================================================
# BUILD 5: TRAINING LOOP + STEP-BY-STEP LOGGING
# =============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Fungsi bantu untuk menghitung rata-rata loss (tanpa update model)
def estimate_loss():
    model.eval()  # mode evaluasi
    hasil = {}
    for split in ["train", "val"]:
        total = 0
        for _ in range(20):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            total += loss.item()
        hasil[split] = total / 20
    model.train()  # kembali ke mode training
    return hasil

print("\n" + "=" * 45)
print("  TRAINING LOG")
print("=" * 45)
print(f"{'Step':<8} {'Train Loss':>12} {'Val Loss':>12}")
print("-" * 34)

# SUBMIT: Training log dengan minimal 20 langkah
log = []

for step in range(MAX_STEPS + 1):

    # Catat loss setiap EVAL_EVERY langkah
    if step % EVAL_EVERY == 0:
        losses = estimate_loss()
        log.append({"step": step, "train": losses["train"], "val": losses["val"]})
        print(f"{step:<8} {losses['train']:>12.4f} {losses['val']:>12.4f}")

    if step == MAX_STEPS:
        break

    # Ambil data → hitung loss → perbarui model
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)

    optimizer.zero_grad()  # reset gradien
    loss.backward()        # hitung arah perbaikan
    optimizer.step()       # terapkan perbaikan

# =============================================================
# SUBMIT: Initial loss dan final loss
# =============================================================
initial_loss = log[0]["train"]
final_loss   = log[-1]["train"]
drop = initial_loss - final_loss
pct  = drop / initial_loss * 100

print("\n" + "=" * 45)
print("  RUN SUMMARY — WEEK 2")
print("=" * 45)
print("Device         :", device)
print("Vocab size     :", vocab_size)
print("Block size     :", BLOCK_SIZE)
print("Batch size     :", BATCH_SIZE)
print("Embed dim      :", EMBED_DIM)
print("Total params   :", total_params)
print("Training steps :", MAX_STEPS)
print("Log recorded   :", len(log), "titik")
print()
print("Loss awal  :", round(initial_loss, 4))
print("Loss akhir :", round(final_loss, 4))
print(f"Penurunan  : {drop:.4f} ({pct:.1f}%)")
print("Status     : SELESAI")
print("=" * 45)

# =============================================================
# SUBMIT: Tensor shape notes
# =============================================================
print("""
--- CATATAN TENSOR SHAPES ---
x input       : [32, 8]     -> 32 sampel, tiap 8 token
y target      : [32, 8]     -> 32 label, tiap 8 token
embed output  : [32, 8, 32] -> tiap token jadi vektor 32 angka
logits output : [32, 8, 65] -> skor untuk 65 karakter
logits flat   : [256, 65]   -> setelah diratakan untuk hitung loss
-----------------------------
""")
