# Gunakan path lengkap ke tempat hasil_output.txt Anda berada
dataset = open("/home/eljo/Pusri/runLlm/data/hasil_output.txt", "r", encoding="utf-8").read()
kandunganKarakter = sorted(set(dataset))
banyakKarakter = len(kandunganHuruf)

print(kandunganHuruf)
print(banyakKarakter)