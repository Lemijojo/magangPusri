import pandas as pd

# Gunakan jalur lengkap dari /home sampai ke nama filenya
baca = pd.read_csv("/home/eljo/Pusri/runLlm/data/tokopedia-product-reviews-2019.csv")

# Tentukan juga outputnya mau disimpan di folder yang sama
baca.to_csv('/home/eljo/Pusri/runLlm/hasil_output.txt', sep='\t', index=False)

print("Konversi CSV ke TXT selesai!")