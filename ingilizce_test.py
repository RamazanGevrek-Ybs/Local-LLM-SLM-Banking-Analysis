import time
import psutil
import csv
import os
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# EKRAN KARTI (VRAM) ÖLÇÜMÜ
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) # RTX 3050'yi yakalar
    vram_olcum_aktif = True
except Exception as e:
    print("NVIDIA VRAM ölçümü başlatılamadı. Sadece Sistem RAM'i ölçülecek.")
    vram_olcum_aktif = False

def anlik_vram_gb():
    if vram_olcum_aktif:
        info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        return round(info.used / (1024 ** 3), 2)
    return 0.0

# DOSYA OKUMA VE AYARLAR
LOCAL_CLIENT = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def dosya_oku(dosya_adi):
    script_klasoru = os.path.dirname(os.path.abspath(__file__))
    dosya_yolu = os.path.join(script_klasoru, "Test_Verileri", dosya_adi)
    
    try:
        with open(dosya_yolu, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"HATA: '{dosya_yolu}' bulunamadı! 'Test_Verileri' klasöründe olduğundan emin ol."

# ALTIN STANDART (REFERANS) CEVAPLAR (İNGİLİZCE)

REFERANS_1 = """
[GEMINI CEVABI]:
1. Smurfing/Structuring Suspicion: The five consecutive 4,999.00 TRY Instant Transfers to CryptoExchange TR indicate an attempt to evade legal reporting thresholds by breaking down a larger sum (smurfing).
2. Money Laundering Suspicion: The 150,000.00 TRY OUTBOUND SWIFT to Cambodia is an unusually large, middle-of-the-night transaction to a high-risk jurisdiction.
3. Card Skimming / Impossible Travel Suspicion: The physical POS transaction in Istanbul at 14:00 followed by physical ATM withdrawals in Antalya at 15:15 represents an impossible geographical displacement, strongly indicating card skimming.

[CHATGPT CEVABI]:
1. Multiple instant transfers of 4,999 TRY to CryptoExchange within minutes are suspicious due to structuring behavior to avoid detection thresholds.
2. The 150,000 TRY outbound SWIFT transfer to Cambodia is suspicious due to high amount and unusual international destination.
3. Three consecutive 10,000 TRY ATM withdrawals in Antalya within minutes are suspicious due to rapid cash extraction in a different city, indicating possible fraud or account takeover.
"""

REFERANS_2 = """
[
  {
    "Customer_Name": "Elif Yilmaz",
    "Requested_Amount": 350000,
    "Monthly_Net_Income": 58000
  },
  {
    "Customer_Name": "Kemal Demir",
    "Requested_Amount": 900000,
    "Monthly_Net_Income": 120000
  },
  {
    "Customer_Name": "Zeynep Celik",
    "Requested_Amount": 450000,
    "Monthly_Net_Income": 100000
  }
]
"""

REFERANS_3 = """
[GEMINI CEVABI]:
MAIL 1 - VIOLATION: The employee sent a confidential VIP customer risk report to their personal external email, constituting a severe internal data leak.
MAIL 2 - NO VIOLATION: This is a standard, automated legal notification about the GDPR privacy policy update without any sensitive personal data.
MAIL 3 - VIOLATION: Sending the unmasked PII of 5,000 customers to a third-party vendor is a massive GDPR breach, compounded by the cybersecurity failure of sharing the file's password in the same email.

[CHATGPT CEVABI]:
MAIL 1: Sending confidential VIP customer financial data to a personal email is a clear internal data leakage and GDPR violation due to unauthorized external transfer.
MAIL 2: No violation, as it is an informational email about privacy policy with no sensitive data shared.
MAIL 3: Sharing unmasked personal data (ID numbers, addresses, spending habits) with a third-party vendor and including the password in the email is a severe GDPR and data security violation.
"""

senaryolar = [
    {
        "id": "Senaryo 1",
        "isim": "Hesap Hareketleri Analizi (İNGİLİZCE)",
        "metin": dosya_oku("scenario1_account_statement.txt"),
        "referans": REFERANS_1
    },
    {
        "id": "Senaryo 2",
        "isim": "Kredi Sözleşmesi Veri Çıkarımı (İNGİLİZCE)",
        "metin": dosya_oku("scenario2_credit_approval.txt"),
        "referans": REFERANS_2
    },
    {
        "id": "Senaryo 3",
        "isim": "DLP Mail İhlal Tespiti (İNGİLİZCE)",
        "metin": dosya_oku("scenario3_dlp_email_logs.txt"),
        "referans": REFERANS_3
    }
]

# EKA PUANI (BENZERLİK) HESAPLAMA
def zeka_puani_hesapla(referans_cevap, test_cevabi):
    if "BURAYA" in referans_cevap or len(referans_cevap) < 10:
        return 0.0
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([referans_cevap, test_cevabi])
        benzerlik = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(benzerlik * 10, 2)
    except:
        return 0.0

# ANA TEST FONKSİYONU
def modele_soru_sor(model_turu, senaryo):
    print(f"\n--- {senaryo['id']} ({senaryo['isim']}) ---")
    
    baslangic_ram = round(psutil.virtual_memory().used / (1024 ** 3), 2)
    baslangic_vram = anlik_vram_gb()
    baslangic_zamani = time.time()
    
    try:
        response = LOCAL_CLIENT.chat.completions.create(
            model="local-model",
            messages=[

                {"role": "system", "content": "You are a senior bank risk and data analyst. Follow the instructions in the provided text strictly."},
                {"role": "user", "content": senaryo["metin"]}
            ]
        )
        cevap = response.choices[0].message.content
    except Exception as e:
        cevap = f"LM Studio Bağlantı Hatası: {e}"
        
    bitis_zamani = time.time()
    bitis_ram = round(psutil.virtual_memory().used / (1024 ** 3), 2)
    bitis_vram = anlik_vram_gb()
    
    gecen_sure = round(bitis_zamani - baslangic_zamani, 2)
    ram_farki = round(bitis_ram - baslangic_ram, 2)
    
    return {
        "cevap": cevap,
        "sure_sn": gecen_sure,
        "ram_farki_gb": ram_farki, 
        "toplam_vram_gb": bitis_vram
    }

# PROGRAM YÖNETİMİ
def ana_program():
    sonuclar = []
    
    print("=== NATIVE LANGUAGE (İNGİLİZCE) PERFORMANS TESTİ ===")
    
    while True:
        model_adi = input("Test ettiğiniz modelin adını yazın (örn: Llama-3-ENG) veya çıkmak için 'bitti' yazın: ")
        
        if model_adi.lower() == 'bitti':
            break
            
        input(f"\nLütfen LM Studio'yu açın, {model_adi} modelini yükleyin, 'Start Server' butonuna basın ve ENTER'a basın...")
        
        for senaryo in senaryolar:
            sonuc = modele_soru_sor(model_adi, senaryo)
            zeka_puani = zeka_puani_hesapla(senaryo["referans"], sonuc["cevap"])
            
            sonuclar.append([model_adi, senaryo["id"], sonuc["sure_sn"], sonuc["ram_farki_gb"], sonuc["toplam_vram_gb"], zeka_puani, sonuc["cevap"]])
            
            print(f"Cevap Alındı! | Süre: {sonuc['sure_sn']} sn | Sistem RAM Farkı: {sonuc['ram_farki_gb']} GB | VRAM: {sonuc['toplam_vram_gb']} GB")
            print(f"Zeka Puanı (Referansa Benzerlik): {zeka_puani} / 10")
        
        print(f"\n✅ {model_adi} testi tamamlandı! EJECT (çıkar) yapmayı unutmayın.")

    dosya_adi = "Yapay_Zeka_Performans_Sonuclari_INGILIZCE.csv"
    with open(dosya_adi, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Model_Adi", "Senaryo", "Yanit_Suresi_Sn", "Sistem_RAM_Farki_GB", "Tuketilen_VRAM_GB", "Zeka_Puani_10_Uzerinden", "Verilen_Cevap"])
        writer.writerows(sonuclar)
    
    print(f"\nİNGİLİZCE TEST BİTTİ! Bütün veriler '{dosya_adi}' dosyasına kaydedildi.")

if __name__ == "__main__":
    ana_program()