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

# 1. DOSYA OKUMA VE AYARLAR
LOCAL_CLIENT = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def dosya_oku(dosya_adi):
    # Scriptin bulunduğu tam klasör yolunu otomatik bulur (Kurşun Geçirmez Taktik)
    script_klasoru = os.path.dirname(os.path.abspath(__file__))
    dosya_yolu = os.path.join(script_klasoru, "Test_Verileri", dosya_adi)
    
    try:
        with open(dosya_yolu, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"HATA: '{dosya_yolu}' bulunamadı!"

# ALTIN STANDART (REFERANS) CEVAPLAR

REFERANS_1 = """
[GEMINI CEVABI]:
1. Smurfing (Şirinler) Şüphesi: 06.04.2026 tarihinde Paribu'ya yapılan peş peşe 4.999 TL'lik transferler, yasal bildirim limitlerine takılmamak için tutarın bölünerek kripto borsasına aktarılması (smurfing) şüphesi taşır.
2. Kara Para Aklama Şüphesi: 09.04.2026'da Kamboçya'ya yapılan 150.000 TL'lik SWIFT işlemi, müşterinin normal profiline uymayan, olağandışı saatte ve yüksek riskli bir ülkeye yapılan yüklü bir transferdir.
3. Kart Kopyalama (Skimming) Şüphesi: 10.04.2026'da İstanbul'da POS işlemi yapıldıktan 1 saat 15 dakika sonra Antalya'da fiziksel ATM'den para çekilmesi coğrafi olarak imkansızdır ve kartın kopyalandığına işaret eder.

[CHATGPT CEVABI]:
1. Aynı gece art arda yapılan 4.999 TL’lik FAST transferleri, işlem limitine takılmamak için bölünmüş (structuring) şüpheli işlemlerdir.
2. Kamboçya’ya yapılan 150.000 TL SWIFT transferi, müşteri profiline göre olağandışı yüksek ve riskli bir yurtdışı işlemidir.
3. İstanbul’da işlem yaptıktan hemen sonra Antalya’da peş peşe ATM çekimleri yapılması, lokasyon uyumsuzluğu nedeniyle kart kopyalama/dolandırıcılık şüphesi taşır.
"""

REFERANS_2 = """
[
  {
    "Musteri_Adi": "Elif Yılmaz",
    "Talep_Edilen_Tutar": 350000,
    "Aylik_Net_Gelir": 58000
  },
  {
    "Musteri_Adi": "Kemal Demir",
    "Talep_Edilen_Tutar": 900000,
    "Aylik_Net_Gelir": 120000
  },
  {
    "Musteri_Adi": "Zeynep Çelik",
    "Talep_Edilen_Tutar": 450000,
    "Aylik_Net_Gelir": 100000
  }
]
"""

REFERANS_3 = """
[GEMINI CEVABI]:
MAİL 1 - İHLAL VAR: Banka personeli, VIP müşteri mevduat ve risk verilerini içeren gizli kurumsal dosyayı kendi kişisel dış e-posta adresine (hotmail) ileterek ciddi bir veri sızıntısı ve güvenlik ihlali yapmıştır.
MAİL 2 - İHLAL YOK: Bu e-posta sadece müşterilere KVKK politikalarındaki güncellemeyi bildiren yasal bir aydınlatma metnidir, herhangi bir kişisel veri içermemektedir.
MAİL 3 - İHLAL VAR: 5.000 müşterinin T.C. kimlik ve harcama verilerinin maskelenmeden üçüncü taraf bir firmaya gönderilmesi büyük bir KVKK ihlalidir; ayrıca parola güvenliği hiçe sayılarak dosyanın şifresi aynı mail içinde açıkça paylaşılmıştır.

[CHATGPT CEVABI]:
Mail 1: İhlal var — Şirket içi hassas müşteri verilerinin kişisel e-posta adresine gönderilmesi veri sızıntısı riskidir.
Mail 2: İhlal yok — Sadece bilgilendirme amaçlı genel bir KVKK duyurusu içeriyor.
Mail 3: İhlal var — Maskelenmemiş kişisel verilerin üçüncü tarafla paylaşılması ve şifrenin açıkça iletilmesi ciddi KVKK ve veri güvenliği ihlalidir.
"""

senaryolar = [
    {
        "id": "Senaryo 1",
        "isim": "Hesap Hareketleri Analizi (Samanlıkta İğne)",
        "metin": dosya_oku("senaryo1_hesap_hareketleri.txt"),
        "referans": REFERANS_1
    },
    {
        "id": "Senaryo 2",
        "isim": "Kredi Sözleşmesi Veri Çıkarımı (Matematiksel Zorluk)",
        "metin": dosya_oku("senaryo2_kredi_sozlesmesi.txt"),
        "referans": REFERANS_2
    },
    {
        "id": "Senaryo 3",
        "isim": "DLP Mail İhlal Tespiti (Yalancı Pozitif Tuzağı)",
        "metin": dosya_oku("senaryo3_mail_dokumu.txt"),
        "referans": REFERANS_3
    }
]

# ZEKA PUANI (BENZERLİK) HESAPLAMA
def zeka_puani_hesapla(referans_cevap, test_cevabi):
    if "BURAYA" in referans_cevap or len(referans_cevap) < 10:
        return 0.0
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([referans_cevap, test_cevabi])
        benzerlik = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(benzerlik * 10, 2) # 10 üzerinden puanlar
    except:
        return 0.0

#  ANA TEST FONKSİYONU
def modele_soru_sor(model_turu, senaryo):
    print(f"\n--- {senaryo['id']} ({senaryo['isim']}) ---")
    
    baslangic_ram = round(psutil.virtual_memory().used / (1024 ** 3), 2)
    baslangic_vram = anlik_vram_gb()
    baslangic_zamani = time.time()
    
    try:
        response = LOCAL_CLIENT.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": "Sen kıdemli bir banka risk ve veri analistisin. Verilen metindeki talimatlara harfiyen uy."},
                {"role": "user", "content": senaryo["metin"]}
            ]
        )
        cevap = response.choices[0].message.content
    except Exception as e:
        cevap = f"LM Studio Bağlantı Hatası: {e}\n(LM Studio'da Server'ı başlattığına emin misin?)"
        
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
    
    print("=== BANKACILIK YAPAY ZEKA PERFORMANS TESTİ ===")
    print("Bu script 'Test_Verileri' klasöründeki dosyaları okur ve LM Studio'daki modele gönderir.\n")
    
    while True:
        model_adi = input("Test ettiğiniz modelin adını yazın (örn: Qwen, Llama-3) veya çıkmak için 'bitti' yazın: ")
        
        if model_adi.lower() == 'bitti':
            break
            
        input(f"\nLütfen LM Studio'yu açın, {model_adi} modelini RAM'e yükleyin, 'Start Server' butonuna basın ve ardından klavyeden ENTER'a basın...")
        
        for senaryo in senaryolar:
            sonuc = modele_soru_sor(model_adi, senaryo)
            zeka_puani = zeka_puani_hesapla(senaryo["referans"], sonuc["cevap"])
            
            sonuclar.append([model_adi, senaryo["id"], sonuc["sure_sn"], sonuc["ram_farki_gb"], sonuc["toplam_vram_gb"], zeka_puani, sonuc["cevap"]])
            
            print(f"Cevap Alındı! | Süre: {sonuc['sure_sn']} sn | Sistem RAM Farkı: {sonuc['ram_farki_gb']} GB | VRAM: {sonuc['toplam_vram_gb']} GB")
            if zeka_puani > 0:
                print(f"Zeka Puanı (Referansa Benzerlik): {zeka_puani} / 10")
            else:
                print("Zeka Puanı: Hesaplanamadı (Referans cevaplar kodun içine girilmemiş)")
        
        print(f"\n✅ {model_adi} testi tamamlandı! Başka bir modele geçmeden önce LM Studio'dan bu modeli EJECT (çıkar) yapmayı unutmayın.")

    dosya_adi = "Yapay_Zeka_Performans_Sonuclari.csv"
    with open(dosya_adi, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Sütun başlıklarına Sistem_RAM_Farki_GB eklendi
        writer.writerow(["Model_Adi", "Senaryo", "Yanit_Suresi_Sn", "Sistem_RAM_Farki_GB", "Tuketilen_VRAM_GB", "Zeka_Puani_10_Uzerinden", "Verilen_Cevap"])
        writer.writerows(sonuclar)
    
    print(f"\nTEST SÜRECİ BİTTİ! Bütün veriler analiz için '{dosya_adi}' dosyasına kaydedildi.")

if __name__ == "__main__":
    ana_program()