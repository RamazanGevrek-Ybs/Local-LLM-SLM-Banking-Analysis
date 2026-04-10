import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11})

def model_ismi_temizle(isim):
    isim = str(isim).upper()
    if 'QWEN' in isim: return 'Qwen (1.5B)'
    elif 'PHI' in isim: return 'Phi-3 (3.8B)'
    elif 'LLAMA' in isim: return 'Llama-3 (8B)'
    return isim

def kisa_senaryo(isim):
    if '1' in isim: return 'Senaryo 1\n(Muhakeme/Fraud)'
    if '2' in isim: return 'Senaryo 2\n(JSON/Veri Çekme)'
    if '3' in isim: return 'Senaryo 3\n(DLP/Hukuki)'
    return isim

def puan_grafik_ciz(csv_dosyasi, dil_baslik, dosya_adi_soneki):
    try:
        df = pd.read_csv(csv_dosyasi)
        df['Model'] = df['Model_Adi'].apply(model_ismi_temizle)
        df['Kisa_Senaryo'] = df['Senaryo'].apply(kisa_senaryo)
        
        df_mean = df.groupby(['Kisa_Senaryo', 'Model'])['Zeka_Puani_10_Uzerinden'].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Kisa_Senaryo', y='Zeka_Puani_10_Uzerinden', hue='Model', 
                         data=df_mean, palette=['#3498db', '#e74c3c', '#2ecc71'])
        
        plt.title(f"Görev ve Model Bazlı Başarı Oranları - {dil_baslik} Testi")
        plt.xlabel("Banka Senaryoları")
        plt.ylabel("Ortalama Zeka Puanı (10 Üzerinden)")
        plt.ylim(0, 10.5) # Y eksenini 10'da sabitliyoruz
        
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(format(p.get_height(), '.2f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', xytext = (0, 8), textcoords = 'offset points',
                            fontsize=10, fontweight='bold')
        
        plt.axhline(y=10, color='gray', linestyle='--', alpha=0.7)
        plt.text(-0.4, 9.7, 'Bulut Modelleri (ChatGPT/Gemini) Referans Seviyesi (10)', color='gray', fontsize=9, fontstyle='italic')

        plt.legend(title='Yerel (Edge) Modeller')
        plt.tight_layout()
        
        resim_adi = f"Makale_Grafik_{dosya_adi_soneki}.png"
        plt.savefig(resim_adi, dpi=300)
        print(f"✓ {dil_baslik} grafiği kaydedildi: {resim_adi}")
        
    except FileNotFoundError:
        print(f"HATA: {csv_dosyasi} bulunamadı!")

puan_grafik_ciz('Yapay_Zeka_Performans_Sonuclari.csv', 'TÜRKÇE', 'TR_Detay')

puan_grafik_ciz('Yapay_Zeka_Performans_Sonuclari_INGILIZCE.csv', 'İNGİLİZCE', 'ENG_Detay')