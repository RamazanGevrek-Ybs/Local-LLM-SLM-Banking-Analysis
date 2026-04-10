import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

def model_ismi_temizle(isim):
    isim = str(isim).upper()
    if 'QWEN' in isim: return 'Qwen (1.5B)'
    elif 'PHI' in isim: return 'Phi-3 (3.8B)'
    elif 'LLAMA' in isim: return 'Llama-3 (8B)'
    return isim

def verileri_hazirla():
    try:
        df_tr = pd.read_csv('Yapay_Zeka_Performans_Sonuclari.csv')
        df_tr['Dil'] = 'Türkçe (TR)'
        
        df_eng = pd.read_csv('Yapay_Zeka_Performans_Sonuclari_INGILIZCE.csv')
        df_eng['Dil'] = 'İngilizce (ENG)'
        
        df_all = pd.concat([df_tr, df_eng], ignore_index=True)
        df_all['Temiz_Model'] = df_all['Model_Adi'].apply(model_ismi_temizle)
        
        df_all['Senaryo'] = df_all['Senaryo'].replace({
            'Senaryo 1': 'S1: Hesap (Muhakeme)',
            'Senaryo 2': 'S2: Kredi (JSON/Mat)',
            'Senaryo 3': 'S3: Mail (DLP/Sızıntı)'
        })
        return df_all
    except FileNotFoundError as e:
        print(f"HATA: CSV dosyaları bulunamadı! Lütfen aynı klasörde olduklarından emin olun.\nDetay: {e}")
        return None

df = verileri_hazirla()

if df is not None:
    print("Veriler başarıyla okundu! Grafikler çiziliyor...")
    
    plt.figure(figsize=(8, 6))
    vram_df = df.groupby('Temiz_Model')['Tuketilen_VRAM_GB'].mean().reset_index()
    vram_df = vram_df.sort_values('Tuketilen_VRAM_GB')
    
    ax = sns.barplot(x='Temiz_Model', y='Tuketilen_VRAM_GB', data=vram_df, palette='viridis')
    plt.title("Modellerin Uç Cihazlarda (Edge) VRAM Tüketimi Karşılaştırması")
    plt.xlabel("Yerel Modeller (Parametre Boyutu)")
    plt.ylabel("Tüketilen VRAM (GB)")
    
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig("Makale_Grafik1_VRAM.png", dpi=300)
    print("✓ Grafik 1 (VRAM Tüketimi) kaydedildi: Makale_Grafik1_VRAM.png")

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Senaryo', y='Zeka_Puani_10_Uzerinden', hue='Dil', data=df, errorbar=None, palette=['#e74c3c', '#3498db'])
    plt.title("Dil Bariyerinin Yapay Zeka Başarısına Etkisi (Zeka Puanları)")
    plt.xlabel("Test Senaryoları")
    plt.ylabel("TF-IDF Benzerlik Puanı (10 Üzerinden)")
    plt.legend(title='Test Dili')
    
    plt.tight_layout()
    plt.savefig("Makale_Grafik2_Zeka_Puanlari.png", dpi=300)
    print("✓ Grafik 2 (Zeka Puanları) kaydedildi: Makale_Grafik2_Zeka_Puanlari.png")

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Temiz_Model', y='Yanit_Suresi_Sn', hue='Dil', data=df, errorbar=None, palette=['#e67e22', '#2ecc71'])
    plt.title("Tokenizasyon Verimsizliği: Türkçe vs İngilizce Ortalama Yanıt Süreleri")
    plt.xlabel("Yerel Modeller")
    plt.ylabel("Yanıt Süresi (Saniye)")
    plt.legend(title='Test Dili')
    
    plt.tight_layout()
    plt.savefig("Makale_Grafik3_Yanit_Suresi.png", dpi=300)
    print("✓ Grafik 3 (Yanıt Süreleri) kaydedildi: Makale_Grafik3_Yanit_Suresi.png")