# SMS Spam Tespiti – Makine Öğrenmesi Projesi

Bu proje, **Python** kullanılarak geliştirilmiş bir **makine öğrenmesi** uygulamasıdır.  
Amaç, gelen bir SMS mesajının **spam (istenmeyen mesaj)** olup olmadığını otomatik olarak tespit etmektir.

Proje kapsamında farklı makine öğrenmesi algoritmaları uygulanmış ve performansları karşılaştırılmıştır.

---

## Veri Seti

Veri seti depoda `SMSSpamCollection` adıyla bulunmaktadır.

- Sekme (`\t`) ile ayrılmış, başlık satırı olmayan bir metin dosyası
- İki sütun: etiket (`ham` / `spam`) ve mesaj metni
- **5574** mesaj: **4827** `ham`, **747** `spam`

Kodda etiketler `ham = 0`, `spam = 1` olarak kodlanır.

---

## Yöntem

Tüm betiklerde ortak adımlar:

1. Veri okunur ve etiketler sayısallaştırılır.
2. Veri %80 eğitim / %20 test olarak ayrılır (`random_state=42`).
3. Mesajlar `TfidfVectorizer(stop_words="english")` ile TF-IDF vektörlerine dönüştürülür.
4. Model eğitilir, test verisi üzerinde sınıflandırma raporu yazdırılır.

### Modeller ve betikler

| Betik | Model | TF-IDF ayarları | Ek çıktılar |
|---|---|---|---|
| `sms.py` | `LogisticRegression()` | varsayılan | Eğitim/test doğruluk grafiği, konsoldan girilen SMS için etkileşimli tahmin |
| `son.py` | `LogisticRegression(class_weight="balanced", max_iter=1000)` | `max_df=0.9`, `min_df=5` | Doğruluk grafiği, ROC eğrisi + AUC, confusion matrix |
| `svm.py` | `SVC(kernel="linear", class_weight="balanced", probability=True)` | `max_df=0.9`, `min_df=5` | Doğruluk grafiği, ROC eğrisi + AUC, confusion matrix |
| `knn.py` | `KNeighborsClassifier(n_neighbors=5)` | `max_df=0.9`, `min_df=5` | Doğruluk grafiği, ROC eğrisi + AUC, confusion matrix |
| `randomforest.py` | `RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)` | `max_df=0.9`, `min_df=5` | Doğruluk grafiği, ROC eğrisi + AUC, confusion matrix |

Confusion matrix hem sayısal hem de satır bazında normalize edilmiş olarak `seaborn` ısı haritası şeklinde çizdirilir.

`gpu-test.py` ise modelleme ile ilgili değildir; TensorFlow'un CUDA desteğiyle derlenip derlenmediğini ve kullanılabilir GPU olup olmadığını kontrol eden yardımcı bir betiktir.

---

## Sonuçlar

Depoda kaydedilmiş bir çıktı bulunmamaktadır. Sınıflandırma raporlarını, AUC skorlarını ve grafikleri görmek için ilgili betiği çalıştırın.

---

## Proje Yapısı

```
.
├── SMSSpamCollection   # Veri seti
├── sms.py              # Lojistik regresyon + etkileşimli SMS tahmini
├── son.py              # Lojistik regresyon (dengeli sınıf ağırlığı) + ROC / confusion matrix
├── svm.py              # SVM (lineer çekirdek)
├── knn.py              # K-En Yakın Komşu
├── randomforest.py     # Random Forest
├── gpu-test.py         # TensorFlow GPU kontrolü
├── requirements.txt
└── README.md
```

---

## Çalıştırma

```bash
pip install -r requirements.txt
python svm.py          # veya knn.py, randomforest.py, sms.py
```

- Betikler `SMSSpamCollection` dosyasını çalışma dizininde arar; komutları depo klasörü içinden çalıştırın.
- **Not:** `son.py` veriyi `veri/SMSSpamCollection` yolundan okur. Depoda `veri/` klasörü bulunmadığından bu betik mevcut haliyle dosyayı bulamaz; çalıştırmak için dosyanın bu yolda bulunması gerekir.
- `sms.py` çalıştırıldığında sınıflandırma raporundan sonra konsoldan SMS girilmesini bekler; çıkmak için `q` yazın. Doğruluk grafiği döngüden çıktıktan sonra gösterilir.

---

## Gereksinimler

- Python 3
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow (yalnızca `gpu-test.py` için gereklidir)
