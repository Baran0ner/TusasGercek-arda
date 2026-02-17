# Performans İyileştirmeleri Raporu

## Özet

Threading tabanlı parallelizasyon Windows'ta beklenen hızlanmayı sağlamadı (GIL sınırlaması).
Bunun yerine **algoritma seviyesinde iyileştirmeler** öneriyoruz.

## Test Sonuçları

**Test Ortamı:**
- CPU: 7 çekirdek
- Test: 36-ply kompozit (0°:12, 90°:8, ±45°:8)
- GA runs: 7

**Sonuçlar:**
```
Serial:   3.19s (score: 92.04/100)
Threading: 3.83s (score: 92.01/100)  → 0.83x (YAVAŞLADI!)
```

**Sebep:** Python GIL (Global Interpreter Lock) threading'i CPU-bound görevlerde etkisiz kılıyor.

---

## Önerilen İyileştirmeler

### 1. ✅ HEMEN UYGULANAB İLİR: Early Stopping İyileştirmeleri

**Mevcut durum:** Stagnation limit = 25 generation
**Öneri:** Adaptive early stopping

```python
# laminate_optimizer.py içinde
if generations_without_improvement >= stagnation_limit:
    # Eğer fitness yüksekse (>90), daha erken dur
    if best_fit > 90:
        break
    # Düşükse daha fazla bekle
    elif best_fit < 85 and generations_without_improvement < stagnation_limit * 1.5:
        continue
```

**Beklenen kazanım:** %10-20 hız artışı

---

### 2. ✅ HEMEN UYGULANABİLİR: Popülasyon Boyutu Optimizasyonu

**Mevcut:** 100-140 popülasyon
**Öneri:** 80-100 yeterli (daha az fitness hesabı)

```python
# Küçültülmüş popülasyon, aynı kalite
population_size = 80 if self.total_plies <= 40 else 100
```

**Beklenen kazanım:** %15-25 hız artışı

---

### 3. ✅ DOĞRULUK KORUMALI: Fitness Caching

**Fikir:** Aynı sequence'i tekrar hesaplama

```python
from functools import lru_cache

@lru_cache(maxsize=5000)
def _calculate_fitness_cached(self, seq_tuple):
    return self.calculate_fitness(list(seq_tuple))

# Kullanım:
fit, det = self._calculate_fitness_cached(tuple(sequence))
```

**Dikkat:** Sadece GA içinde aynı sequence tekrarlanıyorsa yararlı.
**Beklenen kazanım:** %5-15 (tekrar yüksekse)

---

### 4. 🔴 GELECEK: Multiprocessing (Windows Compatible)

**Sorun:** Flask app multiprocessing'le çakışıyor.
**Çözüm:** Sadece standalone optimizasyon scriptlerinde kullan.

**Implementasyon:**
```python
if __name__ == '__main__':
    # Multiprocessing sadece bu blokta çalışır
    with Pool(processes=4) as pool:
        results = pool.map(...)
```

**Faydası:** Batch optimizasyonlar için 3-4x hız

---

### 5. ✅ UZUN VADEL İ: NumPy Vektörizasyonu

**Mevcut kod zaten NumPy kullanıyor:**
```python
actual_spacings = np.diff(indices)  # ✓ İyi
std_dev = np.std(actual_spacings)    # ✓ İyi
```

**İlave optimizasyon:** Grouping hesabını vektörize et
```python
# Şu anki (loop):
for i in range(len(sequence) - 1):
    if sequence[i] == sequence[i+1]:
        count += 1

# Vektörize:
consecutive = np.diff(sequence) == 0
groups = np.where(consecutive)[0]
```

**Beklenen kazanım:** %5-10

---

## Önerilen Uygulama Sırası

### Faz 1: Hızlı Kazançlar (1 saat)
1. Popülasyon boyutunu 80-100'e düşür
2. Adaptive early stopping ekle
3. Test et: Hedef 25-30% hız artışı

### Faz 2: Caching (2 saat)
4. Fitness caching ekle
5. Benchmark: Gerçek kazancı ölç

### Faz 3: Algoritma İyileştirmeleri (1 gün)
6. NumPy vektörizasyonu (grouping, symmetry checks)
7. Mutation operatörlerini profille, yavaş olanları optimize et

### Faz 4: Parallelizasyon (Opsiyonel, batch işlemler için)
8. Standalone batch optimizer scripti (multiprocessing destekli)
9. Flask API'den ayrı çalışır

---

## Benchmark Hedefleri

**Mevcut:** 36-ply → 3-5 saniye
**Hedef (Faz 1):** 36-ply → 2-3 saniye (30% iyileştirme)
**Hedef (Faz 1+2):** 36-ply → 1.5-2.5 saniye (40% iyileştirme)
**Hedef (Multiproc):** 36-ply → 0.8-1.5 saniye (70% iyileştirme, sadece batch)

---

## Sonuç ve Öneriler

✅ **KABUL EDİLEN İYİLEŞTİRMELER:**
- Threading kodu eklendi (parallel=True parametresi) ama GIL yüzünden kazanç yok
- Kod yapısı modüler ve genişletilebilir

🔄 **SONRAK İ ADIMLAR:**
- Popülasyon boyutu ve early stopping optimizasyonları
- Fitness caching (dikkatli kullan, doğruluğu koru)
- NumPy vektörizasyonu

❌ **ÖNERİLMEYEN:**
- Flask içinde multiprocessing (Windows sorunları)
- Asyncio (CPU-bound task için uygunsuz)

---

## Kod Değişiklikleri

### Eklenen Özellikler:
1. `_run_single_ga()` metodu - tek GA run'ı paralel çalıştırmak için
2. `_multi_start_ga(parallel=True/False)` - threading desteği
3. ThreadPoolExecutor entegrasyonu

### Test Sonucu:
✓ Her iki mod da aynı kalitede sonuç veriyor (score diff < 0.1)
✓ Sequence'ler geçerli (simetrik, external ±45°)
✗ Speedup yok (GIL limiti)

---

## Tavsiye

**ÜRETİM ORTAMI İÇİN:**
- `parallel=False` kullan (daha basit, hata riski düşük)
- Algoritma-seviye optimizasyonlara odaklan (Faz 1-2)
- Multiprocessing'i sadece offline batch işlemler için sakla

**GELIŞTIRME ORTAMI İÇİN:**
- Threading kodunu tut (gelecekte Cython/numba ile hızlandırılabilir)
- Performans profiling ile darboğazları tespit et
- Benchmark scriptini düzenli çalıştır
