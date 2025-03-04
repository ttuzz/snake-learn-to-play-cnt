# Snake AI - Eğitim ve Oynatma

## Kaynaklar
- [YouTube Video](https://www.youtube.com/watch?v=rHaxVtZkREw)
- [GitHub Repository](https://github.com/muzafferkadir/snake-learn-to-play)

## Yapılan Güncellemeler
- Eğitime kaldığı yerden devam edilebilmesi için `ai_model` (save, load) ve `train` üzerinde eklemeler yapıldı.
- Google Colab üzerinden eğitilebilmesi için düzenlemeler yapıldı.
- Eğitim, son checkpoint dosyasını alarak devam ediyor.
- `model_virtualization.ipynb` dosyası, checkpointler arasındaki farkı göstermektedir.Checkpoint noktalarını collabu zerine yukleyerek png cıktısı alıp zip olarak pc'ye indiriyor
- `play_ai.py` en yüksek iterasyonu açarak oynatmaktadır.

## Checkpoint Sonuçları
| Model Dosyası               | Tur Sayısı | Ortalama Skor | Düşük Skor | Maksimum Skor |
|-----------------------------|-----------|---------------|------------|--------------|
| `model_checkpoint_1000.pth`  | 24        | 33.58         | -          | -            |
| `model_checkpoint_7700.pth`  | 22        | 39.64         | 17         | 62           |
| `model_checkpoint_8700.pth`  | 21        | 40.66         | 17         | 64           |
| `model_checkpoint_10200.pth` | 76        | 40.7          | 14         | 69           |
| `model_checkpoint_17900.pth` | 25        | 41.72         | 22         | 73           |