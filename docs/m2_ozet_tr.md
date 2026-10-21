# M2 özeti (Türkçe)

Bu özette M2'de ne yaptığımızı, nasıl ölçtüğümüzü ve sonuçları bulacaksın. Notebook'un adım
adım anlatımı: `notebooks/m2_walkthrough_tr.ipynb`. İngilizce karar kaydı:
`docs/decisions/003-m2-agent.md`.

## Ne yaptık?

M1'de her soru makalesiyle birlikte geliyordu. M2'de kullanıcı makaleyi söylemeden sohbet
ediyor ve arada makale değiştiriyor. Bunu bir LangGraph agent'ı çözüyor
(`src/paperlens/graph/`):

| Adım | Ne yapar |
|---|---|
| `condense` | Takip sorusunu ("peki ne kadar büyük?") geçmişle tamamlar. Yalnızca geriye gönderme yapan bir kelime varsa çalışır. |
| `resolve_paper` | Soruda makalenin adı geçiyorsa o makaleyi seçer. Geçmiyorsa 281 makalede arar ve chunk'ları makaleye göre oylar. Açık fark varsa kazananı seçer; fark yoksa aktif makalede kalır, aktif makale de yoksa sorar. |
| `clarify` | "Hangisini kastediyorsunuz?" diye sorar ve durur (`interrupt`). Kullanıcı cevap verince devam eder. |
| `retrieve` / `generate` | Seçilen makalede M1b ayarlarıyla arar ve paragraf atıflı cevap üretir. |

Sohbet hafızasını checkpointer tutuyor: her sohbet bir thread. Her sohbet MLflow'a bir
trace olarak, her node da bir span olarak yazılıyor.

## Nasıl ölçtük?

- **Veri:** M1b'nin aynı 250 sorusu, 3 makalelik 57 sentetik sohbete dağıtıldı. Makale
  değişen turların yarısında makalenin adı geçiyor (3 farklı şablonla), yarısında geçmiyor.
- **Simüle kullanıcı:** agent "hangisi?" diye sorarsa doğru başlıkla cevap veriyor.
- **Karşılaştırılan üç sistem:**
  - `m1b_best`: makale verili, tavan.
  - `m2_corpus_rag`: aynı pipeline tüm makalelerde arıyor, hafıza yok; taban.
  - `m2_agent`.

## Sonuçlar

| Sistem | answer F1 | evidence F1 | doğru makale | "hangisi?" sordu | sormadan yanlış makale |
|---|---|---|---|---|---|
| m1b_best | 0.440 | 0.491 | 1.00 | – | – |
| m2_corpus_rag | 0.216 | 0.181 | 0.46 | %0 | %49 |
| **m2_agent** | **0.290** | **0.293** | **0.68** | %7.6 | **%22** |

- **Agent tabandan anlamlı derecede iyi.** answer F1 farkı +0.074 [+0.036, +0.111], evidence F1
  farkı +0.112. İki güven aralığı da 0'ı içermiyor.
- **Tavana göre −0.151.** Makaleyi doğru bulmak tek başına M1b seviyesine çıkarmıyor.
- **Güçlü yanlar:** makale adı geçtiğinde %100 doğru makaleye gidiyor. İlk turlarda doğru
  makale oranı %98; bunların sadece 19'unda soru sordu. Aynı makaleden devam eden turlarda
  doğru makale oranı 0.26'dan 0.63'e çıkıyor.
- **Bilinçli zayıf yan:** makale adı geçmeden yapılan geçişlerin çoğunu kaçırıyor (0.17).
  Aktif makale varken belirsiz bir soru "aynı makale" sayılıyor, bir insan da öyle varsayardı.
  Kaçırılan bir geçiş sonraki turları da etkiliyor.
- **Soru sormak puanı şişirmiyor.** Soru sorulmayan turlarda da answer F1 0.279.

## Yol üstünde bulduğumuz hata

İlk koşuda answer F1 0.266'ydı. Sebebi: 14B model, "değiştirme" demesine rağmen standalone
soruların %74'ünü yeniden yazıyordu ve çoğuna önceki makalenin başlığını ekliyordu. Oylama
bu metinle yapılınca agent eski makaleden ayrılamıyordu.

Düzeltme:
- Oylama artık kullanıcının yazdığı metinle yapılıyor.
- `condense` yalnızca "it", "that", "them", "there" gibi tetikleyici kelimeler varsa
  çalışıyor. Liste config'de: `agent.condense.triggers`.
- Kullanıcının yazmadığı bir başlık eklenirse yeniden yazım reddediliyor.

Bunlardan sonra değiştirme oranı %1.2'ye, answer F1 0.290'a çıktı.

## Senin ayarlayabileceklerin

- **`agent.route.min_lead` (0.3):** yükseltirsen daha az yanlış makale ama daha çok soru;
  düşürürsen daha çok geçiş yakalanır ama aynı makaledeki turlar başka makaleye kayabilir.
- **`agent.condense.triggers`:** hangi kelimeler takip sorusu sayılsın.
- **`conversations.signal_prob`:** kullanıcının makaleyi ne sıklıkla andığı.

Koşmak için: `make m2`, demo için: `make converse`.

## Sıradaki

- `min_lead` taraması yapılacak.
- Daha zor bir makale-adı şablonu eklenecek (kısaltma ya da başka sözcüklerle anma).
- Ayarlar dondurulduktan sonra test setinde tek koşu yapılacak (senin onayınla).
