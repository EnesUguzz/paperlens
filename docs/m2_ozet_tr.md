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

## Sonuçlar (`min_lead` 0.2)

| Sistem | answer F1 | evidence F1 | doğru makale | "hangisi?" sordu | sormadan yanlış makale |
|---|---|---|---|---|---|
| m1b_best | 0.440 | 0.491 | 1.00 | – | – |
| m2_corpus_rag | 0.216 | 0.181 | 0.46 | %0 | %49 |
| **m2_agent** | **0.304** | **0.294** | **0.68** | %6.8 | **%24** |

- **Agent tabandan anlamlı derecede iyi.** answer F1 farkı +0.088 [+0.052, +0.127], evidence
  F1 farkı +0.113. İki güven aralığı da 0'ı içermiyor.
- **Tavana göre −0.136.** Makaleyi doğru bulmak tek başına M1b seviyesine çıkarmıyor.
- **Güçlü yanlar:** makale adı geçtiğinde %100 doğru makaleye gidiyor. İlk turlarda doğru
  makale oranı %96; bunların sadece 17'sinde soru sordu. Aynı makaleden devam eden turlarda
  doğru makale oranı 0.26'dan 0.61'e çıkıyor.
- **Asıl zayıf yan:** makale adı geçmeden yapılan geçişlerin sadece %19'unu yakalıyor. Aktif
  makale varken belirsiz bir soru "aynı makale" sayılıyor; bir insan da öyle varsayardı.
  Kaçırılan bir geçiş sonraki turları da etkiliyor. Hiçbir eşik bunu çözmüyor: taramadaki en
  iyi değer bile 0.29.
- **Soru sormak puanı şişirmiyor.** Soru sorulmayan turlarda da answer F1 0.294. Değerlendirmede
  simüle kullanıcı her zaman doğru cevap verdiği için soru sormak "bedava". Gerçek kullanıcı
  için öyle değil.

## Eşiği (`min_lead`) nasıl seçtik?

Makale seçimi LLM'e bağlı olmadığı için bütün eşikleri saniyeler içinde, LLM'siz taradık
(`paperlens route-sweep`). Oylamaya 10 chunk girdiği için eşik 0.1'lik adımlarla anlam
taşıyor; 0.2 "birinci makale ikinciden en az 2 chunk önde" demek.

| min_lead | 0.0 | 0.1 | **0.2** | 0.3 | 0.4 | 0.5 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| doğru makale | 0.564 | 0.632 | **0.676** | 0.676 | 0.672 | 0.664 | 0.652 | 0.644 | 0.632 |
| "hangisi?" sordu | %0 | %4.4 | **%6.8** | %7.6 | %8.4 | %8.4 | %8.8 | %9.2 | %9.2 |
| adsız geçiş yakalandı | 0.29 | 0.21 | **0.19** | 0.17 | 0.14 | 0.13 | 0.10 | 0.07 | 0.06 |

- **0.2–0.4 arası bir plato:** doğru makale oranı neredeyse aynı. 0.2, platonun en az soru
  soran ve en çok geçiş yakalayan ucu.
- **0.8 reddedildi:** daha az doğru makale, daha çok soru, geçişlerin sadece %7'si
  yakalanıyor.
- **0.2 / 0.3 / 0.5'i LLM ile de koştuk:** answer F1 sırasıyla 0.304 / 0.290 / 0.292. Bu fark
  seçimde belirleyici olmadı. Makale seçimi aynı olan iki koşu bile 0.003 oynadı, ve eşiği
  birkaç değere bakarak seçtik. İkinci bir bağımsız agent'ın görüşü de buydu.
- **Not:** 0.2 ve 0.5 koşularının bir kısmında GPU başka bir yükle paylaşıldı. Bu yüzden o
  koşuların gecikme ölçümleri geçersiz; gecikme aynı kodla temiz koşan 0.3 koşusundan
  raporlandı. Cevaplar ve makale seçimi bundan etkilenmedi.

## Yol üstünde bulduğumuz hata

İlk koşuda answer F1 0.266'ydı. Sebebi: 14B model, "değiştirme" demesine rağmen standalone
soruların %74'ünü yeniden yazıyordu ve çoğuna önceki makalenin başlığını ekliyordu. Oylama
bu metinle yapılınca agent eski makaleden ayrılamıyordu.

Düzeltme:
- Oylama artık kullanıcının yazdığı metinle yapılıyor.
- `condense` yalnızca "it", "that", "them", "there" gibi tetikleyici kelimeler varsa
  çalışıyor. Liste config'de: `agent.condense.triggers`.
- Kullanıcının yazmadığı bir başlık eklenirse yeniden yazım reddediliyor.

Bunlardan sonra değiştirme oranı %1.2'ye düştü, answer F1 0.290'a çıktı (eşik o sırada
0.3'tü).

## Senin ayarlayabileceklerin

- **`agent.route.min_lead` (0.2):** yükseltirsen daha çok soru sorar ve daha çok adsız geçiş
  kaçırır; 0.1 ve altına düşürürsen genel sorular agent'ı aktif makaleden koparır. Denemek
  için önce `uv run --extra ml paperlens route-sweep` (LLM'siz, saniyeler), sonra seçtiğin
  değerle `paperlens eval`.
- **`agent.condense.triggers`:** hangi kelimeler takip sorusu sayılsın.
- **`conversations.signal_prob`:** kullanıcının makaleyi ne sıklıkla andığı.

Koşmak için: `make m2`. Demo için: `make converse`. Tarama için: `make route-sweep`.

## Sıradaki

- Daha zor bir makale-adı şablonu eklenecek (kısaltma ya da başka sözcüklerle anma).
- Adsız geçişleri yakalamak için oylamadan başka bir sinyal denenebilir (örneğin sorunun aktif
  makaleye ne kadar benzediği).
- `min_lead` 0.2 ile dondurulup test setinde tek koşu yapılacak (senin onayınla).
