# M1b özeti: neler eklendi, ne öğrendik

M1b'nin sorusu: **M1a'daki baseline RAG'i hangi değişiklikler gerçekten iyileştirir?** Her değişiklik baseline'dan tek bir ayar farkıyla, aynı 250 QASPER sorusunda ölçüldü. Kod üzerinde adım adım görmek için: `notebooks/m1b_walkthrough_tr.ipynb`. Resmî karar kaydı (İngilizce): `docs/decisions/002-m1b-pipeline.md`.

## Tek cümlelik sonuç

Cevap doğruluğunu (answer_f1) kanıtlanmış biçimde artıran bir değişiklik bulunmadı. **Kanıt göstermeyi** (evidence_f1) ise iki değişiklik açıkça artırdı: **bölüm chunking** ve **paragraf atfı**. İkisinin birleşimi M2'nin karşılaştırma noktası olacak aday pipeline.

## İki temel kavram

### Bölüm chunking (`chunking.strategy: section`)

Chunking, makaleyi aramada kullanılacak parçalara bölmek demek.

- **fixed (baseline):** Metin 200 kelimelik pencerelerle kesilir, pencereler 40 kelime örtüşür. Paragraf sınırına bakılmaz. Bir chunk bir paragrafın ortasında başlayıp başka bir paragrafın ortasında bitebilir.
- **section (aday):** Bütün paragraflar 200 kelimeye kadar aynı chunk'a doldurulur ve bölüm sınırı asla geçilmez. Her chunk tam paragraflardan oluşur. Tek bir paragraf 200 kelimeden uzunsa yalnızca o bölünür.

Örnek makalede (soru 16, SnapCaptions) fixed chunking 43 paragrafı birden fazla chunk'a dağıttı; section chunking yalnızca 2 paragrafı böldü. Gold paragraf, fixed'de formüllerle dolu "Modality Attention" bölümünün sonuna yapışık geliyor. Section'da ise kendi bölümünün ("SnapCaptions Dataset") tek ve tam paragrafı olarak geliyor.

### Paragraf atfı (`generation.cite_unit: paragraph`)

Model cevabını verirken kanıtını prompt'taki numaralarla gösterir. Evidence-F1, atıf yapılan **paragraf id'lerinin** gold kanıt paragraflarıyla örtüşmesini ölçer.

- **chunk atfı (baseline):** Her chunk tek bir numara alır (`[1]`…`[5]`). Model `[1]` dediğinde o chunk'taki bütün paragraflar kanıt sayılır.
- **paragraf atfı (aday):** Her chunk'ın içindeki her paragraf parçası ayrı numaralanır. Model tam olarak kanıt olan paragrafı gösterebilir.

Soru 16'da üç kurulum da doğru cevabı verdi ("10K user-generated image (snap) and textual caption pairs"):

| kurulum | atıf | kanıt sayılan paragraflar | evidence_f1 |
|---|---|---|---|
| baseline (fixed + chunk atfı) | `c15` | 5 paragraf parçası, 1'i gold | 0.33 |
| fixed + paragraf atfı | `[5]` | yalnız gold paragraf | 1.00 |
| aday (section + paragraf atfı) | `[n]` | yalnız gold paragraf | 1.00 |

Retrieval iki atıf modunda da aynı. Değişen sadece prompt'taki numaralandırma ve atfın paragraf id'lerine çevrilmesi.

### Evidence-F1 tavanı

Tavan, getirilen chunk'larla kusursuz atıf yapılsaydı ulaşılabilecek en iyi Evidence-F1 değeri. Chunk atfında tavan chunk genişliğiyle sınırlı: gürültülü bir chunk'ı atfetmek precision'ı düşürür. Paragraf atfında tavan yalnızca retrieval'a bağlı. Baseline'da tavan .524, adayda .770.

## M1b'de eklenenler

| # | Ne | Nerede | Config |
|---|---|---|---|
| 1 | Resmî QASPER evaluator ile eşlik testi (bizim F1'imiz resmî script ile aynı mı) | `tests/official/`, `tests/test_official_parity.py` | — |
| 2 | Yeni metrikler: gold_recall, cevap uzunluğu, doc_hit_rate, atıf yapılan paragraflar | `evals/records.py` | — |
| 3 | Daha kısa cevap isteyen prompt v2 | `prompts/answer_v2.md` | `generation.prompt_template` |
| 4 | LLM-as-judge: kurallar önce, kalanlar qwen3:14b'ye | `evals/judge.py`, `prompts/judge.md` | `judge.*` |
| 5 | QueryPlan: aramanın hangi metinlerle yapılacağını aramadan önce belirleyen yapı; tek yerden pipeline kurulumu | `retrieval/query.py`, `pipelines/factory.py` | — |
| 6 | Cross-encoder reranker (bge-reranker-v2-m3) | `retrieval/reranker.py` | `rerank.*`, `retrieval.candidate_k` |
| 7 | Multi-query (soruyu 3 şekilde yeniden yazma) ve HyDE (uydurma cevap paragrafıyla arama) | `retrieval/query.py`, `prompts/multi_query.md`, `prompts/hyde.md` | `query.*` |
| 8 | Paragraf atfı: her chunk içindeki paragraf parçaları (`Chunk.segments`) | `domain.py`, `ingestion/chunkers.py`, `pipelines/baseline.py` | `generation.cite_unit` |
| 9 | Semantik chunking: cümle embedding'leri arasındaki anlam uzaklığına göre bölme | `ingestion/chunkers.py` | `chunking.strategy: semantic`, `chunking.semantic_*` |
| 10 | arXiv'den PDF indirme ve Docling ile ayrıştırma | `ingestion/pdf.py`, `ingestion/docling_parser.py`, `paperlens prepare-pdfs` | `data.pdf_*` |
| 11 | Gold kanıtı PDF paragraflarına hizalama (rapidfuzz) | `ingestion/align.py`, `ingestion/sources.py` | `data.text_source`, `data.align_*` |
| 12 | Deneyler, `make m1b`, README, karar kaydı | `experiments/configs/m1b/`, `m1b_best.yaml`, `m1b_section_best.yaml` | — |

## Sonuçlar (250 soru)

Judge değerleri son inceleme turundaki kural düzeltmesinden sonraki değerlerdir. Δ sütunları baseline'a göre farktır; köşeli parantezde paired bootstrap ile %95 güven aralığı (GA) var. **GA 0'ı içeriyorsa fark bu örneklemde kanıtlanmamıştır.**

| koşu | answer_f1 | Δ answer_f1 [%95 GA] | Δ evidence_f1 [%95 GA] | judge | ev_f1 / tavan | R@5 | p95 ms |
|---|---|---|---|---|---|---|---|
| baseline | .419 | — | — | .500 | .297 / .524 | .703 | 1851 |
| prompt_v2 | .421 | +.003 [−.019, +.026] | −.005 [−.023, +.013] | .528 | .292 / .524 | .703 | 1624 |
| rerank | .387 | −.032 [−.069, +.005] | −.012 [−.043, +.018] | .496 | .285 / .526 | .731 | 2897 |
| hybrid_rerank | .387 | −.032 [−.069, +.005] | −.012 [−.043, +.018] | .496 | .285 / .526 | .731 | 2905 |
| multi_query | .427 | +.008 [−.027, +.045] | −.026 [−.063, +.010] | .496 | .271 / .505 | .672 | 2917 |
| hyde | .414 | −.005 [−.034, +.025] | −.024 [−.050, +.000] | .484 | .273 / .529 | .713 | 4124 |
| cite_paragraph | .421 | +.002 [−.018, +.023] | **+.158 [+.120, +.198]** | .524 | .455 / .787 | .703 | 1809 |
| chunk_semantic | .427 | +.009 [−.030, +.045] | **+.116 [+.071, +.160]** | .548 | .413 / .658 | .667 | 1501 |
| chunk_section_200 (M1a) | .442 | +.024 [−.013, +.058] | **+.128 [+.079, +.172]** | .556 | .425 / .664 | .665 | 1451 |
| source_docling | .404 | −.015 [−.057, +.025] | **−.044 [−.083, −.007]** | .532 | .253 / .484 | .660 | 3339 |
| m1b_best (semantic + paragraf) | .435 | +.016 [−.024, +.053] | **+.173 [+.122, +.224]** | .540 | .469 / .767 | .667 | 1538 |
| **m1b_section_best (section + paragraf)** | **.440** | +.022 [−.015, +.056] | **+.194 [+.144, +.242]** | **.568** | **.491** / .770 | .665 | 1642 |

20'den fazla karşılaştırma yapıldı. Bu kadar karşılaştırmada birkaç tanesi şans eseri "anlamlı" çıkabilir. Bu yüzden büyük ve tutarlı farklara güveniyoruz. evidence_f1'deki +.12–.19'luk kazançlar bu türden; source_docling'deki −.044 ise sınırda.

## Elenenler ve nedenleri

- **Reranker (rerank, hybrid_rerank):** Recall@5 .703'ten .731'e çıktı ama answer_f1 .419'dan .387'ye düştü ve her soruya yaklaşık 1 saniye eklendi. İki koşunun birebir aynı çıkmasının nedeni: retrieval tek makale içinde yapılıyor ve bir makalede ortalama 23.8 chunk var. 30 adaylık havuz zaten neredeyse bütün makaleyi kapsıyor; 281 makalenin 234'ünde chunk sayısı 30 veya altında. Yani havuzu hangi yöntemin doldurduğu fark etmiyor. Bütün korpusta arama yapılacak M2'de yeniden denenmeli.
- **Multi-query ve HyDE:** Kazanç yok, gecikme 1,5–2 kat arttı. Tek makale içinde yeniden yazılmış sorguların bulacağı yeni bir aday pek yok. Bunlar da M2'de korpus modunda tekrar denenecek.
- **prompt_v2:** Cevaplar daha kısa (15.5 → 12.1 kelime) ama gerçekten cevapsız soruları tanıma oranı (refusal_recall) .270'ten .216'ya düştü.
- **Semantik chunking:** İyi çalıştı ama M1a'daki bölüm chunking'den her metrikte biraz geride kaldı. Üstelik her cümleyi embed etmek gerekiyor, yani daha pahalı ve daha karmaşık.
- **Docling PDF:** Gold kanıtın %90.4'ü PDF paragraflarına hizalanabildi. Kalan yaklaşık %10'luk kanıt hiçbir zaman bulunamayacağı için evidence_f1 düştü. 278 makale ayrıştırıldı, 1 makalenin PDF'i arXiv'de yok. Gerçekçi bir senaryo olarak değerli, ama ölçüm için QASPER metniyle kalmak doğru.

## Birleşim etkileri toplanmıyor

Değişikliklerin tek tek kazançları birleşince toplanmıyor. Paragraf atfı tek başına evidence_f1'e +.158, semantik chunking tek başına +.116 ekliyor; ikisi birlikte +.173 ekliyor, toplam olan +.274 değil. Bölüm chunking ile birleşim de benzer: +.194. Nedeni şu: daha iyi chunking ile paragraf atfı aynı sorunu, yani gürültülü chunk'ı atfetmeyi çözüyor. Biri çözünce diğerinin düzelteceği şey azalıyor.

## Son incelemede bulunan ve düzeltilen hatalar

- **İndeks anahtarı:** Semantik chunking'in ayarları bütün indeks anahtarlarına girmişti. Bu yüzden eski indeksler bulunamıyor ve `paperlens rescore` çöküyordu. Artık bu ayarlar yalnızca semantik stratejide anahtara giriyor.
- **Docling indeks anahtarı:** Hangi PDF'lerin ve hangi içerikle ayrıştırıldığı artık anahtara giriyor. Önceden PDF seti değişince eski indeks fark edilmeden yeniden kullanılabiliyordu.
- **Judge kuralı:** Annotatörlerin bir kısmı "cevaplanamaz", bir kısmı cevap verdiğinde, cevaplanabilir referansla eşleşen bir cevap da otomatik olarak yanlış sayılıyordu. Artık bu durum LLM'e gidiyor. Bütün koşuların judge accuracy'si 0.016–0.032 arttı.
- **Judge önbelleği:** Kurallar artık önbellekteki LLM kararlarından önce uygulanıyor. `judge` ve `rescore` komutları aynı sırayı izliyor.

## M2'ye ne taşınıyor

- **Karşılaştırma noktası (baseline):** bölüm chunking + paragraf atfı (`experiments/configs/m1b_section_best.yaml`). Kesin karar kaydı senin onayınla `m1b_best` olarak güncellenecek.
- **M2'nin konusu:** LangGraph ile agentic pipeline ve korpus modu, yani soru hangi makalede bilinmeden 281 makalenin hepsinde arama. Reranker, multi-query ve HyDE orada yeniden denenecek, çünkü aday havuzu artık tek bir makale değil.

## Komutlar

```bash
make notebook                 # defterleri JupyterLab'de aç (http://localhost:8888)
uv run paperlens trace --config experiments/configs/m1b_section_best.yaml --index 16
uv run paperlens summarize --out evals/results/SUMMARY.md
make m1b                      # bütün M1b deneylerini baştan koş (GPU + Ollama, saatler sürer)
```
