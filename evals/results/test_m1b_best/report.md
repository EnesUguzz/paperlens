# Run `test_m1b_best`

No reference run: showing absolute metrics only.

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.458 | - | - |
| answer_f1.abstractive | 0.168 | - | - |
| answer_f1.extractive | 0.551 | - | - |
| answer_f1.gold.abstractive | 0.317 | - | - |
| answer_f1.gold.extractive | 0.504 | - | - |
| answer_f1.gold.unanswerable | 0.557 | - | - |
| answer_f1.gold.yes_no | 0.370 | - | - |
| answer_f1.unanswerable | 0.630 | - | - |
| answer_f1.yes_no | 0.357 | - | - |
| answer_words_mean | 12.692 | - | - |
| completion_tokens_mean | 47.960 | - | - |
| context_words_mean | 692.988 | - | - |
| data.questions_dropped | 0.000 | - | - |
| data.unmatched_evidence_rate | 0.039 | - | - |
| error_rate | 0.000 | - | - |
| evidence_f1 | 0.530 | - | - |
| evidence_f1_ceiling | 0.741 | - | - |
| false_refusal_rate | 0.088 | - | - |
| gold_recall | 0.523 | - | - |
| index.num_chunks | 12142.000 | - | - |
| latency_p50_ms | 986.683 | - | - |
| latency_p95_ms | 1929.805 | - | - |
| n.abstractive | 52.000 | - | - |
| n.extractive | 143.000 | - | - |
| n.gold.abstractive | 51.000 | - | - |
| n.gold.extractive | 139.000 | - | - |
| n.gold.unanswerable | 33.000 | - | - |
| n.gold.yes_no | 27.000 | - | - |
| n.unanswerable | 27.000 | - | - |
| n.yes_no | 28.000 | - | - |
| n_questions | 250.000 | - | - |
| prompt_tokens_max | 2332.000 | - | - |
| prompt_tokens_mean | 1785.672 | - | - |
| query_error_rate | 0.000 | - | - |
| refusal_recall | 0.515 | - | - |
| retrieval.doc_hit_rate | 1.000 | - | - |
| retrieval.mrr | 0.503 | - | - |
| retrieval.n | 226.000 | - | - |
| retrieval.ndcg_at_1 | 0.376 | - | - |
| retrieval.ndcg_at_3 | 0.475 | - | - |
| retrieval.ndcg_at_5 | 0.510 | - | - |
| retrieval.recall_at_1 | 0.345 | - | - |
| retrieval.recall_at_3 | 0.569 | - | - |
| retrieval.recall_at_5 | 0.660 | - | - |

## Lowest-scoring questions

- **What language is this dataset in?** (`00c443f8d32d6baf7c7cea8f4ca9fa749532ccfd`, answer_f1 0.00)
  - gold: English | English
  - predicted: English-language
  - top context: [The Phoenix Dataset] The Phoenix dataset is an attempt to take both the new advances in event data described above, along with decades of knowledge regarding best practices, in order to create a new iteration of event data. The dataset makes use of 450 English-language news sites, which are each sc

- **Do they compare to abstractive summarization methods?** (`0f9c1586f1b4b531fa4fd113e767d06af90b1ae8`, answer_f1 0.00)
  - gold: Yes | Yes | Yes
  - predicted: We observe that nn-we consistently outperforms the purely abstractive model.
  - top context: [Introduction] rush2015neural propose a neural attention model for abstractive sentence compression which is trained on pairs of headlines and first sentences in an article. In contrast, our model summarizes documents rather than individual sentences, producing multi-sentential discourse. A major ar

- **What categories of fake news are in the dataset?** (`1100e442e00c9914538a32aca7af994ce42e1b66`, answer_f1 0.00)
  - gold: Satire/Parody, Misleading Content, Imposter Content, False Connection | Satire/Parody, Misleading Content, Imposter Content, False Connection
  - predicted: 2-way, 3-way, and 5-way classification categories
  - top context: [Related Work] A variety of datasets for fake news detection have been published in recent years. These are listed in Table TABREF1, along with their specific characteristics. When comparing these datasets, a few trends can be seen. Most of the datasets are small in size, which can be ineffective fo

- **Which datasets do they experiment on?** (`11ef46187a5bf15e89d63220fdeaecbeb92d818e`, answer_f1 0.00)
  - gold: Indian Language Corpora Initiative (ILCI) corpus, OpenSubtitles2016 section of the OPUS corpus | multilingual Indian Language Corpora Initiative (ILCI) corpus BIBREF25, OpenSubtitles2016 section of the OPUS corpus collection BIBREF26
  - predicted: Unanswerable
  - top context: [Experimental Setup] We trained translation systems over the following basic units: character, morpheme, word, orthographic syllable and BPE unit. In this section, we summarize the languages and writing systems chosen for our experiments, the datasets used and the experimental configuration of our t

- **How is cluster purity measured?** (`130d73400698e2b3c6860b07f2e957e3ff022d48`, answer_f1 0.00)
  - gold: V-measure | V-measure BIBREF16
  - predicted: Low values of cluster purity indicate that biased words are less clustered following bias mitigation.
  - top context: [Results ::: Indirect bias] Figure FIGREF30 shows the V-measures of the clusters of the most biased words in Wikipedia for each embedding. Gigaword patterns similarly (see appendix). Figure FIGREF31 shows example tSNE projections for the Gigaword embeddings (“$\mathrm {V}$” refers to their V-measure
