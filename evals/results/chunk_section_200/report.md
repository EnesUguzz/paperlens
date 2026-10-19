# Run `chunk_section_200`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `chunking.chunk_overlap` | 40 | 0 |
| `chunking.strategy` | fixed | section |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.442 | 0.419 | +0.024 |
| answer_f1.abstractive | 0.249 | 0.225 | +0.024 |
| answer_f1.extractive | 0.522 | 0.473 | +0.049 |
| answer_f1.gold.abstractive | 0.332 | 0.302 | +0.030 |
| answer_f1.gold.extractive | 0.510 | 0.464 | +0.045 |
| answer_f1.gold.unanswerable | 0.387 | 0.411 | -0.024 |
| answer_f1.gold.yes_no | 0.426 | 0.464 | -0.038 |
| answer_f1.unanswerable | 0.400 | 0.458 | -0.058 |
| answer_f1.yes_no | 0.452 | 0.486 | -0.034 |
| completion_tokens_mean | 49.180 | 51.800 | -2.620 |
| context_words_mean | 689.316 | 998.848 | -309.532 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.425 | 0.297 | +0.128 |
| evidence_f1_ceiling | 0.663 | 0.471 | +0.192 |
| false_refusal_rate | 0.061 | 0.042 | +0.019 |
| index.num_chunks | 8360.000 | 6676.000 | +1684.000 |
| latency_p50_ms | 943.976 | 1042.795 | -98.818 |
| latency_p95_ms | 1451.214 | 1850.713 | -399.500 |
| n.abstractive | 55.000 | 55.000 | +0.000 |
| n.extractive | 143.000 | 144.000 | -1.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 25.000 | 24.000 | +1.000 |
| n.yes_no | 27.000 | 27.000 | +0.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2394.000 | 2577.000 | -183.000 |
| prompt_tokens_mean | 1791.016 | 2228.868 | -437.852 |
| refusal_recall | 0.243 | 0.270 | -0.027 |
| retrieval.mrr | 0.484 | 0.552 | -0.068 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.322 | 0.409 | -0.087 |
| retrieval.ndcg_at_3 | 0.423 | 0.492 | -0.069 |
| retrieval.ndcg_at_5 | 0.474 | 0.527 | -0.053 |
| retrieval.recall_at_1 | 0.264 | 0.335 | -0.070 |
| retrieval.recall_at_3 | 0.524 | 0.603 | -0.079 |
| retrieval.recall_at_5 | 0.665 | 0.703 | -0.038 |

## Biggest regressions

- **What is the size of bioScope corpus?** (`45be665a4504f0c7f458cf3f75a95d5a75eefd42`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Unanswerable
  - predicted: BioScope Abstracts sub corpora (BA): 11871 sentences, BioScope Full Papers sub corpora (BF): 2670 sentences
  - top context: [Introduction] This task was the CoNLL-2010 Shared Task (BIBREF0), which had 3 different subtasks. Task 1B was speculation cue detection on the BioScope Corpus, Task 1W was weasel identification from Wikipedia articles, and Task 2 was speculation scope resolution from the BioScope Corpus. For each t

- **Is fine-tuning required to incorporate these embeddings into existing models?** (`b06512c17d99f9339ffdab12cedbc63501ff527e`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: we can just use Dolores embeddings as a drop-in replacement. We just initialize the corresponding embedding layer with Dolores embeddings.
  - top context: [Introduction] Knowledge Embeddings learned using Dolores can easily be used as input representations for predictive models on knowledge graphs. More importantly, when existing predictive models use input representations for entities and relations, we can easily replace those representations with Do

- **what evaluation metrics are reported?** (`d9412dda3279729e95fcb35cbed09e61577a896e`, answer_f1 1.00 -> 0.00)
  - gold: precision, recall, F1 and accuracy | precision, recall, F1 , accuracy 
  - predicted: P/R/F1, Acc, MaxFreq
  - top context: [Case Study] We visualize the heat maps of attention results. As shown in Fig. FIGREF47 , deeper background color represents larger attention score. The attention score is calculated with Eq. ( EQREF15 ). We take the average of the resulting INLINEFORM0 attention matrix over the time dimension to ob

- **what dataset was used?** (`e48e750743aef36529fbea4328b8253dbe928b4d`, answer_f1 1.00 -> 0.00)
  - gold: WASSA-2017 Shared Task on Emotion Intensity
  - predicted: dev data set BIBREF19, train and dev sets
  - top context: [Parameter Optimization] In order to find the optimal parameter values for the EmoInt system, an extensive grid search was performed through the scikit-Learn framework over all subsets of the training set (shuffled), using stratified 10-fold cross validation and optimizing the Pearson's Correlation 

- **What is the size of SFU Review corpus?** (`e86130c5b9ab28f0ec539c2bed1b1ae9efb99b7d`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Unanswerable
  - predicted: SFU Review Corpus (SFU)
  - top context: [Introduction] This task was the CoNLL-2010 Shared Task (BIBREF0), which had 3 different subtasks. Task 1B was speculation cue detection on the BioScope Corpus, Task 1W was weasel identification from Wikipedia articles, and Task 2 was speculation scope resolution from the BioScope Corpus. For each t
