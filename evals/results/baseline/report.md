# Run `baseline`

No reference run: showing absolute metrics only.

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.419 | - | - |
| answer_f1.abstractive | 0.225 | - | - |
| answer_f1.extractive | 0.473 | - | - |
| answer_f1.gold.abstractive | 0.302 | - | - |
| answer_f1.gold.extractive | 0.464 | - | - |
| answer_f1.gold.unanswerable | 0.411 | - | - |
| answer_f1.gold.yes_no | 0.464 | - | - |
| answer_f1.unanswerable | 0.458 | - | - |
| answer_f1.yes_no | 0.486 | - | - |
| answer_words_mean | 15.492 | - | - |
| completion_tokens_mean | 51.800 | - | - |
| context_words_mean | 998.848 | - | - |
| error_rate | 0.000 | - | - |
| evidence_f1 | 0.297 | - | - |
| evidence_f1_ceiling | 0.524 | - | - |
| false_refusal_rate | 0.042 | - | - |
| gold_recall | 0.494 | - | - |
| index.num_chunks | 6676.000 | - | - |
| latency_p50_ms | 1042.795 | - | - |
| latency_p95_ms | 1850.713 | - | - |
| n.abstractive | 55.000 | - | - |
| n.extractive | 144.000 | - | - |
| n.gold.abstractive | 58.000 | - | - |
| n.gold.extractive | 131.000 | - | - |
| n.gold.unanswerable | 37.000 | - | - |
| n.gold.yes_no | 24.000 | - | - |
| n.unanswerable | 24.000 | - | - |
| n.yes_no | 27.000 | - | - |
| n_questions | 250.000 | - | - |
| prompt_tokens_max | 2577.000 | - | - |
| prompt_tokens_mean | 2228.868 | - | - |
| refusal_recall | 0.270 | - | - |
| retrieval.doc_hit_rate | 1.000 | - | - |
| retrieval.mrr | 0.552 | - | - |
| retrieval.n | 230.000 | - | - |
| retrieval.ndcg_at_1 | 0.409 | - | - |
| retrieval.ndcg_at_3 | 0.492 | - | - |
| retrieval.ndcg_at_5 | 0.527 | - | - |
| retrieval.recall_at_1 | 0.335 | - | - |
| retrieval.recall_at_3 | 0.603 | - | - |
| retrieval.recall_at_5 | 0.703 | - | - |

## Lowest-scoring questions

- **Do they evaluate only on English datasets?** (`00c57e45ac6afbdfa67350a57e81b4fad0ed2885`, answer_f1 0.00)
  - gold: Yes | Yes
  - predicted: Unanswerable
  - top context: [Feature Contribution] classifier produced consistent F1-scores, even slightly above the baseline for depressive symptoms and minor fluctuations of change in recall and precision when removing other feature groups suggesting that the contribution of non-lexical features to classification performance

- **How do they define robustness of a model?** (`02428a8fec9788f6dc3a86b5d5f3aa679935678d`, answer_f1 0.00)
  - gold: ability to accurately classify texts even when the amount of prior knowledge for different classes is unbalanced, and when the class distribution of the dataset is unbalanced | Low sensitivity to bias in prior knowledge
  - predicted: robustly against baselines
  - top context: [Introduction] can boost the performance remarkably, making the model more robust. More attractively, we do not need manual annotation to label these neutral feature words in our proposed approach. More specifically, we explore three regularization terms to address the problem: (1) a regularization 

- **what was the baseline?** (`0752d71a0a1f73b3482a888313622ce9e9870d6e`, answer_f1 0.00)
  - gold: system presented by deri2016grapheme | wFST
  - predicted: an approach based on adapting high-resource monolingual g2p models to low-resource languages
  - top context: [Baseline] or for which language distance metrics could not be computed. We train the LangID and NoLangID versions of our model each on three subsets of the Wiktionary data: LangID-High and NoLangID-High: Trained on data from the 85 languages for which BIBREF13 used non-adapted wFST models. LangID-A

- **On top of BERT does the RNN layer work better or the transformer layer?** (`0810b43404686ddfe4ca84783477ae300fdd2ea4`, answer_f1 0.00)
  - gold: Transformer over BERT (ToBERT) | The transformer layer
  - predicted: Unanswerable
  - top context: [Method ::: BERT] obtain two kinds of representation from BERT: pooled output from last transformer block, denoted by H, and posterior probabilities, denoted by P. There are two variants of BERT - BERT-Base and BERT-Large. In this work we are using BERT-Base for faster training and experimentation, 

- **What evaluation metric is used?** (`0ab3df10f0b7203e859e9b62ffa7d6d79ffbbe50`, answer_f1 0.00)
  - gold: average classification accuracy | average classification accuracy, execution performance
  - predicted: classifying a piece of text by its specific language label
  - top context: [Methodology] winning language is at least one more than the number of words considered to be from the language scored in second place. The stacked classifier is tested against three public LID implementations BIBREF17, BIBREF23, BIBREF8. The LID implementation described in BIBREF17 is available on 
