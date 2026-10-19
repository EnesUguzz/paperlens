# Run `model_qwen3_8b`

No reference run: showing absolute metrics only.

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.403 | - | - |
| answer_f1.abstractive | 0.239 | - | - |
| answer_f1.extractive | 0.461 | - | - |
| answer_f1.gold.abstractive | 0.306 | - | - |
| answer_f1.gold.extractive | 0.449 | - | - |
| answer_f1.gold.unanswerable | 0.332 | - | - |
| answer_f1.gold.yes_no | 0.500 | - | - |
| answer_f1.unanswerable | 0.318 | - | - |
| answer_f1.yes_no | 0.517 | - | - |
| completion_tokens_mean | 43.948 | - | - |
| context_words_mean | 998.848 | - | - |
| error_rate | 0.000 | - | - |
| evidence_f1 | 0.265 | - | - |
| evidence_f1_ceiling | 0.471 | - | - |
| false_refusal_rate | 0.047 | - | - |
| index.num_chunks | 6676.000 | - | - |
| latency_p50_ms | 684.335 | - | - |
| latency_p95_ms | 972.827 | - | - |
| n.abstractive | 58.000 | - | - |
| n.extractive | 141.000 | - | - |
| n.gold.abstractive | 58.000 | - | - |
| n.gold.extractive | 131.000 | - | - |
| n.gold.unanswerable | 37.000 | - | - |
| n.gold.yes_no | 24.000 | - | - |
| n.unanswerable | 22.000 | - | - |
| n.yes_no | 29.000 | - | - |
| n_questions | 250.000 | - | - |
| prompt_tokens_max | 2042.000 | - | - |
| prompt_tokens_mean | 1676.652 | - | - |
| refusal_recall | 0.162 | - | - |
| retrieval.mrr | 0.552 | - | - |
| retrieval.n | 230.000 | - | - |
| retrieval.ndcg_at_1 | 0.409 | - | - |
| retrieval.ndcg_at_3 | 0.492 | - | - |
| retrieval.ndcg_at_5 | 0.527 | - | - |
| retrieval.recall_at_1 | 0.335 | - | - |
| retrieval.recall_at_3 | 0.603 | - | - |
| retrieval.recall_at_5 | 0.703 | - | - |

## Lowest-scoring questions

- **How does proposed qualitative annotation schema looks like?** (`005cca3c8ab6c3a166e315547a2259020f318ffb`, answer_f1 0.00)
  - gold: The resulting taxonomy of the framework is shown in Figure FIGREF10 | FIGREF10
  - predicted: The proposed qualitative annotation schema includes categories such as Redundancy, synonyms and paraphrases (labelled Synonym), syntactic features like ambiguous relative clauses, appositions and adverbial phrases (RelAdvApp), and features introducing variance such as switching between verbal and nominal styles (Nominalisation) and from passive to active voice (Voice).
  - top context: [Application of the Framework ::: Annotation Task] development sets of the described datasets. Training, development and the (hidden) test set are drawn from the same distribution defined by the data collection method of the respective dataset. For those collections that contain multiple questions o

- **Do the errors of the model reflect linguistic similarity between different L1s?** (`007b13f05d234d37966d1aa7d85b5fd78564ff45`, answer_f1 0.00)
  - gold: Unanswerable
  - predicted: The errors of the model may reflect linguistic similarity between different L1s as the model could be learning topic associations instead of linguistic patterns.
  - top context: [Introduction] includes texts produced by adults and children, as well as a spoken subset. The aforementioned Portuguese learner corpora contain very useful data for research, particularly for Native Language Identification (NLI), a task that has received much attention in recent years. NLI is the t

- **Do they evaluate only on English datasets?** (`00c57e45ac6afbdfa67350a57e81b4fad0ed2885`, answer_f1 0.00)
  - gold: Yes | Yes
  - predicted: Unanswerable
  - top context: [Feature Contribution] classifier produced consistent F1-scores, even slightly above the baseline for depressive symptoms and minor fluctuations of change in recall and precision when removing other feature groups suggesting that the contribution of non-lexical features to classification performance

- **what levels of document preprocessing are looked at?** (`06eb9f2320451df83e27362c22eb02f4a426a018`, answer_f1 0.00)
  - gold: raw text, text cleaning through document logical structure detection, removal of keyphrase sparse sections of the document | Level 1, Level 2 and Level 3.
  - predicted: four levels
  - top context: [Figures and Tables] for each level (σ1) and the standard deviation across the three levels for each model (σ2). α and β indicate significance at the 0.05 level using Student’s t-test against level 1 and level 2 respectively. Table 5: Difference in f-score between our re-implementation and the origi

- **what was the baseline?** (`0752d71a0a1f73b3482a888313622ce9e9870d6e`, answer_f1 0.00)
  - gold: system presented by deri2016grapheme | wFST
  - predicted: Baseline or for which language distance metrics could not be computed.
  - top context: [Baseline] or for which language distance metrics could not be computed. We train the LangID and NoLangID versions of our model each on three subsets of the Wiktionary data: LangID-High and NoLangID-High: Trained on data from the 85 languages for which BIBREF13 used non-adapted wFST models. LangID-A
