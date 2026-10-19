# Run `topk_3`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `retrieval.top_k` | 5 | 3 |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.403 | 0.419 | -0.016 |
| answer_f1.abstractive | 0.215 | 0.225 | -0.010 |
| answer_f1.extractive | 0.429 | 0.473 | -0.044 |
| answer_f1.unanswerable | 0.640 | 0.458 | +0.182 |
| answer_f1.yes_no | 0.443 | 0.486 | -0.043 |
| completion_tokens_mean | 49.716 | 51.800 | -2.084 |
| context_words_mean | 599.680 | 998.848 | -399.168 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.300 | 0.297 | +0.004 |
| false_refusal_rate | 0.075 | 0.042 | +0.033 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 882.162 | 1042.795 | -160.633 |
| latency_p95_ms | 1521.302 | 1850.713 | -329.412 |
| n.abstractive | 57.000 | 55.000 | +2.000 |
| n.extractive | 143.000 | 144.000 | -1.000 |
| n.unanswerable | 25.000 | 24.000 | +1.000 |
| n.yes_no | 25.000 | 27.000 | -2.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 1913.000 | 2577.000 | -664.000 |
| prompt_tokens_mean | 1649.192 | 2228.868 | -579.676 |
| refusal_recall | 0.405 | 0.270 | +0.135 |
| retrieval.mrr | 0.533 | 0.552 | -0.020 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.409 | 0.409 | +0.000 |
| retrieval.ndcg_at_3 | 0.560 | 0.560 | +0.000 |
| retrieval.recall_at_1 | 0.335 | 0.335 | +0.000 |
| retrieval.recall_at_3 | 0.603 | 0.603 | +0.000 |

## Biggest regressions

- **Do they experiment with the toolkits?** (`5f2bade0881c719ab026bc2e2962e2ada96cdb25`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: Unanswerable
  - top context: [Abstract] We present GluonCV and GluonNLP, the deep learning toolkits for computer vision and natural language processing based on Apache MXNet (incubating). These toolkits provide state-of-the-art pre-trained models, training scripts, and training logs, to facilitate rapid prototyping and promote 

- **did they compare with other extractive summarization methods?** (`c2cb6c4500d9e02fc9a1bdffd22c3df69655189f`, answer_f1 1.00 -> 0.00)
  - gold: No
  - predicted: Unanswerable
  - top context: [Introduction] aims are critical steps in ultimately developing methods that can automate discharge summary creation. In the broader field of summarization, automization was meant to standardize output while also saving time and effort. Pioneering strategies in summarization started by extracting "s

- **what dataset was used?** (`e48e750743aef36529fbea4328b8253dbe928b4d`, answer_f1 1.00 -> 0.00)
  - gold: WASSA-2017 Shared Task on Emotion Intensity
  - predicted: remaining dataset
  - top context: [Regression] remaining dataset. The predictions are averaged across all the folds to generalize the solution and prevent over-fitting. As described in Section SECREF6 , different combinations of feature extractors were used. After performing feature extraction, the data was then passed to various re

- **What are the baselines?** (`eb2d5edcdfe18bd708348283f92a32294bb193a5`, answer_f1 1.00 -> 0.00)
  - gold: a score of 40 | KG-A2C, A2C, A2C-chained, A2C-Explore
  - predicted: Unanswerable
  - top context: [Introduction and Background] both an agents' state and action spaces are in textual natural language. An example of a one turn agent interaction in the popular text-game Zork1 can be seen in Fig. FIGREF1. Text-adventure games provide us with multiple challenges in the form of partial observability,

- **What modalities are supported by Macaw?** (`6ecb69360449bb9915ac73c0a816c8ac479cbbfc`, answer_f1 0.89 -> 0.00)
  - gold: text, speech, image, click, etc
  - predicted: multi-turn, multi-modal, and mixed-initiative interactions
  - top context: [Limitations and Future Work] Macaw by supporting more actions and even more advanced techniques for the developed actions. [leftmargin=*] Clarification and Preference Elicitation: Asking clarifying questions has been recently recognized as a necessary component in a conversational system BIBREF3, B
