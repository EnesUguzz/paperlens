# Run `chunk_fixed_100`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `chunking.chunk_overlap` | 40 | 20 |
| `chunking.chunk_size` | 200 | 100 |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.417 | 0.419 | -0.002 |
| answer_f1.abstractive | 0.216 | 0.225 | -0.010 |
| answer_f1.extractive | 0.465 | 0.473 | -0.008 |
| answer_f1.unanswerable | 0.577 | 0.458 | +0.119 |
| answer_f1.yes_no | 0.425 | 0.486 | -0.061 |
| completion_tokens_mean | 48.932 | 51.800 | -2.868 |
| context_words_mean | 510.892 | 998.848 | -487.956 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.343 | 0.297 | +0.046 |
| false_refusal_rate | 0.070 | 0.042 | +0.028 |
| index.num_chunks | 13291.000 | 6676.000 | +6615.000 |
| latency_p50_ms | 880.222 | 1042.795 | -162.573 |
| latency_p95_ms | 1514.128 | 1850.713 | -336.585 |
| n.abstractive | 56.000 | 55.000 | +1.000 |
| n.extractive | 142.000 | 144.000 | -2.000 |
| n.unanswerable | 26.000 | 24.000 | +2.000 |
| n.yes_no | 26.000 | 27.000 | -1.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 1780.000 | 2577.000 | -797.000 |
| prompt_tokens_mean | 1546.804 | 2228.868 | -682.064 |
| refusal_recall | 0.378 | 0.270 | +0.108 |
| retrieval.mrr | 0.507 | 0.552 | -0.046 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.352 | 0.409 | -0.057 |
| retrieval.ndcg_at_3 | 0.536 | 0.560 | -0.025 |
| retrieval.ndcg_at_5 | 0.636 | 0.649 | -0.013 |
| retrieval.recall_at_1 | 0.294 | 0.335 | -0.041 |
| retrieval.recall_at_3 | 0.537 | 0.603 | -0.066 |
| retrieval.recall_at_5 | 0.673 | 0.703 | -0.030 |

## Biggest regressions

- **Do they experiment with the toolkits?** (`5f2bade0881c719ab026bc2e2962e2ada96cdb25`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: Unanswerable
  - top context: [Abstract] We present GluonCV and GluonNLP, the deep learning toolkits for computer vision and natural language processing based on Apache MXNet (incubating). These toolkits provide state-of-the-art pre-trained models, training scripts, and training logs, to facilitate rapid prototyping and promote 

- **how small of a dataset did they train on?** (`8060a773f6a136944f7b59758d08cc6f2a59693b`, answer_f1 1.00 -> 0.00)
  - gold: 1000 hours data | 23085 hours of data
  - predicted: Unanswerable
  - top context: [Experimental setup] INLINEFORM4 , and send to corresponding worker. In the aggregation phase, INLINEFORM5 -th worker computed INLINEFORM6 splits of model INLINEFORM7 and send updated model INLINEFORM8 back to workers. As a result, all workers participate in aggregation and no computing resource is 

- **Does the model have attention?** (`b9d168da5321a7d7b812c52bb102a05210fe45bd`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: seq2seq model with attention
  - top context: [Baseline] few details regarding model size and training schedule: the number of LSTM layers is one; embedding size, LSTM layer size and attention layer size is 100; models are trained for 20 epochs; on every epoch, training data is subsampled at a rate of 0.3; LSTM dropout is applied at a rate 0.3;

- **did they compare with other extractive summarization methods?** (`c2cb6c4500d9e02fc9a1bdffd22c3df69655189f`, answer_f1 1.00 -> 0.00)
  - gold: No
  - predicted: Unanswerable
  - top context: [Related Work] which made them crude at best. More advanced extractive heuristics like topic modeling BIBREF7 , cue word dictionary approaches BIBREF8 , and title methods BIBREF9 for scoring content in a sentence followed soon after. For example, topic modeling extends initial frequency methods by a

- **what dataset was used?** (`e48e750743aef36529fbea4328b8253dbe928b4d`, answer_f1 1.00 -> 0.00)
  - gold: WASSA-2017 Shared Task on Emotion Intensity
  - predicted: 2 Billion tweets
  - top context: [Regression] remaining dataset. The predictions are averaged across all the folds to generalize the solution and prevent over-fitting. As described in Section SECREF6 , different combinations of feature extractors were used. After performing feature extraction, the data was then passed to various re
