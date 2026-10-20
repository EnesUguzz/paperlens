# Run `retr_bm25`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `retrieval.method` | dense | bm25 |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.370 | 0.419 | -0.049 |
| answer_f1.abstractive | 0.252 | 0.225 | +0.027 |
| answer_f1.extractive | 0.403 | 0.473 | -0.070 |
| answer_f1.gold.abstractive | 0.278 | 0.302 | -0.024 |
| answer_f1.gold.extractive | 0.398 | 0.464 | -0.066 |
| answer_f1.gold.unanswerable | 0.408 | 0.411 | -0.003 |
| answer_f1.gold.yes_no | 0.375 | 0.464 | -0.089 |
| answer_f1.unanswerable | 0.478 | 0.458 | +0.020 |
| answer_f1.yes_no | 0.370 | 0.486 | -0.116 |
| answer_words_mean | 15.196 | 15.492 | -0.296 |
| completion_tokens_mean | 51.184 | 51.800 | -0.616 |
| context_words_mean | 974.236 | 998.848 | -24.612 |
| data.questions_dropped | 0.000 | 0.000 | +0.000 |
| data.unmatched_evidence_rate | 0.046 | 0.046 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.270 | 0.297 | -0.027 |
| evidence_f1_ceiling | 0.471 | 0.524 | -0.053 |
| false_refusal_rate | 0.061 | 0.042 | +0.019 |
| gold_recall | 0.442 | 0.494 | -0.053 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| judge.accuracy | 0.472 | 0.500 | -0.028 |
| judge.accuracy.gold.abstractive | 0.448 | 0.310 | +0.138 |
| judge.accuracy.gold.extractive | 0.496 | 0.595 | -0.099 |
| judge.accuracy.gold.unanswerable | 0.486 | 0.486 | +0.000 |
| judge.accuracy.gold.yes_no | 0.375 | 0.458 | -0.083 |
| judge.agreement_on_high_f1 | 0.933 | 1.000 | -0.067 |
| judge.error_rate | 0.000 | 0.000 | +0.000 |
| judge.llm_calls | 184.000 | 176.000 | +8.000 |
| latency_p50_ms | 953.122 | 1042.795 | -89.673 |
| latency_p95_ms | 1766.440 | 1850.713 | -84.273 |
| n.abstractive | 61.000 | 55.000 | +6.000 |
| n.extractive | 139.000 | 144.000 | -5.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 23.000 | 24.000 | -1.000 |
| n.yes_no | 27.000 | 27.000 | +0.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2526.000 | 2577.000 | -51.000 |
| prompt_tokens_mean | 2170.632 | 2228.868 | -58.236 |
| query_error_rate | 0.000 | 0.000 | +0.000 |
| refusal_recall | 0.270 | 0.270 | +0.000 |
| retrieval.doc_hit_rate | 1.000 | 1.000 | +0.000 |
| retrieval.mrr | 0.437 | 0.552 | -0.115 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.300 | 0.409 | -0.109 |
| retrieval.ndcg_at_3 | 0.377 | 0.492 | -0.115 |
| retrieval.ndcg_at_5 | 0.430 | 0.527 | -0.097 |
| retrieval.recall_at_1 | 0.247 | 0.335 | -0.088 |
| retrieval.recall_at_3 | 0.465 | 0.603 | -0.137 |
| retrieval.recall_at_5 | 0.606 | 0.703 | -0.097 |

## Biggest regressions

- **Which real-world datasets are used?** (`3941401a182a3d6234894a5c8a75d48c6116c45c`, answer_f1 1.00 -> 0.00)
  - gold: Tweets related to CyberAttack and tweets related to PoliticianDeath | cyber security (CyberAttack), death of politicians (PoliticianDeath)
  - predicted: multiple real-world datasets
  - top context: [Abstract] more transparent. We empirically demonstrate the merits of our approach, both in terms of accuracy and interpretability, on multiple real-world datasets and show that our approach improves the state of the art by 24.3%. Event detection on microblogging platforms such as Twitter aims to de

- **Do they experiment with the toolkits?** (`5f2bade0881c719ab026bc2e2962e2ada96cdb25`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: Unanswerable
  - top context: [Introduction] way movies are made, diseases are diagnosed, and play a growing role in understanding and communicating with humans. Such development is made possible by deep learning frameworks, such as Caffe BIBREF0 , Chainer BIBREF1 , CNTK BIBREF2 , Apache (incubating) MXNet BIBREF3 , PyTorch BIBR

- **Is WordNet useful for taxonomic reasoning for this task?** (`75ff6e425ce304a35f18c0230c0d13d3913a31a9`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Yes
  - predicted: WordNet is an English lexical database consisting of around 117k concepts, which are organized into groups of synsets that each contain [...] complex taxonomic relations.
  - top context: [Introduction] interesting nuances to the overall positive trend. For example, the performance of even the best QA models degrades substantially on our hyponym probes (by 8-15%) when going from 1-hop links to 2-hops. Further, the accuracy of even our best models on the WordNetQA probe drops by 14-44

- **How many different characters were in dataset?** (`808f0ad46ca4eb4ea5492f9e14ca043fe1e206cc`, answer_f1 1.00 -> 0.00)
  - gold: 45,821 characters | 45,821 characters
  - predicted: five
  - top context: [Evaluation ::: CSM Evaluation] by the model for $c$. We use $N = 100$, and our model achieves 25.08% recall. To inspect the CSM performance, we use the T-distributed Stochastic Neighbor Embedding (t-SNE) BIBREF35 to reduce each high-dimensionality data point to two-dimensions via Kullback-Leibler D

- **What do they mean by a person's industry?** (`935d6a6187e6a0c9c0da8e53a42697f853f5c248`, answer_f1 1.00 -> 0.00)
  - gold: the aggregate of enterprises in a particular field | the aggregate of enterprises in a particular field
  - predicted: a user's industry
  - top context: [Abstract] Automatic profiling of social media users is an important task for supporting a multitude of downstream applications. While a number of studies have used social media content to extract and study collective social attributes, there is a lack of substantial research that addresses the dete
