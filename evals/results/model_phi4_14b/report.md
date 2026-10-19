# Run `model_phi4_14b`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `llm.extra_body` | {} | None |
| `llm.extra_body.reasoning_effort` | None | none |
| `llm.model` | ministral-3:14b | phi4:14b |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.399 | 0.419 | -0.019 |
| answer_f1.abstractive | 0.257 | 0.225 | +0.031 |
| answer_f1.extractive | 0.478 | 0.473 | +0.004 |
| answer_f1.gold.abstractive | 0.302 | 0.302 | -0.001 |
| answer_f1.gold.extractive | 0.472 | 0.464 | +0.008 |
| answer_f1.gold.unanswerable | 0.418 | 0.411 | +0.007 |
| answer_f1.gold.yes_no | 0.208 | 0.464 | -0.255 |
| answer_f1.unanswerable | 0.522 | 0.458 | +0.063 |
| answer_f1.yes_no | 0.222 | 0.486 | -0.264 |
| completion_tokens_mean | 53.708 | 51.800 | +1.908 |
| context_words_mean | 998.848 | 998.848 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.287 | 0.297 | -0.009 |
| evidence_f1_ceiling | 0.471 | 0.471 | +0.000 |
| false_refusal_rate | 0.080 | 0.042 | +0.038 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 1208.441 | 1042.795 | +165.646 |
| latency_p95_ms | 2365.049 | 1850.713 | +514.336 |
| n.abstractive | 62.000 | 55.000 | +7.000 |
| n.extractive | 138.000 | 144.000 | -6.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 23.000 | 24.000 | -1.000 |
| n.yes_no | 27.000 | 27.000 | +0.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2014.000 | 2577.000 | -563.000 |
| prompt_tokens_mean | 1642.284 | 2228.868 | -586.584 |
| refusal_recall | 0.297 | 0.270 | +0.027 |
| retrieval.mrr | 0.552 | 0.552 | +0.000 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.409 | 0.409 | +0.000 |
| retrieval.ndcg_at_3 | 0.492 | 0.492 | +0.000 |
| retrieval.ndcg_at_5 | 0.527 | 0.527 | +0.000 |
| retrieval.recall_at_1 | 0.335 | 0.335 | +0.000 |
| retrieval.recall_at_3 | 0.603 | 0.603 | +0.000 |
| retrieval.recall_at_5 | 0.703 | 0.703 | +0.000 |

## Biggest regressions

- **Do they use attention?** (`45e9533586199bde19313cd43b3d0ecadcaf7a33`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: the attention function BIBREF11 is used to compute the similarity score between passages and questions
  - top context: [Model] layer, and also with the output of the first contextual encoding layer as the input of its second encoding layer. Thus, we obtain the final representation of the contextual encoding layer by a concatenation of the outputs of two BiLSTM: INLINEFORM0 for questions and INLINEFORM1 for passages.

- **Do they experiment with the toolkits?** (`5f2bade0881c719ab026bc2e2962e2ada96cdb25`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: We evaluate popular or state-of-the-art models on standard benchmark data sets. In the experiments, we compare model performance between GluonCV/NLP and other open source implementations with Caffe, Caffe2, Theano, and TensorFlow, including ResNet and MobileNet for image classification (ImageNet), Faster R-CNN for object detection (COCO), Mask R-CNN for instance segmentation, Simple Pose for pose estimation (COCO), textCNN for sentiment analysis (TREC), and BERT for question answering (SQuAD 1.1), sentiment analysis (SST-2), natural language inference (MNLI-m), and paraphrasing (MRPC).
  - top context: [Abstract] We present GluonCV and GluonNLP, the deep learning toolkits for computer vision and natural language processing based on Apache MXNet (incubating). These toolkits provide state-of-the-art pre-trained models, training scripts, and training logs, to facilitate rapid prototyping and promote 

- **Do they inspect their model to see if their model learned to associate image parts with words related to entities?** (`7cd22ca9e107d2b13a7cc94252aaa9007976b338`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: For interpretability, visual tags (label output of InceptionNet) are presented instead of actual feature vectors used.
  - top context: [Figures and Tables] embeddings, character embeddings, and visual features are merged with modality attention. Bi-LSTM/CRF takes as input each token and produces an entity label. Table 1: NER performance on the SnapCaptions dataset with varying modalities (W: word, C: char, V: visual). We report pre

- **Do they report results only on English datasets?** (`8bb0011ad1d63996d5650770f3be18abdd9f7fc6`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Yes
  - predicted: The passages do not specify whether the results are reported only on English datasets.
  - top context: [Data Enrichment Method] WordNet-based data enrichment method, which is aimed at extracting inter-word semantic connections from each passage-question pair in our MRC dataset. The extraction is performed in a controllable manner, and the extracted results are provided as general knowledge to our MRC

- **Does the model have attention?** (`b9d168da5321a7d7b812c52bb102a05210fe45bd`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: The attention matrix indicates that this character should be based on the final state of the encoder, which contains information about the final character of the input form, and the past and future context.
  - top context: [Baseline] few details regarding model size and training schedule: the number of LSTM layers is one; embedding size, LSTM layer size and attention layer size is 100; models are trained for 20 epochs; on every epoch, training data is subsampled at a rate of 0.3; LSTM dropout is applied at a rate 0.3;
