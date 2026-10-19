# Run `model_phi4_14b`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `llm.model` | qwen3:8b | phi4:14b |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.399 | 0.403 | -0.004 |
| answer_f1.abstractive | 0.257 | 0.239 | +0.018 |
| answer_f1.extractive | 0.478 | 0.461 | +0.017 |
| answer_f1.unanswerable | 0.522 | 0.318 | +0.204 |
| answer_f1.yes_no | 0.222 | 0.517 | -0.295 |
| completion_tokens_mean | 53.708 | 43.948 | +9.760 |
| context_words_mean | 998.848 | 998.848 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.287 | 0.265 | +0.023 |
| false_refusal_rate | 0.080 | 0.047 | +0.033 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 1208.441 | 684.335 | +524.105 |
| latency_p95_ms | 2365.049 | 972.827 | +1392.222 |
| n.abstractive | 62.000 | 58.000 | +4.000 |
| n.extractive | 138.000 | 141.000 | -3.000 |
| n.unanswerable | 23.000 | 22.000 | +1.000 |
| n.yes_no | 27.000 | 29.000 | -2.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2014.000 | 2042.000 | -28.000 |
| prompt_tokens_mean | 1642.284 | 1676.652 | -34.368 |
| refusal_recall | 0.297 | 0.162 | +0.135 |
| retrieval.mrr | 0.552 | 0.552 | +0.000 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.409 | 0.409 | +0.000 |
| retrieval.ndcg_at_3 | 0.560 | 0.560 | +0.000 |
| retrieval.ndcg_at_5 | 0.649 | 0.649 | +0.000 |
| retrieval.recall_at_1 | 0.335 | 0.335 | +0.000 |
| retrieval.recall_at_3 | 0.603 | 0.603 | +0.000 |
| retrieval.recall_at_5 | 0.703 | 0.703 | +0.000 |

## Biggest regressions

- **Did they used dataset from another domain for evaluation?** (`1dac4bc5af239024566fcb0f43bb9ff1c248ecec`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: Yes
  - top context: [Results] Pointer-Gen+ARL-SEN achieve comparable fluency performance to Pointer-Gen and Pointer-Gen+RL-ROUGE. Test set headlines achieve the best performance among all models, but the difference is not statistically significant. Also, we observe that fine-tuning on sensational headlines will hurt th

- **Do they use attention?** (`45e9533586199bde19313cd43b3d0ecadcaf7a33`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: the attention function BIBREF11 is used to compute the similarity score between passages and questions
  - top context: [Model] layer, and also with the output of the first contextual encoding layer as the input of its second encoding layer. Thus, we obtain the final representation of the contextual encoding layer by a concatenation of the outputs of two BiLSTM: INLINEFORM0 for questions and INLINEFORM1 for passages.

- **Do the authors mention any downside of lemmatizing input before training ELMo?** (`5aa12b4063d6182a71870c98e4e1815ff3dc8a72`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: lemmatization brings along both advantages and disadvantages for WSD with ELMo
  - top context: [Introduction] if the field of applying deep learning to NLP tasks can be summarised as follows: as minimal pre-processing as possible. It is widely believed that lemmatization or other text input normalisation is not necessary. Advanced neural architectures based on character input (CNNs, BPE, etc)

- **Do they experiment with the toolkits?** (`5f2bade0881c719ab026bc2e2962e2ada96cdb25`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: We evaluate popular or state-of-the-art models on standard benchmark data sets. In the experiments, we compare model performance between GluonCV/NLP and other open source implementations with Caffe, Caffe2, Theano, and TensorFlow, including ResNet and MobileNet for image classification (ImageNet), Faster R-CNN for object detection (COCO), Mask R-CNN for instance segmentation, Simple Pose for pose estimation (COCO), textCNN for sentiment analysis (TREC), and BERT for question answering (SQuAD 1.1), sentiment analysis (SST-2), natural language inference (MNLI-m), and paraphrasing (MRPC).
  - top context: [Abstract] We present GluonCV and GluonNLP, the deep learning toolkits for computer vision and natural language processing based on Apache MXNet (incubating). These toolkits provide state-of-the-art pre-trained models, training scripts, and training logs, to facilitate rapid prototyping and promote 

- **Do they report results only on English datasets?** (`8bb0011ad1d63996d5650770f3be18abdd9f7fc6`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Yes
  - predicted: The passages do not specify whether the results are reported only on English datasets.
  - top context: [Data Enrichment Method] WordNet-based data enrichment method, which is aimed at extracting inter-word semantic connections from each passage-question pair in our MRC dataset. The extraction is performed in a controllable manner, and the extracted results are provided as general knowledge to our MRC
