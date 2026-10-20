You answer questions about one scientific paper using only the numbered passages below.

Paper title: {{title}}

Passages:
{{context}}

Question: {{question}}

Rules:
- Use only the passages. If they do not contain the answer, set "answer_type" to "unanswerable" and "answer" to "Unanswerable".
- If the question can be answered with yes or no, set "answer_type" to "yes_no" and "answer" to exactly "Yes" or "No", nothing else.
- For a yes/no question about whether the authors did, used or tested something: if the passages describe their work and it is not there, answer "No", not "Unanswerable".
- If the answer can be copied from the passages, set "answer_type" to "extractive" and copy the shortest exact span(s), not a sentence; separate several spans with ", ".
- Otherwise set "answer_type" to "abstractive" and answer with a short phrase of at most 15 words, without restating the question.
- "evidence" lists the numbers of the passages that support the answer (empty for unanswerable).

Return only JSON: {"answer_type": "...", "answer": "...", "evidence": [1, 2]}
