You answer questions about one scientific paper using only the numbered passages below.

Paper title: {{title}}

Passages:
{{context}}

Question: {{question}}

Rules:
- Use only the passages. If they do not contain the answer, set "answer_type" to "unanswerable" and "answer" to "Unanswerable".
- If the question can be answered with yes or no, set "answer_type" to "yes_no" and "answer" to "Yes" or "No".
- If the answer can be copied from the passages, set "answer_type" to "extractive" and copy the shortest exact span(s); separate several spans with ", ".
- Otherwise set "answer_type" to "abstractive" and answer in one short sentence.
- "evidence" lists the numbers of the passages that support the answer (empty for unanswerable).

Return only JSON: {"answer_type": "...", "answer": "...", "evidence": [1, 2]}
