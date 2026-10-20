You grade an answer to a question about a scientific paper.

Question: {{question}}

Reference answers written by annotators (any one of them counts as correct):
{{gold}}

Answer to grade: {{prediction}}

The answer is correct if it states the same fact as at least one reference answer. Wording, length and extra correct detail do not matter. It is incorrect if it contradicts the references, names different items, gives a different number, or misses the main point.

Return only JSON: {"reason": "one short sentence", "correct": true}
