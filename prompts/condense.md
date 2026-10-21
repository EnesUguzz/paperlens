You rewrite the user's latest message in a chat about scientific papers into one standalone question.

Conversation so far:
{{history}}

Latest message: {{question}}

Rules:
- If the latest message already makes sense on its own, return it unchanged, word for word.
- Otherwise replace pronouns and vague references ("it", "they", "that model", "the dataset") with what they refer to in the conversation.
- Keep the user's wording and language; do not answer the question.
- Do not add a paper title or a paper name that the user did not write in the latest message.

Return only JSON: {"question": "..."}
