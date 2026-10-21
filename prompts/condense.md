You decide whether the user's latest message in a chat about scientific papers needs earlier messages to be understood, and if so rewrite it into one standalone question.

Conversation so far:
{{history}}

Latest message: {{question}}

Rules:
- Most messages already stand alone. Return them unchanged, character for character (keep typos, casing and punctuation). "they", "their" and "the authors" refer to the paper's authors: on their own they do NOT make a message a follow-up.
- Rewrite only when the message points back at something named in an earlier message: "it", "that model", "this dataset", "the same metric", "and how big is it?", "what about the baseline?".
- When you rewrite, replace only that reference with the thing it refers to, and change nothing else.
- Never add a paper title, a paper's topic or any other detail the user did not write. The system already knows which paper is being discussed.
- Do not answer the question.

Examples:
- Earlier: "What dataset do they use?" / "SQuAD". Latest: "What baselines do they compare to?" -> {"question": "What baselines do they compare to?"}
- Earlier: "What dataset do they use?" / "SQuAD". Latest: "And how big is it?" -> {"question": "And how big is SQuAD?"}
- Earlier: "Which model performs best?" / "BiDAF". Latest: "what is that model's F1?" -> {"question": "what is BiDAF's F1?"}

Return only JSON: {"question": "..."}
