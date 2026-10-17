# Positional Encoding to Control Output Sequence Length

- Authors: Sho Takase, Naoaki Okazaki.
- Venue/year: NAACL 2019, N19-1401.
- URL: https://aclanthology.org/N19-1401/
- Inspected full text: https://aclanthology.org/N19-1401.pdf
- Accessed: 2026-10-17.
- Type: original peer-reviewed paper.
- Credibility: high within the headline-generation setting. Recency: historical. Bias: author experiments.

Verified excerpt, section 2: “we assign the length of the correct output to len.”

The method conditions decoder positional features on a requested length, with character-oriented experiments. It supports matching training controls to the supervised output. Replacing the rotary positional system of a pretrained decoder is a different and riskier change; it is not recommended for the first SEAM experiment.
