# Source note: Google Research ROUGE tokenizer

- **URL:** https://raw.githubusercontent.com/google-research/google-research/master/rouge/tokenize.py
- **Type:** Primary library implementation, retrieved 2026-10-17.
- **Credibility:** High for this implementation's behavior.
- **Recency:** Current retrieved snapshot; mutable master link, not a pinned release.
- **Bias:** Low for the tokenization mechanics; this source does not validate Vietnamese metrics.

Verified literal source pattern:

> `[^a-z0-9]+`

The default tokenizer lowercases and replaces characters outside that pattern's
allowed set. This supports rejecting the default tokenizer as a Unicode word
tokenizer. It does not prove how the server's separate Perl ROUGE installation
handles UTF-8. The new training proxy preserves Unicode whitespace tokens;
exact final-metric parity remains unestablished.
