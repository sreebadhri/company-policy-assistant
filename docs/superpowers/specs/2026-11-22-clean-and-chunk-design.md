# Step 2: clean and chunk (approved)

**Why:** repeated headers make chunks look alike, and chunks covering several
topics match a question less clearly.

**Goal:** turn the loaded pages into small, citable chunks.
**Not in this step:** embeddings, Chroma, agent changes.
**Must not happen:** lost policy text, or chunks without a source.

**Approaches**
- A (chosen): RecursiveCharacterTextSplitter, about 400 characters with 80
  overlap. Standard and works on any document; can cut mid-topic.
- B: split at topic headings. Cleanest, but heading detection is fragile.
- C: one chunk per page. No work, but mixes topics.

**Design**
- `clean_document()` removes only the known aio PDF header/footer and keeps
  title and section as metadata. Other documents pass through unchanged.
- `split_documents()` keeps source, page, title and section as metadata.
- `ingest.py` prints chunk count, sizes and a before/after example.
- Add `langchain-text-splitters` to requirements.
- Split preference: line breaks after a full stop, then any line break. Measured
  on our PDFs: 0 of 23 chunks cut mid-sentence, versus 8 of 17 with defaults.

**Update (2026-11-23):** chunks first had a "title > section" line at the top. We
removed it after comparing search on the 15 answerable test questions: right page
in the top 4 went from 14 to 15, and the mean gap to the nearest wrong page from
0.239 to 0.320. Small test set, so a signal rather than proof.
