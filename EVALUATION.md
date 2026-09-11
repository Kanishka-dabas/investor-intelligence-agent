# EVALUATION.md

## Investor Intelligence Agent — Evaluation Report

This document summarizes the evaluation methodology, results, and key
findings from testing the v2 RAG + LangGraph agentic pipeline against a
curated question set covering three ingested companies (Apple, Alphabet,
Amazon).

---

## 1. Methodology

### Test set
- **16 questions** in `evaluation/test_set.yaml`, covering:
  - **Exact-number lookups** (5): revenue/net income for a specific quarter
  - **Comparative** (3): cross-company ranking and comparison questions
  - **Adversarial / period-confusion** (2): designed to catch the model
    picking the wrong column from a multi-period financial table (e.g.
    "six months" vs "three months")
  - **Qualitative** (1): open-ended growth-driver questions
  - **Out-of-scope** (2): questions unrelated to ingested financial data,
    to verify the system says "not found" rather than hallucinating
  - **MCP-routing** (2) and **RAG-routing** (1): verifies the LangGraph
    router sends live-market-data questions to the MCP node and
    report-based questions to the RAG node

### Generation vs. judge separation
- **Generation**: Groq (`openai/gpt-oss-20b`), `temperature=0` for
  deterministic, repeatable outputs on factual queries.
- **Judge** (for the LLM-as-judge faithfulness/relevancy scoring in
  `evaluation/generation_metrics.py`): Gemini (`gemini-flash-latest`),
  kept on a fully separate provider so judge-model token usage never
  competes with generation-model quota.

**Deviation from the original plan:** the plan called for wrapping RAGAS
metrics via `ragas.llms.LangchainLLMWrapper`. In practice, the installed
`ragas` version depended on a `langchain_community.chat_models.vertexai`
import path that no longer exists in current `langchain-community`
releases (the package is being sunset upstream), and pinning a
compatible `langchain-community` version created an unresolvable
`langsmith` version conflict with the rest of the stack. Rather than
keep chasing a fragile dependency chain, faithfulness/relevancy scoring
was reimplemented directly as a lightweight LLM-as-judge prompt against
Gemini (see `llm_judge_evaluation` in `generation_metrics.py`). This
keeps the intent of the plan (separate judge provider, structured
scoring) without the brittle dependency.

### Retrieval metrics
Implemented in `evaluation/retrieval_metrics.py`: Hit Rate@3, Hit
Rate@5, Precision@3, Precision@5, and MRR. Relevance is approximated by
company-name matching against the retrieved chunk's source filename,
rather than manually labeled relevant-chunk-IDs — a pragmatic proxy
given the size of the corpus, not a ground-truth relevance judgment.

### Robustness
`evaluation/run_eval.py` wraps each question in its own try/except and
saves results to disk after every question (`save_intermediate`), so a
single failure — or a slow/failing external call — never loses
already-completed work. This was validated in practice, not just in
theory (see Finding 4 below).

---

## 2. Final Results (16/16 questions completed)

| Type | Count | Result |
|---|---|---|
| Exact-number | 5 | 5/5 correct after fixes (see findings below) |
| Comparative | 3 | Correct — router sent all to RAG, answers matched ranked KPI data |
| Adversarial | 2 | Both eventually correct; one required updating the expected answer after data changes (Finding 3) |
| Qualitative | 1 | Reasonable, grounded answer citing Google Services/Cloud growth drivers |
| Out-of-scope | 2 | Both correctly declined / stated the topic or company wasn't in scope |
| MCP-routing | 2 | Router correctly classified both as `mcp`; one call failed due to an OAuth token expiry (Finding 4) |
| RAG-routing | 1 | Correctly routed to `rag` |

**16/16 questions completed without a crash.** Effective correctness on
scored questions: 7/7 after accounting for two scoring-artifact "false
failures" described below (not real errors).

---

## 3. Key Findings (bugs found and fixed during evaluation)

### Finding 1 — Period confusion on multi-column financial tables
**Symptom:** Asked for Apple's net income for "the quarter," the system
initially returned $101,464M — the **nine-month cumulative** figure —
instead of the correct $29,789M quarterly figure. The same failure mode
appeared later for Alphabet, returning a **six-month cumulative** figure
instead of quarterly.

**Root cause:** `pymupdf4llm`'s markdown conversion split multi-word
table headers like "Three Months Ended" across separate cells (e.g.
"Three Mon" / "ths Ended"), so column-to-period alignment was corrupted
before the LLM ever saw it. No amount of prompt engineering reliably
fixes a genuinely misaligned source table.

**Fix:** Two layers —
1. Added an explicit instruction in both the RAG chat prompt and the KPI
   extraction prompt telling the model to identify the correct period
   column and prefer the most recent single quarter over cumulative
   figures.
2. For KPI extraction specifically, added `extract_layout_text()` (using
   `pdfplumber`'s `layout=True` mode) as an alternative to markdown
   conversion — this preserves spatial column alignment and avoids the
   header-splitting problem entirely. `find_financial_statements_section()`
   locates the actual financial-statement table (not a Table-of-Contents
   mention of it) by requiring a dollar-amount to appear near the
   section header match.
3. Set `temperature=0` on both Groq and Gemini generation calls —
   non-zero temperature was independently causing inconsistent answers
   to the *same* question across repeated runs, compounding the period-
   confusion issue.

### Finding 2 — Company-specific terminology mismatch hurts retrieval
**Symptom:** Querying "Apple revenue" retrieved Alphabet/Amazon chunks
instead of Apple's, even though Apple's data was present in the corpus.

**Root cause:** Apple's filings use "**Net sales**," not "Revenue" —
Alphabet and Amazon both use "Revenue." Both dense (semantic) and sparse
(BM25) retrieval favored the term that literally appeared in the query.

**Fix:** Added lightweight synonym expansion (`expand_query_with_synonyms`
in `hybrid_retriever.py`) that appends known synonyms ("net sales,"
"total sales") to the query before embedding, when a mapped term is
detected. This is a pragmatic workaround, not a general solution — a
more robust fix (noted as future work) would be LLM-based query
rewriting or a per-company terminology mapping table.

This was compounded by a **data volume imbalance**: Apple initially had
only 7 chunks ingested (from a 1-page financial-summary slide) versus
342 for Alphabet, making Apple systematically under-represented in
retrieval. Re-ingesting Apple's full 10-Q (14 chunks, much richer
context) alongside the synonym fix resolved this.

### Finding 3 — Corrupted markdown tables can make correct data unretrievable, not just mis-extracted
**Symptom:** Even after fixing Finding 1's prompt-level period confusion,
Alphabet's net-income question still failed at the *retrieval* stage —
the chunk containing the clean "Revenues $119,796 / Net income $112,193"
table was never surfacing in the top-k results at all. The only chunk
containing "$112,193" in the corpus was an unrelated Stockholders' Equity
table where the number appeared incidentally.

**Root cause:** The markdown-chunked version of the income statement table
was structurally corrupted (same header-splitting issue as Finding 1),
which apparently also degraded its embedding quality enough to rank it
out of the top 10 results for a direct, on-topic query.

**Fix:** For each company, in addition to the markdown-derived RAG
chunks, one additional "clean" chunk is now inserted directly from
`extract_layout_text()` + `find_financial_statements_section()` — the
same layout-preserving extraction used for KPI extraction. This
guarantees at least one well-formed, correctly-aligned copy of the core
financial statement is retrievable, without needing to fix or replace
the markdown chunker itself (a larger undertaking left as future work).

This finding illustrates a distinct failure mode worth separating from
Finding 1: a **generation-time** bug (right chunk retrieved, wrong number
read from it) versus a **retrieval-time** bug (right chunk never
surfaced at all). Both were present in this pipeline, from the same
underlying table-corruption root cause.

### Finding 4 — MCP OAuth token expiry mid-run
**Symptom:** During one eval run, the MCP client's cached OAuth token
failed to refresh (`Token refresh failed: 401`), triggered a fresh
browser-based OAuth flow, and then timed out after 300 seconds waiting
for interactive authorization (since the eval script runs unattended).

**Result:** The single MCP-routing question failed and was logged with
its error message; the eval run continued and completed the remaining
questions normally. This validated the fault-tolerance design (Section
1) under a real, unplanned failure rather than only a simulated one.

**Implication for production/AWS deployment:** interactive OAuth is not
viable for a headless server. This needs to be resolved before
deployment — either a service-account-style credential if the MCP
server supports one, or a documented manual re-authorization step as
part of deployment runbooks. Flagged as a blocker for the AWS deployment
step, not resolved in this evaluation cycle.

### Finding 5 — Scoring-artifact false failures (not real errors)
Two questions were flagged as "hallucination check failed" by the
automated spot-check, but manual review showed both answers were
correct:
- **q5** (Amazon net income): answer was "$62.647 billion" — correct,
  but the spot-check only matched the literal digit string "62647"
  and missed the decimal-billions phrasing.
- **q10** (adversarial, Apple nine-month revenue): the original expected
  answer assumed nine-month data was *not* ingested, so the "correct"
  behavior was to decline. After re-ingesting Apple's full 10-Q (which
  does include nine-month cumulative figures), the system correctly
  answered with the cumulative figure — the test set's expected answer
  was stale, not the system's output.

**Takeaway:** the `hallucination_spot_check` substring match is a useful
fast signal but has real limitations (number-formatting variance) and
the test set itself needs to be kept in sync with what data has actually
been ingested. Both are noted as maintenance items rather than pipeline
bugs.

---

## 4. Caveats and Known Limitations

- Retrieval "relevance" in `retrieval_metrics.py` is approximated by
  company-name string matching against source filenames, not manually
  labeled ground-truth relevant chunks. This is directionally useful
  but not a rigorous IR benchmark.
- The synonym-expansion fix for terminology mismatch (Finding 2) is a
  small hardcoded dictionary, not a general solution — a new company
  using yet another synonym for "revenue" (e.g. "turnover") would not
  be covered without adding it explicitly.
- The "clean chunk" injection (Finding 3) is a workaround around the
  markdown chunker's table-corruption issue, not a fix to the chunker
  itself. A more thorough fix would improve `pymupdf4llm` table handling
  or post-process markdown tables to repair split headers generically.
- MCP live-data questions depend on interactive OAuth, which is not
  compatible with unattended/scheduled eval runs or headless production
  deployment (Finding 4) — this remains open.
- The judge-model scoring (`llm_judge_evaluation`) has not yet been run
  across the full test set in this evaluation cycle; RAGAS's dependency
  issues were discovered and worked around, but a full faithfulness/
  relevancy scoring pass is a natural next step before relying on those
  scores for reporting.
- Test set size (16 questions) is at the lower end of the originally
  planned 15–25 range. Expanding coverage — especially more adversarial
  period-confusion cases, given how much this class of bug turned up in
  practice — is recommended before treating this suite as a stable
  regression benchmark.