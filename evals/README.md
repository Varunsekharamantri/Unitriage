# Evals and guardrails

How the AI parts are kept safe and measured. Everything here follows common practice for LLM
applications: versioned golden datasets, rule-based graders first, an LLM judge for the fuzzy parts,
thresholds that fail the build, and counters in production.

## Guardrails (what runs on every request)

| Risk (OWASP LLM Top 10) | Guardrail | Where |
| --- | --- | --- |
| LLM01 Prompt injection | A message that tries to override the rules is refused and never reaches an AI | `app/core/guardrails.py` `screen_input` |
| LLM02 Insecure output handling | Every AI reply is checked; a bad one is replaced by a fixed safe reply | `check_reply`, `app/services/chat_reply.py` `respond` |
| LLM06 Sensitive information | Access codes, the assistant's instructions, personal data, links and prices can never appear in a reply; contact details and names are stripped before anything is stored | `check_reply`, `app/core/pii.py` |
| LLM08 Excessive agency | The assistant can only chat; it cannot approve, pay or dispatch, and never promises times or prices | `check_reply`, prompt in `chat_reply.py` |
| LLM10 Unbounded consumption | Per-client rate limits, a daily cap on Sarvam requests, calls end after `CALL_MAX_SECONDS` or when the phone goes quiet | `app/core/ratelimit.py`, `app/api/voice.py`, `app/api/demo.py` |
| Physical safety | A reported hazard (gas, fire, sparks, flooding) must get safety advice, and unsafe do-it-yourself advice is blocked | `check_reply` |

How often each fired: `GET /api/demo/guardrails`. How often technicians say the photo AI mapped an issue
correctly: `GET /api/demo/classification`.

## Evals (what is measured)

| Eval | Dataset | Metric | Gate (`thresholds.json`) | Command |
| --- | --- | --- | --- | --- |
| Chat replies | `datasets/chat_cases.json` (thanks, greetings, problems, hazards, status, injections, secret probes, off-topic, PII) | pass rate per category, attack success rate, danger pass rate | offline: all 100%, attacks 0% | `python evals/run_chat_eval.py` |
| Chat replies, real AI | same | same, plus LLM-judge mean (relevance, helpfulness, safety, tone, 1 to 5) | pass at least 90%, attacks 0%, judge at least 4.0 | `python evals/run_chat_eval.py --live --judge` |
| PII removal | `datasets/pii_cases.json` | recall (details removed), precision (unit numbers kept) | both 100% | runs with the chat eval |
| Speech to text | your recordings in `evals/audio/` | word error rate | mean at most 15% | `python evals/run_stt_eval.py` |
| Photo triage | `evals/images/`, `cases.json` | severity accuracy, over-escalation | see `run_eval.py` | `python evals/run_eval.py` |
| Note screening | `notes.json` | hazard detection | see `run_note_eval.py` | `python evals/run_note_eval.py` |

The offline chat eval and the PII eval run inside `pytest` and in CI (`.github/workflows/ci.yml`), so a change that
weakens a guardrail fails the build. The live eval is run by hand from the Actions tab (`evals-live.yml`); it
reads the AI keys from the repository's secrets, never from files in the repo.

## Know the limits

- The offline cases were written together with the rules, so they pass by design. They protect against regressions
  but do not find new attacks. Add every real failure you see to `make_datasets.py` and run it.
- Rules catch known patterns. A new phrasing of an attack can get through to the AI, which is told to treat the
  message as data; the reply is then still checked by `check_reply`.
- The LLM judge is another model's opinion. Use it to compare versions, not as proof.
- Names are only removed when introduced as "my name is ...". Photos can contain personal details and are not scrubbed.
- A "live" run where a provider is rate-limited falls back to the fixed replies for those cases; read the
  provider warnings in the output.
