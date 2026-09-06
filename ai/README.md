# ai — model clients, prompts, and structured outputs

Every call to a hosted or local model lives here: the vision-language capture supervision (§6.5),
the prose-to-spec parser and placement notes (§7.4), the capture agent's tool-calling loop (§6.7),
the shape-completion client (§6.6), and the report narrative.

**Why grouped by technology rather than by consumer.** Prompts, API-key handling, retry and
timeout behaviour, structured-output parsing and cost accounting are the same problems whichever
module is calling. Scattering them across `recon/`, `inspect/` and `service/` means three copies of
each, and three places a key can leak from. See D-025.

**What does *not* live here.** Orchestration that decides *when* to call a model belongs to the
module that owns the decision. This directory owns the call, the prompt and the parse — not the
policy.

**Tests:** `ai/tests/`. Model responses are mocked; no test makes a live API call.

**Not yet populated.** Work begins with stories A-01 onward; see `../stories/README.md`.
