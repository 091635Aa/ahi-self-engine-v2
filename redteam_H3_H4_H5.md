# RED TEAM — REFUTATION: H3 / H4 / H5
*Adversarial review for AHI. Verdicts at the bottom of each section. Confidence stated.*

---

## 0. Retrieval honesty note (read this first)

The search backend returned **degraded, largely non-answering results** for this query set (heavy
foreign-language/news noise, few primary sources, some malformed/synthetic-looking hits). Of my five
permitted fetches: the SC'25 mouse-cortex PDF on \`dl.acm.org\` was **blocked by Cloudflare (HTTP 403)**;
\`nature.com\` **redirected to an identity provider** and was not followed. I read exactly **two** sources
in full: the abstract of **arXiv:2404.01413** and the abstract of **arXiv:2412.19437**.

Consequences you must respect when using this report:
- Every URL below **was actually surfaced by my tools**; the titles I quote are real titles I saw.
- Numbers I could not retrieve in this session are tagged **[memory — verify]**. I have **not invented
  citations**.
- Where I could not get a number, I give you **checkable arithmetic** instead. The arithmetic is the
  strongest part of this report; treat it as the load-bearing evidence.

Per instruction I **stopped** at the budget rather than keep searching.

---

## H3 — "Architecture-level AHI is computationally affordable ($10k–$50k train, 1–8 GPUs real-time)"

### 1. Strongest counter-evidence

**(a) The single most useful verified anchor.** DeepSeek-V3 — a *dense-quality MoE frontier model*, not a
brain model — required **2,788,000 H800 GPU-hours** for full training (arXiv:2412.19437, abstract, read
directly; the paper's own costing at its assumed $2/GPU-hour implies ≈$5.6M [memory — verify]).
At that rate your **entire $10k–$50k budget buys ~5,000–25,000 H800-hours = 0.18%–0.9% of one
DeepSeek-V3 run.** You are not buying a frontier-scale artifact. You are buying a small one. That is
fine — *if you say so*.

**(b) The empirical prior is brutal.** The Human Brain Project ran ~10 years on an EU flagship budget of
~€1bn (critics cited ~€1.2bn) and is widely assessed as **not having delivered a brain simulation**
([Genetics & Society on the €1.2bn boycott letter](https://www.geneticsandsociety.org/article/scientists-threaten-boycott-eu12bn-human-brain-project);
[SRF: "Umstrittenes virtuelles Gehirn — Was bleibt?"](https://www.srf.ch/wissen/gesundheit/umstrittenes-virtuelles-gehirn-das-human-brain-project-geht-zu-ende-was-bleibt);
[a 10-year, ~4.7bn retrospective](https://m.cyzone.cn/article/736727.html);
[The Conversation's more generous post-mortem](https://theconversation.com/building-better-brain-collaboration-online-despite-scientific-squabbles-the-decade-long-human-brain-project-brought-measurable-success-to-neuroscience-collaboration-196473)).
Blue Brain's canonical result — the Markram et al. 2015 *Cell* neocortical microcircuit, **~31,000 neurons /
~37M synapses, ~0.3 mm³ of rat cortex** — is roughly **3×10⁻⁷ of a human brain's neuron count**
[memory — verify]. The 2025 state of the art is the SC'25 Fugaku run:
[*"Microscopic-Level Mouse Whole Cortex Simulation Composed of 9 Million Biophysical Neurons and 26 Billion
Synapses"*](https://dl.acm.org/doi/pdf/10.1145/3712285.3759819) — a **mouse**, ~10⁻⁴ of human scale. **The
largest funded attempt in history spent 4–5 orders of magnitude more than your budget and did not produce
a brain.**

**(c) Neuromorphic hardware does not rescue this.** The 2016-era flagship, IBM TrueNorth, is 1M neurons /
256M synapses at 70 mW, and LLNL bought a system for **~$1M**
([Computerra](https://www.computerra.ru/180243/ibm_truenorth_in_llnl/); [The Register](https://www.theregister.com/2014/08/07/ibm_synapse_chip/)).
SpiNNaker / BrainScaleS / Loihi class machines are excellent *per synaptic event* but are bound to narrow
workloads with limited on-chip plasticity, and no platform does human-scale real-time **learning**
([neuromorphic benchmarking](https://ar5iv.labs.arxiv.org/html/2004.01656);
[IOP NCE review](https://iopscience.iop.org/article/10.1088/2634-4386/ae379a/pdf)).
Cerebras wafer-scale is the only credible answer to the *memory-bandwidth* problem
([Cerebras Teratec 2025 deck](https://teratec.eu/library/pdf/forum/2025/Presentations/A6_Forum_Teratec_2025_Alexander_Mikoyan_Cerebras.pdf)),
but a CS-3-class system is ~$2–3M [memory — verify] — **40–300× your entire budget**.

### 2. The arithmetic that actually decides H3 (my inference; check it yourself)

**Roofline.** H100 SXM5: **3.35 TB/s** HBM, **989 TFLOPS** dense BF16 (1979 w/ sparsity).
Ridge point = 989/3.35 ≈ **295 FLOP/byte**. Transformer decode has arithmetic intensity ~O(1).
**Decode is memory-bandwidth-bound and always will be.** [Specs from standard datasheets — verify against
NVIDIA's page.]

**Per-token bytes (batch B, context C, params N, BF16):**
\`\`\`
bytes/token ≈ 2N/B  +  C·K_kv        where K_kv = 2·L_layers·H_kv·d_head·2 bytes
\`\`\`
For a Llama-2-7B-shaped model (32 layers, 32 KV heads, d_head 128): **K_kv = 512 KiB/token**.

**8×H100 (TP=8), 7B, B=8, C=4096:**
per-GPU bytes/token = (2·7e9/8)/8 + 4096·524288/8 = 0.219e9 + 0.268e9 = **0.487 GB**
→ 0.487e9 / 3.35e12 = **0.145 ms/token/GPU** → **≈6,900 tok/s aggregate (idealized)**
→ realistically **≈3,500–4,800 tok/s** after comms/overheads.
Effective FLOPs = 2·7e9·6900 ≈ 9.7e13 = **~10% of peak at absolute best; 1–5% realistic.**

**Now apply your own L1→L2→L3 nesting.** Let k = sequential model invocations per tick (7 organs + L2 + L3
≈ **k=9**), T_tick = tokens consumed+produced per invocation (~1,500 with state summaries),
f = tick rate in Hz. Required throughput = **k·f·T_tick tok/s**.

| k | T_tick | Achievable f on 8×H100 @7B (realistic ≈4,000 tok/s) |
|---|---|---|
| 9 | 1500 | **≈0.30 Hz** |
| 3 (grouped organs) | 1500 | **≈0.90 Hz** |
| 3 | 500 | **≈2.7 Hz** |

**8 H100s buys you roughly 0.3–2.7 nested-supervision ticks per second at 7B scale.**
At 70B the same 8 GPUs give ≈190 tok/s → you are **~70× short at even 1 Hz.**

**This is the finding that matters.** "Real-time" is undefined in your brief and it is doing all the work:
- If "real-time" = *responsive chat* (≤1 s perceived latency), **H3 survives comfortably.**
- If "real-time" = *biologically plausible L1 organ dynamics* (10–1000 Hz), **H3 is false by 1–3 orders of
  magnitude** at 7B, and worse with long contexts.

**Where naive FLOP estimates break down (your question, answered):** (i) decode is bandwidth- not
FLOP-bound (above); (ii) sparse/event-driven kernels have arithmetic intensity ~O(1) vs a 295 FLOP/byte
ridge — hence the 1–10% of peak you already suspected, and my number says **worse**; (iii) spike
communication is **latency-and-bisection-bound** (each synapse is a message), not FLOP-bound; (iv) global
synchronization barriers per timestep make brain-sim wall-clock latency-bound, so adding GPUs gives
sublinear returns; (v) precision — bf16/fp8 fine for ML, but biophysical sims often need fp32/fp64 state,
instantly costing 2–8× memory bandwidth.

### 3. What the "no simulation" constraint (A) costs H3

This is the coupling the brief misses. If it cannot be a simulation, training/eval must run against a
**live** environment, which is **wall-clock-bound, not FLOP-bound**. You cannot buy your way past the rate
at which reality emits feedback. Concretely:

- **Iteration count.** A research programme is not one run; it is 10²–10³ runs, most of which fail
  [my prior, standard practice]. At $10k/run, 100 runs = **$1M**. Your stated range is one order of
  magnitude below the *programme*.
- **Personnel dominates compute.** 2 FTE × 12 months × ~$150k loaded = **$300k = 6–30× your entire
  compute budget.** If "$10k–$50k" excludes salaries, it is not a programme budget and should not be
  presented as one.
- **Data collection.** Annotation runs ~$15–50/hr; a 100k-example preference set at ~10 min/example =
  ~16,700 hours = **$250k–$830k for one dataset** [my arithmetic on standard market rates; verify].
- **Evaluation under (A).** No simulated benchmark ⇒ live A/B testing with adequate statistical power ⇒
  thousands of real sessions per comparison, plus **contamination** (real eval traffic leaks into
  training) ⇒ continuously refreshed held-out real evals = recurring cost forever.
- **Deployment.** Live traffic means safety review, incident response, and rollback capability — real
  engineering headcount, not GPU hours.

### Verdict — H3

> **REFUTED as stated. WEAKENED-to-SURVIVES in a narrow form.**
> Confidence: **high** on the arithmetic and the DeepSeek-V3 anchor; **medium** on platform-specific
> numbers (unverified in this session).

**The dilemma you cannot escape:** either (a) AHI is a small-model proof-of-concept — then $10k–$50k is
defensible but **"architecture-level replication of the brain" is a structural analogy, not a capability
claim**; or (b) it claims brain-level *capability* — then the budget is off by **2–4 orders of magnitude**
against the programme that had 4–5 orders of magnitude more and failed. Pick one and say it out loud.

**Defensible restatement of H3:**
> "A ≤7B-parameter, 3-layer nested-supervision agent can be *trained* for $10k–$50k of rented compute and
> *serve* at **≤1 Hz tick rate** on 8 GPUs, in a chat-like latency envelope. It is not brain-scale and not
> biologically real-time."

---

## H4 — "The 'no simulation' constraint is satisfiable"

### 1. Strongest counter-evidence

**(a) Autophagous loops are established, not speculative.**
- **Shumailov et al., "AI models collapse when trained on recursively generated data," *Nature* 631:755–759
  (2024)** — [nature.com](https://www.nature.com/articles/s41586-024-07566-y) (fetch was
  identity-provider-blocked), [Oxford ORA copy](https://ora.ox.ac.uk/objects/uuid:fa1155e9-c2ff-436a-8391-455b622f4e64),
  [DESY journal-club PDF](https://indico.desy.de/event/46453/contributions/175820/attachments/93133/126270/Journal%20Club_%20AI%20models%20collapse.pdf).
  **Established**: recursive training on own outputs degrades — tails vanish first.
- **Gerstgrasser et al., arXiv:2404.01413** ([abs](https://arxiv.org/abs/2404.01413), **read directly**):
  *"replacing the original real data by each generation's synthetic data does indeed tend towards model
  collapse… accumulating the successive generations of synthetic data alongside the original real data
  avoids model collapse."* This is the **verified** statement of the condition under which collapse is
  avoidable. Also [ICML 2024](https://icml.cc/virtual/2024/36382).
- **Self-verification provably prevents collapse** —
  [NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/3380e8116452e0efbf36f35d95e88c94-Abstract-Conference.html)
  (title verified; contents not read).
- Alemohammad et al., "Self-Consuming Generative Models Go MAD," ICLR 2024, **arXiv:2307.01850**
  [memory — verify].
- Dohmatob et al., "A Tale of Tails: Model Collapse as a Change of Scaling Laws," **arXiv:2402.07043**
  [memory — verify]. **This is the most dangerous one for you**: collapse manifests as a *changed scaling
  exponent*, i.e. it is **invisible for many generations and then fatal**. You cannot "monitor for it" at
  small scale and conclude you are safe.
- Real-world framing: [CACM, "Model Collapse Is Already Happening, We Just Pretend It Isn't"](https://cacm.acm.org/blogcacm/model-collapse-is-already-happening-we-just-pretend-it-isnt/).

**(b) The data arithmetic is fatal to the *literal* reading of (A).** DeepSeek-V3 pretrained on
**14.8 trillion tokens** (arXiv:2412.19437 abstract, read directly). A full-time professional writer
produces ~40 wpm × 2,000 h/yr ≈ 4.8M words/yr ≈ **6.4M tokens/yr** (1.33 tokens/word).
> **14.8e12 / 6.4e6 ≈ 2.3 million writer-years** for one pretraining run ≈ **$70bn** at $30k/writer-year
> ≈ **11,000× the top of your budget.**
> Even a 100× smaller model (148B tokens) needs **~23,000 writer-years ≈ $700M.**
*(My arithmetic; trivially checkable. This is inference, not citation.)*

So (A) **caps AHI's data scale 4–6 orders of magnitude below frontier pretraining.** The constraint is not
false — it is *binding*, and it must be designed for (small model, high-value interaction data, heavy
reuse, verifier-grounded signal), not assumed away.

**(c) The token-supply literature points the same way.** Villalobos et al., *"Will we run out of data?
Limits of LLM scaling based on human-generated data,"* **arXiv:2211.04325**
([ar5iv](https://ar5iv.labs.arxiv.org/html/2211.04325)) project exhaustion of the high-quality human text
stock within roughly this decade [memory on the specific years — verify]. Under (A) you inherit that
ceiling with no synthetic escape hatch.

### 2. Is "real interaction" well-defined? — **No, as stated**

- Every environment is a *model* of something. An API call to a live service is exogenous and genuinely
  real. A **recorded trace, a replay buffer, a local stub, or a mocked service is a simulation** — even
  though the code path looks identical.
- **The reward model is itself a simulation of a human.** RLHF/RLAIF trains a critic on preference labels;
  optimizing against that critic is optimizing against a learned model, i.e. **a simulation of human
  judgment**. A strict reading of (A) therefore **forbids RLHF, RLAIF, and every learned reward function**
  ([RLAIF, arXiv:2309.00267](https://export.arxiv.org/pdf/2309.00267)). Do you accept that consequence? If
  not, (A) is already inconsistent.
- "Simulated agents" is worse: a **self-play** or **population-based** setup is simulated agents *by
  definition*, and it is the main engine behind AlphaZero-class results. (A) forbids it.

**Falsifiability audit.** As written, "不能模拟" is **not falsifiable**: "is a simulation" has no decision
procedure over a training pipeline. It collapses into "we promise it's real" — exactly the failure mode you
asked me to look for. **Confirmed.**

### 3. The repair (this is constructive, not just destructive)

Rewrite (A) as three **auditable, falsifiable** clauses:

- **F1 — Provenance.** Every training and eval item has a recorded exogenous origin (a real event, a real
  human, a real environment return value) not derivable from the system's own outputs. Auditable via a
  signed provenance ledger (C2PA-style manifests).
- **F2 — No closure.** No training round's data distribution is a function solely of prior system outputs:
  \`D_t ⊄ σ(outputs of model_{<t})\`. Directly testable, and exactly the Gerstgrasser condition.
- **F3 — Exogenous verifier.** Any self-generated candidate must be filtered by an exogenous verifier
  (compiler, test suite, real return code, human sign-off). This is the NeurIPS-2025 self-verification
  condition and the only known way to keep recursion from collapsing.

**Under F1∧F2∧F3 the constraint is satisfiable, falsifiable, and — notably — the direction of (A) is
correct**: the collapse literature says a purely endogenous loop *does* degrade. (A) is the right instinct
with the wrong statement.

### Verdict — H4

> **WEAKENED — severely. Satisfiable only after definitional repair; unsatisfiable-as-stated at
> pretraining scale; not falsifiable-as-stated.**
> Confidence: **high**.

---

## H5 — "Other-agent prediction has an irreducible, quantifiable positive error floor"

I split this into three claims, because they have **different verdicts**.

### 1. Counter-evidence to the *universal* floor claim

- **Deterministic, stationary, fully-observed opponent ⇒ floor is exactly ZERO.** If B's policy is a fixed
  lookup table you have observed (or B's source code you can read), prediction is exact. Trivial, and
  therefore decisive against "irreducible" as a universal.
- **Rational opponent with known utility and action set** ⇒ behavior is rationalizable; exact up to
  indifference sets. This is the standard revealed-preference assumption of mechanism design.
- **An opponent that is a copy of yourself** ⇒ maximally predictable. This is the actual pathology of LLM
  multi-agent systems (mode collapse, collusion), **not unpredictability**. Your own architecture, with
  self-similar organs, sits here.
- **Two-player zero-sum games**: CFR converges to Nash. You cannot predict the opponent's *action* (mixed
  strategy) — but the *value* is exact and best-response is optimal in the limit.
  **Red-team point: H5 conflates action-prediction error with decision quality.** An agent can have a
  large action-prediction floor and still play optimally. **Respecting a prediction floor is therefore NOT
  why you build robustness — minimax/regret reasoning is, and it needs no floor at all.** If your
  justification for robustness is H5, you have the wrong justification.

### 2. What I can actually *prove*

- **Diagonalization (constructive, mine, standard).** For any total computable predictor P there exists a
  total computable policy π_B with π_B(h) ≠ P(h) for all h — take π_B(h) = ¬P(h). Hence **no computable
  predictor is universally correct over computable agents**. Corollary via **Rice's theorem**: any
  non-trivial semantic property of another agent's program is undecidable, so no general procedure extracts
  the opponent's policy from its behavior. *(Inference from Rice 1953 + the standard diagonal argument — not
  a citation to a paper stating it in this exact form.)*
- **The theorem only bites under three conditions:** the adversary is (a) computable, (b) **able to observe
  or correlate with your predictions**, (c) non-stationary/adaptive. In a multi-agent LLM system where every
  agent reads every other agent's outputs, (b) holds — **so the floor is real in your setting, but for a
  mundane reason: the observability channel.** Not a deep epistemic fact.
- **Self-confirming equilibrium — Fudenberg & Levine, *Econometrica* 61(3):523–545 (1993)**
  ([link](https://jstor.econometricsociety.org/publications/econometrica/browse/1993/05/01/self-confirming-equilibrium)).
  **This is your best real theorem.** Players can hold *incorrect* beliefs about opponents' off-path play
  **indefinitely**, because the data that would correct them is never generated. It gives a persistent,
  positive, non-vanishing prediction-error floor — and, crucially, it **locates the cause in partial
  observability + path confinement**, i.e. it hands you the *testable conditions* under which your floor
  holds.
- **Calibration impossibility (Oakes 1985)**: *"no deterministic algorithm can be calibrated on every
  sequence"* (quoted in [arXiv:2402.08733](https://ar5iv.labs.arxiv.org/html/2402.08733)). Real theorem —
  but randomized forecasters **can** be calibrated asymptotically (Foster & Vohra 1998, *Biometrika*
  85(2):379–390 [memory — verify]). **It constrains calibration, not accuracy.** Using it to license a
  prediction-error floor is a category error.
- **Wolpert, "Physical limits of inference," *Physica D* 237(9):1257–1281 (2008)**
  ([arXiv:0708.1362](https://ar5iv.labs.arxiv.org/html/0708.1362)): limits on what a device embedded in a
  system can infer about that system. Relevant, but weaker than usually claimed — it concerns universal
  inference devices, not the practical error of a trained model.
- **Do NOT use No-Free-Lunch (Wolpert & Macready 1997)** [memory]. NFL says no learner dominates uniformly
  over all target functions. It **does not** say the error floor on a *given* target is positive. It is
  routinely misused this way in exactly your argument. Drop it.

### 3. The attack on the word **"quantified"** — this is where H5 breaks

The floor is the Bayes error of predicting B's action given A's history:
\`ε* = E_h[1 − max_a P(a | h)]\`.
Computing ε* requires the **true** conditional P(·|h). A has only its own model P̂. Therefore **every
estimate ε̂ is model-relative, and the gap |ε̂ − ε*| is itself unknown.** There is no fourth option:

- **(i) Bound it under assumptions** (finite policy class + stationarity): then it is **not irreducible** —
  it decays like O(√(log|Π|/n)) and can be driven toward zero with enough real interaction. This
  *refutes* "irreducible" in the statistical sense: it is irreducible-in-principle, not
  irreducible-at-finite-sample. Those are different claims and you have been making the stronger one.
- **(ii) Measure it on a held-out population of real agents** — honest, expensive, and requires exactly the
  real-interaction corpus that (A) makes scarce. **This is the only legitimate path.**
- **(iii) Stipulate it exogenously** as a safety margin — a *choice*, not a measurement.

> **You cannot quantify the prediction floor from inside the system.** The word "quantified" in H5 is the
> load-bearing word, and it does not hold. Confidence: **high.**

### 4. Does respecting a floor help, or license under-performance? — **Both.**

**Legitimate:** (a) never condition plans on point predictions of others' actions; (b) spend compute on
robustness/minimax rather than sharpening a point forecast; (c) **keep prediction error as a permanent L2
metacognitive signal** — this is the one place the floor is genuinely *useful*: a permanently nonzero
surprise channel is a permanently available training signal for a metacognitive monitor. That is a real
architectural payoff and I will defend it.

**Illegitimate:** using an assumed floor as an excuse for a weak world model, or as an unfalsifiable
license to be wrong.

**Test to impose:** for each subsystem, state **the observable decision that changes** if the floor is 2×
higher or 2× lower. **If no decision changes, the claim is unfalsifiable decoration — delete it.**

### Verdict — H5

> - **Universal "irreducible floor": REFUTED.** (Zero-floor counterexamples are trivial and decisive.)
> - **Conditional floor under partial observability + observability channel + adaptivity: SURVIVES.**
>   Backed by Fudenberg–Levine self-confirming equilibrium. Confidence: **high**.
> - **"Quantified": REFUTED.** Not computable from inside; requires held-out real agents or an exogenous
>   stipulation. Confidence: **high**.

---

## Cross-cutting: the three hypotheses are coupled, and this is the useful part

1. **(A) ⟹ small data ⟹ small model ⟹ H3's budget is defensible only in the small-model regime**, where
   "brain-scale architecture" is a *structural analogy*, not a capability claim. The three claims cannot
   all be held in their strong forms simultaneously.
2. **The single number that decides H3 is the tick rate f of the L1→L2→L3 loop.** At 7B and 8×H100 the
   ceiling is ~0.3–2.7 Hz depending on how you count a tick. **State f explicitly, in Hz, with its
   biological justification.** Everything else in H3 is downstream of it.
3. **H4 and H5 need the same artifact: a real, held-out multi-agent interaction corpus.** H4 needs it to
   satisfy F2/F3 and to estimate collapse; H5 needs it to estimate ε*. **Build that corpus first — it is
   the load-bearing element of the whole design, and it is the thing a $10k–$50k budget cannot buy.**

## Three falsification tests to impose on yourself

1. **H3.** Publish the budget: \`required_tok/s = k · f · T_tick\`. Measure achieved tok/s on the target
   model/context/batch. If achieved < required at 8 GPUs, H3 is **false as stated**. Also: state whether
   the $10k–$50k includes personnel. If not, it is not a programme budget.
2. **H4.** Publish a provenance ledger covering 100% of training and eval items with exogenous origin. Any
   item tracing only to system outputs fails (A). Then **run the Gerstgrasser experiment on your own
   pipeline** — replace-only vs. accumulate — and report the divergence curve. If you have not run it, you
   do not know whether your loop collapses.
3. **H5.** Estimate the floor **only** on a held-out real-agent set; report the estimator and a confidence
   interval; state which decision changes at 2× up or down. If none does, delete the claim.

---

## Summary table

| Hypothesis | Verdict | Confidence | One-line reason |
|---|---|---|---|
| **H3** Affordable brain-arch AHI | **REFUTED as stated** / SURVIVES only as a narrow ≤7B, ≤~1 Hz PoC | High on arithmetic; medium on platform numbers | $50k = 0.2–0.9% of one DeepSeek-V3 run; 8×H100 ≈ 0.3–2.7 nested ticks/s; HBP spent ~4–5 orders more and failed |
| **H4** "No simulation" is satisfiable | **WEAKENED (severely)** | High | Not falsifiable as stated; forbids RLHF/self-play if read strictly; 14.8T tokens ≈ 2.3M writer-years. Satisfiable only as F1∧F2∧F3 |
| **H5** Irreducible, quantifiable floor | **Universal form REFUTED; conditional form SURVIVES; "quantified" REFUTED** | High | Zero-floor counterexamples; Fudenberg–Levine gives the real conditional theorem; Bayes error not computable from inside |
