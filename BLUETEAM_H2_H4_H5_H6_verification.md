# BLUE TEAM — VERIFICATION REPORT: H2 (federation + nested supervision → evaluative self-awareness), H4 (no-simulation / real-grounding path), H5 (irreducible error floor in other-agent prediction), H6 (independent test methods cross-validate)

## 0. Provenance & method limits — READ FIRST

**Budget:** 8/8 `web_search` used; 5/5 `web_fetch` attempts used — **all 5 returned no content** (transport-level `fetch failed` / `unsupported content type "application/pdf"`). This is the *same* failure the H1/H3 blue team hit.

**Recovery (this is new relative to the H1/H3 report):** `curl` from the shell **does** reach the network. All abstracts below marked [S] were retrieved this session via:
- `api.openalex.org/works` (titles + abstract-inverted-index),
- `api.crossref.org/works/<DOI>` (DOI → title/venue/date existence checks),
- `arxiv.org/abs/<id>` HTML (abstracts),
- `api.semanticscholar.org` (citation counts; fully rate-limited after the first few calls).

**Still hard-blocked at the network layer:** science.org (full text / PDF), creativemachineslab.com (TLS reset), waymo.com (TLS reset), pubmed.ncbi.nlm.nih.gov (HTTP 203 challenge), r.jina.ai proxy (empty). So: **no full text of any Science / Science Robotics paper was read.** Only publisher-supplied abstracts.

**Tags:**
- **[S]** = verbatim/paraphrase-quote retrieved **this session** (API or HTML), URL cited.
- **[S-sec]** = retrieved this session, but secondary source (course slides, survey) — weaker.
- **[K]** = my prior knowledge of the canonical literature. Probably right in outline, **NOT verified this session**.
- **[I]** = my own inference / arithmetic / design judgment.

Do not let [K] items into a design doc as settled. §6 lists the URLs that would convert each [K]→[S].

---

## 1. H2 — self-models + multi-layer nested supervision ⇒ "evaluative self-awareness"

### 1.1 What is actually IMPLEMENTED (retrieved evidence)

| System | What the self-model is | What it concretely buys | Numbers | Tag |
|---|---|---|---|---|
| **Bongard, Zykov & Lipson, *Science* 314:1118, 2006, "Resilient Machines Through Continuous Self-Modeling"** (DOI 10.1126/science.1133687, verified Crossref; 718 cites OpenAlex / 808 S2) | A learned model of **its own morphology** inferred from **actuation–sensation relationships** ("indirectly infer its own structure") | Uses the self-model to **generate forward locomotion**; when **a leg part is removed** it **re-adapts the self-models, producing qualitatively different, alternative gaits** — autonomous recovery from unanticipated morphological damage | Abstract gives no case count | [S] abstract |
| Same robot, third-party course slides (SLC, 2009) | Self-model = **16 numerical parameters**, learned by a **genetic algorithm**; robot knows it has **9 body parts** but not the attachment topology; sensors: 2 tilt, 8 joint-angle, 8 motors | "Learns to walk based on self-model"; "can recover from damage" | 16 params / 9 parts / 8 joints | [S-sec] |
| **Chen, Kwiatkowski, Vondrick & Lipson, *Science Robotics* 7(68):eabn1944, 2022, "Fully body visual self-modeling of robot morphologies"** | Instead of forward kinematics, a **query-driven self-model answering space-occupancy queries conditioned on the robot's state**; continuous in space, memory-efficient, **fully differentiable, kinematic-aware** | Enables **motion planning and control tasks without prior knowledge of which morphology aspects matter**; replaces "try it in physical reality" with internal simulation of candidate actions | **"a visual self-model is accurate to about 1% of the workspace"** | [S] abstract |
| **MUSE — "Metacognition for Unknown Situations and Environments", *Neural Networks* 2025** (DOI 10.1016/j.neunet.2025.108131, 0 cites yet) | Explicit **metacognitive layer**: **competence awareness** (self-assessment) + **self-regulation** (strategy selection), implemented twice: **world-model-based** and **LLM-based** | "Continually learns to assess its competence on a given task and **uses this self-assessment to guide iterative cycles of strategy selection**"; better **out-of-distribution** task solving than model-based RL and prompt-only LLM agents | "high competence awareness and **significant improvements**"; **no effect sizes in abstract** | [S] abstract |
| **Reflexion** (arXiv 2303.11366) | **Linguistic** self-evaluation: agent "verbally reflect[s] on task feedback signals", stores reflective text in **episodic memory**, re-decides next trial | Self-evaluation **changes behaviour** without weight updates; accepts external **or internally simulated** feedback | **91% pass@1 on HumanEval** | [S] abstract |
| **Self-Refine** (arXiv 2303.17651) | The **same LLM** is generator, critic and refiner; iterative self-feedback | Consistent gains across 7 tasks with **no training, no RL, no extra data** | **~20% absolute average** improvement; human-preferred | [S] abstract |
| **DreamerV3** (arXiv 2301.04104) | **NOT a self-model** — a learned model *of the environment* (observations/rewards) used for imagination | >150 tasks, one config; first to collect Minecraft diamonds from scratch | — | [S] abstract + **[I] classification** |
| **JEPA family** — note: the URL I retrieved (arXiv 2301.08243) is **I-JEPA**, "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" | Predicts **representations of target blocks in an image**; a *world*/representation model, **not an agent self-model** | Semantic image representations without hand-crafted augmentations | ViT-H/14, 16×A100 <72 h | [S] abstract + **[I] classification** |

### 1.2 The honest classification everyone gets wrong
**Every "world model" citation in the AHI brief (Dreamer, V-JEPA) is an *environment* model, not a *self*-model.** [I] They predict the world; they do not take the agent's own body/policy/cognition as their object. The **only** retrieved systems that genuinely self-model are: Lipson-lineage robots (Bongard 2006; Chen 2022), MUSE (competence self-assessment), and the LLM self-critique loops (Reflexion, Self-Refine) — and the latter are **not grounded in real interaction**, which collides with constraint (A).

### 1.3 Minimum architecture that has demonstrably produced functional self-modelling
From Bongard 2006 + Chen 2022 + MUSE, the common minimum is [I]:
1. a **closed interaction loop** producing the agent's own (command, sensed-consequence) stream;
2. a **learned regressor whose input is the agent's own candidate action** and whose output is a **prediction about the agent itself** (structure, kinematics, occupancy, or competence);
3. a **search/optimizer over that regressor whose output is executed** (GA in 2006; differentiable planning in 2022; strategy selection in MUSE).

Condition (3) is what separates *functional* self-modelling from a decorative self-report head. Both Lipson-lineage systems satisfy all three; **no retrieved system requires more than one self-model layer.**

### 1.4 Verdict: **PARTIAL**
- **SUPPORTED (weak form):** functional self-modelling is real, replicated across independent labs/lines, and *does* drive behaviour that a fixed reactive controller cannot produce — specifically, **autonomous recovery from unmodelled morphological damage** (Bongard 2006 [S]) and **planning without physical trial** (Chen 2022 [S]).
- **UNSUPPORTED (the actual H2 claim as written):** I retrieved **zero** evidence that a **federation + multi-layer *nested* supervision hierarchy (L1→L2→L3)** produces *more* than a single self-model layer. The closest real system, **MUSE, has 2 layers (policy + metacognition), not 3** [S]. Reflexion/Self-Refine are 2-layer loops with [S] gains, but they are ungrounded and their gains are contested (§1.5).
- **Quantification gap (state plainly):** no retrieved study performs an **A/B of "explicit self-model" vs "reactive policy at matched capacity"** with effect sizes. Chen 2022's quantified result (~1% workspace accuracy) is an *accuracy* number, not a *capability delta*. Bongard 2006's capability delta is real but **qualitative** in the abstract. So "self-model yields capabilities beyond a reactive policy" is **supported by demonstration, not by measurement.**
- **The "prefrontal L3 supervisor" has no published analogue I could find.** [I] Treat it as an untested architectural bet, not a replication of known results.

### 1.5 Contested / must-not-cite-as-settled
- **LLM self-critique gains are contested** [K]: "Large Language Models Cannot Self-Correct Reasoning Yet" (Huang et al., arXiv 2310.01798) reports intrinsic self-correction degrading reasoning performance; part of Self-Refine's gain may be extra inference compute. **Not verified this session.**
- Bongard 2006's citation count differs by source (718 OpenAlex vs 808 S2) — minor, but a reminder these APIs disagree.
- Tani's predictive-coding robotics work: I retrieved **only a slide-deck URL** (ircn.jp Tani PDF), no primary text. **The Tani leg of H2 is unverified this session.**

---

## 2. H4 — "no simulation / real-grounding" training path

### 2.1 Retrieved hard numbers for real-interaction learning

| Evidence | Retrieved numbers | Tag |
|---|---|---|
| **Open X-Embodiment / RT-X** (arXiv 2310.08864) | Dataset assembled from **22 different robots**, **21 institutions**, **527 skills**, **160,266 tasks**; RT-X shows **positive transfer** — a single high-capacity model improves *multiple* robots by leveraging other platforms' experience | [S] abstract |
| **RoboCat** (arXiv 2306.11706) | Multi-embodiment, multi-task generalist; adapts to new tasks/robots with **only 100–1000 examples**; "a trained model itself can be used to generate data for subsequent training iterations, thus providing a basic building block for an autonomous improvement loop"; evaluated on **three real robots** | [S] abstract |
| **Chatbot Arena** (arXiv 2403.04132) | **240K+ real user votes**, pairwise, crowdsourced; **crowdsourced human votes agree with expert raters** — real-user preference collection is credible at scale | [S] abstract |
| **Cicero / Diplodocus** (arXiv 2210.05492) | **200-game no-press Diplomacy tournament, 62 human participants**, beginner→expert; two Diplodocus agents scored above **all** participants who played >2 games | [S] abstract |
| **Waymo / Tesla fleet mileage** | Only **search-result snippets** (waymo.com/safety/impact; a Zenodo CAV-assurance paper snippet "Injury crashes: 0.41/million miles (Waymo)"; a Yahoo article on "Tesla reaches major Cybercab milestone of 1 million unsupervised miles"). **waymo.com was TLS-blocked; nothing verified.** | [S-snippet, UNVERIFIED] |

### 2.2 Does real grounding *prevent* the failure modes synthetic data causes?
**No retrieved evidence.** This is the weakest link in H4 and I want it stated bluntly [I]:
- What is established [S]: real-robot data **aggregates and transfers across embodiments** (OXE/RT-X), and self-generated **real** rollouts can close an improvement loop (RoboCat).
- What is **not** established by anything I retrieved: a controlled comparison showing a specific failure mode that appears under synthetic data and disappears under real-interaction grounding. The sim-to-real gap is real [K], but "real data prevents synthetic-data pathologies" is a *plausible thesis*, not a retrieved result.
- **Counter-evidence to the strict form of constraint (A)** [S]: **RoboCat — the flagship "self-improving real robot" system — trains across simulated *and* real arms.** AlphaStar's league machinery ran in simulation (§3.2) [K]. **No retrieved system achieved superhuman multi-agent competence from real interaction alone.** A hard "zero simulation" constraint therefore costs you the self-play/league/exploration machinery that the strongest multi-agent results depend on. Recommend restating (A) as "no *simulated evaluation*; simulation may be a training scaffold" unless the user truly means zero sim.

### 2.3 Cost/scale reality check
- Real-mile flywheels require **owned fleets**; the retrieved evidence for their scale is snippet-level only. I could **not** retrieve a defensible cost-per-useful-sample figure for Tesla/Waymo-style loops or for AlphaFold-style wet-lab loops this session [I-limitation].
- The one retrievable *structural* fact [S]: OXE needed **21 institutions** to reach 160k tasks — i.e., the community's answer to the cost of real data was **federate, not build one giant lab.**

### 2.4 Verdict: **PARTIAL**
Feasible and demonstrated at **moderate** scale with hard numbers (robotics + real-user preference + a real human tournament). **Unsupported**: that real grounding prevents synthetic failure modes; that a *no-simulation-at-all* path reaches frontier multi-agent performance; any credible cost model for fleet-scale flywheels from what I retrieved.

---

## 3. H5 — irreducible positive error floor in other-agent prediction

### 3.1 Formal results retrieved

| Result | Content | Tag |
|---|---|---|
| **Wolpert & Macready, "No free lunch theorems for optimization", *IEEE TEC* 1(2), 1997** (DOI 10.1109/4235.585893; **14,318 cites**) | "for any algorithm, any elevated performance over one class of problems is offset by performance over another class"; results include **time-varying problems** and **a priori head-to-head minimax distinctions** between algorithms | [S] abstract |
| **Perdomo, Zrnic, Mendler-Dünner & Hardt, "Performative Prediction" (arXiv 2002.06673, ICML 2020)** | "When predictions support decisions they may influence the outcome they aim to predict… When ignored, performativity surfaces as **undesirable distribution shift**"; introduces **performative stability** — predictions "calibrated **not against past outcomes, but against the future outcomes that manifest from acting on the prediction**"; **performative prediction strictly subsumes strategic classification** | [S] abstract |
| **Bernstein, Givan, Immerman & Zilberstein, "The Complexity of Decentralized Control of Markov Decision Processes", *Math. of OR* 27(4), 2002** (DOI 10.1287/moor.27.4.819.297) | Decentralized (multi-agent) control under partial observability is **computationally intractable** — the canonical NEXP-completeness result | [S] DOI/title; **[K]** the exact NEXP/decidability statement |
| **Hypergame theory survey (arXiv 2507.19593, 2025)** | "Classical game-theoretic models typically assume rational agents, complete information, and common knowledge of payoffs — assumptions that are **often violated in real-world MAS** characterized by uncertainty, **misaligned perceptions, and nested beliefs**"; hypergames model agents holding **divergent beliefs about the structure, payoffs, or available actions** ("perceptual games") | [S] abstract |
| **Albrecht & Stone, "Autonomous agents modelling other agents: A comprehensive survey and open problems", *Artificial Intelligence* 258, 2018** (DOI 10.1016/j.artint.2018.01.002; 428 cites) | The field's own catalogue of opponent-modelling methods **and open problems** | [S] DOI/title/cites; [K] contents |
| **Papoudakis et al., "Dealing with Non-Stationarity in Multi-Agent Deep RL" (arXiv 1906.04737)** | "additional learning problems arise **due to the continually changing decision-making policies of agents**"; surveys centralized training, **learning representations of the opponent's policy**, meta-learning, communication | [S] abstract |
| **de Weerd, Verbrugge & Verheij, "Higher-order theory of mind in the Tacit Communication Game", *BICA* 2014** (DOI 10.1016/j.bica.2014.11.010) | Recursive nested-belief agents | [S] existence only — **abstract NOT retrievable**. The widely cited claim that higher-order ToM is **not monotonically beneficial** is **[K], unverified here**. |

### 3.2 Empirical: does *calibrated/approximate* modelling of others beat confident point prediction?

| Evidence | Content | Tag |
|---|---|---|
| **Cicero / RL-DiL-piKL (arXiv 2210.05492)** — **the strongest retrieved empirical support** | Self-play alone "is insufficient for achieving optimal performance in domains involving cooperation with humans". Their fix **regularizes a reward-maximizing policy toward a human imitation-learned policy** — i.e. an **approximate, deliberately regularized model of human play**, not an oracle — yielding a **no-regret guarantee under a modified utility**, and an agent that both "provides a model of human play" **and** "responds well to this human model". Validated in a **real 200-game tournament vs 62 humans**. | [S] abstract |
| **AlphaStar, *Nature* 2019** (DOI 10.1038/s41586-019-1724-z; 3,528 cites) | Grandmaster-level StarCraft II. Its architecture addresses non-stationary/unknown opponents with a **league of main agents + main exploiters + league exploiters** rather than one confident opponent model | [S] DOI/title; **[K]** league composition |
| **Libratus / DeepStack (Science 2018)** | Reported to be superhuman **without real-time opponent modelling**, via self-play/equilibrium approximation | **[K] only** — my Crossref DOI check for 10.1126/science.aao1734 **failed**, so I could not verify even the DOI this session. Do not cite from this report. |

### 3.3 Verdict: **SUPPORTED (weak/formal form) — PARTIAL to UNSUPPORTED (quantitative form)**
- **SUPPORTED [S]:** (i) no prior-free predictor dominates across all opponent/problem classes (NFL); (ii) others' policies change **because** you act, so the prediction target is *performative*, and ignoring this shows up as distribution shift; (iii) classical common-knowledge/rationality assumptions are routinely violated in real MAS (hypergame survey); (iv) optimal decentralized control is intractable.
- **PARTIAL/UNSUPPORTED:** I did **not** retrieve any theorem giving an **explicit positive minimax lower bound on opponent-prediction error**. The formal floor is *structural* (no universal dominance; target moves with your policy), not *numeric*. [I] If AHI's design doc says "there is a proven positive error floor ε>0 for predicting others", that is **stronger than the literature I could retrieve**; the defensible phrasing is "no universally accurate predictor exists, and the target is endogenous."
- **Strongest empirical result retrieved:** **Cicero** — an *approximate, regularized* model of humans, validated against **62 real humans** [S]. That is direct evidence that calibrated/approximate other-modelling beats oracle-assumption-free play in a real interaction setting. Its scope limit: **no-press Diplomacy** (no communication), so it dodges the hardest nested-ToM part.
- **Cheap mathematical floor you can state as yours [I]:** over the class of *all* binary sequences, any deterministic predictor's worst-case error is 1/2 (diagonal/counting argument); with an assumption class you get a floor only relative to that class. This is a trivial fact — but it is exactly why "irreducible error" must be **stated relative to an assumption class**, not absolutely. That framing is the honest version of constraint (B) 不能预测别人是绝对准确的.

---

## 4. H6 — can multiple independent test methods cross-validate one architecture?

### 4.1 Retrieved methodology anchors

| Anchor | Why it matters here | Tag |
|---|---|---|
| **Platt, "Strong Inference", *Science* 146:347, 1964** (DOI 10.1126/science.146.3642.347) | The canonical rule: progress comes from **multiple competing hypotheses + crucial experiments that *exclude***, not from accumulating agreeing observations | [S] DOI/title |
| **Kriegeskorte, Simmons, Bellgowan & Baker, "Circular analysis in systems neuroscience: the dangers of double dipping", *Nature Neuroscience* 2009** (DOI 10.1038/nn.2303) | The canonical proof that **selecting** data with a statistic and then **testing** on the same data is invalid — the operational definition of a *non-independent* test | [S] DOI/title |
| **Poldrack, "Can cognitive processes be inferred from neuroimaging data?", *TICS* 2006** (DOI 10.1016/j.tics.2005.12.004) | **Reverse inference**: a shared substrate/method does not license a cognitive conclusion; tests sharing a localizer share its assumptions | [S] DOI/title |
| **Kahneman & Klein, "Conditions for intuitive expertise: A failure to disagree", *American Psychologist* 2009** (DOI 10.1037/a0016755) | The canonical **adversarial collaboration** — and its title is the honest outcome: agreeing on a *method* did **not** force agreement on the *conclusion* | [S] DOI/title |
| **Oberauer & Lewandowsky, "Addressing the theory crisis in psychology", *Psychonomic Bull. & Rev.* 2019** (DOI 10.3758/s13423-019-01645-2) | Verbal theories are too vague to be falsifiable; **model comparison requires formal specification** | [S] DOI/title |
| **"From Adversarial Tests to Prediction Ledgers: A Role-Typed Framework…"** (Zenodo) | A framework requiring each test to have a **distinct role** pre-declared | [S-snippet, **UNREAD**] |

### 4.2 Genuinely INDEPENDENT test families for AHI — and why each is independent
"I independent" = the family's error is **not** a function of the architecture's own fitting procedure, and it can **falsify** rather than decorate. [I]

**Tier 1 — genuinely independent (use these; agreement here is informative)**
1. **Exogenous held-out task stream.** Tasks generated by the *world* (real users, real robots, real opponents) that AHI did not author and cannot inspect at train time. Independence source: the data-generating process is outside the system. *Requires constraint (A) to be meaningful — this is where "no simulation" actually pays.*
2. **Interventional ablation with capacity-matched controls.** Lesion L1/L2/L3 *and* a compute/parameter-matched control; require a **double dissociation** (lesion A breaks task X but not Y; lesion B the reverse). Independence: inference comes from **do(X)**, not from fit to observations.
3. **Exploitability against a dedicated adversary.** An opponent population *optimized against AHI* (AlphaStar-style exploiters [K]) reports **exploitability**, not self-reported competence. Independence: the adversary is chosen by an external optimizer, and the metric is minimax, not fit.
4. **Cross-embodiment / cross-operator transfer.** Performance on robots/hosts configured by *another* lab (the OXE pattern: 21 institutions, 22 robots [S]). Independence: you do not control the other embodiment's idiosyncrasies.
5. **Calibration of the system's own uncertainty** (ECE/Brier/reliability diagrams on its self-assessments), scored on **rollouts the metacognitive layer never trained on**. Independence: different dependent variable (metacognitive accuracy), *provided* item 6's circularity is avoided.

**Tier 2 — conditionally independent (only if pre-registered and out-of-sample)**
6. **Pre-registered out-of-sample predictions about *itself*** — the MUSE-style competence claim [S] tested on held-out rollouts. **Circular if** the competence estimator is trained on the same rollouts it is scored on (= double dipping, Kriegeskorte [S]).
7. **Human/biological correspondence** (error patterns, reaction-time distributions, neural signatures). **Circular if** the comparison dataset is the one used to fit parameters (model mimicry). Must use an adversarial collaboration protocol — and expect a *failure to disagree* (Kahneman & Klein [S]).
8. **Formal identifiability / null-model exclusion.** Prove AHI's behaviour class is not realizable by a capacity-matched one-layer reactive policy [I]. Independent of data, but shares your formalism — so it constrains your *theory*, not your *implementation*.

**Tier 3 — NOT independent (agreement is circular; do not count these as separate evidence)**
- Task X and a cosmetic variant X′ scored by the **same pipeline/metric** — one test wearing two hats.
- Two LLM-judge or self-report scores — both inherit the same judge biases (and, if the judge *is* the system, literal self-grading).
- An "ablation" that also **removes capacity** — confounded with model size, not evidence about the module.
- Two neural/behavioural measures sharing the same **functional localizer or preprocessing** (Poldrack [S]).
- Any family whose data was used for **selection/tuning** and then reused for scoring (Kriegeskorte [S]).

### 4.3 Verdict: **SUPPORTED as methodology — with a hard caveat**
Multi-method cross-validation is legitimate **only** when the families' errors are decorrelated; the canonical citations show the default failure is *circularity* [S] and that adversarial collaboration **frequently fails to converge even with agreed methods** [S]. Practical rule for AHI [I]: **count a family as independent only if you can name, in advance, the observation that would make it fail** — otherwise it is Tier 3.

---

## 5. One-line verdicts

| Hypothesis | Verdict | Single strongest citation |
|---|---|---|
| **H2** federation + nested supervision ⇒ evaluative self-awareness | **PARTIAL** — self-modelling is real and functional (damage recovery, planning without trial, competence-aware strategy selection); the **multi-layer *nested*** part and any **quantified** self-model-vs-reactive gain are **UNSUPPORTED** by retrieved evidence. Closest real system has **2** layers, not 3. | Bongard/Zykov/Lipson, *Science* 2006, [10.1126/science.1133687](https://doi.org/10.1126/science.1133687) |
| **H4** no-simulation / real-grounding path | **PARTIAL** — works at moderate scale with hard numbers (22 robots/21 institutions/160,266 tasks; 100–1000-shot adaptation; 240K real votes; a real 62-human tournament). **Unsupported:** that real grounding *prevents* synthetic-data failure modes; that zero-simulation reaches frontier multi-agent performance; any cost model for fleet flywheels. | Open X-Embodiment, [arXiv:2310.08864](https://arxiv.org/abs/2310.08864) |
| **H5** irreducible error floor for other-agent prediction | **SUPPORTED (structural), PARTIAL→UNSUPPORTED (numeric)** — no universal predictor (NFL); the target is **performative**; classical common-knowledge assumptions routinely violated. No retrieved theorem yields an explicit positive ε. Best empirical: approximate/regularized human model beat 62 real humans. | Wolpert & Macready 1997 [10.1109/4235.585893](https://doi.org/10.1109/4235.585893); Perdomo et al. [arXiv:2002.06673](https://arxiv.org/abs/2002.06673); Cicero [arXiv:2210.05492](https://arxiv.org/abs/2210.05492) |
| **H6** independent methods cross-validate one architecture | **SUPPORTED as method, with caveat** — real independence comes from **exogenous data, interventions, adversaries, and foreign embodiments**; fit-based agreement is circular (double dipping) and adversarial collaboration often ends in a *failure to disagree*. | Kriegeskorte 2009 [10.1038/nn.2303](https://doi.org/10.1038/nn.2303); Platt 1964 [10.1126/science.146.3642.347](https://doi.org/10.1126/science.146.3642.347) |

---

## 6. Conversion list — [K] → [S] (fetch these next session, in this order)
1. `https://www.science.org/doi/10.1126/science.1133687` (full text via a non-CF-blocked mirror or institutional proxy) — get the **number of damage cases recovered** for Bongard 2006.
2. `https://www.science.org/doi/10.1126/scirobotics.abn1944` — Chen 2022 **capability A/B**, not just the ~1% accuracy.
3. `https://www.science.org/doi/10.1126/science.aao1734` (Libratus) — verify DOI + whether opponent modelling was used.
4. `https://doi.org/10.1038/s41586-019-1724-z` (AlphaStar) — league composition, verbatim.
5. `https://doi.org/10.1016/j.bica.2014.11.010` (de Weerd) — does higher-order ToM actually fail to help?
6. `https://doi.org/10.1287/moor.27.4.819.297` — exact complexity class statement.
7. `https://waymo.com/safety/impact/` + Tesla fleet-mile disclosures — real-mile flywheel scale/cost.
8. `https://www.creativemachineslab.com/uploads/6/9/3/4/69340277/task-agnostic_self-modeling_machines.pdf` (Kwiatkowski & Lipson focus article) — the "task-agnostic self-model" claim in the authors' own words.
9. `https://arxiv.org/abs/2310.01798` — the anti-self-correction result that contests Reflexion/Self-Refine gains.

---

## 7. What I would tell the architect if asked directly [I]
1. **H2 as written is a bet, not a replication.** Keep the self-model (evidence: real). Drop the framing that *nesting* L1/L2/L3 is what produces evaluative self-awareness — nothing retrieved supports nesting as the causal ingredient, and MUSE gets competence-awareness with **one** metacognitive layer. If L3 exists, its justification must be a *predicted behavioural difference* vs. a 2-layer version, pre-registered.
2. **Constraint (A) as literally stated will cost you the machinery that works.** RoboCat (the best "self-improving real robot" result) trains with simulation in the loop; AlphaStar's robustness came from a simulated league. Rewrite (A) as "no simulated *evidence*, simulation permitted as a training scaffold" — or accept that you are knowingly forgoing the strongest multi-agent results.
3. **Constraint (B) is right, but phrase it correctly.** The proven claim is "no universally accurate predictor; the target is endogenous to your policy" (NFL + performativity). "There is a proven positive error floor ε" is stronger than anything I retrieved. Build the metacognitive layer to output **calibrated** uncertainty and score it on held-out rollouts — otherwise it is self-grading.
4. **For H6, the only cheap genuine independence you have is intervention and adversaries** (ablation with capacity-matched controls + an optimized exploiter population). Everything derived from your own rollouts is one test, not many.
