# BLUE TEAM — VERIFICATION REPORT: H1 (architecture-level brain replication) & H3 (computational affordability)

## 0. Provenance & method limits — READ FIRST
Budget used: **8/8 web_search, 5/5 web_fetch. All 5 fetches returned NO content** (nature.com 302-redirects to idp.nature.com paywall; alleninstitute.org and dcs.warwick.ac.uk failed at the network layer). **No primary full text was read in this session.** Every fact below is tagged:

- **[S]** = verbatim in a search-result title/snippet this session (URL cited).
- **[K]** = my prior knowledge of the canonical literature. Citable and almost certainly right in outline, but **NOT verified this session** — re-check before building on it.
- **[I]** = my own inference / arithmetic.

Do not let [K] items enter a design doc as settled. §7 lists the exact URLs to fetch to convert [K]→[S].

---

## 1. Master table — project → scale → hardware → real-time factor → functional result

| Project | Neurons / synapses | Hardware | Real-time factor | Functional result | Tag |
|---|---|---|---|---|---|
| **FlyWire adult fly whole-brain connectome** (Nature 634, Oct 2024) | 139,255 neurons, ~50M synapses, ~7,000 cell types | n/a (anatomy) | n/a | Complete wiring diagram of an animal that walks, flies, navigates, learns, courts | [S] |
| **Shiu et al., "A Drosophila computational brain model reveals sensorimotor processing", Nature 634 (2024)** | ~125k–139k neurons (press vs. connectome count) | **single laptop-class machine** (Berkeley headline: "Researchers Simulate a Fly Brain on a Laptop") | not verified; plausibly ≈1× with reduced (LIF-like) dynamics | Connectome-constrained whole-brain model **predicted and reproduced sugar-taste sensorimotor processing** (proboscis motor output); used to identify circuit elements | [S] |
| **snedea/flybrain** (open-source, third-party) | "139K LIF neurons from the FlyWire FAFB connectome", interactive | commodity PC/laptop | "interactive" | Interactive whole-fly-brain LIF simulation | [S] (not peer-reviewed) |
| **Eon Systems "digital fly" (2025)** | whole fly brain in a simulated body | press says laptop-class | press claims real-time embodiment | Press: digital fly grooms/forages in a physics sim | [S] press only — **UNVERIFIED / contested** |
| **MICrONS** (Nature 2025, s41586-025-08790-w) | 1 mm³ mouse visual cortex; ~200,000 neurons; ~523M synapses; ~4 km axons | offline EM + GPU training | n/a | Structure→function: a model fit to this volume **predicts responses to novel visual stimuli better than prior functional models** | [S] existence, [K] "better than" |
| **Mouse whole-cortex simulation (Nov 2025)** — SC/ACM paper title: "Microscopic-Level Mouse Whole Cortex Simulation Composed of 9 Million Biophysical Neurons and 26 Billion Synapses on the Supercomputer…" | **9–10M neurons, 26B synapses** | supercomputer at **~4×10^17 ops/s (~400 PFLOPS)** (Fugaku-class) | **not verified in session** | Full mouse cortex, biophysically detailed, cortex-wide activity | [S] title/scale/hardware; [K]/unverified: RTF |
| **Blue Brain rat somatosensory column** (Markram et al., Cell 2015) | 31,000 morphologically detailed neurons, ~37M synapses, ~200 cell types | Blue Gene/Q | slower than real time | Reproduced in-vivo-like activity states and sensory-evoked dynamics | [K] |
| **HBP (2013–2023)** | program, not one model | ~€1.19B total / ~€406M EU core; 10 yrs; ~19 institutions; EBRAINS | n/a | Delivered atlases, EBRAINS platforms, hippocampus/cortex models; **no whole-human-brain simulation, no cognitive function** | [K] |
| **K computer NEST benchmark** (Jordan et al., Front. Neuroinform. 2018) | **1.73 billion neurons, 10.4 trillion synapses** | K computer, 82,944 nodes, 10.5 PFLOPS | **1 s biological = ~1 day wall (~1/86,400)** *with synaptic plasticity* | Generic scaling demonstration, not a brain | [K] |
| **Spaun** (Eliasmith et al., Science 2012) | **2.5 million neurons, 8 anatomically mapped subsystems** | **one machine with a GPU** | reported ≈real time | One model performs **8 diverse cognitive tasks** incl. inductive reasoning, working memory, symbol binding | [K] |
| **C. elegans** | **302 neurons, ~7,000 synapses** (full connectome since 1986) | laptop | ≫1× | The organism's **entire behavioural repertoire** (locomotion, chemotaxis, escape, habituation) | [K] |
| **Drosophila larva whole-brain connectome** (Winding et al., Science 2023) | **3,016 neurons, ~548,000 synapses** | laptop | ≫1× | Complete larval CNS; used for connectome-constrained models of **learned** behaviour | [K] |
| **The Virtual Brain / EPINOV clinical trial** | region-level (neural-mass) models on the ~1,000-region human connectome, ~10²–10³ nodes | workstation/cluster | ≫1× (neural mass is cheap) | Regional models **reproduce individual resting-state networks** (Honey et al. 2009) and were used **prospectively to guide epilepsy surgery** (~350 patients, France) | [K] |
| **Corticostriatal micro-assemblies** (Nat. Commun. 2025, s41467-025-67076-x; Salk/Sejnowski) | ~10³-scale "micro-assemblies", not whole regions | small GPU | — | Reproduces in-vivo corticostriatal dynamics and **discovers a neural code** | [S] title-level |
| **DishBrain / Cortical Labs** (Kagan et al., Neuron 2022) | **~800,000 real neurons** (human iPSC + mouse primary) on HD-MEA | real biological substrate + closed-loop electronics | **true real time (it IS wetware)** | **Learned Pong in ~5 min** of closed-loop stimulation with free-energy/dopamine-like feedback | [S] |
| **Cortical Labs CL1** (Mar 2025, commercial) | ~800k neurons per unit | biological + silicon, reported ~US$20k–35k/unit | real time | Biological computer running game environments (press: "Doom") | [S] press/commercial |
| **FinalSpark Neuroplatform** (2024) | 4 organoids × ~10,000 neurons ≈ **40,000 neurons**; 24/7 remote access, ~10 universities | biological + silicon | real time | Remotely accessible wetware computing service; company claims ~10⁶× better energy efficiency than GPUs | [S]; the 10⁶× figure is **company marketing, contested** |
| **Zebrafish larva** | ~100,000 neurons, whole-brain light-sheet imaging at cellular resolution | microscope | real time (imaging) | A 10⁵-neuron vertebrate brain imaged **in toto**, driving hunting/escape/learning | [S]/[K] |

---

## 2. H1 verdict: **PARTIAL — SUPPORTED at sensorimotor/mesoscale; UNSUPPORTED for cognition**

### Strongest real support
1. **A whole-brain, connectome-constrained model produced a real functional prediction** (fly; Shiu 2024) [S]. This is the single best existing datapoint for "region/module-level replication captures functionally relevant computation": the architecture *was* the wiring diagram, and the model predicted motor-neuron activity that was then confirmed. It is a strict improvement over "just fit a black box".
2. **Structure→function at 1 mm³ of mammalian cortex** (MICrONS 2025): a model built from the real connectome predicts responses to **novel** stimuli [S]. Function was not hand-authored; it fell out of measured connectivity.
3. **Regional models reproduce system-level functional networks** — resting-state networks and travelling waves emerge from connectome-constrained neural-mass models (TVB; Honey 2009) and have been used **prospectively in a surgical trial** [K]. Clinical actionability is much stronger evidence than a correlation plot.
4. **Whole-cortex scale is now routinely simulated** (9–10M neurons / 26B synapses, 2025) [S] — architecture-level simulation is no longer the bottleneck it was in 2015.
5. **Function exists at absurdly small scale** (C. elegans 302 neurons → full behavioural repertoire; larva 3,016 → learned behaviour; Spaun 2.5M → 8 cognitive tasks) [K]. If your target function is insect-scale sensorimotor + simple learning, architecture-level replication at 10⁵–10⁶ neurons already has direct precedent.

### Where the support is thin — say this out loud
- **No architecture-level model has demonstrated anything resembling general cognition, language, or open-ended reasoning.** The fly model reproduces *specific sensorimotor pathways*, not the fly. The mouse and human-scale efforts have produced *dynamics*, not *competence*.
- **Connectomes are necessary but demonstrably not sufficient**: the fly model required cell-type-specific physiological parameters, neurotransmitter-sign assignment, and per-pathway tuning [K, and implied by [S] framing]. Learning rules, neuromodulation, glia, dendritic computation and inhibition balance are absent from a static wiring diagram.
- **HBP is the negative datapoint you must confront**: ~€1.19B, 10 years, no functional cognition [K]. This is *consistent with* "detail ≠ function".
- **Competing explanations are alive**: DishBrain shows real function from **unstructured** tissue at 8×10⁵ neurons [S], and LLMs show function from **non-brain** architectures at ~10¹¹ parameters [K]. So region-level architecture is neither necessary nor proven sufficient for function. A blue-team honest framing: *"architecture-level replication is a defensible prior, not an established law."*
- **No validated scaling law maps neuron count to capability.** Nobody can tell you "10⁷ neurons buys you X". Any AHI roadmap that assumes a scale threshold is assuming, not citing.

---

## 3. H3 verdict: **SUPPORTED at AHI-relevant scale; UNSUPPORTED at human scale**

### The compute ladder [K numbers + [I] arithmetic]
| Scale | Neurons / synapses | Plausible hardware | Real-time? | Approx. cost of real-time [I] |
|---|---|---|---|---|
| Worm | 3×10² / 7×10³ | laptop | yes, ≫1× | ~$0 |
| Larva | 3×10³ / 5.5×10⁵ | laptop | yes | ~$0 |
| **Adult fly (whole brain)** | **1.4×10⁵ / 5×10⁷** | **1 GPU** | **~1× plausible** | **~$0.3–3/h** |
| Region-scale cognitive model (Spaun) | 2.5×10⁶ | 1 GPU | ~1× (reported) | ~$0.3–3/h |
| Mouse cortex (2025) | 9×10⁶ / 2.6×10¹⁰ | ~400 PFLOPS supercomputer | ≪1× | not rentable at this fidelity |
| Generic 1.73B-neuron net | 1.73×10⁹ / 1.04×10¹³ | 10.5 PFLOPS (K) | 1/86,400 | ~10²–10³ GPU-days per bio-second |
| **Human brain** | **8.6×10¹⁰ / ~6×10¹⁴** | **~10⁶ GPUs at fly-model efficiency** | 1× is 6 orders of magnitude away | **~$10⁵/h [I]** |

Arithmetic [I]: fly = 5×10⁷ synapses; human ≈ 6×10¹⁴ synapses → **~10⁷× the fly's synapse count**. At the *same* per-synapse efficiency as a single-GPU fly simulation, human-scale real time needs ~10⁶ GPUs. A 10× algorithmic win still leaves 10⁵ GPUs ≈ **$10⁵/hour** at ~$1–3/GPU-hr. **Whole-human-brain real time is not purchasable; a 10⁶–10⁷-neuron architecture-level AHI is purchasable today.**

Cloud GPU price anchors [K — verify, prices move weekly]: H100 80GB ≈ **$2–4/GPU-hr** (RunPod/Vast/Lambda community-tier ~$1.9–2.8; AWS p5 8×H100 list ≈$98/hr ⇒ ~$12/GPU-hr); A100 80GB ≈ $1.5–2.5; B200 ≈ $5–12; RTX 4090/5090 ≈ $0.3–0.6. Sustained throughput for point-neuron spiking sim (GeNN/Brian2CUDA/NEST-GPU class) ≈ **10⁸–10⁹ synaptic events/s per datacenter GPU** [K — verify; this single number decides your hardware budget].

### Smallest scale at which the *target function* has been demonstrated
- **Working memory / decision-making**: a ~2,000-neuron cortical-column model reproduces delay-period persistent activity and Weber-like decisions (Wang 2002) [K].
- **Navigation / vector computation**: insect central-complex models with ~10³–10⁴ neurons [K].
- **Learning in a closed loop, real time**: **~8×10⁵ real neurons** (DishBrain) [S].
- **Multi-task cognitive flexibility**: **2.5×10⁶ neurons** (Spaun) [K].
→ The affordable "function" regime starts at **10³** and is well-established by **10⁶**.

### The constraint you have not priced: (A) kills sample throughput, not FLOPs
This is my most actionable finding [I]. Under constraint (A) — real interaction only — experience throughput is bound by wall clock: a 100 Hz sensorimotor loop yields ~8.6×10⁶ steps/day; a 1 kHz loop ~8.6×10⁷. Simulated-agent RL routinely does 10⁹–10¹¹ steps. **Constraint (A) costs you 3–5 orders of magnitude in data, so the binding constraint on AHI is sample efficiency and real-time interaction bandwidth, not GPU dollars.** Corollary: budget for a high-bandwidth real sensor/actuator rig, and treat any architecture claim that needs >10⁷ real-interaction steps as unfundable.

Wetware is the cheapest way to satisfy (A) literally: CL1 ~US$20k–35k/unit and FinalSpark remote organoid access [S] give you **real** neurons in a **real** closed loop at 10⁴–10⁶ cells *today*, with no simulation of the substrate at all. For an architecture that must "not simulate" (不能模拟), a hybrid wetware/silicon loop is the strongest — and most under-explored — option.

---

## 4. Contested claims (do not present as settled)
1. **Eon Systems' "digital fly in a virtual body"** — press-level only, no peer review in my sources [S]. Also: a *simulated* body arguably violates constraint (A).
2. **Real-time factor of the fly model** — the "laptop" framing is a press headline; the actual neurons/s and time-step are unverified here [S shaky].
3. **DishBrain "learning"/"sentience"** — the Neuron 2022 title claims sentience; the learning effect is small-n and has been argued to be passive adaptation [K, contested].
4. **FinalSpark's "10⁶× more energy-efficient than GPU"** — company claim [S], no independent replication in my sources.
5. **Mouse whole-cortex real-time factor** — unverified; "on a supercomputer" ≠ real time.
6. **Whether biophysical detail is needed at all** — point-neuron vs multi-compartment is an open fight; the affordable results (fly, Spaun) all use *simplified* units.

---

## 5. What would actually settle H1 and H3
**To settle H1:** a pre-registered, connectome-constrained model that reproduces **≥3 unlearned behaviours across modalities** (sensory, motor, navigation) *with no per-behaviour hand-tuning*, **plus a lesion-correspondence test** (silence region R in the model → the same deficit as silencing R in vivo). Lesion correspondence is the causal test; correlation of activity is not. Absent that, H1 stays PARTIAL.
**To settle H3:** one systems paper reporting, for a 10⁶–10⁸-neuron architecture-level model in a **closed real-interaction loop**: (neurons, synapses, hardware, real-time factor, **US$ per biological hour**). That paper does not exist. **"Cost per biological hour of closed-loop brain-scale interaction" is a publishable, unclaimed measurement** — building it would be genuine novel evidence for AHI.

---

## 6. Tension with constraint (B)
Nothing in the H1 evidence above touches other-agent prediction — the fly/mouse/worm models have no social layer. So this evidence **cannot be borrowed** to justify an other-modeling module. Independent evidence ([K], not verified here): human false-belief/ToM accuracy is bounded (~60–80% on standard tasks), and machine ToM models (e.g. ToMnet 2018) are only partially predictive — consistent with constraint (B), but that is red-team territory and needs its own sourcing.

---

## 7. Verification queue — exact URLs to fetch (failed this session)
1. https://www.nature.com/articles/s41586-024-07763-9 — **highest priority**: fly model neuron count, time-step, hardware, real-time factor. (Paywall: use PMC/europepmc mirror.)
2. https://vcresearch.berkeley.edu/news/researchers-simulate-fly-brain-laptop-human-brain-next — the laptop/real-time claim and any human-scale extrapolation quoted.
3. https://alleninstitute.org/news/one-of-worlds-most-detailed-virtual-brain-simulations-is-changing-how-we-study-the-brain/ — mouse whole-cortex hardware + real-time factor.
4. https://dl.acm.org/doi/pdf/10.1145/3712285.3759819 — the 9M-neuron/26B-synapse mouse cortex systems paper (hardware, RTF, parallel efficiency).
5. https://www.nature.com/articles/s41586-025-08790-w — MICrONS numbers and the prediction benchmark.
6. https://dcs.warwick.ac.uk/~feng/papers/NCS_24.pdf — "On the path toward brain-scale simulations": compute/real-time scaling estimates.
7. Jordan et al. 2018, Front. Neuroinform. (K-computer 1.73B neurons / 10.4T synapses / 1 day per bio-second) — confirm the exact benchmark.
8. Eliasmith et al., Science 2012 (Spaun) — confirm "real time on one GPU".
9. A current GPU price page (RunPod/Vast/Lambda) + a GeNN/Brian2CUDA throughput paper — replace my [K] price and events/s figures.

## 8. Bottom line
- **H1: PARTIAL.** Architecture-level replication has *real*, recent, functional wins (fly whole brain → sensorimotor prediction; MICrONS → structure predicts function; regional models → resting-state networks + a clinical trial). It has **zero** wins on general cognition, no scale→capability law, and a €1.19B counterexample (HBP) showing detail alone buys infrastructure, not function.
- **H3: SUPPORTED for 10³–10⁷ neurons** (single-GPU to small-cluster; ~$0.3–10/h; precedent: fly brain, Spaun, DishBrain). **UNSUPPORTED for human-scale real time** (~10⁶ GPUs, ~$10⁵/h [I]). The real cost driver under constraint (A) is **real-interaction sample throughput (10⁶–10⁷ steps/day), not FLOPs** — architect for sample efficiency and high-bandwidth sensing, and consider wetware to satisfy "不能模拟" literally.
