# Mission 04: Direct Preference Optimization (DPO), Verbosity Regularization & Steering Vector Control

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Module:** Week 4 — Alignment Pipelines, Preference Optimization & Steering Core  
**Waypoint:** Betelgeuse & Orion Molecular Cloud Complex (`SEC-04-ORI`, 642 ly from Sol)  
**IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | PyTorch 2.4+ / Modern LLMs  
**Instructor Reference Repository:** [https://github.com/Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  

---

## Mission Folder Structure & Role Deliverables

This repository folder contains the starter assets and templates for **Mission 4**.

```
mission-04/
├── README.md                              <- Mission overview & squad sign-off status
├── ps04_briefing.md                        <- Full mission specification & problem statement
├── 01_DPO_Loss_Core.py                    <- PyTorch DPO Loss implementation (Systems Architect)
├── 02_Implicit_Reward_Evaluator.py        <- Bradley-Terry scoring & evaluation (Protocol Officer)
├── 03_Length_Bias_and_SimPO.py            <- Length bias & SimPO loss (Data Specialist)
└── 04_Activation_Steering_and_Audit.py    <- Activation steering & Alignment Tax audit (Security Sentinel)
```

---

## Station Role Deliverables Checklist

Before transmitting your team's repository link on the course website submission portal, complete all deliverables and record digital sign-offs from all four station officers:

- [x] **Systems Architect (`01_DPO_Loss_Core.py`)** - Fahad **:** Pure PyTorch `DPOLoss` module with vectorized log-probability gather and frozen reference policy gradients.
- [x] **Communications & Protocol Officer (`02_Implicit_Reward_Evaluator.py`)** - Madhu **:** Implicit reward extraction, pairwise preference accuracy, and Bradley-Terry calibration curve.
- [ ] **Data & Payload Specialist (`03_Length_Bias_and_SimPO.py`):** Verbosity correlation analysis and reference-free length-normalized SimPO loss implementation.
- [x] **Security & Ethics Sentinel (`04_Activation_Steering_and_Audit.py`)** - Janusz **:** Linear Artificial Tomography (LAT) steering vector extraction and Alignment Tax capability retention evaluation.

---

## Running the DPO loss core (Systems Architect)

```bash
# from the repo root (same .venv as Missions 1-3; CPU-only PyTorch is enough, no API key needed)
source .venv/bin/activate && pip install -r mission-04/requirements.txt
cd mission-04
python 01_DPO_Loss_Core.py                   # demo: log-prob gather, loss, numerical stability, gradient test
python 01_DPO_Loss_Core.py --selftest        # 38 checks: gather oracle, loss algebra, gradients, frozen ref, integration
python 01_DPO_Loss_Core.py --train           # 150-step DPO run on synthetic preference pairs + beta sweep
python 01_DPO_Loss_Core.py --check-signoffs  # gate: all four boxes above must be ticked
```

Each run writes `logs/<timestamp>-ps04-<mode>.json`. Everything runs on CPU in under 10 s.

**What `01_DPO_Loss_Core.py` implements** (pure PyTorch, no TRL):

1. `get_batch_logps(logits, labels, label_pad_token_id=-100, average_log_prob=False)`: one `torch.gather` over
   `log_softmax(logits[:, :-1])` at `labels[:, 1:]`, masked where the label is `-100` (prompt and padding), summed
   per sequence, or averaged per response token for SimPO. Verified against an explicit token loop and against
   `F.cross_entropy(ignore_index=-100)`; a row with no response tokens returns 0, never NaN.
2. `dpo_log_ratios(...)` returns `Delta_w = log pi(y_w|x) - log pi_ref(y_w|x)` and `Delta_l` likewise.
3. `DPOLoss(beta)` computes `-log sigmoid(beta (Delta_w - Delta_l))` with `F.logsigmoid` (finite at z = -1000,
   where `-log(sigmoid(z))` is `inf`), detaches the reference terms, and returns the implicit rewards
   `r = beta * Delta`, their margin, pairwise accuracy and the Bradley-Terry probability as metrics. Options:
   `label_smoothing` (conservative DPO) and `reference_free`. `from_logits(...)` takes raw logits and labels.
4. Empirical gradient test (`gradient_test`): a `TinyCausalLM` policy (Mission 3's `MultiHeadSelfAttention` is
   its mixing layer when `mission-03/` is present) and a frozen deep copy as `pi_ref`. After `backward()` all
   14 policy parameters carry finite non-zero gradients, no reference parameter has `requires_grad` or `.grad`,
   one SGD step changes the policy's SHA-256 parameter digest and leaves the reference digest bitwise identical,
   and autograd equals the analytic `dL/d log pi(y_w) = -beta (1 - sigma(z)) / B` to 0 error. `gradcheck` in
   float64 passes on the loss itself.

**Training result** (`--train`, 150 AdamW steps, batch 64, beta 0.1, synthetic 4-topic preference world): validation
loss 0.693 -> 0.031, pairwise accuracy 99 %, implicit reward margin +7.2, reference frozen throughout. The beta sweep
shows beta as the KL leash: after the same 150 steps the chosen log-prob drifts from the reference by -20.7 nats at
beta 0.05, -3.2 at beta 0.1 and +4.9 at beta 0.5 (the small-beta runs lower the likelihood of *both* responses,
the known DPO failure mode that SimPO's length normalisation in Part 3 addresses).

**Integration of the four stations.** The self-test imports the other scripts at run time:

| Station file | What the DPO core checks with it | If absent |
|---|---|---|
| `02_Implicit_Reward_Evaluator.py` | `implicit_rewards_from_logps` returns the same `r_w`, `r_l`, `P(y_w > y_l)` as the `DPOLoss` metrics, and Part 2 reports that it loaded `get_batch_logps` / `DPOLoss` from this file | check skipped |
| `03_Length_Bias_and_SimPO.py` | `SimPOLoss(beta=2, gamma=0.5)` fed with `get_batch_logps(..., average_log_prob=True)` equals `-log sigma(beta (avg_w - avg_l) - gamma)`; `get_batch_logps_and_lengths` supplies `|y|` for the length-bias correlation | check skipped |
| `04_Activation_Steering_and_Audit.py` | `extract_steering_vector` and `ActivationSteeringHook` are present | check skipped |

---

## How to Submit
1. Push your squad's completed solution repository to GitHub.
2. Go to the course website portal under **Week 4 (Mission 4) -> Problem Set ps04 & Submission**.
3. Paste your public squad GitHub repository URL and click **Transmit Solution Link to Database**.
