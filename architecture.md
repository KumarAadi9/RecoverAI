# RecoverAI — Architecture

## Product thesis

**Failed payments are not one problem.** Some are temporary and clear on a quick retry, some need
the right timing, and some need the customer to act (an expired card, a failed authentication).
RecoverAI turns failure events into prioritised recovery decisions and measures them against baselines.

## Decision pipeline

1. **Detect:** ingest failed payment events.
2. **Classify:** describe each case (failure reason, method, customer history, attempt number).
3. **Score:** the model estimates the success probability of every candidate action and timing.
4. **Decide:** choose the best expected value (`probability x amount - cost`), then apply merchant policy.
5. **Act:** run the action. In this prototype the outcome is sampled from the simulated world.
6. **Audit:** record every decision and outcome in the events table.

Not implemented: online learning. The model is trained at startup from simulated attempt logs.

## Components

- `backend/simulation.py`: the simulated payments world and its hidden recovery rules.
- `backend/ml_engine.py`: gradient-boosted model, time-based evaluation, candidate scoring.
- `backend/policy.py`: guardrails (high-value approval, retry limit, confidence floor).
- `backend/main.py`: API and the agent cycle.
- `backend/backtest.py`: compares naive retries, a rules baseline and RecoverAI.

## Guardrails

- Payments of INR 1,00,000 or more require merchant approval.
- At most 3 automatic attempts per payment.
- Configurable confidence floor; the default was chosen on a validation set.
- Every automated action creates an audit record.

## Metrics

- Revenue recovered as a share of failed revenue
- Attempts and customer requests per 100 payments (effort)
- Revenue recovered per attempt
- Calibration, PR-AUC and Brier score of the model

See the README for results and limitations.
