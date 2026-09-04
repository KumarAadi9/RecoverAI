# RecoverAI — AI Revenue Recovery

RecoverAI is an internship-ready product prototype for the **AI Revenue Recovery** track. It is designed as a merchant-facing control center that identifies revenue at risk, predicts recovery likelihood, recommends the next-best action, and demonstrates how an agent can execute recovery workflows.

## Run locally

This version is deliberately dependency-light so you can demo it immediately.

```bash
cd recoverai
python3 -m http.server 4173
```

Open `http://localhost:4173`.

## Included product flows

- Merchant overview with revenue-at-risk, recoverable revenue and recovery-rate KPIs.
- Failure mix and recovery trend visualizations.
- Searchable failed-payment queue with AI probability and next-best action.
- Click any payment to open an AI analysis modal with reasoning and expected recovery value.
- Autonomous Recovery Agent page with live activity and simulated execution.
- Recovery simulator to model monthly revenue upside.
- Merchant automation settings and AI guardrails.
- One-click failed-payment simulation for demo storytelling.

## AI approach in the prototype

The browser implementation uses a deterministic scoring engine so the demo remains self-contained. It combines:

- failure type
- customer payment history
- historical success rate
- transaction value
- retry count / context

In production, this score would be replaced by a trained probability model and a policy layer connected to payment and messaging APIs. The LLM should explain decisions and orchestrate tools, not blindly invent the financial score.

## Generate a 50k synthetic dataset

```bash
python3 scripts/generate_dataset.py
```

This produces `payments_50000.json` for model-training or backend experiments.

## Suggested production architecture

Frontend → API → feature store / PostgreSQL → recovery model → policy engine → payment retry / customer notification tools → event log → feedback loop.
