# RecoverAI — Architecture & Internship Story

## Product thesis

**Failed payments are not one problem.** Some are temporary, some require customer action, and some should be escalated. RecoverAI turns failure events into prioritized recovery decisions.

## Decision pipeline

1. **Detect:** ingest failed payment events.
2. **Classify:** map the failure into a recoverability class.
3. **Score:** estimate recovery probability and expected recoverable value.
4. **Decide:** select the next-best action under merchant policy.
5. **Act:** execute a retry, request a payment-method update, notify the customer, or escalate.
6. **Learn:** store outcome and recalibrate the model.

## Example decision

- Payment: ₹24,500
- Failure: bank decline
- Customer success rate: 92%
- Recovery probability: 86%
- Expected value: ₹21,070
- Action: retry in 2 hours

## Guardrails

- High-value transactions require approval.
- Low-confidence payments can be left untouched.
- Every decision exposes its main signals.
- All automated actions should create an audit record.

## Metrics

- Revenue at risk
- Recoverable revenue
- Revenue recovered
- Recovery rate
- Recovery by failure reason
- Recovery by action
- False-positive / wasted-retry rate
- Customer contact rate
