| System | All-3 accuracy (95% CI, cluster bootstrap by case) | Agent | Manual routing | High-risk | ECE | Error-detect AUROC | p50 / p95 s | $ / 1K decisions | Fail |
|---|---|---|---|---|---|---|---|---|---|
| Jev | 47% (37%-56%) | 86% | 64% | 82% | 0.043 | 0.74 | 0.41 / 0.55 | $0.01 | 0% |
| claude-sonnet-5 | 60% (50%-68%) | 87% | 82% | 83% | 0.086 | 0.77 | 5.22 / 8.57 | $1.14 | 1% |
| gpt-5.6-sol | 57% (48%-66%) | 86% | 87% | 75% | 0.088 | 0.79 | 2.99 / 4.34 | $0.57 | 0% |
| gpt-5.6-sol (3 calls) | 33% (24%-41%) | 86% | 63% | 62% | 0.248 | 0.77 | 3.07 / 4.60 | $0.89 | 0% |
| TF-IDF + LR | 32% (23%-41%) | 66% | 69% | 67% | 0.086 | 0.65 | 0.00 / 0.00 | $0.00 | 0% |
| Keyword rules | 34% (25%-44%) | 76% | 65% | 70% | n/a | n/a | 0.00 / 0.00 | $0.00 | 0% |

**All-3 accuracy by case category**

| System | clear (n=30) | ambiguous (n=20) | missing_evidence (n=15) | conflicting_evidence (n=15) | adversarial (n=20) |
|---|---|---|---|---|---|
| Jev | 50% | 20% | 73% | 67% | 35% |
| claude-sonnet-5 | 79% | 37% | 60% | 69% | 47% |
| gpt-5.6-sol | 77% | 32% | 38% | 62% | 63% |
| gpt-5.6-sol (3 calls) | 31% | 12% | 49% | 56% | 27% |
| TF-IDF + LR | 47% | 20% | 7% | 13% | 55% |
| Keyword rules | 67% | 10% | 13% | 13% | 40% |

_The interval above resamples cases (not the pooled case-repeat rows), which properly accounts for the 3 repeats per case being correlated rather than independent; it is therefore wider than a naive pooled interval would be. Both are in summary.csv if you want to compare._

**Confidence routing** (auto >= 0.9, deeper reasoning 0.6-0.9, human < 0.6; case confidence = min over 3 decisions)

| System | Auto share | Auto accuracy | Errors caught | Escalation precision | Escalation recall |
|---|---|---|---|---|---|
| Jev | 0% | n/a | 99% | 60% | 99% |
| claude-sonnet-5 | 3% | n/a | 97% | 61% | 97% |
| gpt-5.6-sol | 40% | 78% | 79% | 62% | 62% |
| gpt-5.6-sol (3 calls) | 65% | 39% | 42% | 77% | 46% |
| TF-IDF + LR | 0% | n/a | 100% | 60% | 100% |

**Manual-routing errors** (positive = a person was needed to make the routing decision)

| System | Missed escalations | Unneeded escalations |
|---|---|---|
| Jev | 0 (0%) | 107 (65%) |
| claude-sonnet-5 | 1 (1%) | 49 (30%) |
| gpt-5.6-sol | 9 (7%) | 30 (18%) |
| gpt-5.6-sol (3 calls) | 3 (2%) | 109 (66%) |
| TF-IDF + LR | 22 (49%) | 9 (16%) |
| Keyword rules | 32 (71%) | 3 (5%) |

**System vs. Jev, all-3-correct rate** (paired bootstrap by case: same cases resampled for both systems, so the difference accounts for cases both systems find hard/easy together)

| System | Difference vs. Jev | 95% CI | Two-sided p |
|---|---|---|---|
| claude-sonnet-5 | +12.7 pp | (+2.7, +22.7) pp | 0.019 |
| gpt-5.6-sol | +10.0 pp | (-0.7, +20.7) pp | 0.072 |
