# Maintenance Advisory - 10 MVA 33/11 kV Transformer
**As of:** 2018-06-26   |   **Health index:** 45.0 / 100   |   **Band:** Poor   |   **Urgency:** HIGH - within 2 weeks

**Component scores:** thermal 82.9, loading 85.0, oil 65.0, dga 58.0

## Top risk
Thermal fault 300-700 C (ML: T2, Duval Triangle 1: T2)

**Top drivers of the score**
1. Buchholz Alarm (cap) = True (-22.0 pts)
2. Moisture (ppm) = 34.6 (-4.6 pts)
3. CH4 (ppm) = 446.4 (-4.4 pts)

## Recommended action
Typical of bad joints, circulating currents in core/clamps or OLTC contacts: confirm with lab DGA, IR thermography of bushings/OLTC, winding resistance + core-insulation (megger) test at the next planned outage; cap loading at 0.8 pu until resolved. Buchholz gas alarm in last 30 days: sample Buchholz gas and analyse with oil DGA before any re-energisation after trip.

## Cost-benefit (12-month expected cost, assumptions in config.yaml)
| | Value |
|---|---|
| Cost if the transformer fails (repair + 720 h outage) | INR 3.48 crore |
| Probability of failure in 12 months at this band (assumed) | 12% |
| Expected loss - no action | INR 41.8 lakh |
| Cost of recommended action | INR 13.6 lakh |
| Expected loss - after action | INR 12.5 lakh |
| **Net benefit of acting now** | **INR 15.7 lakh** (benefit/cost = 2.2) |

Insulation life used over the monitored period: 1.79 % of 180,000 h (IEC 60076-7); max hot-spot 137 C.

*Prototype output on public + synthetic data; confirm with lab DGA and site inspection before acting.*
