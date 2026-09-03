# Theme relevance methodology v5

Research-only. Production trading rules remain unchanged.

## Source priority

1. Company official website
2. Company IR HTML / earnings presentation / earnings release / integrated report / annual report
3. EDINET annual report XBRL/CSV fallback when an API key is configured
4. If a revenue ratio cannot be verified, keep it `unknown`; never impute it from keywords.

## Extraction order

The system first reconstructs disclosed business segments and their revenue shares. It does **not** attach a nearby percentage directly to a theme keyword.

Preferred evidence:

- Explicit segment revenue composition (`segment A 30%, segment B 70%`)
- External-customer segment sales table (`segment sales / consolidated sales`)

Order composition, customer composition, geographic composition, margins, growth rates, ROE, PBR, LTV, visitor mix, and similar percentages are not revenue exposure.

## Theme mapping

After segment reconstruction, a 124-theme label is mapped conservatively to one or more disclosed segments.

- Broad theme `建設` may aggregate disclosed `土木` + `建築` segments.
- Specific theme `下水道` does not inherit the entire `土木` segment unless the company separately discloses a sewage/wastewater revenue amount or share.
- Cross-cutting themes such as AI, DX, data center, autonomous driving, etc. remain `unknown` when the company discusses them strategically but does not disclose revenue attributable to them.
- Macro-benefit themes (FX, rates, defensive consumption, inbound) require a separate sensitivity model; segment sales alone do not prove the earnings sensitivity.

## QA fixtures

Parser changes must continue to pass these controls:

- IKK: care 3.0%, food 2.0%
- Sata Construction: construction 98.8%, sewerage unknown
- Yondenko: solar 2.0%, leasing 1.5%
- meito: food about 82.4%
- Hibiya Engineering: data-center order mix must not be treated as revenue mix
- Cross Cat: income/margin percentages must not be treated as IT/cloud revenue mix

## Candidate scoring after revenue mapping

`55% revenue exposure + 30% business directness + 15% growth relevance`

Revenue exposure maps 80%+ sales share to 100 points and scales linearly below 80%.

All v5 outputs are research candidates until sampled source QA is complete.
