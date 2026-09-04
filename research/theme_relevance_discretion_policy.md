# Theme relevance discretionary review policy

## Purpose
When a stock-theme pair cannot be assigned a disclosed revenue share, do not leave the theme relevance permanently unscored. Move the pair to discretionary research and assign a final relevance score from 0 to 100.

## Non-negotiable rule
Never fabricate an undisclosed revenue percentage. `revenue_share_pct` remains blank/unknown unless an official source provides the figure or the figure can be directly calculated from disclosed sales amounts.

## Final relevance score for unknown-revenue pairs
Research the company independently using, in priority order:
1. Official company/business pages
2. Product/service pages
3. Medium-term management plan / integrated report
4. Securities report / earnings materials
5. Subsidiary/business portfolio descriptions
6. Major customers, contracts, orders, installed base, production capacity, R&D/capex priorities
7. Reliable external sources only when official information is insufficient

Assess:
- Current business centrality to the theme
- Economic significance even when exact revenue mix is undisclosed
- Breadth and depth of products/services tied to the theme
- Commercial evidence (customers, orders, installed base, recurring sales)
- Strategic growth commitment (capex, R&D, MTP targets)
- Whether the theme is direct/core vs peripheral/association-only

## Score bands
- 80-100: 主力テーマ — the theme can reasonably describe a core business or major growth pillar
- 60-79: 有力関連 — meaningful commercial exposure but not the dominant/core business
- 40-59: 補助関連 — real business exposure, but economically limited or secondary
- 0-39: ノイズ候補 — weak, indirect, occasional, legacy, or association-only exposure

## Confidence
- A: official quantitative evidence or clearly dominant single-business structure
- B: strong official qualitative evidence with meaningful commercial activity, exact revenue share unknown
- C: indirect evidence or limited economic scale; use conservative score

## Data fields
Keep disclosed and discretionary information separate:
- `revenue_share_pct`: disclosed/calculable official value only
- `revenue_share_status`: disclosed / calculated / unknown
- `discretion_relevance_score`: 0-100 when discretionary review is used
- `discretion_confidence`: A/B/C
- `discretion_reason`: concise rationale
- `discretion_sources`: supporting URLs
- `final_relevance_score`: disclosed-revenue model score when reliable, otherwise discretionary score
- `final_score_source`: `structured_revenue` or `chatgpt_discretion`

## Review flow
1. Run v6 structured revenue extraction.
2. Accept structured rows after QA.
3. Send all remaining `revenue_share_unknown` rows to discretionary review rather than leaving them permanently unscored.
4. Research and assign a 0-100 relevance score.
5. Store rationale/source/confidence.
6. Escalate only genuinely ambiguous or contradictory cases for manual review.

This policy is research-only and must not change the frozen production trading strategy.