# Fama–French five factors — interpretation for Indian large-cap context

## Not a trading signal by itself
Factor regressions **attribute** realised returns to systematic tilts (size, value, profitability, investment) plus market beta. They do not automatically produce alpha trades without model of time-varying premia.

## Factor intuition (pedagogical)
**SMB** — small minus big: historical premium small caps over large (varies by era and region). On NIFTY-heavy universes cross-section may be truncated because mega-caps dominate—interpret coefficients cautiously.

**HML** — high book-to-market minus low: value vs growth proxy in factor construction style.

**RMW** — robust minus weak profitability: quality / earnings power dimension.

**CMA** — conservative minus aggressive investment: firms that expand assets aggressively vs those that rein conservatively.

## Implementation notes in research stacks
Monthly factor returns often built from sorted portfolios; alignment with daily BL optimisation layers requires resampling or bridging assumptions when mixing horizons.

## Usage in mixed AI storytelling
Show factor loadings to explain **risk exposures** of BL-optimised sleeve vs benchmark—not to claim factors predict next month deterministically.
