# Gap ratio by number of missing teeth: is DENTEX's lower ratio a patient-mix effect?

Approved 2026-10-07, written before the analysis ran. Reported only; it
does not change any graded result (Sections 85 and 86 stand).

## Question

Shared errors sit next to a missing tooth 5.3 times as often as teeth all
three detectors get right on UFBA (44.7% against 8.5%, Section 69), but
3.5 times on DENTEX trained within DENTEX (37.0% against 10.5%, Section
86). DENTEX X-rays may simply have more missing teeth, which puts more
correct teeth next to gaps. Is the ratio similar when X-rays with the same
number of missing teeth are compared?

## Data

- UFBA: seed 0 confidence-first per-tooth table (Section 64, kernel
  `tooth-numbering-cv-confmatch-s0`), 11,602 teeth; the shared-error set
  must reproduce 161.
- DENTEX: Section 86 per-tooth table (trained within DENTEX), 18,095 teeth,
  303 shared errors; Section 85 (UFBA-trained) also shown.

Definitions as Sections 69, 85 and 86: shared error = all three detectors
wrong with the same answer and position-only wrong; all-right = all three
correct; next to a gap = an arch neighbor (positions 1 to 8) has no label
in that X-ray.

## Analysis

- Missing teeth per X-ray: positions 1 to 7 with no label (third molars
  left out, since they are often absent by development). Strata: 0, 1 to
  2, 3 to 5, 6 or more.
- Per dataset and stratum: X-rays, shared errors, share of shared errors
  next to a gap, share of all-right teeth next to a gap, ratio.
- Standardized ratio: DENTEX's per-stratum shares weighted to UFBA's
  stratum mix (shared errors by UFBA's shared-error mix, all-right teeth by
  UFBA's all-right mix), so both datasets have UFBA's patients. 95% CI from
  10,000 bootstrap draws of DENTEX X-rays (UFBA weights held fixed).

## Reading, fixed in advance

- If the standardized DENTEX ratio falls inside the UFBA CI (4.2 to 6.6),
  the lower DENTEX ratio is reported as mostly patient mix.
- If it stays near 3.5, the paper says the gap link is weaker on DENTEX
  for reasons other than the number of missing teeth.
- In between, both are reported as is.
