# Gap intervention: decision rule (frozen 2026-10-01, approved as written)

Follows RESULTS.md Section 69: joint failures sit next to missing teeth 5
times as often as correct teeth, and 93% of those get the missing tooth's
number. That is an association. This tests whether making a gap causes it.

**Method (T4, inference only, seed 0 fold models, confidence-first).**
- Targets: in each X-ray, up to 3 teeth T drawn at random (seed 0) among
  teeth that are not third molars and have both arch neighbors labeled.
- Removal: erase T by inpainting its own tooth mask (dilated 3 px) with
  OpenCV's Telea method, so the slot looks like empty bone rather than a
  black box. One removal per inference.
- Measured on each neighbor N of T (both sides) that the detector numbers
  correctly on the intact image: after removal, is N given T's number
  (gap fill), another wrong number, the right number, or missed?
- Control: the teeth two positions away from T (N2), same measures. And a
  sham: inpaint a patch of T's mask shape over bone just above (upper
  arch) or below (lower arch) the root of T, where there is no tooth.

**Measure.** g = share of neighbors N given T's number after removal,
95% CI by X-ray bootstrap; same for N2 and for N under the sham.

For scale: in the real data roughly 7% of teeth next to a gap are joint
failures (approximate, from Section 69 counts).

**Rule** (per detector):
1. g >= 5%, and the CI of g minus the N2 rate and of g minus the sham rate
   both exclude 0: removing a tooth makes its neighbor take its number.
   Gap filling is causal for this detector.
2. g < 1%: an empty slot alone does not cause gap filling. The Section 69
   association then comes from what goes with real gaps (teeth drifting
   or tipping into the space, or looking like the missing tooth), which
   an inpainted gap does not reproduce.
3. Otherwise: report as measured.

**Limits.** Inpainting is not an extraction: real gaps come with drift,
tipping and bone change, which this does not create, so rule 2 does not
rule out gap-related causes. Teeth near inpainted regions can be missed;
missed neighbors are reported separately, not counted as gap fills.

**Cost.** About 1,200 removals plus 1,200 shams per detector, three
detectors: about 30 to 45 minutes on a T4. Needs the tooth masks
(Dataset/bb_u_net_dataset/labels, already in the repo).
