# Tipping intervention: decision rule (frozen 2026-10-06)

Approved by the user on the condition that the design be as accurate and
strong as possible. Change from the draft, made before any run: a
cut-and-paste-back control (repaste) is added, so paste artifacts can be
told apart from the tip itself.

Follows RESULTS.md Section 76, where sliding the distal neighbor sideways
into an erased tooth's space made all three detectors give it the missing
number (20.8%, against 0.18% with the space open). Real teeth next to a
gap mostly tip rather than slide: the crown leans into the space while
the root apex stays put. This test tips instead of sliding.

**Targets and mover.** As Section 76: the Section 71 targets, the distal
neighbor N of each, scored only if all three detectors number N right on
the intact image. Seed 0 fold models, confidence-first.

**Conditions.**
- open: T erased (as Sections 71 and 76).
- half tip, full tip: T erased, then N cut out (mask dilated 3 px), its
  old place inpainted, and N pasted back rotated about its root apex
  (the mask's top point for upper teeth, bottom point for lower teeth)
  toward the gap. Full tip is the smallest angle, searched in 0.5 degree
  steps up to 45 degrees, at which N's mask touches the mesial neighbor M
  at the crown; half tip is half that angle. Targets that never touch by
  45 degrees are skipped and counted.
- repaste (control): T erased, N cut out and pasted back unmoved, with
  the same inpainting, so only the paste itself differs from open.
- N's labeled box becomes the box around the rotated mask.

**Measure.** j = share of movers given T's number by all three
detectors, with 95% CIs from 10,000 bootstrap draws of X-rays, paired by
mover. Also: all three plus the position-only model, each detector alone,
and the missed share.

**Rule** (on j, full tip against open; the 5% bar is the same anchor as
Section 76, half the about 11% per-neighbor rate at really closed gaps).
1. The CI of j(full tip) minus j(open) excludes 0 and j(full tip) is at
   least 5%: tipping a tooth into the space causes shared errors at about
   the real rate, as sliding did.
2. The CI includes 0: tipping alone does not; the Section 76 result then
   depends on moving the whole tooth.
3. The CI excludes 0 but j(full tip) is under 5%: a partial cause.

Paste check, reported with the grade: if j(repaste) minus j(open) has a
CI excluding 0 and is more than a third of j(full tip) minus j(open), the
grade is reported as confounded by paste artifacts.

**Also reported, no rule.** j(full tip) against j(slide, Section 76)
on the movers both scored; the tip angles used.

**Limits.** Rotation in the image plane only (no rotation about the long
axis). A pasted tooth still leaves edges, and the root is rotated with the
crown, which a real tipped tooth's root is not quite. Examples are saved
for a visual check before the numbers are read.

**Cost.** About 1.5 hours on a T4.

## Dated note 2026-10-06: replication on the seed 1 models (approved, frozen)

The same procedure on the seed 1 split models and seed 1 position-only
models, same targets. Graded: replicated if the CI of j(full tip) minus
j(open) excludes 0 and j(full tip) falls inside the seed 0 95% CI (7.73
to 11.03%, RESULTS.md Section 80); otherwise both are reported with the
spread. The paste check is reported as for seed 0.

## Dated note 2026-10-06: replication on the seed 2 models (approved, frozen)

The same procedure on the seed 2 split models and seed 2 position-only
models, same targets. Graded: replicated if the CI of j(full tip) minus
j(open) excludes 0 and j(full tip) falls inside the seed 0 95% CI (7.73
to 11.03%, RESULTS.md Section 80); otherwise both are reported with the
spread. The paste check is reported as for seed 0.
