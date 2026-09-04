# Preregistration: π0.5 curvature removal and action rescue

**Frozen:** 2026-09-04, after the midpoint-curvature result and before sampling actions for these interpolated states

For the same 12 directed LIBERO Goal contrasts and initial states 28–32, insert
the A/B midpoint into the 512 valid image positions at the output of layer 5,
leaving all non-image positions in the B context.

Compare three conditions at the output of layer 8:

1. native midpoint output;
2. linearized output, replacing the image field with the mean of the A-image and
   B-image endpoint outputs;
3. rescue, adding the measured curvature back to that linearized field.

Use identical action noise. Rescue must reproduce the native midpoint prefix cache
and action bitwise. The primary effect is the action distance between native and
linearized midpoint states, divided by the action distance between the two image
endpoints over the first ten actions.

The result passes if the median ratio is at least `0.10`, at least 8 of 12 directed
cells have median ratio at least `0.10`, and every rescue is bitwise exact. A pass
means downstream action computation is sensitive to curvature on this interpolated
instruction chord. It does not prove that clean task behavior naturally depends on
the same component. No parameter search or additional branch follows failure.
