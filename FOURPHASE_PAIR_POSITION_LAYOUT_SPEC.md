# Fourphase Pair-Position Layout Spec

This document defines the direct `E1`-`E4` decoder models for the 6-axis to fourphase converter.

It supersedes the older idea that different layouts are only tetrahedral permutations. In this spec:

- `layout_model` changes the decoder math.
- `wiring_map` is a separate optional remap applied after decoding.
- `E1` through `E4` are fixed physical channels.

## Hardware Contract

FOC-Stim/ReStim fourphase accepts direct `E1`-`E4` coordinates, but the position must remain on the valid fourphase manifold.

Valid direct coordinates must satisfy all of these:

- each component is in `[0, 1]`
- at least one component is exactly `1`
- the largest component does not exceed the sum of the other three

The final decoder output must always be passed through the same constraint repair as `constrain_fourphase_coordinates()` in [funscript_converter.py](funscript_converter.py#L214).

Practical implication:

- The hardware supports real direct fourphase layout-specific decoders.
- The hardware does not support arbitrary 4-independent-corner control.
- The position space is still effectively a constrained 3-DOF manifold plus overall amplitude.

## Upstream Control Axes

The upstream 6-axis mixer is assumed to provide three normalized control values:

- `u in [-1, 1]`: axial top-to-bottom travel
- `s in [-1, 1]`: left-to-right bias
- `r in [-1, 1]`: rotational branch bias

Default sign convention for this spec:

- `u > 0` means toward the top of the physical layout
- `u < 0` means toward the bottom or rear of the physical layout
- `s < 0` means left bias
- `s > 0` means right bias
- `r < 0` means left rotational branch
- `r > 0` means right rotational branch

## Realizable Anchor Set

Only realizable FOC-Stim landmarks are used as decoder anchors.

```text
CENTER = [1, 1, 1, 1]

A  = [1,   1/3, 1/3, 1/3]
B  = [1/3, 1,   1/3, 1/3]
C  = [1/3, 1/3, 1,   1/3]
D  = [1/3, 1/3, 1/3, 1]

AB = [1, 1, 0, 0]
AC = [1, 0, 1, 0]
AD = [1, 0, 0, 1]
BC = [0, 1, 1, 0]
BD = [0, 1, 0, 1]
CD = [0, 0, 1, 1]

ABC = [1, 1, 1, 0]
ABD = [1, 1, 0, 1]
ACD = [1, 0, 1, 1]
BCD = [0, 1, 1, 1]
```

## Common Helpers

```text
lerp(a, b, t) = a * (1 - t) + b * t

pm(x, left, right) = lerp(left, right, (x + 1) / 2)

step(p, anchor, gain) =
    constrain_fourphase_coordinates(
        lerp(p, anchor, clamp(gain, 0, 1))
    )
```

Default v1 gains:

- `SIDE_GAIN = 0.45`
- `ROT_GAIN = 0.35`

These are tuning constants, not hardware constraints.

## Pair-Position Family

All three layouts are treated as one family. They differ only by where the parallel pair sits along the top-to-bottom spine.

### Pair At Top

Legacy intent: `Tip / Sides + Base`, revised into a `T` shape.

Fixed physical meaning:

- `E1 = top-left`
- `E2 = top-right`
- `E3 = middle-center`
- `E4 = bottom-center`

Anchor selection:

```text
TOP_ANCHOR    = AB
MID_ANCHOR    = C
LOW_ANCHOR    = D

SIDE_LEFT     = A
SIDE_RIGHT    = B

ROT_LEFT      = ACD
ROT_RIGHT     = BCD
```

Intended feel:

- axial motion travels top pair -> middle center -> bottom center
- side motion biases `E1` versus `E2` without collapsing the center spine
- rotation leans into the left or right branch while keeping the spine active

### Pair At Middle

Legacy intent: `Triangle + Behind`.

Fixed physical meaning:

- `E1 = top-center`
- `E2 = middle-left`
- `E3 = middle-right`
- `E4 = bottom/rear-center`

Anchor selection:

```text
TOP_ANCHOR    = A
MID_ANCHOR    = BC
LOW_ANCHOR    = D

SIDE_LEFT     = B
SIDE_RIGHT    = C

ROT_LEFT      = ABD
ROT_RIGHT     = ACD
```

Intended feel:

- axial motion travels top center -> middle pair -> bottom/rear center
- side motion biases `E2` versus `E3` while keeping top and rear active
- rotation arcs around the `E2`/`E3` pair plane using the shared top/rear spine

### Pair At Bottom / Rear

Legacy intent: `Tip / Base + Bipolar`, reinterpreted as a bottom or rear pair model.

Fixed physical meaning:

- `E1 = top-center`
- `E2 = middle-center`
- `E3 = bottom-left` or `rear-left`
- `E4 = bottom-right` or `rear-right`

Anchor selection:

```text
TOP_ANCHOR    = A
MID_ANCHOR    = B
LOW_ANCHOR    = CD

SIDE_LEFT     = C
SIDE_RIGHT    = D

ROT_LEFT      = ABC
ROT_RIGHT     = ABD
```

Intended feel:

- axial motion travels top center -> middle center -> bottom/rear pair
- side motion biases `E3` versus `E4` while keeping the upper spine present
- rotation leans into the left or right lower branch using the shared upper spine

## Common Decoder

All three layouts use the same decoder structure. Only the anchors change.

### 1. Axial Base

```text
if u >= 0:
    p = lerp(CENTER, TOP_ANCHOR, u)
elif u >= -0.5:
    p = lerp(CENTER, MID_ANCHOR, -2 * u)
else:
    p = lerp(MID_ANCHOR, LOW_ANCHOR, -2 * u - 1)
```

Interpretation:

- positive `u` moves from center toward the top feature of the layout
- mild negative `u` moves from center toward the middle feature of the layout
- deep negative `u` moves from the middle feature toward the low feature of the layout

### 2. Side Bias

```text
p = step(p, pm(s, SIDE_LEFT, SIDE_RIGHT), SIDE_GAIN * abs(s))
```

Interpretation:

- side bias is always applied through the parallel pair for that layout
- side bias should sharpen left versus right, but should not hard-mute the opposite side

### 3. Rotation Bias

```text
p = step(p, pm(r, ROT_LEFT, ROT_RIGHT), ROT_GAIN * abs(r))
```

Interpretation:

- rotation does not create a separate 4D mode
- rotation is a branch bias inside the same valid fourphase manifold
- each layout uses branch faces that keep the non-pair spine electrodes involved

### 4. Final Constraint

```text
E = constrain_fourphase_coordinates(p)
```

The constrained output is the direct `E1`-`E4` command sent downstream.

## What Changes vs. Current Converter

Current converter behavior in [funscript_converter.py](funscript_converter.py#L34) treats layouts as permutations of one shared tetrahedral decoder.

This spec changes that design:

- `layout_model` selects one of three pair-position decoders
- `wiring_map` optionally remaps the final `E1`-`E4` outputs for user wiring corrections
- the UI should stop presenting layout as only a swap or permutation concept

## UI Implications

The layout selector should represent decoder geometry, not only electrode swapping.

Recommended layout names:

- `Pair At Top`
- `Pair At Middle`
- `Pair At Bottom / Rear`

Optional legacy subtitles:

- `Pair At Top (legacy Tip / Sides + Base)`
- `Pair At Middle (legacy Triangle + Behind)`
- `Pair At Bottom / Rear (legacy Tip / Base + Bipolar)`

Recommended separate control:

- `Wiring Map` or `Channel Remap`

That remap is applied only after the direct `E1`-`E4` decode.

## Implementation Note

This document is a decoder spec, not a final tuning sheet.

Expected follow-up tuning points:

- exact upstream 6-axis mix into `u`, `s`, and `r`
- sign convention for `r` if the subjective clockwise/counterclockwise feel needs inversion
- numeric tuning of `SIDE_GAIN` and `ROT_GAIN`
- optional nonlinear easing on the axial path