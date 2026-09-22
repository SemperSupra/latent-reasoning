# WP5e fixed-compute training-data sweep

WP5d established that a width-64, four-layer direct model can nearly memorize
the 768-example training set while failing to generalize. WP5e therefore moves
to data diversity.

Only the unique training-pool size varies:

- 768
- 3,072
- 12,288

Architecture is frozen at width 64 / four layers. The number of optimizer
updates is also fixed (240 in the full run), with batch size 128. Every
configuration therefore processes the same 30,720 training examples, avoiding
the confound where a larger dataset also receives more optimizer compute.

Training pools are sampled without replacement from the finite procedural
six-action universe. Validation and OOD generation remain unchanged.

If no data pool qualifies but validation improves materially with pool size,
the next experiment may vary training compute at the smallest promising pool.
If validation does not improve, the next variable should be task curriculum /
representation rather than blindly increasing data.
