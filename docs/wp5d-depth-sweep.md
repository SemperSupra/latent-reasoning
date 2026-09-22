# WP5d direct-baseline depth sweep

WP5c showed that increasing transformer width from 32 to 64 to 128 did not
establish a sufficiently learnable baseline.

WP5d changes the next architecture variable in isolation:

- width fixed at 64;
- heads fixed at 4;
- feed-forward multiplier fixed at 2;
- task/data/optimizer/epochs unchanged;
- transformer layers swept over 1, 2, and 4.

The same qualification gate applies: mean training accuracy >= 0.90 and mean
validation accuracy >= 0.75 across the full five-seed run.

If depth also fails, the next experiment moves to data size/curriculum rather
than combining width and depth changes.
