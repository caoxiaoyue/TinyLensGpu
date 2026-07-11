# Require unit-amplitude lens-light bases in joint inversion

Joint semi-linear inversion will treat every parametric lens-light component as a unit-amplitude basis: its single supported intensity parameter must be static and equal to one, while the fitted lens-light intensity supplies the only amplitude scale. Both dense and operator probability models will validate this boundary and expose the same positive `lens_light_regularization` parameter, defaulting to `1e-6`, so the weak zero-order prior has consistent and visible semantics rather than depending on hidden component scaling.
