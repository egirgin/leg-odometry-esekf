# SSL NN Training Module

This module is for self-supervised neural training only.

Current scope:
- train a self-supervised neural backbone on precomputed instant features,
- emit artifacts with metadata and scaler sidecars compatible with the existing neural stack,
- keep architecture and feature contracts aligned with the supervised NN pipeline.

Out of scope for now:
- runtime detector integration,
- EKF contact factory wiring,
- changes to detector class topology.

## Compatibility Guardrails (Important)

Future intent is to use this model as a contact detector in the same main EKF loop.

Do not change these contracts lightly:
1. Keep metadata keys required by neural runtime loaders:
   - architecture
   - feature_fields
   - history_length
   - instant_dim
   - instant_feature_spec_version
   - robot_kinematics
2. Keep scaler format stable:
   - npz keys: mean, scale
3. Keep checkpoint payload stable:
   - a dict containing state_dict for the deployable backbone
4. Keep feature names from leg_odom.features.instant_spec allowed fields.
5. Keep sliding-window padding semantics aligned with training.nn and contact.neural.

## Why This Exists Separately

The existing training.nn module is supervised and label-centric.
This ssl_nn module isolates self-supervised experimentation while preserving future detector compatibility.

## Planned Integration Path

When SSL training is validated:
1. add a runtime detector entrypoint that follows existing BaseContactDetector topology,
2. load SSL-produced checkpoint/meta/scaler with the same runtime safety checks,
3. keep EKF loop interfaces unchanged.

Any future contributor should treat this module as training-only unless integration is explicitly requested.
