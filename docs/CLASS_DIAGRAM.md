# Leg odometry: class and dependency diagrams

Mermaid diagrams for the **`leg_odom`** package. For folder-level narrative, see [ARCHITECTURE.md](../ARCHITECTURE.md). **Submodule-focused UML:** [features](../leg_odom/features/README.md#uml-class-diagrams-mermaid), [training](../leg_odom/training/README.md#uml-class-diagrams-mermaid), [contact](../leg_odom/contact/README.md#uml-class-diagrams-mermaid).

## Core classes (simplified UML)

```mermaid
classDiagram
    direction TB
    class BaseLegOdometryDataset {
        <<abstract>>
        +__len__()
        +__getitem__(index) LegOdometrySequence
    }
    class CachedSingleSequenceDataset {
        <<abstract>>
    }
    class TartangroundDataset
    class OcelotDataset
    class LegOdometrySequence {
        frames
        median_rate_hz
        position_ground_truth
        sequence_name
        meta
    }

    BaseLegOdometryDataset <|-- CachedSingleSequenceDataset
    CachedSingleSequenceDataset <|-- TartangroundDataset
    CachedSingleSequenceDataset <|-- OcelotDataset
    BaseLegOdometryDataset ..> LegOdometrySequence : returns

    class BaseKinematics {
        <<abstract>>
        +fk(leg_id, q)
        +J_analytical(leg_id, q)
    }
    class AnymalKinematics
    class Go2Kinematics
    BaseKinematics <|-- AnymalKinematics
    BaseKinematics <|-- Go2Kinematics

    class BaseContactDetector {
        <<abstract>>
        +update(step) ContactEstimate
        +reset()
    }
    class ContactDetectorStepInput
    class ContactEstimate
    BaseContactDetector ..> ContactEstimate : produces
    BaseContactDetector ..> ContactDetectorStepInput : consumes

    class ErrorStateEkf {
        +imu_predict(...)
        +update_zupt(...)
    }
```

Concrete detectors include GRF threshold, GMM+HMM, and neural classifiers implementing `BaseContactDetector`.

## Run-time factories (experiment YAML)

```mermaid
flowchart LR
    subgraph cfg [experiment YAML]
        dk[dataset.kind]
        rk[robot.kinematics]
        cd[contact.detector]
    end
    build_ds[build_leg_odometry_dataset]
    build_kin[build_kinematics_backend]
    build_ct[build_contact_stack]
    TartangroundDataset[TartangroundDataset]
    OcelotDataset[OcelotDataset]
    AnymalKinematics[AnymalKinematics]
    Go2Kinematics[Go2Kinematics]
    ContactStack[ContactStack]

    dk --> build_ds
    build_ds --> TartangroundDataset
    build_ds --> OcelotDataset
    rk --> build_kin
    build_kin --> AnymalKinematics
    build_kin --> Go2Kinematics
    cd --> build_ct
    build_ct --> ContactStack
```

## Data pipeline: precompute, training, EKF

```mermaid
flowchart LR
    subgraph offline [Offline]
        pre[precompute_contact_instants]
        npz[precomputed_instants.npz]
        nn[train_contact_nn]
        gmm_fit[train_gmm]
        pt[neural .pt meta scaler]
        gmm_npz[gmm weights .npz]
    end
    subgraph runtime [Runtime]
        yaml[experiment YAML]
        main[main.py EKF]
    end
    pre --> npz
    npz --> nn
    npz --> gmm_fit
    nn --> pt
    gmm_fit --> gmm_npz
    pt --> yaml
    gmm_npz --> yaml
    yaml --> main
```

**Independent path:** `contact.detector: grf_threshold` needs no precompute or training artifacts.
