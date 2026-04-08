# Ocelot lowstate dataset — export requirements

This document states what we need from you when you deliver trajectory data. If something here conflicts with your logger, tell us early so we can agree on a mapping.

---

## Scope

- **One trajectory** = **one directory** (one run).
- **Required in that directory:** `lowstate.csv` — a single table where IMU, foot loads, and joint state are already **time-aligned** (one row per sample instant; no separate IMU vs kinematics files for this product).

---

## `lowstate.csv`

### File format

- UTF-8, **comma-separated** CSV.
- **Header row required** (exact column names below).
- **No BOM.**
- **One row per time step**, sorted in **non-decreasing** time. If two rows share the same `sec`, use `nanosec` as the tie-breaker so ordering is unambiguous.

### Time (required — exact column names)

| Column     | Meaning |
| ---------- | ------- |
| `sec`      | Integer or float: whole seconds part of the timestamp (epoch or any monotonic clock **consistent within the file**). |
| `nanosec`  | Fractional sub-second part in **nanoseconds** (same convention as our Tartanground/ANYmal split exports). |

### IMU (required — exact column names)

| Column     | Unit / semantics |
| ---------- | ---------------- |
| `gyro_x`, `gyro_y`, `gyro_z` | Angular rate. **rad/s** preferred. If you can only export **deg/s**, say so in a short note alongside the delivery (e.g. README or email); downstream tools may auto-detect. |
| `accel_x`, `accel_y`, `accel_z` | **Specific force** in **body FLU** (forward–left–up): gravity is **included**. When the base is level and still, magnitude should be near **9.81 m/s²** and **+Z** should be positive. |

**Important:** Do **not** send gravity-compensated (linear-only) acceleration **unless** you also provide full body orientation (quaternions) on the same rows. This contract assumes **no** orientation columns on `lowstate.csv`.

### Foot vertical load (required if the signals exist — exact column names)

Provide **four** columns, **0-based** indices (not 1-based):

- `foot_force_0`, `foot_force_1`, `foot_force_2`, `foot_force_3`

**Unit:** Newtons (vertical load proxy per foot), unless you agree otherwise **in writing** for the project.

Map your internal leg IDs to `0 … 3` **once** and use the **same** mapping for **every** sequence you deliver.

### Joints (required — twelve motors, ANYmal-style)

For each motor index **`i` from 0 to 11**:

| Column            | Unit   |
| ----------------- | ------ |
| `motor_i_q`       | rad    |
| `motor_i_dq`      | rad/s  |
| `motor_i_tau_est` | Nm     |

**Optional:** If you have joint acceleration, you **may** include `motor_i_ddq` (rad/s²). It is **not** part of the minimum contract.

### What we do **not** require on `lowstate.csv`

- Body orientation / quaternions.
- World position or world velocity.

---

## Optional: `ground_truth.csv`

Include this **only** when you have an external reference pose for that trajectory. If there is no reference, **omit the file**.

- **Location:** Same directory as `lowstate.csv`.
- **Filename:** `ground_truth.csv` (fixed name).

### Time

- Same **`sec`** and **`nanosec`** column names and the **same clock epoch** as `lowstate.csv` for that run, so timestamps can be aligned numerically across files.

### Content

| Column   | Required | Notes |
| -------- | -------- | ----- |
| `p_x`, `p_y`, `p_z` | Yes (when file is present) | Position in metres, **world frame** you use for evaluation (document the frame if it is non-standard). |
| `heading` | No | If present, you **must** document **unit** (radians vs degrees) and **definition** (e.g. yaw about world +Z, and what “zero” means). |

**Sampling rate** may differ from `lowstate.csv`; that is acceptable.

---

## Delivery checklist

- [ ] `lowstate.csv` present with all **required** columns and exact spellings.
- [ ] Time column values monotonic; rows ordered accordingly.
- [ ] IMU is **specific force** FLU as above (not silent switch to linear accel without orientation).
- [ ] Foot columns named `foot_force_0` … `foot_force_3` with a **stable** leg-to-index mapping across sequences.
- [ ] All required numeric fields finite (no empty cells in required columns).
- [ ] If reference pose exists: `ground_truth.csv` with `p_x`, `p_y`, `p_z` + time columns; `heading` documented if included.

---

## Related (maintainers)

Downstream code expects these names to match shared column constants in `leg_odom/io/columns.py` (0-based foot indices, `sec` / `nanosec`, motor prefixes).
