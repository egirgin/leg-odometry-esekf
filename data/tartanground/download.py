# Add the cloned directory to Python's path so that 'tartanair' can be found
import sys
import os
#sys.path.insert(0, os.path.abspath('.')) # Add current directory (tartanairpy) to sys.path

import tartanair as ta
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Target Environments (From your priority list)
PRIORITY_CANDIDATES = [
    'ForestEnv', 'GreatMarsh', 'OldTownSummer', 'Downtown', 'Gascola',
    'ModernCityDowntown', 'ModularNeighborhood', 'NordicHarbor',
    'OldTownFall', 'SeasonalForestAutumn', 'SeasonalForestSpring',
    'SeasonalForestWinter'
]

# 2. Specific Data Targets
# [CRITICAL]: 'P2000' is the ANYmal trajectory ID.
#TARGET_TRAJ = ['P2000']
TARGET_TRAJ = []
TARGET_VERSION = ['anymal']
# [CRITICAL]: Requesting 'rosbag' (joints), 'imu', and 'meta' to see what sticks.
#TARGET_MODALITIES = ['rosbag', 'imu', 'meta']
TARGET_MODALITIES = ['rosbag', 'imu', 'meta', 'image']
#CAMERA_NAMES = []
CAMERA_NAMES = ['lcam_front']
DOWNLOAD_ROOT = './anymal'

def download_data():
    print(f"[ACTION] -> Initializing Data Download...")
    try:
        ta.init(DOWNLOAD_ROOT)
    except Exception as e:
        print(f"[FATAL] Init failed: {e}")
        return

    results = {}

    for env in PRIORITY_CANDIDATES:
        print(f"\n[TEST] {env:<25} ... ", end="")

        target_dir = os.path.join(DOWNLOAD_ROOT, env)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        try:
            ta.download_ground(
                env=[env],
                version=TARGET_VERSION,
                traj=TARGET_TRAJ,
                modality=TARGET_MODALITIES,
                camera_name=CAMERA_NAMES,
                unzip=True
            )

            data_path = os.path.join(DOWNLOAD_ROOT, env)
            found_files = []

            if os.path.exists(data_path):
                # Pass 1: Conditional ZIP Cleanup
                for root, _, files in os.walk(data_path):
                    for f in files:
                        if f.endswith('.zip'):
                            base_name = f[:-4]
                            extracted_path = os.path.join(root, base_name)
                            
                            # Delete ONLY if the extracted counterpart exists
                            if os.path.exists(extracted_path):
                                os.remove(os.path.join(root, f))
                                print(f"   > Cleaned ZIP: {f} (extracted version found)")

                # Pass 2: Build Final Manifest
                for root, _, files in os.walk(data_path):
                    for f in files:
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(full_path, data_path)
                        found_files.append(rel_path)

            if found_files:
                print("✅ DATA FOUND")
                results[env] = found_files
                print(f"   > Manifest for {env}:")
                for f in found_files:
                    print(f"     - {f}")
            else:
                print("❌ EMPTY (No files retrieved)")

        except (IndexError, ValueError, KeyError):
            print("❌ UNAVAILABLE (API Index Error)")
        except Exception as e:
            print(f"⚠️ ERROR: {e}")

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print("\n" + "="*50)
    print("AVAILABILITY REPORT")
    print("="*50)

    if results:
        for env, files in results.items():
            print(f"\n[{env}]")
            has_bag = any('bag' in f for f in files)
            print(f"  - Status: {'🟢 COMPLETE' if has_bag else '🟡 PARTIAL (No Bag)'}")
            print(f"  - Files: {len(files)} items")
            if has_bag:
                print(f"  - Bag File: {[f for f in files if 'bag' in f][0]}")
    else:
        print("❌ CRITICAL: No data found for trajectory in any priority environment.")

if __name__ == "__main__":
    download_data()
