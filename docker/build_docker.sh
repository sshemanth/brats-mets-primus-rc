#!/bin/bash
# Build, test, save and upload the BraTS-METS ensemble Docker image.
# usage:  nohup ~/brats_package/docker/build_docker.sh > ~/logs/build_docker.log 2>&1 &
# Every step prints a line starting with "==>" so the log is easy to follow.
set -euo pipefail

# ------------------------------------------------------------------ paths (edit here if needed)
PKG=~/brats_package/docker                                   # Dockerfile + infer.py
TRAINER_DIR=~/nnUNet-primus/nnunetv2/training/nnUNetTrainer  # where both custom trainers live
RES=/data/nnUNet_results/Dataset101_BraTSMET
RESENC_DIR=$RES/nnUNetTrainerCoreCavityV3ResEncL__nnUNetResEncUNetLPlans__3d_fullres
PRIMUS_DIR=$RES/nnUNetTrainerPrimusV3S_RegionAwareRC__nnUNetResEncUNetLPlans__3d_fullres
BUILD=/data/docker_build                                     # build context (on /data: the root disk is almost full)
TEST=/data/docker_test                                       # small test input/output
BUCKET=gs://brats_data_imp/brats_mets_2026_project/06_docker_image
TEST_INPUT=gs://brats_data_imp/final_backup/brats_docker_test_multi/input
IMAGE=brats-mets-ensemble:fold0
TAR=/data/brats_mets_ensemble_fold0.tar.gz

# use sudo for docker only if the user is not in the docker group
DOCKER="docker"; docker info >/dev/null 2>&1 || DOCKER="sudo docker"

echo "==> 1. checks"
command -v docker >/dev/null || { echo "docker is not installed"; exit 1; }
for f in "$RESENC_DIR/plans.json" "$RESENC_DIR/fold_0/checkpoint_final.pth" \
         "$PRIMUS_DIR/plans.json" "$PRIMUS_DIR/fold_0/checkpoint_final.pth" \
         "$TRAINER_DIR/nnUNetTrainerCoreCavityV3ResEncL.py" "$TRAINER_DIR/nnUNetTrainerPrimusV3S_RegionAwareRC.py"; do
  [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done

# Docker stores images under its data-root (usually /var/lib/docker on the small root disk).
# The image is several GB, so move the data-root to /data if it is still on the root disk.
ROOTDIR=$($DOCKER info -f '{{.DockerRootDir}}')
if [[ "$ROOTDIR" != /data/* ]]; then
  echo "==> moving Docker data-root from $ROOTDIR to /data/docker (keeps existing daemon settings)"
  sudo mkdir -p /data/docker /etc/docker
  sudo python3 - <<'PY'
import json, os
p = "/etc/docker/daemon.json"
cfg = json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) else {}
cfg["data-root"] = "/data/docker"          # merge: the NVIDIA runtime settings stay untouched
json.dump(cfg, open(p, "w"), indent=2)
PY
  sudo systemctl restart docker
  echo "    Docker data-root now: $($DOCKER info -f '{{.DockerRootDir}}')"
fi

echo "==> 2. prepare build context in $BUILD"
rm -rf "$BUILD"; mkdir -p "$BUILD/trainers" "$BUILD/models/resencl/fold_0" "$BUILD/models/primus/fold_0"
cp "$PKG/Dockerfile" "$PKG/infer.py" "$BUILD/"
cp "$TRAINER_DIR/nnUNetTrainerCoreCavityV3ResEncL.py" "$TRAINER_DIR/nnUNetTrainerPrimusV3S_RegionAwareRC.py" "$BUILD/trainers/"
for pair in "resencl:$RESENC_DIR" "primus:$PRIMUS_DIR"; do
  name=${pair%%:*}; src=${pair#*:}
  cp "$src"/*.json "$BUILD/models/$name/"                                  # plans.json, dataset.json, ...
  cp "$src/fold_0/checkpoint_final.pth" "$BUILD/models/$name/fold_0/"
done
du -sh "$BUILD"

echo "==> 3. read the exact library commits used for training"
set +u; source ~/primus-env/bin/activate; set -u    # venv scripts may use unset variables
NNUNET_COMMIT=$(git -C ~/nnUNet-primus rev-parse HEAD)
DNA_COMMIT=$(pip freeze | grep -i "dynamic.network.architectures @" | sed 's/.*@//' || true)
[ -n "$DNA_COMMIT" ] || { echo "    WARNING: dynamic-network-architectures commit not found, using main"; DNA_COMMIT=main; }
echo "    nnU-Net $NNUNET_COMMIT | dynamic-network-architectures $DNA_COMMIT"

echo "==> 4. build image $IMAGE"
$DOCKER build -t "$IMAGE" --build-arg NNUNET_COMMIT="$NNUNET_COMMIT" --build-arg DNA_COMMIT="$DNA_COMMIT" "$BUILD"

echo "==> 5. test on the sample cases from the bucket"
rm -rf "$TEST"; mkdir -p "$TEST/input" "$TEST/output"
gcloud storage cp "$TEST_INPUT/*" "$TEST/input/"
$DOCKER run --rm --gpus all --network none -v "$TEST/input":/input:ro -v "$TEST/output":/output "$IMAGE"
ls -l "$TEST/output"
python3 - <<PY
# check every output has exactly the geometry of its input T1c
import glob, os, SimpleITK as s, numpy as np
bad = 0
for f in sorted(glob.glob("$TEST/output/*.nii.gz")):
    c = os.path.basename(f)[:-7]
    p, r = s.ReadImage(f), s.ReadImage(f"$TEST/input/{c}-t1c.nii.gz")
    ok = p.GetSize() == r.GetSize() and np.allclose(p.GetSpacing(), r.GetSpacing()) \
         and np.allclose(p.GetOrigin(), r.GetOrigin()) and np.allclose(p.GetDirection(), r.GetDirection())
    print(c, "OK" if ok else "GEOMETRY MISMATCH", np.unique(s.GetArrayFromImage(p)))
    bad += not ok
print("test cases with problems:", bad)
PY

echo "==> 6. save image to $TAR and upload"
$DOCKER save "$IMAGE" | gzip > "$TAR"
ls -lh "$TAR"
gcloud storage cp "$TAR" "$BUCKET/"
cp "$BUILD/Dockerfile" "$BUILD/infer.py" /tmp/ && gcloud storage cp /tmp/Dockerfile /tmp/infer.py "$BUCKET/"

echo "==> DOCKER DONE  (load later with: gunzip -c $(basename "$TAR") | docker load)"
