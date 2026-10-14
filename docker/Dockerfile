# BraTS-METS 2026 inference image: ResEnc-L + Primus (fold 0) ensemble.
# Build with build_docker.sh, which prepares this folder (models/ and trainers/) and passes the
# exact nnU-Net / dynamic-network-architectures commits used during training.

# PyTorch + CUDA runtime base image (GPU support comes from the NVIDIA container toolkit on the host).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# git is needed to install the two libraries from GitHub at fixed commits.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Pin the library versions to what the models were trained with (filled in by build_docker.sh).
ARG NNUNET_COMMIT=master
ARG DNA_COMMIT=main
RUN pip install --no-cache-dir "git+https://github.com/MIC-DKFZ/nnUNet.git@${NNUNET_COMMIT}" \
 && pip install --no-cache-dir --force-reinstall --no-deps \
        "git+https://github.com/MIC-DKFZ/dynamic-network-architectures.git@${DNA_COMMIT}"

# nnU-Net rebuilds each network from its trainer class, so both custom trainers must be inside the package.
COPY trainers/ /tmp/trainers/
RUN TRAINER_DIR=$(python -c "import nnunetv2, os; print(os.path.join(os.path.dirname(nnunetv2.__file__), 'training', 'nnUNetTrainer'))") \
 && cp /tmp/trainers/*.py "$TRAINER_DIR"/ && rm -rf /tmp/trainers

# Trained models: plans.json, dataset.json and fold_0/checkpoint_final.pth for each model.
COPY models/ /opt/models/
COPY infer.py /opt/app/infer.py

# nnU-Net warns if these are unset; they are not used for inference.
ENV nnUNet_raw=/tmp/nnUNet_raw \
    nnUNet_preprocessed=/tmp/nnUNet_preprocessed \
    nnUNet_results=/opt/models \
    PYTHONUNBUFFERED=1

# Default: read /input, write /output (override with --input/--output if the challenge uses other paths).
ENTRYPOINT ["python", "/opt/app/infer.py"]
CMD ["--input", "/input", "--output", "/output"]
