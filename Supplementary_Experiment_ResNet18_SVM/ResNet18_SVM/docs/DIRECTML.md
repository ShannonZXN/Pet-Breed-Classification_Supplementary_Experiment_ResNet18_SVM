# Windows / AMD extraction with DirectML

The supplied script sets `DEVICE = "directml"`. This guide preserves its
separate Windows DirectML installation route, including the pinned
`torch-directml==0.2.5.dev240914` dependency. The pin is retained for continuity;
it is not a claim that it is the newest release.

DirectML is a PyTorch backend for compatible DirectX 12 hardware. Availability
and operator behavior depend on the installed package, hardware, and driver.
The target Radeon RX 5700 XT was not available during local verification.
Use the real sample check below to establish whether your configuration works
and to measure extraction time. SVM fitting stays on CPU.

## 1. Create a separate Python 3.12 environment

With 64-bit Python 3.12 installed, run in the project folder:

```powershell
py -0p
py -3.12 -m venv .venv-dml
.\.venv-dml\Scripts\python.exe -m pip install --upgrade pip
.\.venv-dml\Scripts\python.exe -m pip install -r requirements-directml.txt
```

`py -3.12` selects an existing interpreter; it does not convert a newer
interpreter into 3.12. This environment can coexist with a CPU/CUDA environment.
The pinned release provides a Windows CPython 3.12 wheel in its published files.

Let `torch-directml` choose its matching torch/torchvision dependencies.
`requirements-directml.txt` retains `numpy>=1.26,<2` for that older stack.
Do not independently install the latest CPU/CUDA torch pair into this same
environment. If dependency resolution fails, inspect the pinned release's
metadata and use an interpreter/platform for which it provides a wheel.

## 2. Configure the existing inputs

At the top of `resnet18_svm.py`:

```python
IMAGE_FOLDERS = {
    "original": Path(r"D:\Pets\original"),
    "foreground": Path(r"D:\Pets\background_removed_all"),
}
DEVICE = "directml"
BATCH_SIZE = 1
NUM_WORKERS = 0
```

Use actual paths. Supply your exact manifest as described in
[Step 1 of the run guide](../START_HERE.md#step-1--supply-the-exact-split-membership).
The source archive did not include it.

`DEVICE="auto"` means CUDA when available, otherwise CPU. It does not
automatically select DirectML. An explicitly requested DirectML initialization
failure is reported as a failure rather than silently selecting another backend.

## 3. Verify device operations and the actual extractor

```powershell
.\.venv-dml\Scripts\python.exe resnet18_svm.py check
.\.venv-dml\Scripts\python.exe self_test.py
.\.venv-dml\Scripts\python.exe resnet18_svm.py audit --dataset both
.\.venv-dml\Scripts\python.exe resnet18_svm.py smoke --dataset both
```

`environment_report.json` should show a passing device tensor check and
`selected_backend: "directml"`. The selected device may appear as
`privateuseone:0`; `cuda_available: false` is compatible with using DirectML.

The basic check probes one tensor operation; the smoke check exercises the
pretrained network on native-size images. Each branch's `step2_report.json`
should show `passed: true`. Inspect `inference_seconds` in its extraction
metadata rather than assuming GPU acceleration improves runtime.

If extraction fails, keep the report, inspect the error, and resolve the
installation/operator issue or deliberately select `DEVICE="cpu"`. An out-of-
memory problem should not be worked around by resizing inputs if the intended
experiment requires native resolution.

## 4. Continue the normal experiment

```powershell
.\.venv-dml\Scripts\python.exe resnet18_svm.py train --dataset foreground
.\.venv-dml\Scripts\python.exe resnet18_svm.py train --dataset original
.\.venv-dml\Scripts\python.exe resnet18_svm.py test --dataset both
```

Results remain under `resnet18_svm_outputs_available/`. The same manifest,
availability policy, feature specification, candidate grid, and separate final
test stages apply. When intentionally recomputing features on another backend,
use `--refresh-cache` and keep its environment metadata.

## Official references

- [Microsoft: enable PyTorch with DirectML on Windows](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows)
- [Pinned torch-directml release and wheel files](https://pypi.org/project/torch-directml/0.2.5.dev240914/)
- [PyTorch installer for a separate CPU/CUDA setup](https://pytorch.org/get-started/locally/)

This package does not assert current ROCm support for a specific GPU; its
provided AMD route uses DirectML. Check current vendor documentation when
deliberately choosing a different backend.
