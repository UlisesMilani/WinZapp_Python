# Third-party notices

## DPDFNet

WinZapp's optional neural noise reduction for outgoing voice recordings uses
DPDFNet 0.6.0, distributed under the Apache License 2.0.

- Source code: https://github.com/ceva-ip/DPDFNet
- Python package: https://pypi.org/project/dpdfnet/0.6.0/
- Pretrained model: Ceva-IP/DPDFNet, `dpdfnet2_48khz_hr.onnx`
- Pinned model revision: c7ac7b249ff5e17fa606794dc4f68ed9a544834f
- Model SHA-256: `7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b`

The ONNX model is downloaded on first use into WinZapp's writable global data
area and verified before inference. Consult the upstream repository and model
card for complete license terms and attribution details.
