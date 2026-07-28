# Third-Party Notices

`sttx` combines the following upstream software, model, and model-conversion
artifacts. Their terms remain separate from the `sttx` project, which makes no
license grant of its own.

## sherpa-onnx

The Python ASR/VAD integration uses **sherpa-onnx** by k2-fsa. The upstream
repository identifies the project and displays an Apache-2.0 license:

- Project: https://github.com/k2-fsa/sherpa-onnx
- License: https://github.com/k2-fsa/sherpa-onnx/blob/master/LICENSE

## Silero VAD

The `silero_vad.onnx` asset is Silero VAD, distributed by sherpa-onnx. Silero's
upstream repository displays the MIT license; retain the upstream notice when
redistributing the asset:

- Project: https://github.com/snakers4/silero-vad
- License: https://github.com/snakers4/silero-vad/blob/master/LICENSE
- sherpa-onnx release tag carrying the asset:
  https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
- Direct asset URL:
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

The exact terms bundled with a particular release asset should be checked
against the accompanying upstream notices.

## NVIDIA Parakeet TDT 0.6B v3

The acoustic model is **NVIDIA Parakeet TDT 0.6B v3**. NVIDIA's model card
states that its governing terms are CC BY 4.0:

- Model card: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- License: https://creativecommons.org/licenses/by/4.0/

## Converted ONNX/int8 repository

The flat ONNX/int8 files are obtained from the converted Hugging Face
repository **csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8**:

- Repository: https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8

That repository page does not expose separate license metadata. Do not treat
the conversion repository as NVIDIA-authored or assign it CC BY 4.0 solely from
its identity; consult the repository's own notices together with the NVIDIA
model-card terms.
