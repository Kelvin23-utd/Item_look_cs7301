# Pivot check, Part 2: which models can do the perception on a Pixel 8?

Research date: 2026-11-16. Target: Google Tensor G3, 8 GB RAM, Android 17, with ARCore running at the same time and supplying camera pose and motion-based depth.

## Read this first

- No page opened in this session publishes a latency for any candidate model on a Pixel 8 or Tensor G3. Every phone number below is for a different chip and is marked as such. The first engineering task is to measure on the Pixel 8 itself.
- Every fact has the URL of the page it came from. Pages were read through a fetch tool that extracts answers from page text; short quotes are reproduced as returned and should be re-checked before they are used in the paper.
- "Not stated" means the page opened did not say; it does not mean the thing does not exist.
- Licences matter for what can be shipped, less for a research prototype. Research-only and copyleft licences are flagged.

## Platform facts that shape the choice

| Fact | Source |
|---|---|
| On a Pixel 10 (Tensor G5), a LiteRT export of YOLO26n detection at 640 px takes 36.6 ms on CPU and 36.5 ms on GPU, so the GPU gave no speed-up; other phones in the same table get 10-16 ms on GPU or NPU. The page notes that "Google's Tensor NPU has no LiteRT row". Different chip from Tensor G3. | https://docs.ultralytics.com/integrations/litert |
| LiteRT's NPU support for Google Tensor is labelled Beta and ahead-of-time only: "Google Tensor SDK doesn't yet support on-device (JIT) compilation." No Tensor chips are named. | https://developers.google.com/edge/litert/next/npu |
| In Google's AI Edge Gallery app, a Pixel 10 Pro Fold (Tensor G5) offered only a CPU backend for Gemma 4 E2B: time to first token 1.65 s, 10.42 tokens/s decode, against 48.55 tokens/s on a Galaxy S26 Ultra GPU. The article says no Android chip in its test used the NPU. Different chip from Tensor G3; text-only benchmark. | Beebom, Arjun Sha, 25 Apr 2026, https://gadgets.beebom.com/stories/i-tested-on-device-ai-on-android-and-iphone-results-not-even-close |
| ARCore depth is "a depth-from-motion algorithm"; "Valid depth data are only available after the user has started moving their device around"; most accurate "half a meter to about five meters away"; featureless surfaces "will be associated with imprecise depth". | https://developers.google.com/ar/develop/depth |
| Google's archived ARCore ML sample (Kotlin) shows the pattern of feeding ARCore camera images to an on-device model and creating anchors from the results. | https://github.com/googlesamples/arcore-ml-sample |

Working assumption that follows (an inference, not a published fact): on Tensor G3, plan for CPU (XNNPACK) inference, treat the GPU delegate as something to test rather than rely on, and do not plan around the NPU.

## A. Detectors that label many objects per image without a per-query prompt

| Model | Size | Licence | Android-runnable export | Published phone latency | Prompt needed? |
|---|---|---|---|---|---|
| YOLOE, prompt-free checkpoints (`yoloe-v8{s,m,l}-seg-pf`, `yoloe-11{s,m,l}-seg-pf`, `yoloe-26{n,s,m,l,x}-seg-pf`) | Prompt-free parameter counts in the YOLOE paper: 11-S 11M, v8-S 13M, 11-M 24M, 11-L 29M. YOLOE-26 paper: n 6.5M, s 16.2M (prompt-free). The Ultralytics page gives 2.3M for prompt-free 26n; the two sources disagree. File sizes not stated. | Ultralytics code and models: AGPL-3.0, or a paid Enterprise licence (https://www.ultralytics.com/license). The YOLOE docs page itself does not state a licence. | Ultralytics lists ONNX, OpenVINO, TensorRT, CoreML, LiteRT and RKNN as static export targets. The YOLOE-26 paper says prompt-free models export like standard YOLO models. The docs do not describe a prompt-free export workflow, so this must be tried. | None on Android. iPhone 12 via CoreML, text-prompt mode only: 11-S 73.3 FPS, v8-S 64.3 FPS (different chip, different mode). Prompt-free speed is reported only on a T4 GPU in PyTorch: 11-S 93.0 FPS, v8-S 95.8 FPS. | No. Built-in vocabulary of 4,585 names taken from the RAM++ tag list. Returns boxes and masks with names. |
| YOLO-World (v2 weights) | Not stated in the README or Ultralytics page | Official repo: "YOLO-World is under the GPL-v3 Licence and is supported for commercial usage." | Official repo: ONNX export and "TFLite Export" with INT8 quantisation. Ultralytics: export supported only for `*-worldv2.pt`. | None | A vocabulary is set once offline and baked into the exported model ("re-parameterizes vocabulary embeddings as parameters into the model"), so no text encoder runs on the phone. Vocabulary is whatever list you choose. |
| OWLv2 base (`google/owlv2-base-patch16-ensemble`) | 0.2B parameters | Apache-2.0 | None mentioned on the model card | None | Yes, text queries per class. Too heavy and no mobile path found. |
| RAM / RAM++ tagger | Swin-Large backbone; sizes not stated | Apache-2.0 | None mentioned | None | No prompt, but outputs tags only (RAM 6,400+ tags; RAM++ 4,585-tag list), no boxes. Needs a separate detector. Server-class backbone. |
| MediaPipe Object Detector (EfficientDet-Lite0, EfficientDet-Lite2, SSD MobileNetV2) | Input 320, 448, 256 px | Not checked | Native Android task | Pixel 6 (older, different chip): Lite0 int8 29.31 ms CPU; Lite0 float32 61.30 ms CPU, 27.83 ms GPU; Lite2 int8 70.91 ms CPU; SSD MobileNetV2 float32 36.30 ms CPU, 24.01 ms GPU | Closed set of 80 COCO labels |
| ML Kit Object Detection plus Image Labeling | Bundled with the SDK | ML Kit terms | Native Android, Gradle dependency | None listed | Detector: on-device, tracking IDs, five coarse categories ("home goods, fashion goods, food, plants, and places"), optional custom LiteRT classifier. Labeler: "more than 400 categories" for a whole image or crop. |

Sources: https://docs.ultralytics.com/models/yoloe , https://arxiv.org/html/2503.07465v1 , https://arxiv.org/html/2602.00168 , https://github.com/AILab-CVC/YOLO-World , https://docs.ultralytics.com/models/yolo-world , https://huggingface.co/google/owlv2-base-patch16-ensemble , https://github.com/xinyu1205/recognize-anything , https://developers.google.com/edge/mediapipe/solutions/vision/object_detector , https://developers.google.com/ml-kit/vision/object-detection , https://developers.google.com/ml-kit/vision/image-labeling

Further facts on YOLOE that affect the plan:

- Accuracy in prompt-free mode is modest: fixed AP on LVIS minival is 20.6 for 11-S, 21.0 for v8-S, 25.5 for 11-M, 26.3 for 11-L (YOLOE paper, Table 3). YOLOE-26 reports prompt-free mAP50-95 of 16.6 (n), 21.4 (s), 25.7 (m).
- The large vocabulary costs time: on CPU with `yoloe-26s-seg.pt`, a forward pass grows "about 19% going from 80 to 1,203 classes" and about 89% at the full 4,585 names (Ultralytics YOLOE page).
- Prompt-free checkpoints cannot be re-targeted: "Prompt-free model does not support setting classes." Text-prompt checkpoints can have a custom class list baked in before export ("Exported models are static"); the text encoder (about 254 MB for YOLOE-26) is then needed only at export time on a computer.
- Authors and venue as shown on the paper page: Ao Wang, Lihao Liu, Hui Chen, Zijia Lin, Jungong Han, Guiguang Ding (Tsinghua), arXiv:2503.07465; no venue printed on v1.

Anything newer: YOLOE-26 (arXiv:2602.00168, Ranjan Sapkota and Manoj Karkee) is the newest opened; it gives no latency on any hardware. OV-DEIM and RT-DETR-World appeared in search snippets as 2026 real-time open-vocabulary detectors but were not opened and no mobile numbers were seen.

## B. Class-agnostic "segment everything" models

| Model | Size | Licence | Android-runnable export | Published phone latency | Segment-everything? |
|---|---|---|---|---|---|
| MobileSAM | 9.66M parameters (encoder 5M, decoder 3.876M) per its README. Qualcomm's export: encoder 26.6 MB, decoder 23.7 MB | Apache-2.0 | ONNX export script in the repo. Qualcomm AI Hub publishes TFLITE, ONNX and QNN builds | Snapdragon 8 Gen 3 NPU (different chip): encoder 79.9 ms (QNN), 172.7 ms (ONNX), 490.3 ms (TFLITE); decoder 3.4-4.1 ms. iPhone 14: 4.9 FPS as reported in the EdgeSAM README. Authors' own figures are for an unnamed GPU (12 ms) and a Mac i5 CPU (about 3 s) | Yes by prompting a grid of points, which multiplies decoder calls; MobileSAMv2 replaces the grid with object-aware prompts |
| EdgeSAM | 9.6M parameters | NTU S-Lab License 1.0 | CoreML and ONNX exports provided; Android and TFLite not mentioned | iPhone 14: 38.7 FPS (different chip) | README documents box and point prompts only |
| EdgeTAM (CVPR 2025) | CoreML parts about 9.6 MB, 2 MB and 8 MB | Apache 2.0 | CoreML only is documented | iPhone 15 Pro Max: 16 FPS for video tracking, 40.4 FPS in its image-segmentation table (different chip) | Yes: "also supports automatic mask generation on images just like SAM" |
| FastSAM-s | 11.8M parameters, 23.9 MB | Not stated on the Ultralytics page | Export marked supported; only an ONNX example is shown | None on a phone. 58.0 ms per image on a 2025 Apple M4 laptop CPU | Yes, an "Everything Prompt" |
| EfficientSAM3 (arXiv:2511.15833, Nov 2025) | Not stated on the abstract page | Not stated | Not stated | None | Newer SAM 3 distillation; release status not verified |

Sources: https://github.com/ChaoningZhang/MobileSAM , https://huggingface.co/qualcomm/MobileSam , https://github.com/chongzhou96/EdgeSAM , https://github.com/facebookresearch/EdgeTAM , https://docs.ultralytics.com/models/fast-sam , https://arxiv.org/abs/2511.15833

Note: the YOLOE checkpoints are segmentation models and return a mask per detection. A separate segment-everything model is only needed for objects YOLOE's vocabulary misses, and can be left out of a first prototype.

## C. Text-image embedding models

| Model | Size (image + text) | Licence | Android-runnable export | Published phone latency | Zero-shot ImageNet top-1 |
|---|---|---|---|---|---|
| MobileCLIP-S0 | 11.4M + 42.4M parameters | Code MIT; "ML models: Apple ML Research Model TOU" | README covers PyTorch, OpenCLIP and an iOS demo app; nothing for Android, TFLite or ONNX | 1.5 ms + 1.6 ms in the README table. The table does not name the device; an iPhone 12 Pro Max is named in the highlights for another variant. Different chip in any case | 67.8% |
| MobileCLIP2-S0 | 11.4M + 63.4M | Same; Hugging Face tag `apple-amlr` | Community ONNX conversion (`plhery/mobileclip2-onnx`): S0 vision encoder 43 MB, 256x256 input, 512-d output, made for transformers.js. Outputs are unnormalised. Not an official export | 1.5 ms + 3.3 ms (same caveat) | 71.5% |
| MobileCLIP2-S2 | 35.7M + 63.4M | Same | Community ONNX: vision 136 MB | 3.6 ms + 3.3 ms (same caveat) | 77.2% |
| SigLIP 2 base (`google/siglip2-base-patch16-224`) | 0.4B parameters | Apache-2.0 | None mentioned | None | Not small; the card names no smaller variant |
| TinyCLIP (e.g. ViT-8M/16 + Text-3M; ViT-39M/16 + Text-19M) | 23.4M total for the smallest | MIT | PyTorch and Safetensors only | None; throughput figures on unnamed hardware | 41.1% (smallest), 63.5% (ViT-39M/16) |
| OpenAI CLIP ViT-B/16, Qualcomm export | 150M parameters, 571 MB | MIT | TFLITE, ONNX and QNN builds | Whole model on Snapdragon NPUs (different chips): 21.1 ms on 8 Gen 3, 17.1 ms on 8 Elite, 12.8 ms on 8 Elite Gen 5 (TFLITE). The same card shows 520-750 ms where it falls back to CPU on automotive chips | Not stated on the card |
| Perception Encoder Core (`facebook/PE-Core-*`) | B/16: 0.09B vision + 0.31B text | Apache-2.0 | None mentioned | None | 78.4% (B/16); the card opened gives no figures for the tiny variant |
| ML Kit GenAI Embedding API (EmbeddingGemma 2) | System model managed by AICore | ML Kit GenAI terms | Android API, preview (`genai-embedding:1.0.0-alpha1`); text and image input, 768-d output | None | Requires an AICore-supported device and Developer Preview enrolment; Pixel 8 is not mentioned |

Sources: https://github.com/apple/ml-mobileclip , https://huggingface.co/apple/MobileCLIP2-S0 , https://huggingface.co/plhery/mobileclip2-onnx , https://huggingface.co/google/siglip2-base-patch16-224 , https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M , https://huggingface.co/qualcomm/OpenAI-Clip , https://huggingface.co/facebook/PE-Core-T16-384 , https://developers.google.com/ml-kit/genai/embedding/android

Notes:

- Licence caution for the MobileCLIP family: the weights licence file itself could not be opened (404 at the path tried). The sibling Apple licence that was opened (FastVLM's LICENSE_MODEL, "Apple Machine Learning Research Model License Agreement") limits use to research and excludes "any commercial exploitation, product development or use in any commercial product or service" (https://github.com/apple/ml-fastvlm/blob/main/LICENSE_MODEL). Assume MobileCLIP weights are research-only until the actual file is read. That is acceptable for a paper prototype.
- Existence proof on Android: TIDY is an open-source (GPL-3.0) Android app doing offline text-to-image search with a quantised OpenCLIP model on ONNX Runtime (https://github.com/slavabarkov/tidy). It gives no speed numbers.

## D. Small vision-language models that can name or describe a crop

| Model | Size | Licence | Android-runnable export | Published phone latency | Notes |
|---|---|---|---|---|---|
| Gemma 3n E2B | "over 5 billion parameters" loaded normally, effective load 1.91B with parameter skipping; file size not stated | Gemma terms (gated download) | LiteRT-LM file (`gemma-3n-E2B-it-int4.litertlm`), Kotlin API; AI Edge Gallery app has an "Ask Image" demo | Samsung S24 Ultra (different chip): CPU 110.5 tokens/s prefill, 16.1 decode; GPU 816.4 prefill, 15.6 decode. Vivo X300 Pro NPU: 1,671.3 prefill, 28.4 decode. No image-encoder timing given | Each image is resized to 256, 512 or 768 px and "encoded to 256 tokens". Google suggests it as the stand-in for Gemini Nano on unsupported phones |
| Gemma 4 E2B | 2,583 MB file | Apache-2.0 (per the model card) | LiteRT-LM; AI Edge Gallery; AICore Developer Preview on supported devices | Samsung S26 Ultra (different chip): CPU 557 tokens/s prefill, 46.9 decode, 1.8 s to first token, 1,733 MB; GPU 3,808 prefill, 52.1 decode, 0.3 s, 676 MB. Pixel 10 Pro Fold, Tensor G5, CPU only: 1.65 s to first token, 10.42 tokens/s (Beebom, 256-token text benchmark) | "the Vision and Audio models are loaded on demand"; released 2 Apr 2026 with E2B and E4B sizes |
| SmolVLM-256M-Instruct | 0.3B parameters | Apache 2.0 | Official ONNX weights; the 500M sibling has GGUF builds for llama.cpp (Q8_0 437 MB, F16 820 MB) | None | "under 1GB of GPU RAM"; 64 visual tokens per 512x512 patch; English only |
| LFM2-VL-450M / LFM2.5-VL-450M | 350M language model + 86M vision encoder | LFM Open License v1.0; commercial use allowed below $10M annual revenue | GGUF for llama.cpp, ONNX, MLX (LFM2.5 card); a vendor SDK is linked but not described | None | Native 512x512; 32-256 image tokens, tunable; LFM2.5 can output bounding boxes |
| Moondream 0.5B | 0.5B parameters; 8-bit: 479 MiB download, 996 MiB memory; 4-bit: 375 MiB, 816 MiB | Apache | Format not stated in the announcement | None in text | Announced 5 Dec 2024 for edge and mobile |
| Moondream 2 | 2B parameters | Apache-2.0 | Quantised builds for llama.cpp are linked | None | Captioning, question answering, detection and pointing; the card calls it previous-generation |
| FastVLM-0.5B | Card shows 0.8B parameters | `apple-amlr`, research only | Apple-silicon export and an iOS demo app; nothing for Android or ONNX in the README | None with a named device ("85x faster Time-to-First-Token" than LLaVA-OneVision-0.5B) | CVPR 2025 |

Sources: https://huggingface.co/google/gemma-3n-E2B-it-litert-lm , https://ai.google.dev/gemma/docs/gemma-3n , https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm , https://developers.googleblog.com/en/blazing-fast-on-device-genai-with-litert-lm/ , https://developer.android.com/blog/posts/announcing-gemma-4-in-the-ai-core-developer-preview , https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct , https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF , https://huggingface.co/LiquidAI/LFM2-VL-450M , https://huggingface.co/LiquidAI/LFM2.5-VL-450M , https://www.liquid.ai/lfm-license , https://moondream.ai/blog/introducing-moondream-0-5b , https://huggingface.co/vikhyatk/moondream2 , https://github.com/apple/ml-fastvlm , https://huggingface.co/apple/FastVLM-0.5B

One relevant finding from a 2026 measurement paper (server GPU, not a phone): in compact vision-language models "end-to-end latency ... is often dominated by CPU-side workload" such as image preprocessing and tokenisation, so a small parameter count does not guarantee a fast response (Huang et al., Sony AI, arXiv:2603.16987, https://arxiv.org/html/2603.16987).

### Is Gemini Nano with image input available to third-party apps on a Pixel 8?

Not according to Google's device list. The ML Kit GenAI overview lists, for the Prompt API, Pixel 9, Pixel 9 Pro, Pixel 9 Pro XL and Pixel 9 Pro Fold under Gemini Nano 3 and the Pixel 10 and Pixel 11 series under Gemini Nano 4. Pixel 8, Pixel 8 Pro and Pixel 8a do not appear anywhere on that page, in the Prompt API lists or in the Image Description list (https://developers.google.com/ml-kit/genai). The Prompt API does take images (`ImagePart(bitmapImage)` in Google's own example) and was announced as an alpha on 30 Oct 2025; that post says it "performs best on the Pixel 10 device series" and tells developers without one to prototype "with Gemma 3n locally" (https://developer.android.com/blog/posts/ml-kit-s-prompt-api-unlock-custom-on-device-gemini-nano-experiences). The AICore Developer Preview is limited to "the devices that support custom generation and agents" (https://developers.google.com/ml-kit/genai/aicore-dev-preview).

Caveat: the fetch tool abbreviated the long device lists, although it reported explicitly that no Pixel 8 model appears. Confirm on the device by calling the API's status check. Plan on the answer being no, and on bundling an open model instead.

## Published phone latencies at a glance

| Model and task | Number | Device and chip | Applies to Pixel 8? |
|---|---|---|---|
| YOLO26n detection, 640 px, LiteRT | 36.6 ms CPU, 36.5 ms GPU | Pixel 10, Tensor G5 | No. Nearest Tensor data point; closed-set nano model, not YOLOE |
| YOLO26n segmentation, 640 px, LiteRT | 51.5 ms GPU | Pixel 10, Tensor G5 | No |
| EfficientDet-Lite0 int8 | 29.31 ms CPU | Pixel 6 | No. Older chip |
| YOLOE-11-S, text-prompt, CoreML | 73.3 FPS | iPhone 12 | No. Different chip and mode |
| YOLOE-11-S, prompt-free | 93.0 FPS | Nvidia T4, PyTorch | No. Server GPU |
| MobileSAM encoder | 79.9-490.3 ms depending on runtime | Snapdragon 8 Gen 3 NPU | No |
| EdgeSAM | 38.7 FPS | iPhone 14 | No |
| CLIP ViT-B/16 whole model | 21.1 ms | Snapdragon 8 Gen 3 NPU | No |
| MobileCLIP2-S0 | 1.5 ms image, 3.3 ms text | Device not named in the table | No |
| Gemma 3n E2B | 110.5 tokens/s prefill, 16.1 decode | Samsung S24 Ultra CPU | No |
| Gemma 4 E2B | 10.42 tokens/s decode, 1.65 s to first token | Pixel 10 Pro Fold, Tensor G5 CPU | No. Nearest Tensor data point; text only |

## Recommendation: the cheapest pipeline to prototype in about a week

One developer, existing Kotlin ARCore app, research prototype. Build in the order below so that there is a working end-to-end loop after two days and each later step is a swap, not a rewrite.

1. Zero-export baseline (day 1-2). ML Kit Object Detection for class-agnostic boxes with tracking IDs, ML Kit Image Labeling on each crop for a name, ARCore depth at the box centre plus camera pose for a 3D point, merge by 3D distance and label, plain text search over names. Follow the structure of Google's archived ARCore ML sample. Run on keyframes (for example when the pose has changed enough), not every frame.
   - Evidence: both ML Kit APIs are on-device Gradle dependencies; no latency is published.
   - Main risk: naming is too coarse to be an inventory (five detector categories, about 400 whole-image labels), and the detector favours prominent objects, so cluttered shelves will be under-detected. Treat it as scaffolding and as the baseline to beat.

2. Detector and namer (day 2-4). Replace step 1's models with one prompt-free YOLOE small checkpoint (`yoloe-11s-seg-pf` or `yoloe-26s-seg-pf`) exported to LiteRT from Python and run on CPU at 640 px, at roughly one keyframe per second. One pass gives boxes, masks and names from 4,585 categories with no text encoder on the phone.
   - Evidence for speed: none on Tensor G3. The nearest published figures are 36.6 ms for the much smaller closed-set YOLO26n on a Pixel 10 CPU, 73.3 FPS for YOLOE-11-S in text-prompt mode on an iPhone 12, and the documented 89% extra forward-pass cost for the full vocabulary on CPU. An honest expectation is hundreds of milliseconds per frame on the Pixel 8 CPU; that is an estimate, not a measurement, and it is acceptable at keyframe rate.
   - Main risks: (a) the docs do not describe exporting a prompt-free checkpoint to LiteRT, so the export may fail or need the ONNX route; (b) prompt-free accuracy is about 21 AP on LVIS minival for the small models, so expect missed and mis-named items, especially small, occluded or unusual household things; (c) AGPL-3.0 means the app's source must be released, or a commercial licence bought.
   - Fallback if export fails: a text-prompt YOLOE or YOLO-World v2 checkpoint with a hand-written household vocabulary baked in before export (YOLO-World documents TFLite export with INT8). This loses the open vocabulary but keeps one-model simplicity.

3. Embedding for search and re-identification (day 4-5). MobileCLIP2-S0 (or MobileCLIP-S0) image encoder on each merged item's best crop, text encoder on the typed query, cosine similarity; reuse the image embedding to decide whether a later sighting is the same item. Run with ONNX Runtime on Android, starting from the community ONNX files.
   - Evidence for speed: 1.5 ms image and 3.3 ms text in Apple's table on an unnamed device (an iPhone is named elsewhere in the README); nothing on Android. The model is small (11.4M image parameters, 43 MB vision file), so CPU inference per crop should be affordable, but that is an inference.
   - Main risks: weights appear to be research-only; the ONNX files are a third-party conversion aimed at browsers and may need re-export or operator fixes for Android; embeddings of small, cluttered crops may not separate similar items; preprocessing and normalisation must match the conversion exactly. If licence matters, TinyCLIP (MIT) is the permissive but less accurate alternative.

4. Optional better names, offline (day 6-7, only if time remains). After a scan, with the AR session paused, caption the best crop of each uncertain item with Gemma 3n E2B or Gemma 4 E2B through LiteRT-LM, or with SmolVLM-256M through ONNX Runtime if memory is tight.
   - Evidence for speed: Gemma 4 E2B decodes at 10.42 tokens/s with 1.65 s to first token on a Tensor G5 CPU (text benchmark); Gemma 3n E2B prefill is 110.5 tokens/s on an S24 Ultra CPU and an image costs 256 tokens, which suggests seconds per crop on a faster chip than the Pixel 8's. Expect several seconds per item on Tensor G3; that is an estimate.
   - Main risks: memory (a 2.6 GB model next to ARCore in 8 GB; do not run both at once), heat and battery during a batch pass, invented names, and no accelerator path on Tensor through LiteRT-LM as far as the pages opened show. Gemini Nano is not an option on the Pixel 8.

Left out on purpose: a separate segment-everything model. YOLOE already returns masks; add MobileSAM (Apache-2.0, ONNX) or FastSAM only if recall on stored items proves to be the bottleneck.

Risk that cuts across all steps: ARCore's depth is weakest exactly where the idea is most novel. Inside an open drawer or box the camera is often closer than half a metre, moves little, and looks at low-texture surfaces, all conditions the Depth API page flags as poor. 3D positions for items in containers may need to fall back to "in this container" attached to an anchor on the container, rather than a point per item. Also unmeasured anywhere: how much ARCore tracking itself slows these models on Tensor G3, and thermal throttling over a ten-minute walk-through.

## Could not verify

- Any latency for any candidate on Pixel 8 or Tensor G3. Search results pointed to Geekbench AI uploads for a Pixel 8 with the TensorFlow Lite GPU backend, but the pages returned HTTP 403 and they do not name the underlying networks.
- Whether a prompt-free YOLOE checkpoint exports cleanly to LiteRT and what it costs on a phone CPU.
- Whether the LiteRT GPU delegate works and helps on Tensor G3 (the Tensor G5 data suggest it may not).
- The exact text of the MobileCLIP weights licence.
- Whether Pixel 8 can join the AICore Developer Preview in practice (the device list says no).
- Image-input latency for Gemma 3n or Gemma 4 on any phone; all published numbers are text token rates.
- Phone numbers for SmolVLM, Moondream, LFM2-VL and FastVLM on Android; none of their pages gives one. A third-party figure of about 7 s per image for SmolVLM-500M on Android appeared in a search snippet from an app's documentation and was not opened.
- OV-DEIM, RT-DETR-World, Qwen3-VL-2B on Android via llama.cpp, and Moondream 3: seen in search snippets only.
