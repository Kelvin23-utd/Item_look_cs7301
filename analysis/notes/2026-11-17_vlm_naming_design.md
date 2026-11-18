# Workstation VLM naming: fixed design

Written 2026-11-17, before any vision-language model or judge was run on FindBack data. It fixes the
details that `analysis/VLM_NAMING_PLAN.md` leaves to the implementer. Nothing had been run on the
data when this was written except the earlier baselines (Tracks A and B, whose label-free outputs
are in `analysis/notes/2026-11-17_track_b_label_free.md`); no name produced by any model below had
been seen. Any later change is listed at the end of the results note with its reason and whether
results had been seen.

## Question and answer format

Can a 3B–12B vision-language model (VLM) on the lab workstation name the items FindBack stores
better than the phone's 80-class detector, at what cost per item, and is that cost acceptable for a
background naming step (the phone does the real-time work, the workstation names items afterwards)?

## Machine and communication

- **The lab workstation**: a shared Linux machine with two RTX 4090 (24 GB each), reached by SSH.
  Its alias, address, account and paths are never written into the repository: the scripts read the
  SSH alias from `FB_WORKSTATION` and the private working folder from `FB_WS_DIR` (which may contain
  `$USER`, expanded on the workstation).
- **Working folder**: one directory inside the account's own folder on the workstation's large
  data disk (the disk is organised in per-user folders), mode 700, holding everything of this
  experiment in a fixed layout: `data/` (what was copied from the Mac), `code/`, `models/` (every
  weight cache; `HF_HOME` points here), `env/` (the Python environment and `env.sh`),
  `runs/<date>_<name>/` (outputs and logs of one session) and `scratch/` (`HOME`, `XDG_CACHE_HOME`,
  `TMPDIR` and the vLLM, Triton, Torch and CUDA caches of the processes started there). The home disk
  is 98% full and is not written to. One command removes exactly this directory.
- **Disk budget**: at most 80 GB in the folder at any time (the data disk had 326 GB free). Each
  contestant's weights are deleted after its runs; the judges' after judging.
- **Telemetry off**: `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`, `HF_HUB_DISABLE_TELEMETRY=1`.
  Weights only from the publishers' own Hugging Face repositories, at pinned revisions. No cloud API.
- **Files** go by `rsync` over SSH; **runs** start by SSH; each model is served by vLLM bound to
  `127.0.0.1` on the workstation, and the Mac reaches it through an SSH tunnel (`ssh -L`) for the
  round-trip test. Every server is stopped when its runs end.
- **Shared-machine rule**: before every server start the scripts read `nvidia-smi`, `who` and the
  load. A GPU is *idle* when no compute process is listed on it, utilisation is below 5% and less than
  1 GiB is in use. Only idle GPUs are used, pinned with `CUDA_VISIBLE_DEVICES`. If none is idle the
  run waits; if the machine stays busy, the run is not done and this is reported. Added before any
  contestant run, after another user was seen starting a debugging process on both GPUs every few
  minutes, and confirmed by the user ("do not influence the lab mate's work in any way"): a GPU is
  usable only when no process of anyone is on it (owner of every compute process checked); a GPU is
  never shared, so the fixed 90% memory setting stays. While waiting, one `nvidia-smi` query every 5
  minutes; a run starts only after three free polls in a row (at least 10 minutes free), or after one
  free poll when the previous session ran to its end without another process appearing. During a
  session a watchdog checks every 2 s; when another account's process appears on any GPU, the item in
  flight finishes, the server is stopped and the run goes back to waiting. The interrupted attempt is
  kept aside as contaminated and the whole session is repeated later. Every process gets 4 CPU threads
  (`OMP_NUM_THREADS` and similar); nothing stays loaded between sessions. The wait gives up after 6 h;
  the two-GPU judge is dropped if no two-GPU window comes within 2 h of waiting for it.

## What is copied to the workstation

Only what naming needs, made on the Mac by `vlm_prepare.py`: the stored item crops (tapped records'
`crops/`, scan entries' `inventory/`), their downsized copies, one marked keyframe per scan entry (at
two sizes), and the crops of each scan detection in saved keyframes. No events, positions, depth,
maps or embeddings. The judge requests also carry the candidate names, including the eight names the
user typed.

## Items

From one or more pulled folders (`--data DIR`), sessions left out as in Tracks A and B
(`bench_common.DEFAULT_EXCLUDED_SESSIONS`).

| Set | Definition | n (2026-11-17 pull) |
| --- | --- | --- |
| R | Tapped records with a stored crop | 20 |
| R-human | Records whose `register` event has `nameSource = typed` and a descriptive name (not `item N`, `itemN` or `object`) | 8 |
| S | Scan inventory entries | 24 |
| S-frame | Entries with at least one kept detection in a saved keyframe (keyframes with a person were not saved) | 18 |
| V | Every kept detection of an entry in a saved keyframe (for cross-view consistency only) | 54 |

The human reference is the typed name. One obvious typo is corrected before scoring: `suger` →
`sugar`. Two references are `fan`; they may be one fan registered twice and are counted as two items.

## Inputs (four variants)

| Variant | Images | Size |
| --- | --- | --- |
| `crop-full` (primary) | the stored crop | as stored (longer side capped at 960 px; affects 2 records) |
| `crop-320` | the stored crop | longer side at most 320 px |
| `ctx-full` | the stored crop, then the keyframe with the item's box drawn | crop as stored; keyframe as saved (540 × 960 upright) |
| `ctx-320` | the same two images | each at most 320 px on the longer side |

Only downscaling, never upscaling (Lanczos, then JPEG quality 90); an image that needs no resizing
is sent as the original file bytes. The keyframe of an entry is the saved keyframe whose detection of
that entry has the highest quality (box area × score, the app's rule for choosing an entry's crop).
The box is the detection box grown by 8% per side (the app's crop padding), drawn in red after
resizing (line width max(2, longer side / 240) px). `ctx` variants exist for S-frame only.

## Methods

### Vision-language models (contestants)

| Model (key) | Size class | Licence | Weights as served |
| --- | --- | --- | --- |
| Model A (`model_a`) | about 4 B | Apache-2.0 | 16-bit |
| Model B (`model_b`) | about 4–8 B | Apache-2.0 | 16-bit |
| Model C (`model_c`) | about 9 B | Apache-2.0 | 16-bit |
| Model D (`model_d`) | about 12 B | Apache-2.0 | 4-bit weights, 16-bit activations (the publisher's quantisation-aware checkpoint; 16-bit weights would not fit one 24 GB GPU) |

Two families at two sizes each, all current, ungated, from the publishers' own repositories, each at a
pinned revision. The models are named by these aliases only; their repositories and pinned revisions
are kept in a private file outside the repository (`analysis/vlm_models.private.json`, template
`analysis/vlm_models.example.json`), fixed at the date of this note. Every contestant runs on one GPU.

**Serving (identical for all)**: vLLM 0.33.0 OpenAI-compatible server, `--max-model-len 8192`,
`--max-num-seqs 16`, `--gpu-memory-utilization 0.90`, prefix caching off and the multimodal
preprocessing cache off (so repeats are real repeats and timings are not cache hits), at most two
images per prompt, `--seed 0`, CUDA graphs on (vLLM default). Requests: temperature 0 (greedy, no
sampling), `max_tokens` 200, seed 0, thinking disabled (`chat_template_kwargs: enable_thinking =
false`), structured output with the JSON schema below (vLLM `response_format: json_schema`).
No system prompt.

**Prompt, crop variants** (verbatim, from `vlm_common.py`):

> This photo comes from a phone app that remembers where things are in a home. It shows one object
> the app detected; it may be blurry or cut off at the edges.
> Name the object the way its owner would when asking "where is my ...?": a short everyday name of
> one to four words, specific enough to tell it apart from other things in a home (for example "TV
> remote", "reading glasses", "car keys"). Do not use brand names.
> Answer in JSON with these fields: name; category (one of: kitchenware, food or drink, electronics,
> clothing or accessory, bag or container, paper, book or stationery, toiletry or cleaning, toy,
> tool, furniture or fixture, decoration, other); attributes (up to three short visible attributes
> such as colour, material or shape); visible_text (readable text on the object, or an empty
> string); kind (one of: one item, part of an item, several items, furniture or fixture, background
> or unclear).

**Prompt, context variants**: the first paragraph is replaced by "The first image is a close-up of
one object detected by a phone app that remembers where things are in a home; it may be blurry or
cut off at the edges. The second image is the whole camera view at that moment, with the same
object inside a red box. Use the second image for context only." and "Name the object" becomes
"Name the object in the red box". The rest is identical.

**Answer schema**: `name` (string), `category` (enum above), `attributes` (array of strings, at most
3), `visible_text` (string), `kind` (enum above); all required, no other fields, in this order. The
method's name for an item is the `name` field of repeat 1; an unparsable or truncated answer counts
as no name.

### Baselines (no workstation)

| Key | Name of an item | Sets |
| --- | --- | --- |
| `phone` | S: the entry's label as stored by the app (EfficientDet-Lite0 on the phone, label votes). R: the app's EfficientDet-Lite0 `.tflite` run offline on the stored crop, label of the highest-scoring non-person detection at the app's raw threshold 0.20, else no name | R, S |
| `clip` | MobileCLIP2-S2 (the Track A weights) zero-shot over the 165 names of `bench_vocab_household.txt` (items and stop sections), text "a photo of a {name}.", top-1 on the stored crop | R, S |
| `yoloe11s`, `owlv2` | Track B open-vocabulary detectors (`analysis/out/bench_detect/detections.json`): the entry's kept phone detections in saved keyframes are matched to the detector's kept boxes in the same keyframe at IoU ≥ 0.5; the name with the largest summed score wins; no match = no name | S-frame |

The phone and CLIP baselines see the stored crop only, so they are compared with `crop-full`.

## Correctness

### Human reference (R-human, n = 8)

A name *matches* the typed name when, after normalisation, the words of the typed name, or of one of
its listed synonyms, appear as consecutive words in the name. Normalisation: lower case, any
non-letter becomes a space, the words a, an, the, my, your, some, this, that are dropped, each word
is made singular (simple English rules in `vlm_common.singular`). Synonyms, written before any model
output was read, list only other words for the same thing or a more specific kind of it:

| Typed (normalised) | Accepted |
| --- | --- |
| phone | phone, smartphone, cellphone, mobile, iphone |
| cup | cup, mug, teacup |
| chip | chip, crisp |
| holder, battery, fan, sugar | the word itself |

Reported per method and crop variant: matches out of 8, with a Wilson 95% interval.

### Model judges (all of R and S; every number marked "model-judged")

| Judge (key) | Size class | Licence | Weights | GPUs | Role |
| --- | --- | --- | --- | --- | --- |
| Judge A (`judge_a`) | about 14 B | Apache-2.0 | 8-bit as published | 1 | **primary**: a family that is not among the contestants |
| Judge B (`judge_b`) | about 27 B | Apache-2.0 | 8-bit as published | 2 (tensor parallel) | second judge: stronger, but the family of two contestants |

No judge is a contestant. Same serving settings as the contestants; `max_tokens` 400.

**Pool**: per item, the distinct normalised names proposed by every contestant (every variant and
repeat), every baseline, and, for R-human, the typed name. The judge sees no method name and does not
know which name was typed by the user. Names are numbered in an order shuffled per item (seed
20261117); a second pass in the reverse order measures the judge's order sensitivity. The verdicts of
the first order are the ones used.

**Judge's view**: records: the stored crop; entries in S-frame: the stored crop and the marked
keyframe (full size); other entries: the stored crop.

**Judge prompt** (verbatim, `vlm_common.JUDGE_PROMPT`):

> A phone app photographed an object in a home, and several systems proposed a name for it.
> {view}
> For each proposed name, judge whether it names the object correctly:
> - right: a correct name for this object that a person could use to find it;
> - close: related but not quite right (too general, a near miss, or it names a part, the contents or
>   the container of the object);
> - wrong: it names a different kind of object;
> - unclear: the image is too unclear to tell.
> Proposed names:
> n1: ...
> Answer in JSON with one verdict for each name number.

with {view} = "The image is a close-up of the object; it may be blurry or cut off at the edges." or
"The first image is a close-up of the object; it may be blurry or cut off at the edges. The second
image is the whole camera view, with the object inside a red box." Schema: one required enum field
(`right`, `close`, `wrong`, `unclear`) per name number.

**Scores**: *strict* = share of items whose name is judged right; *lenient* = right or close.
Unclear and no name count as not right. Headline: strict, primary judge. Every proportion with a
Wilson 95% interval; each method against `phone` on the same items with an exact McNemar test.

**Judge against the human reference**: on R-human, every pooled (item, name) pair is labelled by the
match rule above and by the judge; reported: the judge's verdicts on the typed names themselves, a
2 × 2 table (match / no match × judged right / not), agreement and Cohen's kappa, strict and lenient.
Also: agreement and kappa between the two judges over all pooled pairs, and each judge's agreement
between the two name orders.

**Ranking rule**: if the two judges order the contestants differently on the headline score, no
winner is claimed.

### Label-free signals

*Same name*: two names are the same when their normalised forms are equal or their head words are
equal (head = last word, after cutting the name at " of ", " for ", " with ", " in ", " on ").

- Agreement between methods: share of items with the same name, for every pair (primary variant).
- Repeat stability: share of (item, variant) with an identical normalised name in all three repeats.
- Batch stability: share of items whose name at concurrency 16 is the same string as at batch 1.
- Cross-view consistency: share of the 54 V crops (other sightings of an entry, cut from the half-size
  keyframes) whose name is the same as the model's name for the entry's stored crop.
- Input sensitivity: share of items with the same name in `crop-320` as in `crop-full`, and in
  `ctx-full` as in `crop-full`.
- The `kind` answers (how often a VLM says the crop is not one item).

## Profiling (per contestant)

All on the workstation unless stated; NVML from `nvidia-ml-py` for power, energy and memory.

- **Start-up**: seconds from launching the server to its health check answering; weight loading
  time and weight memory as logged by vLLM.
- **Idle power**: 15 s with the model loaded and no requests.
- **Sequential runs** (batch 1): per repeat (3), per variant (4), per item in a fixed shuffled order
  (seed 20261117), one streamed request at a time, after 3 warm-up requests. Per request: client
  latency (send to last byte), time to first token, decode time (first to last token), prompt and
  completion tokens, decode tokens per second, the vLLM server-side times from its `/metrics`
  histograms (queue, prefill, decode, end-to-end; read before and after each request), and GPU
  energy from the NVML energy counter over the request (total, and above idle power). The client-side
  JPEG resizing and base64 encoding is timed separately (preprocessing on the sender).
- **Power trace**: GPU power, utilisation and memory sampled every 100 ms for the whole run.
- **Peak GPU memory**: the largest memory in use on the GPU(s) during the run minus the use before
  the server started. vLLM reserves `0.90` of the GPU at start, so this mostly reflects that setting;
  the weight memory is reported next to it.
- **Batched throughput**: 88 requests (`crop-full`, two passes over R and S) at concurrency 4 and
  16; items per second, joules per item and median latency per level; concurrency 1 from the
  sequential runs.
- **Bytes per item**: JPEG bytes of the images, request body bytes (JSON with base64 images) and
  response bytes.
- **Round trip from the Mac**: through `ssh -L` to the server: 20 text-only pings (`max_tokens` 1)
  from the Mac and 20 from the workstation, then every R and S item in `crop-full` from the Mac, one at
  a time. Network and tunnel overhead = Mac latency − workstation latency for the same item. Because
  the naming and judging sessions run in a queue on the workstation that must not depend on the Mac
  (decided before any result; see below), the round trip runs in a separate short session per model
  (start-up, warm-up, idle window, pings, then the Mac's requests) whenever the Mac is connected and a
  GPU is free; the workstation latency of the same item comes from that model's naming session.
- **Phone, from the log** (same pull, same sessions): per scan keyframe `detectMs`, `convertMs`,
  `embedMs`, `totalMs`; per registration `latency`; per Verify `latency.totalMs`.

## Outputs

- `analysis/notes/2026-11-17_vlm_naming_results.md`: numbers only, n beside every number,
  human-referenced and model-judged in separate columns.
- `analysis/out/vlm/quality_cost.png`: primary-judge strict score on S-frame (n = 18, every method and
  variant defined there) against seconds per item (batch 1) and against joules per item.
- Per-request records and traces under `analysis/out/vlm/` (git-ignored; absolute workstation paths
  are replaced by placeholders when results are pulled).

## Contingencies, fixed in advance

- A model that does not start with the settings above is retried with, in order, `--enforce-eager`,
  `--gpu-memory-utilization 0.95`, `--max-model-len 4096`. Each retry is listed. A model that still
  fails is dropped and named.
- If structured output is refused for a model, it is prompted without a schema and the JSON is parsed
  from the text; listed.
- If two idle GPUs are never available, Judge B is not run and only the primary judge judges.
- If the workstation becomes busy or unreachable, the work stops and is reported.

## How the runs are scheduled (fixed before any result)

The free GPU window may come at night, so the sessions run in a queue on the workstation itself
(`vlm_remote.py queue`, started by `vlm_ws.py queue-start`): single instance (a lock file), log in
`runs/queue.log`, a six-line `runs/STATUS` for a person to read, 4 CPU threads per process, no GPU
memory while waiting (it only queries NVML), and it exits by itself when every session is done or
after 24 hours (`vlm_ws.py queue-stop` stops it earlier). Order: Model A, Model B,
Model D, Model C, then the judging pool is built on the workstation from the finished
sessions plus the baseline and typed names (`pool_inputs.json`, names only, unlabelled), then
Judge A, then Judge B (its weights are downloaded only when both GPUs are free, after
deleting weights of finished models to stay within the disk budget). Results are mirrored to the Mac
with `vlm_ws.py pull`; the report uses each model's latest session that ended normally.

## Deviation, 2026-11-17 about 23:15: shared-GPU mode (decided with the user before any model result)

The other account's single process (memory on both GPUs, about 6.8 GB on GPU 0 and 8.9 GB on GPU 1)
had been idle for hours (sampled by the planner at 1 Hz for 10 minutes: 0% utilisation apart from one
1% reading, memory unchanged). The user chose, knowing the risk that their program may resume and
want more memory, to let the models that fit share ONE GPU with it. Nothing of theirs is ever
stopped, signalled or touched. The rule "a GPU is never shared" above is replaced, for these models
only, by:

- **Which GPU**: the one with the most unreserved memory; the other GPU is not touched.
- **Cap**: our process may use at most (unreserved memory before the start − 5 GiB), so that at
  least 5 GiB stays unreserved with our model at its peak. A model is allowed only if its measured
  footprint fits with margin. Model A: 9,297 MiB after weights and context in the smoke run, plus
  a fixed 1 GiB KV cache (`--kv-cache-memory`), plus activations with `--max-num-batched-tokens 4096`
  and the multimodal memory profiled with two 960 × 960 images (`--limit-mm-per-prompt` with sizes);
  planned peak about 11,300 MiB against a cap of about 12,240 MiB. The model's own preprocessing is
  unchanged (every input is at most 960 px), so the answers should not depend on these settings;
  the smoke run's extra 6.4 GiB came from profiling a 16,384-token dummy image. Model D
  (about 10 GB of packed weights plus context, KV cache and activations) does not fit with margin and,
  like the larger models and both judges, waits for a free GPU.
- **Start**: only after the other account's processes look idle for 120 s (1 Hz: memory unchanged,
  no NVML per-process utilisation sample, no new process); what was seen is recorded as the baseline
  (`idle_check.csv`, `status.json` → `shared`). GPU power is measured for 15 s right before loading as
  the energy baseline.
- **Watchdog** (`watchdog.csv`): every 0.1 s for the first 5 minutes (start-up), then every second.
  It stops our server at once (SIGTERM, SIGKILL after 5 s) at the first sign that the other program is
  active: its memory changes on any GPU, a new process of another account appears, its per-process
  utilisation is above 0, or unreserved memory on our GPU falls below 4,864 MiB. The session is then
  marked interrupted, kept aside (`*_shared_contaminated_*`), the GPU is released, and no shared
  attempt is made until the other process has looked idle for 10 minutes (1 Hz). When only the
  memory floor is crossed and the other process is unchanged, our own footprint was too large: the
  model is marked as not fitting and goes back to the normal queue.
- **Labels**: shared sessions are named as such in `queue.log`, `runs/STATUS` and the report (GPU
  shared with an idle process holding X MiB, cap, watchdog samples). Their timing counts only if the
  watchdog saw no activity for the whole session.
- **Judges**: neither judge fits under the cap; they wait for a free GPU. Until a judge has run, the
  results note gives only the human-reference numbers, agreement between methods, the label-free
  signals and the baselines, and says that the model-judged numbers are pending.
