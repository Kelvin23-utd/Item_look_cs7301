# Workstation vision-language model for naming: experiment plan

Status: plan, 2026-11-17. High-level; the implementer owns the detailed design and records it.

## Question

The phone's detector knows 80 names and named about half of the scanned items wrongly
(`analysis/notes/2026-11-16_first_scan_walk.md`). Can a 3B–12B vision-language model on a lab
workstation give better names, at what cost, and is that cost acceptable for a design in which the
phone does the real-time work and the workstation names items in the background?

## Setup

- **Workstation:** a shared lab machine reached over SSH. Its address and account are private:
  they are never written into this repository. Scripts read the SSH alias from the environment
  variable `FB_WORKSTATION`.
- **Communication:** files by `rsync` over SSH; batch runs by SSH; for the timing of a live design,
  a model server on the workstation reached through an SSH tunnel, with the request measured end
  to end from the Mac.
- **Data sent:** only what naming needs from the pulled phone data: item crops, the saved scan
  keyframes (frames with a detected person were never saved by the phone) and the tapped items'
  crops. Kept in one private folder on the workstation, removable with one command.

## Experiment

1. **Models:** three or four open-weight models spread across about 3B to 12B parameters that run
   on one 24 GB GPU, chosen from what is current and checked for licence; plus the cheap baselines
   already on hand: the phone detector's label, the open-vocabulary detectors' names from Track B,
   and CLIP zero-shot naming over the fixed household vocabulary.
2. **Inputs compared:** the item crop alone, and the crop together with the whole frame with the
   item marked, at two image sizes.
3. **Output:** a fixed prompt and a fixed structured answer (name, category, a few attributes,
   visible text), no sampling, repeated runs to measure how stable the name is.
4. **Correctness without asking the user to label:**
   - names the user typed for tapped items are the human reference (small, but real);
   - for everything else, a judging model that is not one of the contestants, with its agreement
     against the human reference reported, and every such number marked "model-judged";
   - agreement between models, as a label-free signal.
5. **Profiling, per model and input:** seconds per item (load, preprocessing, generation), tokens
   per second, peak GPU memory, GPU power sampled during the run and energy per item, throughput
   when batched, bytes sent per item and round-trip time from the Mac. Compare with the phone's
   on-device times from the log.
6. **Result:** a quality-against-cost table and plot (naming quality against seconds and joules per
   item), a recommended model and input for the design, and what remains unproven.

Stretch, only if the above is complete: the same model as a second opinion on suspected duplicate
entries and on the logged Verify captures.

## Rules

- A shared machine: check what is running first, use only idle GPUs, never stop or disturb other
  users' processes, no `sudo`, and stay within a stated disk budget (the home disk is nearly full).
- No invented numbers; every figure comes from a logged run with its sample size. Prompts, models,
  inputs and metrics are fixed before results are read; later changes are listed.
- No photos in git. No address, account name or host alias in git.
