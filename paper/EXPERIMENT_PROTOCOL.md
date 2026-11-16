# FindBack experiment protocol (Nov 16–23, presentation Nov 25)

For the people running the experiments on the Pixel 8. What each experiment proves is in `FindBack/docs/STUDY_SPEC_v0.4.md`.

## Before you start

- **Study settings.** Tap the top status panel → "Study settings…". Type the tag exactly as written below and pick the capture mode. Both are saved in every log line, and the analysis ignores sessions with no tag.
- **Move the phone a little before tapping an item.** A still phone reads depth wrongly ("Depth looks wrong").
- **Locate at the start of every session.** Hold the phone at the home spot → status panel → "Use a spot" → line up the faint picture → Lock.
- **After each day,** plug the phone into the Mac and tell Claude. Claude pulls the data, checks it and reports problems while you can still redo them.

## Schedule

| Day | Do | Time |
| --- | --- | --- |
| Mon 16 | E0: finish the walk test | 15 min |
| Tue 17 | New map, register 20 items in 2–3 rooms, E1 | 75 min |
| Wed 18 | E2 for all items | 60 min |
| Thu 19 | E3 rounds 1–2 | 60 min |
| Fri 20 | E3 rounds 3–4; E4 with Claude over USB | 75 min |
| Sat 21 | Second home if possible (E1 + two E3 rounds, both tagged `home=B;...;split=test`), or two more E3 rounds at another time of day | 90 min |
| Sun 22 – Mon 23 | Analysis, figures, slides and paper; redo anything that failed | |

## E0. Walk test (today, on the map you already made)

Tag `home=A;exp=E0;split=dev`, mode FREE.

1. Walk into a second room. Tap the "Room" chip → "New room…" and name it. Save one or two items there.
2. Swipe the app away and reopen it. Go to the home spot and locate.
3. For each of your items: Items → the item → Find. Follow the arrow until the ring appears. Done → "At the marker" → tap the item's true centre.

Check: the card names the right room, the arrow follows your walked path, and each "Marker error" toast is a sensible number.

## Set up the study map (Tuesday)

1. Status panel → "New map…". This archives the pilot items; nothing is deleted.
2. Set home with a spot. Name each room as you enter it and walk through all rooms once.
3. Save 20 items, 6–8 per room. Choose things you can pick up and put back in exactly the same place. Include at least five pairs that look alike (two mugs, two remotes, two chargers, two books, two bottles).
4. In the save dialog, answer "How often does it move?" honestly and check the room.

## E1. Position accuracy across rooms and restarts

Mode FREE. The tag changes between steps, so set it each time.

First, in the same session in which you saved the items, with tag `home=A;exp=E1;split=dev`: for every item, Items → Find → walk to the ring → Done → "At the marker" → tap the true centre. This is the baseline before any restart.

Then repeat three times, with tag `home=A;exp=E1;split=dev` for the first restart and `home=A;exp=E1;split=test` for the second and third:

1. Swipe the app away, reopen it, go to the home spot and locate.
2. For every item: Items → Find → walk to the ring → Done → "At the marker" → tap the true centre.

Then, standing at the home spot: "Use a spot" → Lock, ten times. Take the phone away and bring it back between locks. This measures how repeatable a lock is.

## E2. Verification benchmark

Tag `home=A;exp=E2;split=dev` for the first 10 items and `home=A;exp=E2;split=test` for the rest. Mode BENCHMARK.

For each item: Items → Find → walk to the ring. Each time you press Verify and close the result, the app asks "What is at the marker right now?". Set the Condition first, then tap the label.

| Capture | What you do | Label | Condition |
| --- | --- | --- | --- |
| 1 | Item in place, normal view | The item | default |
| 2 | Step back about 1 m | The item | far |
| 3 | Move in close | The item | near |
| 4 | View from the side | The item | angled |
| 5 | Lights off or curtains closed | The item | dim |
| 6, 7 | Take the item away; capture from the front and from the side | Nothing | default, angled |
| 8, 9 | Put a clearly different object on the spot | A different object | default, angled |
| 10–13 | Put its look-alike on the spot, if it has one; capture from the front, the side, close and far | A look-alike | default, angled, near, far |

Then put the item back, press Done and choose "Skip (not usable)".

## E3. Controlled paired trials

Tags `home=A;exp=E3;round=1;split=dev` and `home=A;exp=E3;round=2;split=dev` for rounds 1 and 2; `round=3` and `round=4` with `split=test` for the others. Mode PAIRED.

For each round:

1. Ask Claude for the trial sheet. It lists, in random order, what to do with each item: leave it, remove it, move it nearby, or put another object in its place. Some rows say "walk a long loop first".
2. Apply every change on the sheet. Do not open the app while doing this.
3. Swipe the app away, reopen it, set the tag, go to the home spot and locate.
4. Query the items in the sheet's order: Items → Find → follow the guidance. At the ring, press Verify once and close the result. Done → choose what is really true:
   - "At the marker" → tap the item's true centre.
   - "Nearby, but not at the marker" → tap the item where it is now.
   - "Not in this area".
   - "Another object is at the marker".
5. Put every item back exactly where it was saved. The app's memory is not updated in this experiment, so the next round needs the original layout.

Answer the ground truth from what you see, not from what the app said.

## E4. Cost

Claude runs this over USB: three 5-minute recordings of the phone's power rails with tracking only, and three with a verification every 20 seconds. You only need to prop the phone so it sees one saved item, with the app located.

## Optional: five minutes each evening (E5)

This is the only part where things move for real reasons, so even a small amount is valuable.

1. On the study map, also save 6–8 things you use every day (charger, keys, remote, mug, glasses, wallet) where they usually live. Answer "How often does it move?" honestly.
2. Each evening: tag `home=A;exp=E5;split=test`, mode PAIRED. Locate, then query each of these items: Find → Verify once at the ring → Done → the true answer.
3. If an item was somewhere else, find it, then Items → the item → "Update location" and tap it where it is now.

## What changed from the first plan

The full 5-day "live with it" run is not in this schedule: it does not fit before Nov 25, and a query on every item twice a day is not how a phone is used. The short evening run above replaces it. The paper will say that scripted trials show what each policy does when memory is stale, not how often memory is stale in daily life.
