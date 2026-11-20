# Find by name: what was built and what the saved naming outputs can be joined to

Date: 2026-11-19. App v0.9 (commit `a208a53`), harness `analysis/name_search.py` (commit `9fbcec5`). The rule both use is in `FindBack/docs/PLAN_v0.9_find_by_name.md`; both are tested against the same 21 cases.

## Status

| Part | State |
| --- | --- |
| Search by name on the phone (ranked results with photos, "Last seen", item sheet, Find) | built, 210 unit tests pass, installed; **not yet checked on the device screen** (the phone was locked) |
| Extra names on the phone (`names.json`, bound to the exact photo by SHA-256, on/off switch) | built and unit-tested; no names file exists for the phone's current inventory (see below) |
| Join of the saved naming outputs to a scan | checked, table below |
| Comparison on owner-written queries | **not run**: the queries do not exist yet; the sheet for writing them is generated |

## The join

The saved outputs of the four naming models belong to the first scan walk (24 scan entries, one private pull). Each entry's output was joined by entry id and by the SHA-256 of the photo bytes the model was sent against the photo stored in the pull.

| Model | With output | Parse OK | Photo hash equal | Model says "one item" | Usable |
| --- | --- | --- | --- | --- | --- |
| Model A | 24 | 24 | 24 | 23 | 23 of 24 |
| Model B | 24 | 24 | 24 | 23 | 23 of 24 |
| Model C | 24 | 24 | 24 | 22 | 22 of 24 |
| Model D | 24 | 24 | 24 | 22 | 22 of 24 |

- Every loss is an entry the model itself did not call one item (furniture or fixture, several items, part of an item). No output was lost to a failed join.
- Of the usable entries, 18, 17, 19 and 19 have a model name that differs from the phone's own name; the others add only descriptors.
- Each output also carries up to three short attributes (used as descriptors: 69, 59, 66 and 66 in total), a category from a fixed list and any visible text (non-empty for 5, 4, 3 and 3 entries). Category and visible text are not imported.
- Tapped records: 18 of 20 join by hash; 2 were named from resized copies and cannot be tied to the stored photo.

**None of these 24 entries is in the phone's inventory today** (0 of 24; the home map was replaced on 2026-11-19 and the inventory with it, now 82 entries). So the comparison runs offline on the saved scan, and extra names on the phone need a new naming pass over the phone's current photos.

## Comparison protocol (not run yet)

1. The owner fills in the private query sheet: the 24 photos of the scan with codes and no names, labels or model output; per item an optional general word and an optional specific name; three queries for things not in the scan.
2. `name_search.py freeze` records the SHA-256 of the query file. `run` refuses an unfrozen file.
3. `run` scores the same queries on the same 24 entries with phone labels alone and with each model's names file: results shown, the target's rank and match class; results that appear only through extra names and are not the target are listed for the owner to judge.

No ranking on owner queries has been looked at by anyone before the freeze.

## Naming pass over the phone's current inventory (later the same evening)

The existing pipeline ran once more, on the phone's current scan entries only (82 photos, each sent as its original bytes, primary prompt and decoding unchanged, one pass). All four models ran on the university cluster (one GPU each, about 3 minutes per job including server start); the outputs live in a separate output root and the earlier experiment's files are unchanged.

| Model | With output | Photo hash equal | Model says "one item" (usable) | Name differs from the phone's |
| --- | --- | --- | --- | --- |
| Model A | 82 | 82 | 79 | 73 |
| Model B | 82 | 82 | 75 | 70 |
| Model C | 82 | 82 | 76 | 74 |
| Model D | 82 | 82 | 76 | 72 |

Model C's file was pushed to the phone as `names.json`; all 76 of its items still matched the photos on the phone at that time. Whether the names are right has not been judged: the phone's 14 names (26 "bowl", 18 "cup", ...) become mostly distinct specific names, and a first read shows both plausible corrections and at least one name given to three different entries under different phone labels.

Not yet checked on the device screen.

## First look with names the owner had typed earlier (not the frozen comparison)

On the first walk the owner typed a name for 9 tapped items when saving them. Used as search queries over that walk's 44 items (24 scan entries and 20 tapped items, the tapped items carrying only the phone's own label, not the typed name), with the same matching rule:

| | Phone labels alone | Model A | Model B | Model C | Model D |
| --- | --- | --- | --- | --- | --- |
| Target found (of 9) | 0 | 3 | 3 | 4 | 3 |

- Found with every model: two fans and a cup (the cup last of 8 results). Found by Model C only: one snack whose typed name the other models' names did not contain.
- Not found: three typed words the models name differently, one query with a filler word in front of a word the models do use, one misspelt word on a photo most models did not call one item, and one placeholder name.
- 16 of the 44 items have no phone label at all (tapped items the phone's classifier left unnamed).

Limits: n = 9, one home; the names were typed when saving, not when searching; one is a placeholder; this set was looked at before any rule was frozen, so it is a development set. Ideas taken from it (ignoring filler words, tolerating a typo, saving a failed search word as a name for the item then picked) need a fresh test. The query sheet for the 24 scan entries has not been filled in and its comparison has not been run.
