# v0.9: find by name

Goal: after a scan, the user types what they call an item, recognises it by its saved photo among the results, taps it and starts the existing guidance to its last recorded location.

This file is the contract between the app and the offline harness (`analysis/name_search.py`). Both implement the matching rule below and both are tested against the same cases, `FindBack/app/src/test/resources/name_search_cases.json`.

## What already exists (v0.6 to v0.8)

The Items screen lists tapped records and scan entries with photo, name and place, filters them by a text box (substring over name, labels, container, room), and opens an item sheet with a Find button that starts guidance for that exact record or entry. v0.9 changes the order of results, the wording, and adds extra names; it does not change guidance.

## Matching rule

**Normalise** a string: lower case (root locale); every character that is not a letter or digit becomes a space; runs of spaces collapse; trim. **Words** are the pieces between spaces.

**An item's fields**

| Pool | Fields |
| --- | --- |
| naming | the item's name; its extra names (only when extra names are on and bound, see below) |
| other | every detector label of the item (the label and all label-vote keys; a tapped record's raw label); its descriptors (extra names on and bound only); its container; its room name |

**Match class** of an item for a query with at least one word (lower is better):

| Class | Condition |
| --- | --- |
| 0 | the normalised query equals the normalised name or one normalised extra name |
| 1 | every query word is a prefix of some word of the naming pool |
| 2 | every query word is a prefix of some word of the naming or other pool |
| 3 | every query word is a prefix of some word of either pool, or has at least 3 characters and is contained in some word of either pool |
| none | otherwise: the item is not a result |

**Order of results:** class ascending, then last seen descending, then id ascending. An empty query lists every item by last seen descending (as before).

**Via extra:** a result is "via extra" when it is a result with extra names on and not a result with them off. The row says which extra name or descriptor matched.

## Extra names

The phone's own name and labels are never changed. Extra names come from a file, `names.json` in the app's files folder:

```json
{
  "version": 1,
  "source": "model_c:2026-11-18",
  "createdMs": 1795000000000,
  "items": [
    {"kind": "entry", "id": "e98ae991", "cropSha256": "<64 hex>", "names": ["drink can"], "descriptors": ["red"]}
  ]
}
```

- `kind` is `entry` or `record`. `source` is an alias string; it never holds a model's real name.
- **Binding:** an item's extra names are used only while the SHA-256 of the bytes of its current photo file equals `cropSha256`. A scan that replaces the photo unbinds the names; nothing is guessed.
- A settings switch "Extra names" (default on) turns them off so the same query can be tried both ways.
- A missing or unreadable file means no extra names; the app says nothing and logs the reason once.

## Items screen

- A scan entry the user has not renamed shows its first bound extra name as the row and sheet title while extra names are on, followed by the phone's own name in small grey text; the stored name is unchanged.
- Results in the order above. Several items with the same name stay separate rows, each with its own photo.
- The second line of a row starts with "Last seen: " followed by the place and the time of the last sighting; the existing notes ("looked for in the last scan, not seen", "not checked since ...") follow it. The item sheet uses the same wording.
- A "Last scan" chip limits the list to scan entries seen in the most recent finished scan, so one scan can be used as the inventory without deleting anything. It is one of the single-select chips (All, Last scan, one per room) and hides tapped records.
- No result: the existing empty text names the query.

## Logging

The query text is never logged (unchanged rule). `ui_search` gains whether extra names were on, how many items are bound, how many shown rows are via extra, and whether the "Last scan" chip is selected. Opening a row logs its rank in the shown list, the number shown, the query length, its match class and whether it is via extra. `session_start` logs the names file's source, item count and bound count, and the reason when the file is not used. The file is read once at app start.
