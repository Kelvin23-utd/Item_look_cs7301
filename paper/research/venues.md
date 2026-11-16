# Venues for a 6-page workshop paper, and the journal option

Compiled 2026-11-16. Target window: submission between 2027-02-16 and 2027-08-16 (3 to 9 months out).

## Read this first

- **No 2027 workshop call is out yet** for MobiCom, MobiSys, CPS-IoT Week (SenSys) or UbiComp/ISWC. Every workshop deadline below is the **2026 edition's** date. Where I give an expected 2027 date it is a projection from 2026 and is labelled as one.
- **Every fact has the URL of the page it came from.** Dates shown as "A, extended to B" appeared on the page as a struck-through date followed by a new one.
- **Three sources could not be opened** and are flagged where they matter: the ACM policy pages on acm.org (HTTP 403 to automated fetches), the IMWUT pages on dl.acm.org (HTTP 403), and the UbiComp/ISWC 2027 site (HTTP 500). Facts that depend on them come from other pages that quote them, and say so.

## Ranked shortlist

| Rank | Venue | 2026 deadline (2027 not announced) | Expected 2027 deadline (projection) | Length | Review | ACM DL |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | ImmerCom, workshop at MobiCom 2027 (Athens, Oct 18-22, 2027) | June 5, extended to June 26, 2026 | June 2027 | 6 pages plus unlimited references | Single-blind | Yes |
| 2 | HumanSys, co-located with MobiCom in 2026 | July 15, 2026 | July 2027 if it stays with MobiCom | 6 pages including references | Single-blind | Yes |
| 3 | A UbiComp/ISWC 2027 workshop (Seoul, Oct 3-7, 2027): EverydayXR or PEACE if they run again | July 17, 2026 (EverydayXR); July 12, 2026 (PEACE) | Late June to mid July 2027 | 6 pages excluding references | Single-blind | Adjunct proceedings |
| Journal | IMWUT | Feb 1, May 1, Nov 1 each year | Feb 1, 2027 or May 1, 2027 | No limit found; median article is about 30 pages | Anonymized | Yes |

### Fit, and what each needs that an author-evaluated, two-home system paper does not yet have

1. **ImmerCom.** Fit: a mobile XR systems workshop whose 2026 program included persistent scene understanding, anchoring in mobile AR, and a resource policy for lifelogging; a wrong-claims versus captures-per-query trade-off is its kind of result.
   Needs: on-device cost numbers that make the trade-off a systems result (latency and energy of one verification capture, and the cost of always-verify), and localization error measured against an external reference, so that drift is measured and not inferred. The paper's cost and marker-error tables in `paper/generated/` are set up for exactly this but hold no data yet (every cell is TBD as of 2026-11-16). Beyond what is planned, it lacks a baseline the authors did not design, such as an age-only threshold or a Cloud Anchor relocalization, and a second phone model. No participant study is required by the call.
2. **HumanSys.** Fit: human-centered sensing and systems; its topics name camera, AR and mobile platforms, accessibility, and "everyday environments".
   Needs: its call asks for "Real-world deployment experience with human participants" and "Evaluation methodologies and insights of cyber-physical systems with humans in the loop". Two authors in their own homes will read as a pilot. It needs participants who are not authors, a stated ethics approval, and the related work cut to fit six pages including references.
3. **EverydayXR or PEACE at UbiComp/ISWC.** Fit: EverydayXR asked how XR "interaction paradigms fail, degrade, or adapt" under everyday conditions; PEACE listed "When-to-act policies" and "uncertainty-aware perception". Stale object locations and drift are an everyday-conditions failure, and selective verification is a when-to-act policy.
   Needs: a human-facing account (what the user saw, did, and trusted) alongside the policy comparison, and non-author participants or a clearly framed position paper. Both were first editions in 2026, so their return is not assured.

**Journal, IMWUT.** A different paper: roughly five times the length, anonymized, and in this topic area published work reports studies with outside participants (see the IMWUT section). It needs a multi-household study with non-author participants over days or weeks, statistics on human outcomes, ablations of each signal, and a second device.

**Outside the window but the best match for the thesis: HotMobile 2028.** HotMobile asks for papers that "debunk existing assumptions" and says novel ideas "need not be supported by full evaluations". The 2027 deadline was October 9, 2026 and has passed; the 2028 edition is not announced and, on the 2027 pattern, would be due around October 2027.

## Host conferences in 2027

| Conference | Dates and place | Source |
| --- | --- | --- |
| MobiCom 2027 | Oct 18-22, 2027, Athens, Greece | <https://www.sigmobile.org/mobicom/2027/> |
| MobiSys 2027 | June 21-25, 2027, Ho Chi Minh City, Vietnam | <https://www.sigmobile.org/mobisys/2027/> |
| SenSys 2027 (in CPS-IoT Week 2027) | May 17-20, 2027, Boulder, CO, USA (Limelight Boulder) | <https://sensys.acm.org/2027/>, <https://cps-iot-week-2027.sigbed.org/> |
| UbiComp/ISWC 2027 | Oct 3-7, 2027, Seoul Dragon City, Seoul, Republic of Korea | <https://sigchi.org/events/ubicomp-2027/> |
| HotMobile 2027 | Feb 24-25, 2027, Tucson, AZ, USA | <https://hotmobile.org/2027/> |

## MobiCom 2027 workshops

**Status of the 2027 call.** The MobiCom 2027 site says the conference "will include a set of workshops", but its workshop call page returns 404: <https://www.sigmobile.org/mobicom/2027/workshop_cfp.html>. Main-track winter deadlines are listed as TBD.

**2026 schedule for all MobiCom workshops** (<https://sigmobile.org/mobicom/2026/workshop_cfp.html>): proposals March 20, extended to March 23, 2026; workshop paper submissions June 5, 2026; notification July 31; camera-ready August 30; workshops October 30, 2026 in Austin. The 2026 list of workshops is at <https://sigmobile.org/mobicom/2026/workshop_program.html>.

### ImmerCom: ACM Workshop on Mobile Immersive Computing, Networking, and Systems (4th edition in 2026)

| Item | 2026 fact | Source |
| --- | --- | --- |
| Call for papers | <https://immercom.github.io/call.html> | |
| Submission deadline | June 5, extended to June 26, 2026 (AoE) | <https://immercom.github.io/> |
| Notification | July 31, extended to August 15, 2026 | same |
| Camera-ready | August 21, extended to August 30, 2026 | same |
| Workshop date | October 30, 2026, Austin | same |
| Page limit and template | "at most six (6) pages in length, including all figures and tables in two-column 10pt ACM format"; "as many additional pages as needed for references and for supplementary material in appendices"; the page notes the ACM template default is 9pt and must be changed to 10pt | <https://immercom.github.io/submission.html> |
| Blind | "Papers must include author names and affiliations for single-blind peer reviewing by the PC." | same |
| ACM DL | "accepted ImmerCom 2026 papers will be published in the ACM Digital Library as part of the ACM MobiCom 2026 proceedings" | same |
| Submission system | HotCRP, <https://immercom26.hotcrp.com/> | same |

Relevant topics in the 2026 call include "Efficient visual sensing for augmentation and visualization on resource-constrained smartglasses", "Multi-user spatial synchronization and 6DoF pose prediction in AR/MR", "Novel applications of AR/VR/MR", and "Security, privacy, and trust in AR/VR/MR".

Evidence of fit from the 2026 program (<https://immercom.github.io/program.html>): "Towards Persistent Dynamic Scene Understanding for Immersive Computing", "An Empirical Study of Depth Inference Scheduling for Dynamic Anchoring in Mobile AR", and "Resource-Constrained Lifelogging: A Utility-Based Replacement Policy for Video Understanding". All three are in Crossref as 8-page papers in the workshop proceedings (keys `do2026persistent`, `huang2026empirical`, `chang2026lifelogging` in `refs.bib`).

2027: not announced. The workshop has run each year from 2023 to 2026 (past-edition links on its home page).

### HumanSys: International Workshop on Human-Centered Sensing, Modeling, and Intelligent Systems (4th edition in 2026)

| Item | 2026 fact |
| --- | --- |
| Call for papers | <https://humansys-workshop.github.io/2026/> |
| Co-location | "Co-located with ACM MobiCom 2026" |
| Submission deadline | July 15, 2026 AoE, marked firm (an earlier July 8 date is shown before it) |
| Notification, camera-ready, workshop | August 15, August 30, October 30, 2026 |
| Page limit | Full papers "up to 6 pages including references"; experience papers up to 4; extended abstracts up to 2 |
| Template | ACM `acmart` with the `sigconf` option, 10pt, two-column |
| Blind | "HumanSys'26 follows a single-blind review process." |
| ACM DL | "All of the accepted papers (regardless of category) will be included in the ACM Digital Library." |
| Submission system | HotCRP, <https://humansys26.hotcrp.com/> |

2027: not announced. Its host may change: a leftover template list on the CPS-IoT Week 2027 events page names a "3rd" HumanSys, which suggests an earlier edition ran at CPS-IoT Week (<https://cps-iot-week-2027.sigbed.org/events.php>). If it returns there, the deadline would fall around February or March 2027 instead of July.

### Other MobiCom 2026 items

- **XRSecurity** (Second Workshop on Enhancing Security, Privacy, and Trust in Extended Reality (XR) Systems, <https://xrsecurity.github.io/2026/>): paper deadline June 19, 2026; notification July 31; camera-ready August 30; workshop October 30, 2026. "A full paper should NOT exceed 6 pages (US letter size) double column including figures, tables, and references in standard ACM format"; posters and demos 2 pages; HotCRP. The page does not state the blind policy or the proceedings. Its 2026 program included a paper on latency, performance and trust in mixed reality. Fit depends on framing wrong location claims as a trust problem; it is a fourth option if the paper leans that way.
- **MobiCom Work-in-Progress papers** (<https://sigmobile.org/mobicom/2026/wip_cfp.html>): "at most 3 pages, including all figures and references"; deadline June 26, extended twice to July 10, 2026; published "in the main proceedings and the ACM digital library". The 2027 WiP page returns 404.

## MobiSys 2027 workshops

**Status of the 2027 call.** "The proposal submission deadline will be announced." No workshops are listed. <https://www.sigmobile.org/mobisys/2027/call_for_workshops/>

**2026 schedule** (<https://www.sigmobile.org/mobisys/2026/call_for_workshops/>): proposals December 12, 2025; notification December 19, 2025; workshops June 25, 2026 in Cambridge, UK. "Workshop papers will be included with the ACM MobiSys proceedings and published in the ACM Digital Library."

**2026 workshops** (<https://www.sigmobile.org/mobisys/2026/accepted_workshops/>): EMPOWER, EMDL, EIFCom, HeadSys, AutoSys, AutoEdge, Excellence in Edge AI, EdgeSys, NetAISys, EnvSys. None is about phone AR or spatial memory. The nearest is HeadSys.

### HeadSys: Workshop on Head-Based Intelligent Systems (1st edition in 2026)

| Item | 2026 fact |
| --- | --- |
| Call for papers | <https://www.eventcreate.com/e/headsys26> |
| Submission deadline | April 2, extended to April 19, 2026 (AoE) |
| Notification, camera-ready, workshop | April 28, April 30, June 25, 2026 |
| Page limit | "Full papers must be no longer than 6 pages, position papers are limited to 4 pages and abstracts are limited to 2 pages." |
| Template | MobiSys 2026 formatting |
| Blind | "HeadSys follows a single-blinded review process" |
| ACM DL | Not stated on the workshop page; the MobiSys 2026 workshop call says workshop papers go to the ACM DL |

Fit is weak: its topics are head-worn devices (smart glasses, earables, AR/VR/XR headsets). A phone system would have to be argued as a stand-in for glasses.

**Projection for 2027.** MobiSys 2027 has the same June 21-25 dates as 2026, so workshop paper deadlines around April 2027 are likely.

**MobiSys 2027 posters** (<https://www.sigmobile.org/mobisys/2027/call_for_posters/>): "no longer than two 8.5x11-inch pages, including all figures and references"; "Submissions need not to be anonymized"; accepted abstracts are "included in the conference proceedings and the ACM Digital Library". No deadline is posted yet.

## SenSys 2027 and CPS-IoT Week 2027 workshops

**Status of the 2027 call.** Workshop proposals are due November 20, 2026, with notification November 27, 2026, so the list of workshops does not exist yet. The events page says "Please check back later for the list of workshops." <https://cps-iot-week-2027.sigbed.org/calls.php>, <https://cps-iot-week-2027.sigbed.org/events.php>

**SenSys 2027 main track.** Both deadlines have passed (June 5, 2026 and November 5, 2026). For reference, it accepts short papers of up to 6 pages ("visions, experiences, etc., only") and requires full anonymization. <https://sensys.acm.org/2027/cfp.html>

**2026 workshops at CPS-IoT Week** (<https://cps-iot-week2026.inria.fr/events.php>): ENSsys, DESTION, CAADCPS, RAGE, FMSys, TADAM, FACCT. None fits this paper. Two data points for timing and format:

| Workshop | 2026 deadline | Format | Source |
| --- | --- | --- | --- |
| FMSys (Foundation Models for CPS and IoT, 3rd) | March 1, extended to March 7, 2026; notification March 15; workshop May 11, 2026 | Six pages including figures and references; ACM `sigconf`, 9pt; single-blind | <https://fmsys-org.github.io/2026/> |
| RAGE (Real-time And intelliGent Edge computing, 5th) | February 3, extended to February 10, 2026; notification March 2; workshop May 11, 2026 | Not stated on the page | <https://rage-workshop.github.io/2026/> |

**Projection for 2027.** CPS-IoT Week 2027 is six days later than in 2026 (May 11-14, 2026, per the SenSys 2026 site), so workshop paper deadlines between mid February and mid March 2027 are likely. Check the events page from December 2026. A fit is possible only if a human-centered sensing workshop such as HumanSys is hosted there.

## UbiComp/ISWC 2027 workshops

**Status of the 2027 call.** Dates and venue are on the SIGCHI event page. The conference's own 2027 site returned HTTP 500 (<https://www.ubicomp.org/ubicomp-iswc-2027/>), and I found no 2027 call of any kind.

**2026 schedule** (<https://www.ubicomp.org/ubicomp-iswc-2026/workshop-cfp/>): workshop proposals April 30, 2026; accepted proposals notified May 2; calls distributed May 5; camera-ready workshop papers for the ACM DL July 31, 2026; workshops October 11-12, 2026 in Shanghai. "All workshop papers must use the UbiComp/ISWC 2026 templates and formatting instructions." "Organisers can specify page numbers for papers they solicit." "The ACM Digital Library (DL) and the Adjunct Proceedings can include all accepted workshop papers", at the organisers' choice.

**Template** (<https://www.ubicomp.org/ubicomp-iswc-2026/formatting/>): double-column; LaTeX users "should use `documentclass[sigconf]{acmart}`".

**2026 workshops** (26 listed at <https://ubicomp.org/ubicomp-iswc-2026/workshops-and-tutorials-2026/>). The plausible ones for this paper:

| Workshop | 2026 deadline | Length | Blind | Proceedings | Call |
| --- | --- | --- | --- | --- | --- |
| EverydayXR: 1st Workshop on Situational, Contextual, and Environmental Constraints in XR | July 17, 2026 (23:59 AoE); notification July 31; workshop October 12 | Research papers "up to 6 pages, excluding references and appendices"; position papers and research proposals up to 4 | "Each paper will receive two single-blind reviews." | "Authors of accepted papers will be offered the opportunity to publish their work in the ACM Digital Library as the UbiComp Adjunct Proceedings" | <https://sites.google.com/view/everyday-xr/call-for-participation> |
| PEACE 2026: Proactive Embodied Assistants in Context-aware Environments | July 5, extended to July 12, 2026; notification August 5; camera-ready August 20; workshop October 11 | Research papers 4-6 pages excluding references; position and vision papers 2-4 | "Single-blind peer review." | "UbiComp/ISWC 2026 Adjunct Proceedings + ACM Digital Library." | <https://peace-workshop.github.io/> |
| International Workshop on Reality Mediation (IWRM 2026) | July 3, extended to July 10, 2026; notification July 23; camera-ready August 6; workshop October 12 | "4-8 page workshop papers" | Not stated | Not stated | <https://reality-mediation.github.io/2026/> |
| Assistive Agents for All | Two dates shown for each milestone: abstract July 18 and July 25, full paper July 23 and July 30, 2026 | Full papers max 6 pages, short papers max 4; acknowledgments, references and appendix excluded | "All submissions are anonymous" | "permanently archived in the ACM Digital Library as part of the UbiComp proceedings", with an opt-out | <https://insailab.org/ubicomp2026-aaa-workshop/> |
| WearAgent 2026: Interactive AI for Personal Wearable Agents (1st) | Not on the page | Not on the page | Not on the page | Not on the page | <https://wearagent.github.io/> |

Topic lines that match this paper: EverydayXR, "Understanding how current MR interaction paradigms fail, degrade, or adapt under such conditions"; PEACE, "When-to-act policies", "uncertainty-aware perception", "on-device, privacy-aware perception".

### Other UbiComp/ISWC 2026 tracks of similar size

- **Posters and Demos** (<https://www.ubicomp.org/ubicomp-iswc-2026/posters-demos-2026/>): deadline May 23, extended to June 6, 2026; extended abstracts "up to 4 pages long, in the 2-column ACM Proceedings format, excluding references"; "Submissions are NOT anonymous"; included in the Adjunct Proceedings and the ACM DL but "not considered to be archival publications".
- **ISWC Notes and Briefs** (<https://www.ubicomp.org/ubicomp-iswc-2026/iswc-notes-and-briefs/>): deadline May 17, extended to May 24, 2026; notification July 1; "Notes should not exceed 6 pages in the 2-column, ACM proceedings format", references not counted, no appendices; briefs 3 pages; "ISWC follows a double-blind review process"; archived in the ACM DL. Scope is "technology that is worn on or interacts with the body", which a phone system does not obviously meet.

**Projection for 2027.** UbiComp/ISWC 2027 is eight days earlier than 2026. Workshop paper deadlines between late June and mid July 2027 are likely. Whether EverydayXR, PEACE or IWRM run again is unknown.

## HotMobile

**HotMobile 2027** (<https://hotmobile.org/2027/index.php?id=cfp>): the 28th International Workshop on Mobile Computing Systems and Applications, February 24-25, 2027, Tucson.

| Item | Fact |
| --- | --- |
| Paper deadline | October 9, 2026, 11:59pm AoE. **Passed.** Notification December 16, 2026 |
| Page limit | "Submissions should contain 6 or fewer U.S. letter pages in PDF format, including all references, figures and tables." |
| Template | `sigconf` recommended, body font at least 10pt, double column |
| Blind | "The submissions should not be anonymous." |
| Proceedings | "The papers accepted to HotMobile 2027 will be publicly available from the workshop's website for a limited time window, as per ACM's Digital Library's Policy." The call's open-access section treats accepted papers as ACM publications |
| What it wants | Papers that "propose new directions of research, advocate non-traditional approaches", "debunk existing assumptions", or "report unexpected early results"; "Novel ideas need not be supported by full evaluations" |
| **Posters and demos** | **January 11, 2027 AoE**; notification January 15; camera-ready January 20; "an extended abstract no longer than one 8.5x11-inch page, including all figures and references"; "Please do not anonymize your submissions"; <https://hotmobile27-posterdemo.hotcrp.com/> |

The poster deadline is eight weeks away, earlier than the window asked for, and is the only HotMobile submission still open.

**HotMobile 2028.** Not announced; <https://hotmobile.org/2028/> returns 404. If it keeps the 2027 timing, papers would be due in early October 2027, about eleven months from now.

## Other options inside the window

- **MobileHCI 2027 papers** (<https://mobilehci.acm.org/2027/submission/>): September 20-23, 2027, Limassol, Cyprus. Abstracts February 10, 2027; papers **February 17, 2027**; notification April 15; revisions May 13; final notification June 17. "Submissions between 5000 and 8000 words are encouraged"; papers under 5,000 words are treated as short papers. "All submissions need to be fully anonymized." Accepted papers are journal articles in PACM HCI. This is a full review process, not a workshop, and the deadline is at the very start of the window.

## IMWUT (PACM on Interactive, Mobile, Wearable and Ubiquitous Technologies)

I could not open IMWUT's own pages (<https://dl.acm.org/journal/imwut> and its author guidelines return HTTP 403). The facts below come from UbiComp/ISWC pages that describe IMWUT and from Crossref metadata. Confirm on the IMWUT site before relying on them.

**Deadlines.** "IMWUT has 3 submission deadlines a year (February 1, May 1, and November 1)." (<https://ubicomp.org/ubicomp-iswc-2026/iswc_author_guide/>; the same three dates are on <https://ubicomp.org/ubicomp-iswc-2024/imwut_papers/>.) Older pages list four deadlines on the 15th of February, May, August and November (<https://ubicomp.org/ubicomp2021/upapers/>); that schedule has been replaced. The next deadlines are therefore **February 1, 2027** and **May 1, 2027**.

**Publication and presentation.** "IMWUT has 1 volume per year with 4 issues per volume, published in March (Issue 1), June (Issue 2), September (Issue 3), and December 1 (Issue 4)." Papers "submitted on February 1 may be published as soon as in the June issue" if only minor revisions are required (<https://ubicomp.org/ubicomp-iswc-2024/imwut_papers/>). "The UbiComp / ISWC 2026 main technical track will consist of papers in IMWUT 2025 Issue 4 and IMWUT 2026 Issues 1-3." (<https://www.ubicomp.org/ubicomp-iswc-2026/imwut-papers/>) By the same pattern, a paper would need to appear by IMWUT 2027 Issue 3 to be presented at UbiComp/ISWC 2027; that is a projection.

**Length.** I found no stated page limit on any page I could open. Measured from Crossref records for the journal (ISSN 2474-9567):

| Year | Articles | Median pages | Middle half | 10th to 90th percentile |
| --- | --- | --- | --- | --- |
| 2025 | 242 | 29 | 26 to 33 | 24 to 37 |
| 2026 to November 16 | 198 | 31 | 28 to 35 | 25 to 38 |

Queries: <https://api.crossref.org/journals/2474-9567/works?filter=from-pub-date:2025-01-01,until-pub-date:2025-12-31> and the same with 2026 dates. A 6-page workshop paper is about a fifth of a typical IMWUT article.

**Anonymity.** IMWUT has an anonymization policy: the ISWC call says "the anonymization policy of ACM IMWUT should be observed" for its double-blind review (<https://www.ubicomp.org/ubicomp-iswc-2026/iswc-notes-and-briefs/>). I did not see the policy text itself.

**What reviewers expect in evaluation.** I could not open IMWUT's reviewer guidance, so this is inferred from published papers on this topic that I did open, and should be read as an inference:

- FMT, a wearable-camera memory aid in IMWUT 2019 (25 pages), reports results "across all participants" and findings on how and why users would use the system (`li2019fmt`).
- MemPal, an object-finding assistant at IUI 2025, reports "a quantitative and qualitative study with N=15 older adults within their own homes" (`maniar2025mempal`).
- GO-Finder at IUI 2021 "conducted a user study" comparing against unaided search (`yagi2021gofinder`).

Every comparable system paper in this area evaluates with participants other than its authors, in or near real use. An IMWUT submission evaluated only by its authors in two homes would be out of line with that record.

## ACM policy on generative AI and authorship

**Could not open the policy itself.** <https://www.acm.org/publications/policies/new-acm-policy-on-authorship> and the FAQ at <https://www.acm.org/publications/policies/frequently-asked-questions> return HTTP 403 to automated fetches. The sentences below are quoted from ACM conference and SIG pages that reproduce the policy. Open the ACM page in a browser before quoting it in a submission form.

### Can an AI system be listed as an author

No.

- "Generative AI tools and technologies, such as ChatGPT, may not be listed as authors of an ACM published Work."
  Reproduced as ACM's guidance in the SIGCHI Publications Committee post "ACM Publications Policy: Guidance for SIGCHI Venues", July 6, 2023: <https://sigchi.org/news/blog/acm-publications-policy-guidance-for-sigchi-venues/>. The same post lists as an authorship criterion "Authors must be an identifiable human being."
- The criteria that exclude a non-human author, as quoted on a 2027 ACM conference site: "Anyone listed as Author on an ACM manuscript submission must meet all the following criteria: They have made substantial intellectual contributions to some components of the original work described in the manuscript; and They have participated in drafting and/or revision of the manuscript and They are aware the manuscript has been submitted for publication; and They agree to be held accountable for any issues relating to the correctness or integrity of the work".
  <https://2027.sigcse-ts.acm.org/info/authorship-policies>

### What must be disclosed

The rule changed in 2026. The current wording, quoted identically by the SenSys 2027 call and the SIGCSE TS 2027 policy page:

- "When using Artificial Intelligence to conduct research, including the design and methodology of the research project, creation and selection of data sources, designing experiments, generation and collection of data, coding, implementing models, running simulations, data analysis, testing, validating results, deploying software, archiving data and code for reproducibility, or any other aspects of the research lifecycle that are directly relevant to the conclusions of the research underlying the Work, the specific use(s) of AI tools must be described in detail in the methods section of the Work. This includes the creation of artifacts that are directly relevant to the conclusions of the research, such as code, datasets, and charts or figures that rely on the AI tools."
- "When using Artificial Intelligence to assist with writing an ACM submission, ACM no longer requires the disclosure of information regarding the use of AI (as distinct from AI used in the conduct of the research itself, addressed in item 1 above)."
- "All named authors on an ACM submission will be held responsible and accountable for any problematic content contained in the submission regardless of the source of that problematic content" (SIGCSE TS page only).

Sources: <https://sensys.acm.org/2027/cfp.html> and <https://2027.sigcse-ts.acm.org/info/policies-ai>.

The earlier wording, from April 2023, was stricter on writing: "The use of generative AI tools and technologies to create content is permitted but must be fully disclosed in the Work." (SIGCHI post above.) A commentary dated June 9, 2026 describes the change as recent (<https://recommender-systems.com/news/2026/06/09/acm-new-ai-policy-discussed/>). I could not confirm the effective date.

### What this means for the paper

- If AI tools wrote code, ran analysis, or produced figures that the paper's conclusions rest on, describe that use in the methods section. This is required under the current wording.
- AI help with wording alone no longer needs a statement under ACM's rule, but a venue can be stricter. The MobiSys 2027 call says: "Any text in submissions that is generated using large-scale language models (LLMs), such as ChatGPT, Claude, or Google Gemini, must be clearly identified by the authors in the submitted paper. This policy does not apply to using LLMs for editing or proofreading the author's own text." (<https://www.sigmobile.org/mobisys/2027/call_for_papers/>) MobiSys workshops may adopt it; read each workshop call.
- The MobiCom 2027 call says only: "MobiCom adheres to the ACM Policy on Authorship, including guidance on the use of artificial intelligence (AI)." (<https://www.sigmobile.org/mobicom/2027/cfp.html>)
- Not confirmed: whether the sentence barring AI authorship survives word for word in the 2026 text. Search-engine snippets of the live ACM pages show it, and an FAQ line saying generative AI tools cannot be listed as authors "under any conditions", but I could not open those pages.

## Cost: ACM open access charges in 2027

Workshop papers published in the ACM DL fall under ACM's open-access model. The HotMobile 2027 call states that for conferences scheduled in 2027 the subsidised article processing charge is "$500 APC for ACM/SIG members" and "$750 for non-members", and that authors at ACM Open institutions pay nothing (<https://hotmobile.org/2027/index.php?id=cfp>). For 2026 the ImmerCom page gave $250 and $350. Check whether the corresponding author's institution is in ACM Open before choosing a corresponding author.

## Could not find

- Any 2027 workshop list or workshop call for MobiCom, MobiSys, CPS-IoT Week or UbiComp/ISWC.
- A HotMobile 2028 site or date.
- A working UbiComp/ISWC 2027 website.
- IMWUT's own statements on length, template and review criteria.
- The ACM authorship policy page itself, and the date its 2026 revision took effect.
- Calls for EMDL, EdgeSys and the other 2026 workshops not detailed above; I listed them from the host conference pages without opening them.
