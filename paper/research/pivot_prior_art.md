# Pivot check, Part 1: has this been done?

Research date: 2026-11-16. Scope: a phone-only (Pixel 8 class, ARCore, on-device) app that builds an automatically named 3D inventory of household items from one walk-through, lets the user search it by typing, and keeps each entry's "last confirmed" state honest from whatever the camera happens to see later, including items that are only visible while a drawer, box or cabinet is open.

## How to read this file

- Every fact below comes from a page opened during this session; the URL is given next to it. Authors, year and venue are copied from the page, never from memory. Where a page did not state a venue, this file says so.
- Pages were read through a fetch tool that extracts answers from the page text. Short quotes are reproduced as that tool returned them. Re-check any quote against the source before it goes into the paper. Three PDFs were viewed directly as page images (Find My Things, Funk et al., DrawerFinder).
- Long arXiv HTML pages were read only up to the first 100,000 characters (noted per entry where it matters).
- "Not found" means not found by about 30 generic web searches plus the pages opened. The AAAI proceedings site blocked automated reading and the ACM Digital Library was not searched directly, so a miss here is weaker evidence than a database search. Works seen only in search-result snippets are listed separately under "Doubtful or not confirmed" and are not used as evidence.

## Direct answers

### 1. Is there an on-device smartphone system that builds an automatically named 3D object inventory of a home from a walk-through?

None was found. The pieces exist separately, and the closest complete system does the mapping in the cloud.

- Closest system: SemanticXR (ISMAR 2026) builds an open-vocabulary, object-level 3D map and answers queries such as "Where are my keys?". The mapping runs on a workstation GPU; the device keeps only a sparse local object map for queries. The paper states that running its full pipeline on the device "draws 60 W in maximum power mode and takes several seconds to map a single frame" and that "on-device mapping is impractical within XR power budgets". It lists moving objects as future work and evaluates on the Replica dataset, not real homes. https://arxiv.org/html/2606.12849
- Phone systems found do only part of the job. NaviSense (ASSETS 2025) finds one named object at query time on an iPhone using cloud models and keeps no inventory. Find My Things (Microsoft Research) runs on the phone but the user must teach each object with four videos and there is no map. An older ARCore paper (Gupta and Khan, 2022) lifts COCO detections to 3D cuboids on a Galaxy S9 with one tap per object and no persistent map. Google's own ARCore ML sample attaches ML Kit labels to anchors but uses a five-category classifier.
- Products: video-walkthrough inventory apps exist (Scanlily, HomeZada), but their listings describe room-level or shelf-level locations, not 3D positions, and do not say the processing is on-device.

What is left: the on-device combination itself (automatic naming of many items per frame, multi-view merging into 3D entries, typed search, all on a phone SoC next to ARCore), plus evidence that it is feasible in latency, memory and energy. Be aware that the map-and-query recipe is the ConceptGraphs/SemanticXR recipe; on this axis alone the contribution is a constrained-platform port with measurements, not a new method. One demo paper that could not be opened (MemoVision, AAAI-26) may overlap and must be read before any "first" claim; see "Doubtful or not confirmed".

### 2. Is there work on keeping such an inventory truthful over weeks or months from intermittent, opportunistic observations?

Partly, and almost all of it is robotics with robot-collected observations.

- Robots, weeks: Predictive Spatio-Temporal Scene Graphs (RA-L 2026) attaches Bayesian persistence filters to an open-vocabulary scene graph and was evaluated in a real three-week study (3 rooms, 28 receptacles, 21 objects, changes every two hours). Inference ran off-board on a desktop GPU, receptacles were manually annotated boxes, and closed containers are not addressed. https://arxiv.org/html/2605.00121 . The underlying estimators are older: the persistence filter (Rosen, Mason, Leonard, ICRA 2016, https://dspace.mit.edu/handle/1721.1/107620) and Perpetua (IROS 2025, https://arxiv.org/abs/2507.18808).
- Robots, active re-observation: DynaMem removes stale map points by ray-casting and sends the robot to explore least-recently-seen areas. https://arxiv.org/html/2411.04999
- People, opportunistic: Lost & Found (RA-L 2025) updates a scene graph from glasses video, but only for changes that happen in view of the camera during a detected hand interaction: "Our experiments only focus on changes that occur within the egocentric observations". https://arxiv.org/html/2411.19162 . Out of Sight, Not Out of Mind (3DV 2025) keeps 3D tracks of handled objects after they leave view, over seconds to about 12 minutes, and names "over hours and potentially days" as a future goal. https://arxiv.org/html/2404.05072v2 . MemPal (IUI 2025) was tested with retrieval 30 to 40 minutes after placement.
- Benchmarks, days to two weeks: SuperMemory-VQA (arXiv, May 2026) includes an "Object & Location Memory" task on glasses recordings, some "spanning multiple days (up to two weeks)"; the best system scored 61.0% overall and all systems ran on server GPUs or cloud APIs. https://arxiv.org/html/2606.00825v1 . EgoLife is a week-long glasses dataset with question answering. https://arxiv.org/abs/2503.03803
- Products: Google's Find Hub "remembered items" (September 2026) stores where the user says an item is; the article opened describes no automatic sensing or updating. https://www.androidcentral.com/apps-software/no-tracker-tag-gemini-can-now-store-item-locations-right-in-find-hub

Not found: a system that, on a phone or glasses, turns sparse user-driven sightings over weeks or months into per-entry evidence states (seen again here, expected here but not seen, found elsewhere, new). The estimation ideas (persistence filtering, ray-cast negative evidence) are established in robotics; what is open is the human-carried, opportunistic, on-device setting and its evaluation.

### 3. Is there work on items inside containers for object finding?

Yes, in three forms, none of them a handheld phone app.

- Instrumented storage (HCI): DrawerFinder (Komatsuzaki, Tsukada, Siio; IPSJ 73rd National Convention, 2011) puts a visual marker in each storage box and uses fixed cameras above the shelf to photograph the contents automatically whenever a box is pulled out; users browse the photos. https://teapot.lib.ocha.ac.jp/record/4032/files/siio_56.pdf
- Egocentric tracking: Lost & Found models drawers as nodes with "contains" edges, so an object seen being put into a drawer stays findable after the drawer closes; it notes that "the occlusion of the object precludes the naive solution of a rescan". https://arxiv.org/html/2411.19162
- Robots: Articulated 3D Scene Graphs (MoMa-SG, CoRL 2026) links contained objects to the drawer or cabinet door that reveals them, from RGB-D video of a person opening things; it needs 24 GB of GPU memory and leaves later rearrangements to future work. https://arxiv.org/html/2602.16356 . RoboEXP has a robot open things to discover nested objects. https://arxiv.org/abs/2402.15487
- The need is documented: in MemPal's home study a participant wanted "knowing which drawer", which the system could not reliably give (https://arxiv.org/html/2502.01801); DynaMem lists "searching in cabinets or drawers" as future work; the three-week robot study above does not address closed containers.

Not found: container contents captured opportunistically by a handheld phone with no markers or fixed cameras, and an evidence model that treats "not seen" differently when the container was closed.

### 4. Which of the three is the most defensible gap for a workshop paper?

Gap 2, scoped as visibility-aware evidence maintenance from opportunistic phone observations, with containers (gap 3) as its sharpest case and the on-device build (gap 1) as the enabling artifact and measurement. This is a judgment, with these reasons:

- Gap 1 alone is true as far as this search goes, but it is the easiest to attack: the mapping recipe is published, SemanticXR already frames the device-versus-cloud question, commercial apps already turn a video into an item list, and one possibly overlapping demo paper could not be read. A "first on a phone" claim needs hedging and a measured feasibility result to stand.
- Gap 2 has prior art only where a robot controls its own observations or where changes happen in view. The phone setting changes the problem: observations are sparse and user-driven, and absence is only informative if the spot was actually observable. That is a framing a workshop can accept with a small prototype and a short staged study.
- Gap 3 is a real unmet need but, standalone, has the most direct older prior art (DrawerFinder-style capture) and the hardest sensing on a phone (close range, weak depth). It is stronger as the motivating case for gap 2 than as the headline.
- Constraint: nobody can evaluate "months" on a workshop schedule. Claim the mechanism and evaluate it with staged relocations across several sessions or days, and say so.

## The twelve closest works

Each entry: citation as shown on the opened page; URL; what it does; where it runs; moved items; what the idea would add.

### 1. SemanticXR
- Citation: Rahul Singh, Devdeep Ray, Connor Smith, Sarita Adve. "SemanticXR: Low Power and Real-time Queryable Semantic Mapping with an Object-Level Device-Cloud Architecture." arXiv:2606.12849 (v1 11 Jun 2026). arXiv comments field: "Proceedings of the IEEE International Symposium on Mixed and Augmented Reality (ISMAR), 2026".
- URLs: https://arxiv.org/abs/2606.12849 , https://arxiv.org/html/2606.12849 (first 100k characters read). Related demo page: https://immerse.illinois.edu/symposium-posters/82404
- What it does: open-vocabulary, object-level 3D semantic mapping with natural-language queries for XR. The server maps; the device holds a sparse object map that answers queries in under 100 ms for up to 10,000 objects and keeps working through network drops. Models named: RAM, Grounding DINO, MobileSAM, MobileCLIP.
- Where it runs: client is a Jetson Orin in low-power mode as a proxy XR device (also prototyped on Meta Quest 3 and iPad Pro); mapping on a workstation with an RTX 6000 Ada GPU.
- Moved items: not handled; "support for dynamic environments with moving objects" is future work.
- What the idea adds: all perception on a phone with no server; real homes rather than Replica; entries that change state over time; containers. It also has to answer this paper's claim that on-device mapping is impractical, by using lighter models and a lower mapping rate.

### 2. ConceptGraphs
- Citation: Qiao Gu, Alihusein Kuwajerwala, Sacha Morin, Krishna Murthy Jatavallabhula, Bipasha Sen, Aditya Agarwal, Corban Rivera, William Paul, Kirsty Ellis, Rama Chellappa, Chuang Gan, Celso Miguel de Melo, Joshua B. Tenenbaum, Antonio Torralba, Florian Shkurti, Liam Paull. "ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning." arXiv:2309.16650, submitted 28 Sep 2023. The arXiv page lists no venue; check the published venue before citing.
- URLs: https://arxiv.org/abs/2309.16650 , https://arxiv.org/html/2309.16650
- What it does: segments each posed RGB-D frame with a class-agnostic segmenter (SAM), embeds regions with CLIP, merges detections across views into 3D object nodes, then captions each object with a vision-language model (LLaVA) and summarises with GPT-4. This is the template for "many items per frame, merged into named 3D entries".
- Where it runs: desktop GPU plus cloud LLM; the paper gives no timings and notes the computational and economic cost.
- Moved items: minimal; during robot localisation "previously observed objects are removed if they are not observed by the robot and new objects can also be added".
- What the idea adds: the same output on a phone with motion-based depth, without SAM plus cloud LLM; graded evidence states instead of delete-if-unseen; containers.

### 3. Predictive Spatio-Temporal Scene Graphs (PredictiveGraphs)
- Citation: Miguel Saavedra-Ruiz, Charlie Gauthier, Kumaraditya Gupta, Shima Shahfar, Kirsty Ellis, Steven Parkison, Liam Paull. "Predictive Spatio-Temporal Scene Graphs for Semi-Static Scenes." IEEE Robotics and Automation Letters, vol. 11, no. 12, pp. 14524-14531, 2026. arXiv:2605.00121.
- URLs: https://arxiv.org/abs/2605.00121 , https://arxiv.org/html/2605.00121 (first 100k characters read)
- What it does: adds a Bayesian persistence filter (Perpetua*) to a ConceptGraphs-style graph so that each object's presence at each receptacle is a belief that decays and can be predicted forward. Real-world study over three weeks with scheduled changes every two hours.
- Where it runs: wheeled robot with an RGB-D camera; "model inference was performed offboard, on an Nvidia RTX 2080".
- Moved items: yes, this is its purpose, at the level of which receptacle holds which object. Receptacles are large fixed furniture with manually annotated boxes. Closed containers are not addressed.
- What the idea adds: observations that come from a person's incidental camera use rather than a robot's mapping runs; visibility-aware absence (closed drawer is not evidence of absence); no manual receptacle annotation; phone-only compute; user-facing "last confirmed" semantics.

### 4. DynaMem
- Citation: Peiqi Liu, Zhanqiu Guo, Mohit Warke, Soumith Chintala, Chris Paxton, Nur Muhammad Mahi Shafiullah, Lerrel Pinto. "DynaMem: Online Dynamic Spatio-Semantic Memory for Open World Mobile Manipulation." arXiv:2411.04999 (v1 7 Nov 2024, v2 29 May 2025). No venue on the arXiv page.
- URLs: https://arxiv.org/abs/2411.04999 , https://arxiv.org/html/2411.04999
- What it does: a voxel memory with vision-language features that is updated as the robot moves. A voxel between the camera and a newly observed surface is cleared ("that voxel must be unoccupied"), so moved objects disappear from their old place. Queries go to vision-language features or a multimodal LLM.
- Where it runs: Hello Robot Stretch with RGB-D camera; uses CLIP/SigLIP, SAM 2, OWLv2 and cloud LLMs (GPT-4o, Gemini 1.5 Pro); where compute runs is not stated in the text read.
- Moved items: yes, by ray-cast removal plus active exploration of least-recently-seen areas. Cabinets and drawers are named as future work.
- What the idea adds: no active exploration (the user decides where the camera goes); weak motion depth instead of an RGB-D sensor; explicit evidence states shown to the user; containers; phone-only.

### 5. Lost & Found
- Citation: Tjark Behrens, René Zurbrügg, Marc Pollefeys, Zuria Bauer, Hermann Blum. "Lost & Found: Tracking Changes from Egocentric Observations in 3D Dynamic Scene Graphs." arXiv:2411.19162; arXiv comments: "accepted at IEEE Robotics and Automation Letters"; related DOI 10.1109/LRA.2025.3544518.
- URLs: https://arxiv.org/abs/2411.19162 , https://arxiv.org/html/2411.19162
- What it does: starts from a pre-built scene graph of a room and keeps it up to date from glasses video by detecting hand-object interactions and tracking the 6DoF pose of the moved object. Drawers are nodes with "contains" edges, so a robot can later fetch an object hidden in a closed drawer.
- Where it runs: Aria glasses for capture; the initial scan is made with an iPad Pro scanner app and segmented with Mask3D; timings measured on an RTX 4090.
- Moved items: only if the move happens in view during a detected interaction. Stated limits: rigid objects, the object "must remain in the perceptual field", no rescans to correct long-term drift.
- What the idea adds: automatic naming and typed search over the whole inventory; handling of changes nobody filmed (expected here but not seen, found elsewhere); no prior dense scan or desktop GPU; phone-only.

### 6. Out of Sight, Not Out of Mind (OSNOM)
- Citation: Chiara Plizzari, Shubham Goel, Toby Perrett, Jacob Chalk, Angjoo Kanazawa, Dima Damen. "Spatial Cognition from Egocentric Video: Out of Sight, Not Out of Mind." arXiv:2404.05072; arXiv comments: "Accepted at 3DV 2025".
- URLs: https://arxiv.org/abs/2404.05072 , https://arxiv.org/html/2404.05072v2
- What it does: defines the task of knowing where handled objects are in 3D after they leave the camera view, and a method (Lift, Match, Keep) evaluated on 100 long EPIC-KITCHENS videos.
- Where it runs: offline; scene meshes take about 5 hours per video on a GPU; the tracking step itself "runs at 1000fps on a single CPU core". Most results use ground-truth masks.
- Moved items: tracks objects the wearer moves, over 0-60 s and 1-12 minutes. Objects are not named. The paper notes "we cannot distinguish between missing observations and occlusion".
- What the idea adds: named, searchable entries for all visible items, not only handled ones; weeks rather than minutes; an explicit treatment of the missing-versus-occluded ambiguity; a working phone system rather than a benchmark.

### 7. MemPal
- Citation: Natasha Maniar, Samantha W.T. Chan, Wazeer Zulfikar, Scott Ren, Christine Xu, Pattie Maes. "MemPal: Leveraging Multimodal AI and LLMs for Voice-Activated Object Retrieval in Homes of Older Adults." 30th International Conference on Intelligent User Interfaces (IUI '25), March 24-27, 2025, Cagliari, Italy. arXiv:2502.01801.
- URLs: https://arxiv.org/abs/2502.01801 , https://arxiv.org/html/2502.01801 (first 100k characters read)
- What it does: a worn camera logs what the wearer does and where they put things; a voice query returns the last known location. Home study with 15 older adults: retrieval accuracy 0.97 with MemPal versus 0.81 without when only trials with an accurate system response are counted; with all trials the difference was not significant (p=.054). Audio answers were correct 72% of the time; object misidentification accounted for 24% of responses.
- Where it runs: an iPhone on a neck mount "streams camera footage to a server"; GPT-4V and OpenAI embeddings in the cloud.
- Moved items: records placements the camera saw the wearer make; the paper says users "must be consciously aware of placing objects in front of the wearable camera". Locations are text and room labels, not 3D positions. Retrieval was tested 30 to 40 minutes after placement (the paper gives both figures).
- What the idea adds: 3D positions and guidance; no cloud; long-stored items rather than items placed in the last hour; evidence states; which-drawer resolution.

### 8. GO-Finder
- Citation: Takuma Yagi, Takumi Nishiyasu, Kunimasa Kawasaki, Moe Matsuki, Yoichi Sato. "GO-Finder: A Registration-Free Wearable System for Assisting Users in Finding Lost Objects via Hand-Held Object Discovery." arXiv:2101.07314; arXiv comments: "13 pages, 13 figures, ACM IUI 2021".
- URL: https://arxiv.org/abs/2101.07314 (abstract page only)
- What it does: a wearable camera automatically detects and groups objects held in the hand into a visual timeline; the user browses it on a smartphone to see each object's last appearance. No per-object registration.
- Where it runs: wearable camera plus smartphone browser; processing location not stated on the abstract page.
- Moved items: shows the last time the object was in the wearer's hand; nothing more is stated on the page read.
- What the idea adds: items that were never handled on camera (stored things); names and typed search; 3D positions; evidence states.

### 9. NaviSense
- Citation: Ajay Narayanan Sridhar, Fuli Qiao, Nelson Daniel Troncoso Aldas, Yanpei Shi, Mehrdad Mahdavi, Laurent Itti, Vijaykrishnan Narayanan. "NaviSense: A Multimodal Assistive Mobile application for Object Retrieval by Persons with Visual Impairment." The 27th International ACM SIGACCESS Conference on Computers and Accessibility (ASSETS '25), October 26-29, 2025, Denver, CO, USA. arXiv:2509.18672.
- URL: https://arxiv.org/html/2509.18672v1
- What it does: the user says what they want; the phone scans, a vision-language model detects the object with no predefined categories, and ARKit plus LiDAR tracks its 3D position locally while audio and haptics guide the hand. Reported: 95% detection accuracy on 200 cluttered frames; cloud detection 0.706 s per call.
- Where it runs: iPhone 16 Pro; language and vision models (GPT-4o-mini, Moondream 2B) in the cloud. The authors list "relies on cloud-based models, raising privacy and connectivity concerns" as a limitation.
- Moved items: not applicable; it searches the live view and keeps no inventory.
- What the idea adds: memory. The item need not be in view at query time; the app already knows where it was last confirmed. Also on-device perception and Android without LiDAR.

### 10. Find My Things
- Citation: Cecily Morrison, Martin Grayson, Rita Faia Marques, Daniela Massiceti, Camilla Longden, Linda Wen, Edward Cutrell (Microsoft Research). "Understanding Personalized Accessibility through Teachable AI: Designing and Evaluating Find My Things for People who are Blind or Low Vision." The PDF's first pages do not print the venue; the file on Microsoft's server is named FindMyThings-ASSETS-2023.pdf. Confirm the venue from the ACM record before citing.
- URLs: https://www.microsoft.com/en-us/research/uploads/prod/2023/08/FindMyThings-ASSETS-2023.pdf (pages 1-2 viewed), https://www.microsoft.com/en-us/research/story/find-my-things/ (published October 24, 2024)
- What it does: a blind or low-vision user teaches the phone a personal item with four short videos; later the phone finds that item in the camera view and guides the user to it. Now a feature of the Seeing AI app.
- Where it runs: phone; the story page says the model is personalised "on that individual's device in just a couple of seconds".
- Moved items: not applicable; live search only, no stored locations.
- What the idea adds: no teaching step; a stored 3D inventory; evidence states.

### 11. DrawerFinder
- Citation (version opened, in Japanese): Mizuho Komatsuzaki, Koji Tsukada, Itiro Siio. "DrawerFinder: 収納箱に適した物探し支援システムの提案と運用" (English title printed in the footnote: "DrawerFinder: Finding Items in Storage Boxes using Pictures and Visual Markers"). 情報処理学会第73回全国大会 (IPSJ 73rd National Convention), paper 3M-3, pp. 1-603 to 1-604, 2011. An English version at IUI 2011 was seen only in a search snippet; see "Doubtful or not confirmed".
- URL: https://teapot.lib.ocha.ac.jp/record/4032/files/siio_56.pdf (both pages viewed)
- What it does: each drawer-type storage box has an ARToolKit marker inside its top edge. Cameras fixed above the shelf recognise the marker when a box is pulled out and automatically photograph the contents and the person, every 5 seconds while the box is open. Photos are stored by box and time and browsed in a web browser on a PC or phone. Trial deployment on a lab shelf of 12 boxes.
- Where it runs: fixed webcams and a small PC by the shelf; server database.
- Moved items: the newest photo of a box replaces the user's picture of its contents each time it is opened; there are no item-level names or tracking.
- What the idea adds: no markers or fixed cameras; item-level names and search; works for any container the phone happens to see open; a single map across the home.

### 12. Articulated 3D Scene Graphs (MoMa-SG)
- Citation: Martin Büchner, Adrian Röfer, Tim Engelbracht, Tim Welschehold, Zuria Bauer, Hermann Blum, Marc Pollefeys, Abhinav Valada. "Articulated 3D Scene Graphs for Open-World Mobile Manipulation." arXiv:2602.16356; arXiv comments: "10th Conference on Robot Learning (CoRL 2026)".
- URLs: https://arxiv.org/abs/2602.16356 , https://arxiv.org/html/2602.16356 (first 100k characters read)
- What it does: from an RGB-D video of a person opening drawers and doors, it estimates each articulation and links the objects found inside to their container, with CLIP features for open-vocabulary lookup. Releases a dataset of 62 sequences with 600 interactions.
- Where it runs: "The approach requires 24 GB of VRAM."
- Moved items: no; "In future work, we aim to address semi-static rearrangements."
- What the idea adds: phone-scale compute; no depth sensor; contents captured opportunistically rather than in a demonstration session; change over time.

## Products and shipped features

| Product | Source opened | What the page says | Gap to the idea |
|---|---|---|---|
| Scanlily (iOS) | App Store listing, https://apps.apple.com/app/1627766472 | "Record a video while talking about what you see. AI listens to your voice and interprets the video to build a complete, organized inventory automatically." Locations down to aisle, bay and shelf; optional QR labels; 20 free video recognitions, Pro $8.99. Does not say where the AI runs; no AR or 3D. | Narrated video, no 3D position or guidance, no automatic re-confirmation. |
| HomeZada | Press page (undated), https://homezada.com/press/homezada-launches-home-inventory-video-recognition-ai | Room videos are uploaded and "processed through video recognition AI to automatically detect the inventory items"; the homeowner confirms. Room-level; aimed at insurance. | Upload model, room-level only, no change tracking. |
| Google Find Hub "remembered items" | Android Central, Jay Bonggolto, 1 Sep 2026, https://www.androidcentral.com/apps-software/no-tracker-tag-gemini-can-now-store-item-locations-right-in-find-hub | The user tells Gemini where an item is ("remember in Find Hub that I put my passport in my bedroom drawer"), optionally with a photo; rolling out to Android 16 and newer. No automatic detection described. | Manual, per item; nothing re-confirms the entry. This is the shipped baseline the idea must beat on effort and freshness. |
| Project Astra demo | Engadget, Sam Rutherford and Karissa Bell, 14 May 2024, https://www.engadget.com/google-project-astra-hands-on-full-of-potential-but-its-going-to-be-a-while-235607743.html | Demo on a Pixel 8 Pro "tracked and remembered each one's location"; memories "last for a single session, which currently only spans a few minutes"; needs the cloud; research preview. | Minutes, cloud. Later status of Astra-style memory was not verified. |
| Ray-Ban Meta glasses | Meta, 25 Sep 2024, https://about.fb.com/news/2024/09/ray-ban-meta-glasses-new-ai-features-and-partner-integrations/ | "your glasses can remember your spot in long-term parking for you"; voice-set reminders. | User-initiated notes, not an inventory. |
| WildDet3D (iPhone) | App Store listing, https://apps.apple.com/us/app/id6760861157 | "open-vocabulary 3D object detection to your iPhone"; LiDAR and ARKit; type the object names; "Internet connection required"; copyright Allen Institute for AI. | Live detection for typed classes, internet required, no stored inventory described. |
| Google ARCore ML sample | https://developers.google.com/ar/develop/machine-learning , https://github.com/googlesamples/arcore-ml-sample (archived 19 Apr 2026) | Kotlin sample that feeds ARCore camera images to ML Kit Object Detection (five coarse categories) or Cloud Vision and "attaches a label to the object in the virtual scene". | The detect-then-anchor pattern exists as sample code; no inventory, merging, search or persistence. |
| Memories.ai | TechCrunch, Rebecca Szkutak, 16 Mar 2026, https://techcrunch.com/2026/03/16/memories-ai-is-building-the-visual-memory-layer-for-wearables-and-robotics | A visual memory model as infrastructure; second generation to "run on Qualcomm's processors starting later this year". No object-finding feature mentioned. | Infrastructure, not a product; watch as a possible future competitor. |

## Other related work opened (useful for the related-work section)

| Work (as shown on the page) | URL | Why it matters |
|---|---|---|
| Lixing Guo, Tobias Höllerer. "Words into World: A Task-Adaptive Agent for Language-Guided Spatial Retrieval in AR." arXiv:2512.00294, 29 Nov 2025 (no venue listed). | https://arxiv.org/html/2512.00294 | Meta Quest 3 client, models on an edge server; scene graph "tightly coupled to the current user query and viewpoint"; median end-to-end latency 4.74 s in its Table VIII. Per-query, not an inventory. |
| Lukas Schmid, Marcus Abate, Yun Chang, Luca Carlone. "Khronos: A Unified Approach for Spatio-Temporal Metric-Semantic SLAM in Dynamic Environments." Accepted to Robotics: Science and Systems, 2024. arXiv:2402.13817. | https://arxiv.org/abs/2402.13817 | Robot mapping that reasons about short-term and long-term change. |
| Abdelrhman Werby, Chenguang Huang, Martin Büchner, Abhinav Valada, Wolfram Burgard. "Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation." arXiv:2403.17846 (related DOI 10.15607/RSS.2024.XX.077). | https://arxiv.org/abs/2403.17846 | Floor/room/object hierarchy for open-vocabulary maps. |
| Zhijie Yan, Shufei Li, Zuoxu Wang, Lixiu Wu, Han Wang, Jun Zhu, Lijiang Chen, Jihong Liu. "Dynamic Open-Vocabulary 3D Scene Graphs for Long-term Language-Guided Mobile Manipulation." Accepted by IEEE RA-L, 2025. arXiv:2410.11989. | https://arxiv.org/abs/2410.11989 | Local graph updates during robot interaction. |
| Sara Micol Ferraina, Michele Brienza, Francesco Argenziano, Emanuele Musumeci, Vincenzo Suriani, Domenico D. Bloisi, Daniele Nardi. "LOST-3DSG: Lightweight Open-Vocabulary 3D Scene Graphs with Semantic Tracking in Dynamic Environments." arXiv:2601.02905, Jan 2026 (no venue listed). | https://arxiv.org/abs/2601.02905 | Tracks moved objects using word and sentence embeddings instead of dense CLIP features; robot. Relevant to cheap re-identification. |
| Sebastián Barbas Laina, Simon Boche, Sotiris Papatheodorou, Simon Schaefer, Jaehyung Jung, Helen Oleynikova, Stefan Leutenegger. "FindAnything: Open-Vocabulary and Object-Centric Mapping for Robot Exploration in Any Environment." arXiv:2504.08603 (no venue listed). | https://arxiv.org/abs/2504.08603 | Open-vocabulary object-centric mapping aimed at resource-constrained drones. |
| Hanxiao Jiang, Binghao Huang, Ruihai Wu, Zhuoran Li, Shubham Garg, Hooshang Nayyeri, Shenlong Wang, Yunzhu Li. "RoboEXP: Action-Conditioned Scene Graph via Interactive Exploration for Robotic Manipulation." arXiv:2402.15487 (no venue listed). | https://arxiv.org/abs/2402.15487 | Robot explores by opening things; handles nested objects. |
| Johanna Wald, Armen Avetisyan, Nassir Navab, Federico Tombari, Matthias Nießner. "RIO: 3D Object Instance Re-Localization in Changing Indoor Environments." ICCV 2019 (Oral). arXiv:1908.06109. | https://arxiv.org/abs/1908.06109 | 3RScan: 1,482 RGB-D scans of 478 environments with objects that moved between deliberate rescans. |
| David Matthew Rosen, Julian Mason, John J. Leonard. "Towards lifelong feature-based mapping in semi-static environments." Proc. 2016 IEEE ICRA, pp. 1063-1070. | https://dspace.mit.edu/handle/1721.1/107620 | The persistence filter: a recursive Bayesian belief that a feature still exists. Direct ancestor of an evidence state. |
| Miguel Saavedra-Ruiz, Samer B. Nashed, Charlie Gauthier, Liam Paull. "Perpetua: Multi-Hypothesis Persistence Modeling for Semi-Static Environments." Accepted to IROS 2025. arXiv:2507.18808. | https://arxiv.org/abs/2507.18808 | Persistence plus emergence filters; robust to missing observations. |
| Kristen Grauman et al. (85 authors). "Ego4D: Around the World in 3,000 Hours of Egocentric Video." CVPR 2022. arXiv:2110.07058. | https://arxiv.org/abs/2110.07058 | Origin of the episodic-memory benchmark family ("where did I last see X"). |
| Yunhan Zhao, Haoyu Ma, Shu Kong, Charless Fowlkes. "Instance Tracking in 3D Scenes from Egocentric Videos." CVPR 2024. arXiv:2312.04117. | https://arxiv.org/abs/2312.04117 | Benchmark for locating a specific object instance in 3D from egocentric RGB-D video. |
| Samiul Alam, Shakhrul Iman Siam, Michael J. Proulx, James Fort, Richard Newcombe, Hyo Jin Kim, Mi Zhang. "SuperMemory-VQA: An Egocentric Visual Question-Answering Benchmark for Long-Horizon Memory." arXiv:2606.00825, 30 May 2026. | https://arxiv.org/abs/2606.00825 | 52.9 h of glasses video, 4,853 questions with an "unanswerable" option; object-and-location memory is about 19% of questions; best system 61.0% overall. Evidence that the problem is unsolved even with server models. |
| Jingkang Yang, Shuai Liu, Hongming Guo, Yuhao Dong, Xiamengwei Zhang, ..., Ziwei Liu (22 authors). "EgoLife: Towards Egocentric Life Assistant." arXiv:2503.03803. | https://arxiv.org/abs/2503.03803 | Six people, one week, about 300 hours; long-context question answering. |
| Junxiao Shen, John Dudley, Per Ola Kristensson. "Encode-Store-Retrieve: Augmenting Human Memory through Language-Encoded Egocentric Perception." arXiv:2308.05822 (no venue listed). | https://arxiv.org/abs/2308.05822 | Video to text to vector store to LLM answer; the text-log alternative to a 3D inventory. |
| Yoonsang Kim, Yalong Yang, Arie E. Kaufman. "Memento: Towards Proactive Visualization of Everyday Memories with Personal Wearable AR Assistant." IEEE VRW 2026. arXiv:2601.17622. | https://arxiv.org/abs/2601.17622 | Recalls past spoken queries by context; not an object-location inventory. |
| Markus Funk, Robin Boldt, Bastian Pfleging, Max Pfeiffer, Niels Henze, Albrecht Schmidt. "Representing Indoor Location of Objects on Wearable Computers with Head-Mounted Displays." AH '14, March 7-9, 2014, Kobe, Japan. doi:10.1145/2582051.2582069. | https://www.medien.ifi.lmu.de/pubdb/publications/pub/funk2014hmd/funk2014hmd.pdf | Compares a "last seen image" with a map for guiding a user to an object; results favour the last-seen image for fine navigation. Directly relevant to the guidance UI. |
| N. Gupta, N. M. Khan. "Efficient and Scalable Object Localization in 3D on Mobile Device." Journal of Imaging 8(7):188, 2022. doi:10.3390/jimaging8070188. | https://pmc.ncbi.nlm.nih.gov/articles/PMC9323171/ | ARCore plus an on-device COCO detector on a Galaxy S9 to get 3D cuboids; one tap per object; no persistent map. The nearest Android-phone precedent found. |
| Chanyoung Jung, Jeho Lee, Gunjoong Kim, Jiwon Kim, Seonghoon Park, Hojung Cha. "ARIA: Optimizing Vision Foundation Model Inference on Heterogeneous Mobile Processors for Augmented Reality." MobiSys 2025. doi:10.1145/3711875.3729161. | https://api.openalex.org/works/doi:10.1145/3711875.3729161 (bibliographic record only) | Mobile-systems venue precedent for running vision foundation models on phone processors for AR. Only the citation was confirmed; the paper itself could not be opened. |

## Doubtful or not confirmed

These were seen only in search-result snippets, or the page could not be opened. Do not cite without reading them.

- MemoVision. Snippets describe "MemoVision: A Digital Catalog for Everyday Interactions" by Lai Xing Ng, K. T. W. Tang, J. J. W. Tan, a three-page AAAI-26 demo (Proceedings of the AAAI Conference on Artificial Intelligence, 40(48), 41646-41648), with a client-server design in which smart glasses stream camera, depth, pose, gaze and hand data to a server that builds a queryable catalog with 3D bounding volumes. The AAAI pages (https://ojs.aaai.org/index.php/AAAI/article/view/42368) returned an access-denied page twice. This is the most important unread item: it may be the closest "automatic catalog of objects with positions" system. From the snippets it is client-server and glasses-first, which would leave the on-device phone claim intact, but that is unverified.
- DrawerFinder, English version: IUI 2011, pp. 363-366 (snippet only). The Japanese version cited above was read in full. The same snippets mention a predecessor, BoxFinder (Pervasive 2010 adjunct proceedings; also cited in the DrawerFinder reference list), and a follow-up poster, DrawerBrowser (UbiComp 2011).
- FINDrawers: "FINDrawers: Drawer-shaped Device Supporting Object Localization", IPSJ Journal 62(1), 44-52, 2021 (snippet only; the record page https://ipsj.ixsq.nii.ac.jp/records/208972 returned HTTP 503 twice). Snippet: a drawer with a camera and an open/close sensor, searchable by name, colour and time.
- MoMa-LLM (arXiv:2403.08605) and CuriousBot (arXiv:2501.13338): robot search inside cabinets and drawers (snippets only).
- EgoMemReason (arXiv:2605.09874), HD-EPIC (CVPR 2025), AMEGO (arXiv:2409.10917): long-horizon egocentric benchmarks with object-movement questions (snippets only).
- POCD (arXiv:2205.01202) and "Objects Can Move" (arXiv:2208.09870): object-level change detection between visits (snippets only).
- "Where is my Phone? Personal Object Retrieval from Egocentric Images" (arXiv:1608.08139) and the Antonius system cited inside Funk et al.: earlier wearable-camera object finders (not opened).
- Google Android XR glasses memory demo (reported in snippets as recalling objects for up to about 10 minutes) and any 2025-2026 update to Astra-style memory: not opened.
- A claim in one search snippet that Find Hub's remembered items go stale when an item is moved: plausible from the design, but the page (Android Authority) returned HTTP 403.
- ARCore map persistence: one search snippet says ARCore Cloud Anchors can be resolved for up to 365 days and that anchors are otherwise lost on app restart. Not opened. Relevant because the idea assumes a persistent home map with on-device-only processing.

## Searches run

Generic topic queries only (no personal or project names), roughly: smartphone AR find misplaced household objects open-vocabulary 3D map on-device; open-vocabulary 3D scene graph phone ARKit/ARCore; egocentric "where did I leave" 3D tracking; dynamic scene graph change detection egocentric; wearable camera lost-object retrieval older adults; items in boxes and drawers HCI; home inventory app AI video walkthrough; assistant or glasses visual memory of object locations; MobiCom/MobiSys/SenSys/IMWUT object memory; robots and objects inside containers; persistence modelling in semi-static environments; Ego4D episodic memory 3D; object-level change detection between rescans; ARCore ML Kit object anchors; teachable object recognisers; lifelogging lost-object finders; CHI/UIST AR memory assistants; multi-day egocentric object-location benchmarks. No query returned a phone-only system matching the idea.

## Implications for the paper

- Lead with the evidence-maintenance problem in the opportunistic phone setting; present the on-device inventory as the artifact that makes it testable, with measurements.
- Use SemanticXR's "impractical on-device" statement as the foil for the systems contribution, and state exactly what is relaxed (mapping rate, model class, real-time requirement).
- Use Find Hub's manual remembered items as the shipped baseline, and MemPal's "which drawer" quote plus DynaMem's and PredictiveGraphs' stated limits as motivation for containers.
- Before submission: read MemoVision; confirm venues for ConceptGraphs, Find My Things and DrawerFinder (IUI) from the publisher records; re-check every quote.
