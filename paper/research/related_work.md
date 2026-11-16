# Related work, with verified references

Compiled 2026-11-16. BibTeX for every entry below is in `../refs.bib` (111 entries: 108 papers and preprints, 3 vendor documentation pages).

## How to read this file

- **Verification.** Every entry's authors, title, venue, year and pages were taken from a registry record fetched during this session: Crossref (`api.crossref.org/works/<DOI>`), the arXiv API, a PMLR page, or the NeurIPS proceedings page. The URL after "Confirmed:" is the record that was fetched. Nothing bibliographic was filled in from memory.
- **Descriptions.** Where a description gives specifics (numbers, mechanisms), they come from the paper's abstract or full text opened in this session. Entries tagged **[title only]** were confirmed bibliographically but their abstract was not opened, so the description says only what the title says. Read those before leaning on them.
- **Preprints.** Entries whose venue reads `arXiv:...` are not peer reviewed unless a note says otherwise. Several of the closest works are 2026 preprints.
- **Not opened directly.** ACM Digital Library and IEEE Xplore pages refuse automated fetches, so ACM and IEEE papers were confirmed through Crossref and, for abstracts, OpenAlex. DBLP also refused automated fetches and was not used.

## The headline finding

**I did not find a paper that already does what this paper does**: a phone AR object-memory system that decides per query, from signals available before any new capture, whether to verify a remembered location, and that counts the phone's own localization drift as a reason for doubt.

The parts are not new, and the paper should say so:

1. **"A remembered location is a claim about the past."** This is already how two human-facing systems word their answers. MemPal answers "Your [object] was last seen at [time] in the [location]" (`maniar2025mempal`). Linguistic Trajectory Encoding attaches High / Medium / Low confidence labels and, for stale entries, answers "last seen at [location] at [time], current location uncertain" (`xie2026linguistic`, Appendix F). EGOSTREAM defines an Answer Validity Window for exactly this reason (`forte2026egostream`). None of the three decides whether to go and check.
2. **"Use age and how often the object moves to decide when to look again."** This is established in robotics: temporal persistence models for object search (`toris2017temporal`), stationarity scores that decay while an object is out of view (`bogenberger2026glasses`), and a July 2026 preprint that poses budgeted re-perception with a per-object Poisson move rate and an age-of-information style index (`ghosh2026memory`).
3. **"Unverified stale spatial memory is harmful."** An August 2026 preprint shows a memory-trusting agent failing more than twice as often as the same agent with no memory (`sun2026memory`).

What remains distinctive is the combination: the decision is made per query on a phone, a human pays the cost of each capture, the observer's own frame can drift, and the outcome measured is wrong location claims shown to a person against captures per query. Section "Closest prior work" spells out the differences.

## 1. Finding lost or misplaced objects

### Camera- and AR-based finders

- **GO-Finder: A Registration-Free Wearable System for Assisting Users in Finding Lost Objects via Hand-Held Object Discovery.** Yagi et al. 26th International Conference on Intelligent User Interfaces, 2021. `yagi2021gofinder`. Confirmed: <https://api.crossref.org/works/10.1145/3397481.3450664>
  A wearable camera system that needs no object registration: it discovers hand-held objects automatically, groups them into a visual timeline, and lets the user pull up the last appearance of an object in a phone app. A user study found better accuracy and lower mental load. It shows the last sighting as an image and has no notion of whether that sighting still holds. Journal version: `yagi2022gofinder`.
- **MemPal: Leveraging Multimodal AI and LLMs for Voice-Activated Object Retrieval in Homes of Older Adults.** Maniar et al. Proceedings of the 30th International Conference on Intelligent User Interfaces, 2025. `maniar2025mempal`. Confirmed: <https://api.crossref.org/works/10.1145/3708359.3712151>
  A wearable camera and multimodal LLM keep a text diary of the wearer's activities and answer voice queries about misplaced objects. Tested with 15 older adults in their own homes, it improved object finding over no aid. Its answer template is "Your [object] was last seen at [time] in the [location] near [background]", so the wording is already historical, but the system never checks. Its error categories (incorrect location, object misidentified, no object detected) cover perception errors, not objects moved after the last sighting.
- **FMT: A Wearable Camera-Based Object Tracking Memory Aid for Older Adults.** Li et al. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 2019. `li2019fmt`. Confirmed: <https://api.crossref.org/works/10.1145/3351253>
  A body-worn camera records short clips whenever a fiducial marker placed near an object enters view, so older adults can check whether they completed an action and what state an object is in. It captured direct interactions at an average rate of 75.6% across participants. An IMWUT example of marker-triggered capture and of the participant evaluation that venue expects.
- **VisionAId: An Offline-First Multimodal Android Assistant for People with Visual Impairment, Featuring Personalized Object Retrieval.** Florea and Spînu. arXiv:2607.02371, 2026. `florea2026visionaid`. Confirmed: <https://export.arxiv.org/api/query?id_list=2607.02371>
  Preprint. An Android phone app running six on-device models; the user photographs a personal object from several angles and the app later finds that instance and guides the user with AR markers, spatial audio and haptics. It is the always-verify end of the design space on the same platform class: no memory of location, a live search every time.
- **Where is my Phone? Personal Object Retrieval from Egocentric Images.** Reyes et al. Proceedings of the first Workshop on Lifelogging Tools and Applications, 2016. `reyes2016phone`. Confirmed: <https://api.crossref.org/works/10.1145/2983576.2983582>
  A retrieval pipeline for finding the last appearance of a personal object in a large set of wearable-camera images; results are re-ranked by timestamp to favour later detections. An early statement of the "last appearance" formulation.
- **I’m Here!: A Wearable Object Remembrance Support System.** Ueoka et al. Human-Computer Interaction with Mobile Devices and Services, 2003. `ueoka2003imhere`. Confirmed: <https://api.crossref.org/works/10.1007/978-3-540-45233-1_40>
  **[title only]** An early wearable system for remembering where objects were.
- **Representing indoor location of objects on wearable computers with head-mounted displays.** Funk et al. Proceedings of the 5th Augmented Human International Conference, 2014. `funk2014representing`. Confirmed: <https://api.crossref.org/works/10.1145/2582051.2582069>
  Starts from the premise that wearable cameras can recognise known objects and store their indoor location, and asks how to show that location on a head-mounted display and guide the user to it. Reports a prototype.
- **Traceband: locating missing items by visual remembrance.** Tavakolizadeh et al. Proceedings of the adjunct publication of the 27th annual ACM symposium on User interface software and technology - UIST'14 Adjunct, 2014. `tavakolizadeh2014traceband`. Confirmed: <https://api.crossref.org/works/10.1145/2658779.2658800>
  A bracelet that keeps track of commonly used objects the wearer interacts with; missing items are looked up in a web portal. Two-page UIST adjunct paper.

### Assistive object finding and teachable recognizers

- **Understanding Personalized Accessibility through Teachable AI: Designing and Evaluating Find My Things for People who are Blind or Low Vision.** Morrison et al. The 25th International ACM SIGACCESS Conference on Computers and Accessibility, 2023. `morrison2023findmythings`. Confirmed: <https://api.crossref.org/works/10.1145/3597638.3608395>
  Presents Find My Things, an application that blind or low-vision people teach to find their own belongings, and uses it to study how much teaching effort users will trade for personalization. Demo abstract: `wen2024findmythings`.
- **People with Visual Impairment Training Personal Object Recognizers: Feasibility and Challenges.** Kacorri et al. Proceedings of the 2017 CHI Conference on Human Factors in Computing Systems, 2017. `kacorri2017people`. Confirmed: <https://api.crossref.org/works/10.1145/3025453.3025899>
  People with visual impairment train a personal object recognizer in a mobile app from a few snapshots of their own objects. Studies the feasibility and the challenges of the approach.
- **ReCog: Supporting Blind People in Recognizing Personal Objects.** Ahmetovic et al. Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems, 2020. `ahmetovic2020recog`. Confirmed: <https://api.crossref.org/works/10.1145/3313831.3376143>
  A mobile app with camera-aiming guidance so blind users can frame their own objects when training a recognizer.
- **Blind Users Accessing Their Training Images in Teachable Object Recognizers.** Hong et al. Proceedings of the 24th International ACM SIGACCESS Conference on Computers and Accessibility, 2022. `hong2022blind`. Confirmed: <https://api.crossref.org/works/10.1145/3517428.3544824>
  Real-time descriptors that tell a blind user whether a training photo is usable (for example, the object is cropped or too small), replacing the visual inspection that teachable recognizers assume.
- **ORBIT: A Real-World Few-Shot Dataset for Teachable Object Recognition.** Massiceti et al. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2021. `massiceti2021orbit`. Confirmed: <https://api.crossref.org/works/10.1109/iccv48922.2021.01064>
  A few-shot recognition dataset motivated by personalization, built because standard few-shot benchmarks lack the variation that real use has.
- **Where's my stuff? design and evaluation of a mobile system for locating lost items for the visually impaired.** Kientz et al. Proceedings of the 8th international ACM SIGACCESS conference on Computers and accessibility, 2006. `kientz2006stuff`. Confirmed: <https://api.crossref.org/works/10.1145/1168987.1169006>
  FETCH, a mobile system that lets visually impaired users track and locate frequently lost objects around the house using devices they already own. An early user-centred design and evaluation of a home object finder.

### Tag-based finders

- **Who Can Find My Devices? Security and Privacy of Apple’s Crowd-Sourced Bluetooth Location Tracking System.** Heinrich et al. Proceedings on Privacy Enhancing Technologies, 2021. `heinrich2021findmy`. Confirmed: <https://api.crossref.org/works/10.2478/popets-2021-0045>
  Analysis of Apple's offline finding network, in which other people's devices detect a lost device over Bluetooth and report an approximate location to the owner. The baseline alternative to camera memory: it reports a location sensed now, and depends on tags and on third-party devices.
- **Minding the Billions: Ultra-wideband Localization for Deployed RFID Tags.** Ma et al. Proceedings of the 23rd Annual International Conference on Mobile Computing and Networking, 2017. `ma2017minding`. Confirmed: <https://api.crossref.org/works/10.1145/3117811.3117833>
  Ultra-wideband localization for RFID tags that are already deployed, where earlier systems either restricted the environment or needed new tag hardware. Fine-grained tag localization still needs reader infrastructure and a tag on every object.
- **Ferret: RFID Localization for Pervasive Multimedia.** Liu et al. UbiComp 2006: Ubiquitous Computing, 2006. `liu2006ferret`. Confirmed: <https://api.crossref.org/works/10.1007/11853565_25>
  **[title only]** RFID localization for pervasive multimedia.
- **MAX: human-centric search of the physical world.** Yap et al. Proceedings of the 3rd international conference on Embedded networked sensor systems, 2005. `yap2005max`. Confirmed: <https://api.crossref.org/works/10.1145/1098918.1098937>
  Lets people search for and locate objects when they need them instead of organising them beforehand, and reports location relative to identifiable landmarks ("on the dining table") instead of coordinates. A precedent for landmark- or room-level location answers.
- **Objects Calling Home: Locating Objects Using Mobile Phones.** Frank et al. Pervasive Computing, 2007. `frank2007objects`. Confirmed: <https://api.crossref.org/works/10.1007/978-3-540-72037-9_21>
  **[title only]** Locating objects using mobile phones.

**Why this area matters here.** Camera-based finders all answer from the last sighting. The best of them word the answer as "last seen", and none decides whether to check. Tag-based finders sense the object's position when asked, at the price of a tag on every object and, for crowd-sourced networks, a finder device nearby.

## 2. Memory augmentation, lifelogging and egocentric episodic memory

- **SenseCam: A Retrospective Memory Aid.** Hodges et al. UbiComp 2006: Ubiquitous Computing, 2006. `hodges2006sensecam`. Confirmed: <https://api.crossref.org/works/10.1007/11853565_11>
  **[title only]** The wearable camera as a retrospective memory aid.
- **Ego4D: Around the World in 3,000 Hours of Egocentric Video.** Grauman et al. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. `grauman2022ego4d`. Confirmed: <https://api.crossref.org/works/10.1109/cvpr52688.2022.01842>
  A dataset and benchmark suite of 3,670 hours of daily-life egocentric video from 931 camera wearers. Its episodic memory benchmark includes Visual Queries with 3D Localization (as described in `mai2023egoloc`).
- **EgoLoc: Revisiting 3D Object Localization from Egocentric Videos with Visual Queries.** Mai et al. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023. `mai2023egoloc`. Confirmed: <https://api.crossref.org/works/10.1109/iccv51070.2023.00011>
  Visual Queries with 3D Localization: given an egocentric clip and an image crop of an object, find the object's 3D position relative to the camera pose of the query frame. This is the vision community's formal version of "where did I last see it". The target is where the object was in the recording.
- **Instance Tracking in 3D Scenes from Egocentric Videos.** Zhao et al. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. `zhao2024instance`. Confirmed: <https://api.crossref.org/works/10.1109/cvpr52733.2024.02071>
  Motivated by AR/VR devices that could assist a user by recalling the 3D locations of objects of interest; introduces a benchmark for tracking object instances in 3D from egocentric video.
- **Where did I leave my keys? — Episodic-Memory-Based Question Answering on Egocentric Videos.** Barmann and Waibel. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2022. `barmann2022keys`. Confirmed: <https://api.crossref.org/works/10.1109/cvprw56347.2022.00162>
  Episodic-memory question answering on egocentric video for systems that cannot keep the raw input, as in a real deployment with limited storage.
- **OpenEQA: Embodied Question Answering in the Era of Foundation Models.** Majumdar et al. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. `majumdar2024openeqa`. Confirmed: <https://api.crossref.org/works/10.1109/cvpr52733.2024.01560>
  Formulates embodied question answering with two routes to an answer: episodic memory, "exemplified by agents on smart glasses", or active exploration, as with mobile robots. Those are the two options this paper chooses between per query.
- **EgoLife: Towards Egocentric Life Assistant.** Yang et al. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. `yang2025egolife`. Confirmed: <https://api.crossref.org/works/10.1109/cvpr52734.2025.02690>
  Six participants lived together for a week recording their daily activities with wearable glasses, as the data foundation for an egocentric life assistant. `xie2026linguistic` builds its spatial memory benchmark on these recordings.
- **EGOSTREAM: A Diagnostic Benchmark for Streaming Episodic Memory in Egocentric Vision.** Forte et al. arXiv:2605.31557, 2026. `forte2026egostream`. Confirmed: <https://export.arxiv.org/api/query?id_list=2605.31557>
  Preprint. A streaming episodic memory benchmark that introduces the Answer Validity Window: the time span over which an answer stays valid as the scene changes, used to separate model forgetting from world change. Direct support for treating answers as time-limited.
- **Linguistic Trajectory Encoding for Efficient Long-Horizon Spatial Memory in Embodied Agents.** Xie et al. arXiv:2609.04802 (arXiv comment field: accepted at NeurIPS 2026), 2026. `xie2026linguistic`. Confirmed: <https://export.arxiv.org/api/query?id_list=2609.04802>
  Preprint, accepted at NeurIPS 2026 according to its arXiv comment. Per-object timelines compressed into language plus sparse spatial anchors; periods without reliable observation are anchored to the last seen location. Each answer carries a High / Medium / Low confidence label, and a stale entry is reported as "last seen at [location] at [time], current location uncertain". The paper states that a single ego camera cannot see changes that happen off-frame. It labels staleness but does not act on it.
- **Memoro: Using Large Language Models to Realize a Concise Interface for Real-Time Memory Augmentation.** Zulfikar et al. Proceedings of the CHI Conference on Human Factors in Computing Systems, 2024. `zulfikar2024memoro`. Confirmed: <https://api.crossref.org/works/10.1145/3613904.3642450>
  A wearable audio memory assistant in which an LLM infers what the user needs to recall and offers minimal suggestions. Representative of LLM memory assistants; no spatial component.
- **OmniQuery: Contextually Augmenting Captured Multimodal Memories to Enable Personal Question Answering.** Li et al. Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, 2025. `li2025omniquery`. Confirmed: <https://api.crossref.org/works/10.1145/3706598.3713448>
  **[title only]** Augmenting captured multimodal memories with context to answer personal questions.
- **Encode-Store-Retrieve: Augmenting Human Memory through Language-Encoded Egocentric Perception.** Shen et al. 2024 IEEE International Symposium on Mixed and Augmented Reality (ISMAR), 2024. `shen2024encode`. Confirmed: <https://api.crossref.org/works/10.1109/ismar62088.2024.00108>
  Lifelogging on an AR headset, with egocentric video encoded as language so that it can be stored and queried.
- **Vinci: A Real-time Smart Assistant Based on Egocentric Vision-language Model for Portable Devices.** Huang et al. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 2025. `huang2025vinci`. Confirmed: <https://api.crossref.org/works/10.1145/3749513>
  A real-time egocentric vision-language assistant with a memory module, built to run across portable devices including smartphones. A recent IMWUT system paper in this space (33 pages).
- **Gaze Archive: Enhancing Human Memory through Active Visual Logging on Smart Glasses.** Ren and Lu. arXiv:2511.16214, 2025. `ren2025gaze`. Confirmed: <https://export.arxiv.org/api/query?id_list=2511.16214>
  Preprint. Gaze-triggered visual logging on smart glasses, with lab and real-world user studies.
- **Resource-Constrained Lifelogging: A Utility-Based Replacement Policy for Video Understanding.** Chang and Hsu. Proceedings of the 4th ACM Workshop on Mobile Immersive Computing, Networking, and Systems, 2026. `chang2026lifelogging`. Confirmed: <https://api.crossref.org/works/10.1145/3842656.3843830>
  ImmerCom 2026. An edge-offloaded lifelogging system that decides which lifelog items to keep under a storage budget. Evidence that resource-aware lifelogging policies fit that workshop.

**Why this area matters here.** Benchmarks score recall of what was recorded. Two 2026 works (`forte2026egostream`, `xie2026linguistic`) now state explicitly that an answer can expire. Neither closes the loop with a new observation.

## 3. Persistent and shared AR: anchors, relocalization, drift, markers

- **Here To Stay: A Quantitative Comparison of Virtual Object Stability in Markerless Mobile AR.** Scargill et al. 2022 2nd International Workshop on Cyber-Physical-Human System Design and Implementation (CPHS), 2022. `scargill2022here`. Confirmed: <https://api.crossref.org/works/10.1109/cphs56133.2022.9804545>
  Measures virtual object stability in markerless smartphone AR across six user actions and five phone configurations. Position errors exceeded 10 cm in all but the simplest settings, even on the best phones. Direct evidence that the phone's own frame is a source of location error.
- **Here To Stay: Measuring Hologram Stability in Markerless Smartphone Augmented Reality.** Scargill et al. arXiv:2109.14757, 2021. `scargill2021here`. Confirmed: <https://export.arxiv.org/api/query?id_list=2109.14757>
  Preprint. A method for measuring positional error of virtual content on commercial markerless smartphone AR platforms.
- **Localization Limitations of ARCore, ARKit, and Hololens in Dynamic Large-scale Industry Environments.** Feigl et al. Proceedings of the 15th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications, 2020. `feigl2020localization`. Confirmed: <https://api.crossref.org/works/10.5220/0008989903070318>
  ARCore, ARKit and HoloLens in a 1,600 m2 industrial hall against a sub-millimetre optical reference: an average error of about 17 m per 120 m walked, with scaling error up to 14.4 cm per metre. Supports using distance walked since the last fix as a drift signal.
- **MobileARLoc: On-device Robust Absolute Localisation for Pervasive Markerless Mobile AR.** Liu et al. 2024 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops), 2024. `liu2024mobilearloc`. Confirmed: <https://api.crossref.org/works/10.1109/percomworkshops59983.2024.10503320>
  On-device absolute pose regression combined with visual-inertial odometry: the odometry picks out reliable regressor outputs, which are then used to correct odometry drift. A phone-side precedent for correcting drift only when a trustworthy fix is available.
- **Multi-user augmented reality with communication efficient and spatially consistent virtual objects.** Ran et al. Proceedings of the 16th International Conference on emerging Networking EXperiments and Technologies, 2020. `ran2020multiuser`. Confirmed: <https://api.crossref.org/works/10.1145/3386367.3431312>
  Studies how multi-user AR apps communicate to keep virtual objects in the same place for co-located users. Workshop precursor: `ran2019sharear`.
- **Collaborative Augmented Reality on Smartphones via Life-long City-scale Maps.** Platinsky et al. 2020 IEEE International Symposium on Mixed and Augmented Reality (ISMAR), 2020. `platinsky2020collaborative`. Confirmed: <https://api.crossref.org/works/10.1109/ismar50242.2020.00081>
  An end-to-end production system for city-scale shared AR on phones, built on an experience-based mapping framework designed to cope with map updates among other production requirements.
- **LaMAR: Benchmarking Localization and Mapping for Augmented Reality.** Sarlin et al. Computer Vision – ECCV 2022, 2022. `sarlin2022lamar`. Confirmed: <https://api.crossref.org/works/10.1007/978-3-031-20071-7_40>
  **[title only]** A benchmark for localization and mapping for AR.
- **Beyond Controlled Environments: 3D Camera Re-localization in Changing Indoor Scenes.** Wald et al. Computer Vision – ECCV 2020, 2020. `wald2020beyond`. Confirmed: <https://api.crossref.org/works/10.1007/978-3-030-58571-6_28>
  **[title only]** Camera re-localization when the indoor scene has changed since mapping.
- **AprilTag: A robust and flexible visual fiducial system.** Olson. 2011 IEEE International Conference on Robotics and Automation, 2011. `olson2011apriltag`. Confirmed: <https://api.crossref.org/works/10.1109/icra.2011.5979561>
  **[title only]** The AprilTag fiducial system.
- **Automatic generation and detection of highly reliable fiducial markers under occlusion.** Garrido-Jurado et al. Pattern Recognition, 2014. `garrido2014automatic`. Confirmed: <https://api.crossref.org/works/10.1016/j.patcog.2014.01.005>
  **[title only]** A method for generating and detecting fiducial markers that tolerates occlusion.
- **Comparison of Marker-Based and Markerless AR: A Case Study of An Indoor Decoration System.** Cheng et al. Lean and Computing in Construction Congress - Volume 1: Proceedings of the Joint Conference on Computing in Construction, 2017. `cheng2017comparison`. Confirmed: <https://api.crossref.org/works/10.24928/jc3-2017/0231>
  Compares marker-based and markerless AR through an indoor decoration case study, arguing that each suits different situations. I found no rigorous measurement study comparing marker-based and markerless re-registration on phones; see "Could not find".
- **Cloud Anchors allow different users to share AR experiences.** Google. ARCore documentation, Google for Developers, 2026. `google2026cloudanchors`. Confirmed: <https://developers.google.com/ar/develop/cloud-anchors>
  Vendor documentation. "Cloud Anchors are anchors that are hosted on the ARCore API cloud endpoint"; hosting uploads the anchor's data and "requires a working Internet connection".
- **Cloud Anchors developer guide for Android (Kotlin/Java).** Google. ARCore documentation, Google for Developers, 2026. `google2026cloudanchorsguide`. Confirmed: <https://developers.google.com/ar/develop/java/cloud-anchors/developer-guide>
  Vendor documentation. "When using an API Key, Cloud Anchors have a maximum lifetime of 24 hours"; longer lifetimes need keyless authorization. Hosting uploads "visual data, device poses, and the anchor pose to the ARCore API".
- **Create anchors with ARCore for Jetpack XR.** Google. Android XR documentation, Android Developers, 2026. `google2026jetpackxranchors`. Confirmed: <https://developer.android.com/develop/xr/jetpack-xr-sdk/arcore/anchors>
  Vendor documentation. ARCore for Jetpack XR does offer local persistence: `AnchorPersistenceMode.LOCAL` is used "to persist and load anchors from local storage", and a persisted anchor's position is kept in the app's "private app data". The page lists XR headsets and wired XR glasses as the applicable devices. See "Framing implications", item 5.
- **Towards Persistent Dynamic Scene Understanding for Immersive Computing.** Do et al. Proceedings of the 4th ACM Workshop on Mobile Immersive Computing, Networking, and Systems, 2026. `do2026persistent`. Confirmed: <https://api.crossref.org/works/10.1145/3842656.3843838>
  ImmerCom 2026 vision paper. Argues that scene graphs for immersive computing are rebuilt per session and forgotten, and proposes a persistent representation with temporal memory and stable object identities across visits. The nearest paper in the target workshop's own proceedings.
- **An Empirical Study of Depth Inference Scheduling for Dynamic Anchoring in Mobile AR.** Huang et al. Proceedings of the 4th ACM Workshop on Mobile Immersive Computing, Networking, and Systems, 2026. `huang2026empirical`. Confirmed: <https://api.crossref.org/works/10.1145/3842656.3843833>
  **[title only]** ImmerCom 2026. Scheduling of depth inference for anchoring in mobile AR.

**Why this area matters here.** Measurement studies establish that phone AR frames drift by centimetres to metres depending on motion and distance. That is the second way a stored location becomes wrong, and the object-memory literature in areas 1, 2 and 4 does not model it.

## 4. Objects that move: re-localization, change detection, robot maps

### Vision: re-localizing instances and detecting change

- **RIO: 3D Object Instance Re-Localization in Changing Indoor Environments.** Wald et al. 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019. `wald2019rio`. Confirmed: <https://api.crossref.org/works/10.1109/iccv.2019.00775>
  Defines 3D object instance re-localization: given objects in one RGB-D scan, estimate their 6-DoF poses in a later scan of the same room. Introduces the 3RScan dataset.
- **Rescan: Inductive Instance Segmentation for Indoor RGBD Scans.** Halber et al. 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019. `halber2019rescan`. Confirmed: <https://api.crossref.org/works/10.1109/iccv.2019.00263>
  For interiors scanned repeatedly at sparse intervals, infers a temporal model of the scene with instance labels, using the model from past scans to interpret the next one.
- **Objects Can Move: 3D Change Detection by Geometric Transformation Consistency.** Adam et al. Computer Vision – ECCV 2022, 2022. `adam2022objects`. Confirmed: <https://api.crossref.org/works/10.1007/978-3-031-19827-4_7>
  **[title only]** 3D change detection through geometric transformation consistency.
- **Living Scenes: Multi-object Relocalization and Reconstruction in Changing 3D Environments.** Zhu et al. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. `zhu2024living`. Confirmed: <https://api.crossref.org/works/10.1109/cvpr52733.2024.02646>
  Multi-object relocalization and reconstruction for long-term change seen through sparse observations, where most earlier work tracked short-term change from dense observation.
- **Lost & Found: Tracking Changes From Egocentric Observations in 3D Dynamic Scene Graphs.** Behrens et al. IEEE Robotics and Automation Letters, 2025. `behrens2025lost`. Confirmed: <https://api.crossref.org/works/10.1109/lra.2025.3544518>
  Tracks the 6-DoF pose of an object while a person moves it, from egocentric video with hand and camera poses, and updates a scene graph online. The paper notes that rescanning after every change is costly and misses objects put away in drawers. It tracks moves it sees; it has nothing to say about moves it does not see.

### Robot maps that model change over time

- **Temporal persistence modeling for object search.** Toris and Chernova. 2017 IEEE International Conference on Robotics and Automation (ICRA), 2017. `toris2017temporal`. Confirmed: <https://api.crossref.org/works/10.1109/icra.2017.7989365>
  See "Closest prior work".
- **Towards lifelong feature-based mapping in semi-static environments.** Rosen et al. 2016 IEEE International Conference on Robotics and Automation (ICRA), 2016. `rosen2016towards`. Confirmed: <https://api.crossref.org/works/10.1109/icra.2016.7487237>
  The persistence filter: a recursive Bayesian estimator giving, at any moment, a belief that a map feature still exists, from a generative survival model. The cleanest formal statement that map entries have a time-dependent probability of still being true.
- **FreMEn: Frequency Map Enhancement for Long-Term Mobile Robot Autonomy in Changing Environments.** Krajnik et al. IEEE Transactions on Robotics, 2017. `krajnik2017fremen`. Confirmed: <https://api.crossref.org/works/10.1109/tro.2017.2665664>
  Models environment states as periodic processes through their frequency spectra, so a robot can predict the state at a future time. Evaluated over months to years.
- **Where's Waldo at Time t? Using Spatio-Temporal Models for Mobile Robot Search.** Krajnik et al. 2015 IEEE International Conference on Robotics and Automation (ICRA), 2015. `krajnik2015waldo`. Confirmed: <https://api.crossref.org/works/10.1109/icra.2015.7139481>
  Applies the same spectral model to object search: the probability that an object is at a place is a function of time. Search time fell by 25% to 65% against maps that ignore dynamics.
- **3D VSG: Long-term Semantic Scene Change Prediction through 3D Variable Scene Graphs.** Looper et al. 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023. `looper2023vsg`. Confirmed: <https://api.crossref.org/works/10.1109/icra48891.2023.10161212>
  Adds a variability attribute to 3D scene graphs: the likelihood that an object will change position or state in the long term, estimated by a supervised model and evaluated on 3RScan. Using it for active change detection sped the task up by 66% against a change-unaware planner. A learned prior for "how often does this kind of object move".
- **Where Did I Leave My Glasses? Open-Vocabulary Semantic Exploration in Real-World Semi-Static Environments.** Bogenberger et al. IEEE Robotics and Automation Letters, 2026. `bogenberger2026glasses`. Confirmed: <https://api.crossref.org/works/10.1109/lra.2026.3656790>
  See "Closest prior work".
- **Modeling Dynamic Environments with Scene Graph Memory.** Kurenkov et al. Proceedings of the 40th International Conference on Machine Learning, 2023. `kurenkov2023modeling`. Confirmed: <https://proceedings.mlr.press/v202/kurenkov23a.html>
  Poses object search in a changing household as link prediction on a partially observed dynamic graph, with a scene graph memory that accumulates the agent's observations of the changing scene.
- **Proactive Robot Assistance via Spatio-Temporal Object Modeling.** Patel and Chernova. Conference on Robot Learning, 2023. `patel2023proactive`. Confirmed: <https://proceedings.mlr.press/v205/patel23a.html>
  **[title only]** Spatio-temporal object modeling for proactive robot assistance.
- **POCD: Probabilistic Object-Level Change Detection and Volumetric Mapping in Semi-Static Scenes.** Qian et al. Robotics: Science and Systems XVIII, 2022. `qian2022pocd`. Confirmed: <https://api.crossref.org/works/10.15607/rss.2022.xviii.013>
  Object-level change detection for robots that traverse the same space repeatedly, motivated by the damage undetected changes do to map quality and localization.
- **Panoptic Multi-TSDFs: a Flexible Representation for Online Multi-resolution Volumetric Mapping and Long-term Dynamic Scene Consistency.** Schmid et al. 2022 International Conference on Robotics and Automation (ICRA), 2022. `schmid2022panoptic`. Confirmed: <https://api.crossref.org/works/10.1109/icra46639.2022.9811877>
  A multi-resolution volumetric map representation for environments that are subject to long-term change.
- **Khronos: A Unified Approach for Spatio-Temporal Metric-Semantic SLAM in Dynamic Environments.** Schmid et al. Robotics: Science and Systems XX, 2024. `schmid2024khronos`. Confirmed: <https://api.crossref.org/works/10.15607/rss.2024.xx.081>
  Argues that dynamic SLAM has concentrated on robot pose and neglected dense spatio-temporal representations of the scene, and builds one.
- **LTC-Mapping, Enhancing Long-Term Consistency of Object-Oriented Semantic Maps in Robotics.** Matez-Bandera et al. Sensors, 2022. `matez2022ltc`. Confirmed: <https://api.crossref.org/works/10.3390/s22145308>
  Object-oriented semantic maps that stay consistent over long operation, focusing on duplicate object instances and on dynamic scenes.
- **Detection and Tracking of General Movable Objects in Large Three-Dimensional Maps.** Bore et al. IEEE Transactions on Robotics, 2019. `bore2019detection`. Confirmed: <https://api.crossref.org/works/10.1109/tro.2018.2876111>
  Tracks objects with semi-static dynamics when a robot sees only part of a large environment at a time, so that objects "might be moved when the robot is not there". This is the present paper's core problem, stated for a robot.
- **Long-term object search using incremental scene graph updating.** Zhou et al. Robotica, 2023. `zhou2023longterm`. Confirmed: <https://api.crossref.org/works/10.1017/s0263574722001205>
  Household robot search for movable objects such as cups, combining a navigation map, a semantic map and a scene graph that is updated incrementally.
- **Searching objects in large-scale indoor environments: A decision-theoretic approach.** Kunze et al. 2012 IEEE International Conference on Robotics and Automation, 2012. `kunze2012searching`. Confirmed: <https://api.crossref.org/works/10.1109/icra.2012.6224965>
  Decision-theoretic object search in large environments using probabilistic environment models.
- **Artificial Intelligence for Long-Term Robot Autonomy: A Survey.** Kunze et al. IEEE Robotics and Automation Letters, 2018. `kunze2018artificial`. Confirmed: <https://api.crossref.org/works/10.1109/lra.2018.2860628>
  **[title only]** Survey of AI for long-term robot autonomy.
- **OASIS-Map: Object-Level Change Detection in Multi-Session Mapping using Semantic Correspondence Matching.** Oh et al. arXiv:2607.14899, 2026. `oh2026oasis`. Confirmed: <https://export.arxiv.org/api/query?id_list=2607.14899>
  Preprint. Multi-session mapping that associates objects across revisits; reports 0.667 F1 on moved-object association in 3RScan.
- **Build Once, Monitor Continuously: Persistent Semantic Mapping via Autonomous Exploration and Open-Vocabulary Object Updates.** Allu et al. arXiv:2409.15493, 2024. `allu2024build`. Confirmed: <https://export.arxiv.org/api/query?id_list=2409.15493>
  Preprint. Separates one-off geometric mapping from a cheap semantic stage repeated on each revisit.

### Memory that may be stale: verify, re-observe, or trust

- **Searching in Space and Time: Unified Memory-Action Loops for Open-World Object Retrieval.** Chen et al. arXiv:2511.14004, 2025. `chen2025searching`. Confirmed: <https://export.arxiv.org/api/query?id_list=2511.14004>
  See "Closest prior work".
- **Memory for Attention: Language-Conditioned Re-Perception with a Vision--Language--Motion Map.** Ghosh. arXiv:2607.23797, 2026. `ghosh2026memory`. Confirmed: <https://export.arxiv.org/api/query?id_list=2607.23797>
  See "Closest prior work".
- **When Memory Lies: An Empirical Study of Spatial Memory Staleness in VLM Agents.** Sun and Zhang. arXiv:2608.04574, 2026. `sun2026memory`. Confirmed: <https://export.arxiv.org/api/query?id_list=2608.04574>
  See "Closest prior work".
- **SPARK: Self-Correcting Embodied Memory via Physical Re-Observation.** Kurdydyk et al. arXiv:2611.00344, 2026. `kurdydyk2026spark`. Confirmed: <https://export.arxiv.org/api/query?id_list=2611.00344>
  Preprint, 30 October 2026. A critic flags suspect regions of a robot's language-queryable memory, a planner schedules targeted revisits, and a selective merge corrects the memory. Targeted revisits beat untargeted ones under the same budget. Same instinct as selective verification, applied between episodes on outdoor simulated maps.
- **GLOVE: Global Verifier for LLM Memory-Environment Realignment.** Yin and Du. arXiv:2601.19249, 2026. `yin2026glove`. Confirmed: <https://export.arxiv.org/api/query?id_list=2601.19249>
  Preprint. Verifies LLM agent memory by actively probing the environment and comparing retrieved memories with fresh observations, on web navigation, planning and control tasks with injected drift.
- **MemoGuard: An Adaptive Runtime for Guarding Against Memory Traps in Communication-Limited Robot Navigation.** Bhattacharjya et al. arXiv:2607.15589 (arXiv comment field: accepted at IEEE/ACM ESWEEK (CODES) 2026, to appear in IEEE Embedded Systems Letters), 2026. `bhattacharjya2026memoguard`. Confirmed: <https://export.arxiv.org/api/query?id_list=2607.15589>
  Preprint, accepted at ESWEEK (CODES) 2026 according to its arXiv comment. Validates a retrieved episodic memory against explicit conditions before reuse and falls back to costly reasoning only when validation fails; reports both safety violations avoided and fallback calls avoided. A cost-versus-safety evaluation with the same shape as this paper's.

**Why this area matters here.** This is where the paper's decision rule has its real ancestors. Robotics has modelled "probability that the entry is still true, as a function of time and object type" since 2015 to 2017 and has used it to choose where to look. The 2026 preprints add the vocabulary of stale memory and of verification on a budget.

## 5. Deciding when to sense

### Gating an expensive operation on cheap signals

- **Glimpse: Continuous, Real-Time Object Recognition on Mobile Devices.** Chen et al. Proceedings of the 13th ACM Conference on Embedded Networked Sensor Systems, 2015. `chen2015glimpse`. Confirmed: <https://api.crossref.org/works/10.1145/2809695.2809711>
  Continuous object recognition on phones and Glass. The device tracks objects locally using stale hints from the server and sends only selected trigger frames for full recognition. The canonical mobile example of spending the expensive operation only when a cheap signal says the cached answer is no longer good.
- **Frugal following: power thrifty object detection and tracking for mobile augmented reality.** Apicharttrisorn et al. Proceedings of the 17th Conference on Embedded Networked Sensor Systems, 2019. `apicharttrisorn2019frugal`. Confirmed: <https://api.crossref.org/works/10.1145/3356250.3360044>
  For mobile AR: run the detection network only when needed (new objects, or large appearance change) and track cheaply in between. Energy fell by up to 73.3% against running the best network continuously, with small accuracy loss. The closest systems precedent for a selective policy compared against always-on and infrequent baselines.
- **A framework of energy efficient mobile sensing for automatic user state recognition.** Wang et al. Proceedings of the 7th international conference on Mobile systems, applications, and services, 2009. `wang2009framework`. Confirmed: <https://api.crossref.org/works/10.1145/1555816.1555835>
  Hierarchical sensor management on a phone: power only a minimum set of sensors, with duty cycling, to recognise the user's state and detect transitions. Battery life rose by more than 75% in a one-week, ten-user study.
- **MCDNN: An Approximation-Based Execution Framework for Deep Stream Processing Under Resource Constraints.** Han et al. Proceedings of the 14th Annual International Conference on Mobile Systems, Applications, and Services, 2016. `han2016mcdnn`. Confirmed: <https://api.crossref.org/works/10.1145/2906388.2906396>
  Poses "Approximate Model Scheduling": serving a stream of vision requests on cloud-backed mobile devices without straining battery, wireless data and cloud cost budgets.
- **NoScope: optimizing neural network queries over video at scale.** Kang et al. Proceedings of the VLDB Endowment, 2017. `kang2017noscope`. Confirmed: <https://api.crossref.org/works/10.14778/3137628.3137664>
  Motivated by the cost of running a state-of-the-art detector on every video frame; a system for querying video that cuts that cost.
- **Reducto: On-Camera Filtering for Resource-Efficient Real-Time Video Analytics.** Li et al. Proceedings of the Annual conference of the ACM Special Interest Group on Data Communication on the applications, technologies, architectures, and protocols for computer communication, 2020. `li2020reducto`. Confirmed: <https://api.crossref.org/works/10.1145/3387514.3405874>
  Moves frame filtering onto the camera, at the start of the analytics pipeline, where earlier systems filtered on edge or backend servers.
- **DeepCache: Principled Cache for Mobile Deep Vision.** Xu et al. Proceedings of the 24th Annual International Conference on Mobile Computing and Networking, 2018. `xu2018deepcache`. Confirmed: <https://api.crossref.org/works/10.1145/3241539.3241563>
  A cache for deep vision inference on phones that exploits temporal locality in the video stream, trading cacheability against overhead and accuracy loss.
- **Energy characterization and optimization of image sensing toward continuous mobile vision.** LiKamWa et al. Proceeding of the 11th annual international conference on Mobile systems, applications, and services, 2013. `likamwa2013energy`. Confirmed: <https://api.crossref.org/works/10.1145/2462456.2464448>
  Characterises CMOS image sensor energy and finds that sensors are not energy-proportional: energy per pixel is inversely proportional to frame rate and resolution. Useful for costing a capture.
- **Rapid object detection using a boosted cascade of simple features.** Viola and Jones. Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition. CVPR 2001, 2001. `viola2001rapid`. Confirmed: <https://api.crossref.org/works/10.1109/cvpr.2001.990517>
  **[title only]** The boosted cascade of simple features for rapid object detection.

### Selective prediction and the reject option

- **On optimum recognition error and reject tradeoff.** Chow. IEEE Transactions on Information Theory, 1970. `chow1970optimum`. Confirmed: <https://api.crossref.org/works/10.1109/tit.1970.1054406>
  **[title only]** The classical analysis of the trade-off between error and rejection.
- **Selective Classification for Deep Neural Networks.** Geifman and El-Yaniv. Advances in Neural Information Processing Systems 30 (NIPS 2017), 2017. `geifman2017selective`. Confirmed: <https://papers.nips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html>
  Brings selective classification (the reject option) to deep networks, trading coverage for accuracy. Supplies the standard way to plot this paper's result: wrong-claim rate against the fraction of queries answered from memory.
- **Machine learning with a reject option: a survey.** Hendrickx et al. Machine Learning, 2024. `hendrickx2024machine`. Confirmed: <https://api.crossref.org/works/10.1007/s10994-024-06534-x>
  **[title only]** Survey of machine learning with a reject option.

**Why this area matters here.** The selective policy is a cascade whose first stage uses only metadata (age, move frequency, distance walked, tracking interruptions) and whose second stage is a camera capture. Mobile systems papers evaluate such gates as an accuracy-versus-cost curve against always-on and never-on baselines, which is the comparison this paper makes.

## 6. Staleness and trust

### Freshness and age of information

- **Real-time status: How often should one update?** Kaul et al. 2012 Proceedings IEEE INFOCOM, 2012. `kaul2012realtime`. Confirmed: <https://api.crossref.org/works/10.1109/infcom.2012.6195689>
  Introduces a time-average age metric for status update systems and asks how often a source should update.
- **Age of Information: An Introduction and Survey.** Yates et al. IEEE Journal on Selected Areas in Communications, 2021. `yates2021age`. Confirmed: <https://api.crossref.org/works/10.1109/jsac.2021.3065072>
  Survey of age of information for systems in which sources send time-stamped status updates to recipients who want them as timely as possible under resource limits.
- **The Age of Incorrect Information: A New Performance Metric for Status Updates.** Maatouk et al. IEEE/ACM Transactions on Networking, 2020. `maatouk2020age`. Confirmed: <https://api.crossref.org/works/10.1109/tnet.2020.3005549>
  Age of Incorrect Information: extends freshness to "informative" updates, those that bring new and correct information to the monitor, to fix shortcomings of both plain age and plain error penalties. Without a budget, always updating is optimal; under a power budget the optimal policy is a mixture of two Lagrangian policies. The metric matches this paper's notion of a wrong location claim better than plain age does, and the two results mirror the always-verify and selective policies.
- **Synchronizing a database to improve freshness.** Cho and Garcia-Molina. ACM SIGMOD Record, 2000. `cho2000synchronizing`. Confirmed: <https://api.crossref.org/works/10.1145/335191.335391>
  How to refresh a local copy of an autonomous data source: freshness metrics, change models, and refresh policies, checked against data collected from 270 web sites. The original statement of the problem "my copy may be out of date and checking costs something".
- **Effective page refresh policies for Web crawlers.** Cho and Garcia-Molina. ACM Transactions on Database Systems, 2003. `cho2003effective`. Confirmed: <https://api.crossref.org/works/10.1145/958942.958945>
  Keeping local copies of remote sources fresh when the sources change on their own and do not notify the copy, so the copy must decide when to poll. The same structure as an object memory that is never told when an object moves.

### Over-trust in automated and AR guidance

- **Fast, Accurate, But Sometimes Too-Compelling Support: The Impact of Imperfectly Automated Cues in an Augmented-Reality Head-Mounted Display on Visual Search Performance.** Warden et al. IEEE Transactions on Human-Machine Systems, 2023. `warden2023fast`. Confirmed: <https://api.crossref.org/works/10.1109/thms.2023.3302152>
  56 participants searched a 3D room with AR headset cues (arrow, minimap, or target icon); for one group the cue was wrong on 17% of trials. Location cues were the most effective and also produced the most automation bias: participants followed a wrong cue without checking the scene. The strongest single citation for why a wrong location claim in AR is costly.
- **Beyond the Wizard of Oz: Negative Effects of Imperfect Machine Learning to Examine the Impact of Reliability of Augmented Reality Cues on Visual Search Performance.** Raikwar et al. IEEE Transactions on Visualization and Computer Graphics, 2024. `raikwar2024beyond`. Confirmed: <https://api.crossref.org/works/10.1109/tvcg.2024.3372062>
  53 participants with real detector-driven AR cues at 100% or 88.9% accuracy. Performance with imperfect cues was much worse, and participants were often drawn to incorrect cues.
- **Display Signaling in Augmented Reality: Effects of Cue Reliability and Image Realism on Attention Allocation and Trust Calibration.** Yeh and Wickens. Human Factors: The Journal of the Human Factors and Ergonomics Society, 2001. `yeh2001display`. Confirmed: <https://api.crossref.org/works/10.1518/001872001775898269>
  Cueing in an AR-style display helped find expected targets and pulled attention from unexpected ones; both effects shrank when cue reliability dropped from 100% to 75%.
- **Overtrust of robots in emergency evacuation scenarios.** Robinette et al. 2016 11th ACM/IEEE International Conference on Human-Robot Interaction (HRI), 2016. `robinette2016overtrust`. Confirmed: <https://api.crossref.org/works/10.1109/hri.2016.7451740>
  All 26 participants followed a guidance robot in a simulated emergency, though half had just watched it perform poorly.
- **Surgeons blinded by enhanced navigation: the effect of augmented reality on attention.** Dixon et al. Surgical Endoscopy, 2013. `dixon2013surgeons`. Confirmed: <https://api.crossref.org/works/10.1007/s00464-012-2457-3>
  **[title only]** The effect of AR navigation on surgeons' attention.
- **Humans and Automation: Use, Misuse, Disuse, Abuse.** Parasuraman and Riley. Human Factors: The Journal of the Human Factors and Ergonomics Society, 1997. `parasuraman1997humans`. Confirmed: <https://api.crossref.org/works/10.1518/001872097778543886>
  **[title only]** Use, misuse, disuse and abuse of automation.
- **Trust in Automation: Designing for Appropriate Reliance.** Lee and See. Human Factors: The Journal of the Human Factors and Ergonomics Society, 2004. `lee2004trust`. Confirmed: <https://api.crossref.org/works/10.1518/hfes.46.1.50.30392>
  **[title only]** Trust in automation and designing for appropriate reliance.
- **Complacency and Bias in Human Use of Automation: An Attentional Integration.** Parasuraman and Manzey. Human Factors: The Journal of the Human Factors and Ergonomics Society, 2010. `parasuraman2010complacency`. Confirmed: <https://api.crossref.org/works/10.1177/0018720810376055>
  **[title only]** Complacency and bias in human use of automation, integrated through attention.
- **Does automation bias decision-making?** Skitka et al. International Journal of Human-Computer Studies, 1999. `skitka1999does`. Confirmed: <https://api.crossref.org/works/10.1006/ijhc.1999.0252>
  **[title only]** Whether automation biases decision-making.
- **Automation bias: a systematic review of frequency, effect mediators, and mitigators.** Goddard et al. Journal of the American Medical Informatics Association, 2012. `goddard2012automation`. Confirmed: <https://api.crossref.org/works/10.1136/amiajnl-2011-000089>
  **[title only]** Systematic review of automation bias: frequency, effect mediators, mitigators.

### Showing location uncertainty

- **Pedestrian navigation with degraded GPS signal: investigating the effects of visualizing position uncertainty.** Burigat and Chittaro. Proceedings of the 13th International Conference on Human Computer Interaction with Mobile Devices and Services, 2011. `burigat2011pedestrian`. Confirmed: <https://api.crossref.org/works/10.1145/2037373.2037407>
  User study of position-uncertainty visualizations for pedestrian navigation with degraded GPS, compared against a basic display that shows only the last accurate position. The closest HCI precedent found for showing a last-known location together with its uncertainty.
- **An Exploration of Location Error Estimation.** Dearman et al. UbiComp 2007: Ubiquitous Computing, 2007. `dearman2007exploration`. Confirmed: <https://api.crossref.org/works/10.1007/978-3-540-74853-3_11>
  **[title only]** Location error estimation.

**Why this area matters here.** Age-of-information work supplies the cost model, and the AR cueing studies supply the reason to care: people follow a confident AR pointer even when it is wrong, so the number that matters is how often the system points at the wrong place.

## Closest prior work

Five works come closest to the thesis. None is a phone AR system for a human user.

### 1. Temporal persistence modeling for object search (`toris2017temporal`, ICRA 2017)

**What it does.** Addresses object search where "object permanence cannot be assumed and other agents may move objects between locations without the robot's knowledge". It predicts how long an object is expected to remain at a location from sparse earlier observations, using exponential distributions with a Gaussian component, and uses that to suggest where to search. Evaluated on GPS person-tracking data and on multi-object tracking by a mobile robot in a small household over two weeks.

**How this paper differs.** The persistence estimate there ranks places for a robot to search. Here a similar age-and-move-rate estimate gates whether a phone spends a capture before telling a person where the object is. This paper adds the observer's localization drift as a second cause of error and reports wrong claims against captures per query. It should cite this as the origin of the age-based signal.

### 2. Memory for Attention (`ghosh2026memory`, arXiv July 2026, single author, not peer reviewed)

**What it does.** Frames re-perception as "under a limited perception budget, what should the robot attend to right now to keep its own map fresh?" Each object moves as a Poisson process with its own rate; staleness is a function of time since last sighting; the schedule is a square-root law or a Whittle index from the restless-bandit and age-of-information literature. In simulation over 951 movable objects from AI2-THOR, recency of last sighting alone is a strong baseline, and change history improves on a category prior.

**How this paper differs.** That is proactive scheduling in simulation for a robot that can look whenever it likes. This paper decides reactively at query time, on a phone in real homes, where each capture costs a person effort, and it includes pose drift. The probabilistic core (age multiplied by move rate gives the chance the entry is stale) is the same, so this is the work a reviewer is most likely to raise. Its finding that plain recency is hard to beat is a warning: the paper should include an age-only baseline.

### 3. Where Did I Leave My Glasses? (`bogenberger2026glasses`, IEEE RA-L 2026)

**What it does.** A robot keeps a probabilistic stationarity score for every object instance. While an object is out of view the score decays at a rate set by a semantic prior (a chair decays faster than a coffee table), and the robot revisits regions whose entries are probably outdated. In real environments it detected 95% of map changes and was more than 29% more efficient than random and patrol strategies.

**How this paper differs.** The robot's decision is where to go next. A query such as "Find my glasses!" is answered by sending the robot to search; there is no choice between answering from memory and checking, and the cost is robot travel, not a person's effort. Camera poses are "assumed known" from visual-inertial odometry, whereas this paper treats pose drift as a reason to doubt a stored location. The semantic prior on decay rate is the same idea as "how often the object moves".

### 4. STAR: Searching in Space and Time (`chen2025searching`, arXiv November 2025, not peer reviewed)

**What it does.** A service robot answering object retrieval requests keeps a long-term memory that, the paper says, "may be stale". At each step a vision-language model chooses either a temporal action (query memory) or a spatial action (navigate, observe, manipulate). It outperforms scene-graph and memory-only baselines in simulation and on a Tiago robot.

**How this paper differs.** STAR makes the memory-or-look choice with a large model in the loop at every step. This paper makes it with a few scalar signals available before any capture, on the phone, and treats the cost of looking as the quantity to minimise. STAR's memory-only baseline corresponds to this paper's memory-only policy.

### 5. When Memory Lies (`sun2026memory`, arXiv August 2026, not peer reviewed)

**What it does.** Studies what happens when a memory-augmented vision-language agent holds a confident spatial memory that the world now contradicts, in a dynamic gridworld. An agent that trusts raw memory dies more than twice as often as the same agent with no memory. A read-time audit removes much of that cost in text mode but not when visual checking is unreliable.

**How this paper differs.** It is a synthetic study that audits on every read, which is the always-verify policy, with no physical sensing cost. It is useful as independent evidence for the premise that showing an unverified stale location can be worse than showing nothing.

### Nearest human-facing systems

- `maniar2025mempal` and `xie2026linguistic` already present remembered locations as "last seen at [time]". `xie2026linguistic` also labels confidence by recency. Neither verifies.
- `yagi2021gofinder` shows the last appearance as an image and leaves the judgement to the user.
- `forte2026egostream` formalises answer expiry for evaluation only.

### Supporting precedents for each ingredient

| Ingredient of this paper | Closest precedent | Key |
| --- | --- | --- |
| Probability an entry is still true decays with time | Persistence filter | `rosen2016towards` |
| Per-object move frequency as a prior | Spectral occupancy models; learned variability | `krajnik2015waldo`, `looper2023vsg` |
| Spend the expensive sensor only when a cheap signal says so | Trigger frames; DNN only as needed | `chen2015glimpse`, `apicharttrisorn2019frugal` |
| Penalise only information that is old and wrong | Age of Incorrect Information | `maatouk2020age` |
| Abstain or defer instead of answering | Selective classification | `geifman2017selective` |
| The observer's frame drifts | Phone AR stability measurements | `scargill2022here`, `feigl2020localization` |
| Wrong AR pointers get followed | Imperfect AR cue studies | `warden2023fast`, `raikwar2024beyond` |

## Framing implications

1. **Do not claim the "historical claim" idea as new.** `maniar2025mempal`, `xie2026linguistic` and `forte2026egostream` all express it. Claim the step from labelling staleness to acting on it under a cost.
2. **Do not claim age-and-move-rate gating as new.** Cite `toris2017temporal`, `bogenberger2026glasses` and `ghosh2026memory`, and state the three differences: per-query and reactive, human-paid capture cost on a phone, and observer drift as a second staleness source.
3. **Lead with drift.** None of the object-memory or robot-map papers above treats the observer's own localization error as a reason to doubt a remembered location. Combined with `scargill2022here` and `feigl2020localization`, that is the most defensible novelty.
4. **Borrow the evaluation vocabulary.** Present results as a risk-coverage curve (`geifman2017selective`) and relate the wrong-claim metric to Age of Incorrect Information (`maatouk2020age`). Add an age-only baseline, because `ghosh2026memory` found recency alone hard to beat.
5. **Scope the ARCore statement.** "ARCore cannot save its map locally" matches the phone documentation I opened: persistence is through Cloud Anchors hosted on Google's endpoint, with an Internet connection and a lifetime limit (`google2026cloudanchors`, `google2026cloudanchorsguide`), and those pages describe no local alternative. It is not true of ARCore for Jetpack XR, which persists anchors to local storage on XR headsets and wired XR glasses (`google2026jetpackxranchors`). Say "on phones" and cite both.
6. **Motivate the metric with the cueing studies.** `warden2023fast` and `raikwar2024beyond` show people follow wrong AR location cues. That justifies counting wrong location claims and not only search time.
7. **Mark preprints as preprints.** `ghosh2026memory`, `sun2026memory`, `chen2025searching`, `kurdydyk2026spark`, `yin2026glove`, `forte2026egostream`, `florea2026visionaid`, `oh2026oasis`, `allu2024build`, `ren2025gaze` and `scargill2021here` are arXiv only. `xie2026linguistic` and `bhattacharjya2026memoguard` report acceptance in their arXiv comment fields; check for the proceedings versions before camera-ready.

## Unverified: not in refs.bib

Seen in search results or recalled as relevant, but no confirming record was opened, or confirmed and deliberately left out. Do not cite the unconfirmed ones without checking.

| Work | What is unsure |
| --- | --- |
| "Forget-me-not: Intimate computing in support of human memory" (1994) | No Crossref record found; authors, venue and pages unconfirmed. |
| "On the Foundations of Noise-free Selective Classification" (JMLR) | Not in Crossref; the JMLR page was not opened. |
| Edge-SLAM (edge-assisted visual SLAM) | Crossref returned only a 3-page MobiCom 2020 item under this title; the full paper's record was not confirmed. |
| O-STaR (object search with Dirichlet-count belief and temporal adaptation, 2026 poster) | Seen only as a search snippet of a poster PDF. |
| EOC-Bench, SuperMemory-VQA, EgoMonth, Video MindPalace (egocentric memory benchmarks) | Seen only as search snippets; no records opened. |
| "From Seeing to Acting: Smart Glasses as First-Person Intelligence Platforms" (survey) | Search snippet says it lists map staleness and false recall as evaluation criteria; not opened. |
| "Smart Assistive Glasses for Alzheimer's Patients" | Search snippet says it guides the wearer to an object's last seen location; not opened. |
| "Assisting the blind to reach daily objects using smart glasses" (2026) | Search snippet only. |
| Location-uncertainty visualization field studies after `burigat2011pedestrian` | Search snippets only; authors and venues unconfirmed. |
| DynaMem (dynamic spatio-semantic memory for mobile manipulation), OK-Robot, ConceptGraphs | Named in the related work of `bogenberger2026glasses` as map-based systems, the first of which updates its map; not looked up. |
| A commercial phone AR app that records where household items are stored | Seen only in a news search result; not opened. |
| "Robust Localization with Visual-Inertial Odometry Constraints for Markerless Mobile AR" (arXiv:2308.05394) | Confirmed on arXiv, but its successor `liu2024mobilearloc` says it supersedes it, so only the successor is in refs.bib. |
| "Exploring Object Recall in an Object-Rich AR Office Scene" (arXiv:2610.18183) | Confirmed on arXiv; its comment says ISMAR 2026 and IEEE TVCG vol. 32. Left out as tangential (recall of objects seen in AR, not object finding). Its DOI record was not looked up. |
| RPMS (arXiv:2603.17831) and "Memory-Centric Embodied Question Answering" (arXiv:2505.13948) | Confirmed on arXiv; left out as tangential. |

## Could not find

- A peer-reviewed measurement study comparing marker-based and markerless re-registration accuracy on phones. The only hit is the 2017 case study `cheng2017comparison`.
- A measurement study of ARCore Cloud Anchor resolve accuracy or persistence on phones.
- Any human-subject study of how people respond to a stated "last seen" time or a staleness label on an object-finder answer.
- Any phone or smart-glasses object finder that decides between answering from memory and re-detecting. An extended search returned none.
