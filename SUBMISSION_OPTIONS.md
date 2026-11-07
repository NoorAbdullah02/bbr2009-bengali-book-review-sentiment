# Where to submit: options, fees, payment and what to change for each

Information collected on 7 November 2026 from official pages and listings. **Deadlines and fees change; confirm every item on the official site before relying on it.** Items marked (unverified) were not confirmed on an official page.

Current paper: IEEE conference format, A4, 6 pages (`paper/paper.pdf`, Overleaf project in `paper/bbr2009_overleaf.zip`).

## Short recommendation
- Cheapest and easiest from Bangladesh, with an NLP track: **QPAIN 2027** (deadline 20 Dec 2026). It is double-blind, so remove the GitHub link and any identifying text first. Ask the organizers whether online presentation is allowed.
- Safest if you cannot travel: **ICMI 2027** (deadline 15 Dec 2026), hybrid, but about 350 USD for a student online registration plus a page fee.
- Best topic fit overall, ACL format instead of IEEE: **BLP 2027** (deadline 15 Feb 2027).

## Option 1: QPAIN 2027 (IEEE, Bangladesh)
- Name: 2027 IEEE 3rd International Conference on Quantum Photonics, Artificial Intelligence & Networking. Site: https://qpain.org/
- Where and when: Chattogram, Bangladesh, 31 March to 3 April 2027.
- Dates: submission **20 Dec 2026**; notification 5 Feb 2027; camera-ready and registration 20 Feb 2027.
- Submission system: https://myproconf.com/events/app/login.php?url=qpain-2027
- Format: IEEE conference template, **4 to 6 pages**, PDF. **Initial submissions are double-blind.**
- Track 03 "Artificial Intelligence and Machine Learning" lists "Natural Language Processing and Understanding".
- Indexing: accepted and registered papers "will be submitted to IEEE Xplore and indexed by Scopus" (subject to IEEE approval).
- Fees (USD, page /registration-fee): local student author 95 (non-IEEE), 80 (IEEE member); local regular author 110 / 90; international student author 200 / 160; international author 250 / 200; local participant 80. Non-refundable.
- Payment from Bangladesh: Bangladeshi authors may pay the equivalent in taka by local bank transfer, credit or debit card, internet banking, or mobile banking (bKash, Nagad, Rocket, Upay).
- Online presentation: **not stated** on the call for papers or registration page. Ask the organizers (draft email below).
- What to change in the paper: remove the repository link and any text that identifies you (blind review). The paper already says "Anonymous Authors". Keep 6 pages (limit is 4 to 6).
- Unverified: quality of the proceedings, whether "student author" needs proof, final registration steps (page said "available shortly").

## Option 2: ICMI 2027 (IEEE, USA, hybrid)
- Name: IEEE 6th International Conference on Computing and Machine Intelligence. Site: https://icmiconf.com/
- Where and when: Central Michigan University, Mt. Pleasant, Michigan, USA, 24 to 25 April 2027, hybrid.
- Dates: submission **15 Dec 2026**; notification 15 Feb 2027; registration 1 Mar 2027; camera-ready 30 Mar 2027.
- Submission system: Microsoft CMT, https://cmt3.research.microsoft.com/ICMI2027
- Format: 5 to 6 pages, IEEE conference template; **25 (USD) per page above 5**.
- Review: at least three reviewers; double-blind is **not** stated (so author names may be required; check CMT).
- Topics on the welcome page: AI, machine learning, computing, soft computing; NLP is not named (unverified on the full scope page).
- Fees (page author.html, currency not printed, probably USD): online: student non-IEEE 350, student IEEE member 300, regular non-IEEE 400, regular IEEE member 350; in person is 100 to 150 more. A late fee of 100 is mentioned with an odd date (possible typo). All payments non-refundable. A student registration covers one paper and the student must be first author.
- Payment: credit card through an online portal (Stripe, hosted via MRD Forum), then a Google Form with the payment details. On-site payment only for attendees.
- Rule: at least one author must register and present, or the paper is not included in IEEE Xplore.
- Camera-ready: IEEE PDF eXpress (conference ID 73128X), then upload in CMT and fill the IEEE copyright form.
- Contact: info@icmiconf.com
- Total for a student, online, non-IEEE, 6 pages: about 375 USD (about 45,000 taka at 120 taka per dollar; check the current rate).
- Check before paying: look the conference up on https://conferences.ieee.org/ (conference ID 73128X).

## Option 3: BLP 2027, third Bangla Language Processing Workshop (ACL format)
- Co-located with NAACL 2027, San Francisco, 1 to 5 June 2027. Site: https://blp-workshop.github.io/
- Dates: call published 10 Oct 2026; submission **15 Feb 2027**; notification 26 Mar 2027; camera-ready 16 Apr 2027.
- Format: ACL template, not IEEE; the paper would need to be converted (usually 8 pages for long papers). ACL reviews are normally double-blind (remove the repository link and use an anonymous mirror).
- Best topic fit (Bengali NLP only), proceedings in the ACL Anthology. Fees not checked (unverified).

## Option 4: LoResLM 2027 at EACL 2027 (ACL format)
- Language Models for Low-Resource Languages, Athens, 9 to 14 March 2027. Submission deadline **7 Jan 2027** (listing on wikicfp, unverified). Long papers up to 8 pages, ACL format.

## Not possible or not suitable now
- ICEEICT 2027 (MIST, Dhaka, 28 to 30 Jan 2027): hard deadline 10 Nov 2026 (too soon).
- STI 2026 and ICCIT 2026 (Bangladesh): deadlines already passed (July to September 2026).
- ICECE 2026: deadline passed; no 2027 edition found.
- Others seen in listings, not checked: ICDRET 2027 (UIU, 27 to 29 Mar 2027, listing says deadline 15 Nov 2026), QPAIN-like events, IJCNN 2027 (6 pages, IEEE format, deadline reported as 31 Jan 2027 in one tracker; very competitive), IEEE COMPSAC 2027 (15 Feb 2027, not NLP).

## Checklist before any submission
1. Read the official call for papers: page limit, template, blind policy, fees, presentation mode.
2. Convert the paper to the venue template (`latex/conference/main.tex`).
3. Blind venue: remove the GitHub link and any identifying text; use an anonymous mirror (https://anonymous.4open.science) or write that the code will be released after review. Keep the author line anonymous.
4. Non-blind venue: add real author names and affiliations (a Word or LaTeX edit of the author block only).
5. Run `python src/audit_paper.py` (needs no GPU) and compile from `paper/overleaf/`.
6. Supervisor review and proofreading; make sure the institution accepts the venue.
7. Decide who pays and how (see each option above). Never send card details outside the official payment page.
8. Registration is paid only after acceptance (all the IEEE options above), and at least one author must register and present.

## Draft email to the QPAIN organizers (copy and send from your own account)
> Dear organizers, I plan to submit a paper to QPAIN 2027. Could you please tell me (1) whether authors who cannot travel to Chattogram may present online, and what the registration fee would be in that case, and (2) whether the student author rate applies to bachelor's and master's students? Thank you.

## Draft email to the ICMI organizers (info@icmiconf.com)
> Dear organizers, I plan to submit a paper to ICMI 2027 and would register as an online student author from Bangladesh. Could you please confirm (1) the currency of the listed fees, (2) whether a bank transfer or invoice is possible in addition to card payment, (3) whether the review is double-blind, and (4) the charge for a sixth page? Thank you.

## Honest note
Neither the paper nor any venue choice can guarantee acceptance. On the official BanglaBook benchmark the result is 0.6 points below the published one, and the label-noise audit used one language-model annotator; the paper says both openly.
