"""데이터 계약 형식의 샘플(기사 36건, 유저 2명)을 만든다. 실행: python -X utf8 crystal/web/sample/make_sample.py"""
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = int(datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc).timestamp())
DAY = 86400

A = []


def art(aid, label, p_fake, title, body, kw_title=(), kw_body=(), rat=None, verdict="", event=-1):
    A.append(dict(aid=aid, label=label, p=p_fake, title=title, body=body, kt=list(kw_title), kb=list(kw_body),
                  rat=rat or {}, verdict=verdict, event=event))


art("s01", 1, 0.913,
    "Avery Lane Secretly Married in Las Vegas, Insider Claims — ‘She Didn’t Tell Anyone’",
    "Pop star Avery Lane has reportedly tied the knot in a secret Las Vegas ceremony, according to an insider who claims the singer “didn’t tell anyone, not even her family.”\n"
    "The shocking news comes just weeks after Lane was spotted at a friend’s wedding in Nevada, and sources say the couple exchanged vows at a private chapel late Saturday night. “Everyone is stunned,” the source added. Fans flooded social media with reactions, and some speculated that the singer may already be expecting.\n"
    "Neither Lane nor her representatives have confirmed the ceremony, but the insider insists the marriage is “100 percent real” and that an official announcement is imminent.",
    ["Secretly Married", "Insider Claims"],
    ["reportedly", "according to an insider", "shocking news", "sources say", "stunned", "may already be expecting", "100 percent real", "imminent"],
    {"according to an insider": "This phrase attributes the article’s central claim to an anonymous, unverifiable source. Fabricated celebrity stories commonly rely on such sourcing because it cannot be checked; verified coverage of this event instead names the venue and Lane’s representative.",
     "Secretly Married": "The headline asserts a secret marriage as fact while the body never names anyone who can confirm it; secrecy framing lets the claim avoid verification.",
     "Insider Claims": "Attributing the headline claim to an unnamed “insider” is a hallmark of fabricated celebrity reporting; no named person or outlet stands behind it.",
     "reportedly": "A hedge word that presents the claim without any attribution, so the article commits to nothing checkable.",
     "shocking news": "Emotional framing that pushes the reader toward reaction rather than evidence; verified coverage of this event is descriptive, not sensational.",
     "sources say": "Plural anonymous sourcing with no names, outlets or documents; the same event is covered by a named representative in verified reporting.",
     "may already be expecting": "A speculative claim stacked on top of the unverified marriage claim, with no source at all.",
     "100 percent real": "An insistence on truth by an anonymous insider substitutes assertion for evidence."},
    "The article is labeled Fake because its central claim — a secret wedding — rests on anonymous sourcing (“insider”, “sources say”) and emotional framing (“shocking”, “stunned”), whereas verified coverage of the same event reports that Lane attended a friend’s wedding and that her representative denied any marriage.",
    0)

art("s02", 0, 0.059,
    "Avery Lane attends longtime friend’s Las Vegas wedding; rep denies marriage rumors",
    "Avery Lane spent the weekend in Las Vegas for the wedding of her longtime friend and former bandmate, the singer’s representative, Dana Whitfield, confirmed on Monday. Photos shared by the couple show Lane among the guests at the ceremony at the Solaris Resort on Saturday.\n"
    "Asked about online rumors that Lane herself had married, Whitfield said in a statement that “Avery was a guest, not a bride,” adding that the singer has no plans to marry. Lane is scheduled to begin recording her next album in November.",
    ["rep denies"],
    ["the singer’s representative, Dana Whitfield, confirmed on Monday", "ceremony at the Solaris Resort on Saturday", "Whitfield said in a statement", "has no plans to marry"],
    {"rep denies": "The headline attributes the denial to Lane’s representative, a named and accountable source, which the model associates with verified reporting.",
     "the singer’s representative, Dana Whitfield, confirmed on Monday": "A named spokesperson, a specific day and an on-record confirmation: checkable attribution that fabricated stories about this event lack.",
     "ceremony at the Solaris Resort on Saturday": "The venue and date are specific and verifiable, unlike the “private chapel” in the fake version.",
     "Whitfield said in a statement": "A direct, on-record statement from an identified representative.",
     "has no plans to marry": "A direct denial from the subject’s representative that contradicts the fabricated marriage claim."},
    "The article is labeled True because every claim is attributed to a named representative (Dana Whitfield) with a specific venue, day and on-record statement, and it directly addresses and denies the rumor rather than amplifying it.",
    0)

art("s03", 0, 0.114,
    "‘The Long Harbor’ sequel confirmed for 2027 release, studio announces at press event",
    "Meridian Pictures confirmed on Tuesday that a sequel to ‘The Long Harbor’ will open in theaters in March 2027. The announcement was made by studio chief Elena Park during the company’s annual press event in Los Angeles.\n"
    "Park said director Tomas Reyes will return, and that production is scheduled to begin in January. The original film earned 412 million dollars worldwide, according to the studio’s published figures.",
    [], ["Meridian Pictures confirmed on Tuesday", "studio chief Elena Park", "according to the studio’s published figures"])

art("s04", 1, 0.834,
    "You Won’t Believe Who Just Quit ‘Midnight Court’ Mid-Season — Cast ‘In Total Chaos’",
    "A major star of ‘Midnight Court’ has allegedly walked off the set in the middle of filming, and insiders say the rest of the cast is “in total chaos.” The actor, whose name is being withheld, reportedly stormed out after a heated argument with producers.\n"
    "Sources close to the production claim the show may be cancelled entirely. “Nobody knows what happens next,” one source said. The network has not commented, which fans are reading as confirmation.",
    ["You Won’t Believe", "In Total Chaos"],
    ["allegedly walked off", "insiders say", "whose name is being withheld", "reportedly stormed out", "Sources close to the production claim", "reading as confirmation"],
    {"You Won’t Believe": "Clickbait phrasing that promises a revelation while the article never actually names the person, a pattern typical of fabricated entertainment stories.",
     "In Total Chaos": "An emotional quote attributed to unnamed insiders, used to dramatize rather than document.",
     "allegedly walked off": "The core event is hedged with “allegedly” and never attributed to anyone identifiable.",
     "insiders say": "Anonymous plural sourcing; verified coverage of this event quotes the network and the actor by name.",
     "whose name is being withheld": "Withholding the central name removes the only checkable fact in the story.",
     "reportedly stormed out": "A dramatic action attributed to no one, hedged with “reportedly”.",
     "Sources close to the production claim": "Anonymous sourcing for the most consequential claim (cancellation), which the network later contradicted.",
     "reading as confirmation": "Treats silence as evidence, a reasoning move the model associates with fabricated stories."},
    "The article is labeled Fake because the departure is described through anonymous insiders and hedges (“allegedly”, “reportedly”), the actor is never named, and the network’s silence is presented as confirmation, whereas verified coverage names the actor and quotes the network confirming a planned exit after the season finale.",
    1)

art("s05", 0, 0.098,
    "Singer Jonah Reyes announces fall tour dates across 22 cities",
    "Jonah Reyes will play 22 cities this fall, beginning September 12 in Austin and ending November 3 in Seattle, his label Northline Records said on Monday. Tickets go on sale Friday through the tour’s official site.\n"
    "The tour supports his third album, released in June. Reyes confirmed the dates on his verified social media account and thanked fans for their support.",
    [], ["his label Northline Records said on Monday", "Tickets go on sale Friday", "confirmed the dates on his verified social media account"])

art("s06", 1, 0.760,
    "Shocking: Reality Star Allegedly Caught Faking Charity Donations, Source Says",
    "Reality star Kira Dawson was allegedly caught faking a charity donation she promoted on camera, according to a source who spoke on condition of anonymity. The source claims the check she posed with was “never cashed.”\n"
    "Fans are outraged, and some are demanding she be removed from the show. Dawson has not responded, and people familiar with the matter say the charity is “furious.”",
    ["Shocking", "Allegedly Caught", "Source Says"],
    ["allegedly caught faking", "on condition of anonymity", "The source claims", "people familiar with the matter say"],
    {"Shocking": "Sensational headline framing with no factual content.",
     "Allegedly Caught": "The accusation is hedged and never attributed to anyone who can be checked.",
     "Source Says": "Anonymous single-source attribution in the headline itself.",
     "allegedly caught faking": "The central accusation is hedged with “allegedly” and attributed to an anonymous source; the charity later confirmed the donation.",
     "on condition of anonymity": "An unnamed source for an accusation about a specific person is a strong fabricated-story signal.",
     "The source claims": "The only evidence (“never cashed”) is an unverifiable claim by the anonymous source.",
     "people familiar with the matter say": "A second layer of anonymous sourcing presenting an emotion (“furious”) as fact."},
    "The article is labeled Fake because its accusation rests entirely on anonymous sources and hedged language (“allegedly”), with no document or named party, whereas the charity itself later confirmed on record that the donation was received in full.",
    2)

art("s07", 0, 0.073,
    "Director Mara Ellison to receive lifetime achievement award at film festival",
    "The Riverside Film Festival announced on Wednesday that director Mara Ellison will receive its lifetime achievement award at the closing ceremony on October 28. Festival director Luis Ortega said Ellison “changed how a generation of filmmakers thinks about silence on screen.”\n"
    "Ellison, 71, has directed 14 feature films. The festival will screen a restored print of her 1989 debut as part of the tribute, according to the published program.",
    [], ["announced on Wednesday", "Festival director Luis Ortega said", "according to the published program"])

art("s08", 0, 0.141,
    "‘Harbor Lights’ cast reunites for charity table read, organizers confirm",
    "The cast of ‘Harbor Lights’ will reunite for a live table read on November 9 to benefit the Coastal Food Bank, organizers confirmed on Thursday. The event will stream on the food bank’s official channel.\n"
    "Food bank director Priya Nair said all proceeds will go to its winter program. Four of the five original leads have confirmed their attendance, she added.",
    [], ["organizers confirmed on Thursday", "Food bank director Priya Nair said", "have confirmed their attendance"])

art("s09", 0, 0.082,
    "No, Avery Lane did not get married in Las Vegas, representative confirms",
    "Rumors that pop star Avery Lane secretly married in Las Vegas over the weekend are false, her representative Dana Whitfield told reporters on Monday. Lane was in the city as a guest at a friend’s wedding at the Solaris Resort.\n"
    "Whitfield said the singer “finds the rumor funny but wants it corrected.” No marriage license has been filed in Clark County under Lane’s name, county records show.",
    ["representative confirms"],
    ["her representative Dana Whitfield told reporters on Monday", "county records show"],
    {"representative confirms": "The headline cites a named representative as the source of the correction.",
     "her representative Dana Whitfield told reporters on Monday": "Named, on-record attribution with a date, which the model associates with verified reporting.",
     "county records show": "A reference to public records that can be checked independently."},
    "The article is labeled True because it attributes the denial to a named representative, cites public county records, and names the venue, directly contradicting the anonymous claims of the fake story.",
    0)

art("s10", 0, 0.118,
    "Avery Lane celebrates friend’s wedding weekend in Nevada",
    "Avery Lane joined about 80 guests at the Solaris Resort in Las Vegas on Saturday for the wedding of her former bandmate, the couple said in a post on their verified account. Lane performed one song at the reception.\n"
    "The singer’s representative confirmed her attendance and said Lane returned to Los Angeles on Sunday to continue work on her album.",
    [], ["the couple said in a post on their verified account", "The singer’s representative confirmed her attendance"],
    {"the couple said in a post on their verified account": "The account of the event comes from the couple themselves on a verified channel, a checkable source.",
     "The singer’s representative confirmed her attendance": "An on-record confirmation from an identified representative."},
    "The article is labeled True because its claims come from the couple’s verified account and the singer’s representative, with a named venue and date.",
    0)

art("s11", 1, 0.871,
    "Leaked: The Secret Diet That Made This Actress Lose 30 Pounds in 10 Days",
    "An actress reportedly lost 30 pounds in just 10 days using a secret diet that doctors don’t want you to know about, according to a leaked message seen by this site. Insiders say studios are now demanding all their stars use it.\n"
    "The diet allegedly involves a single ingredient available in any kitchen. We cannot reveal the actress’s name for legal reasons.",
    ["Leaked", "Secret Diet"],
    ["doctors don’t want you to know about", "according to a leaked message", "Insiders say", "cannot reveal the actress’s name"])

art("s12", 0, 0.067,
    "Streaming service renews ‘Northbound’ for a third season, network says",
    "Streamer Lumen has renewed the drama ‘Northbound’ for a third season, the company said in a press release on Monday. Production will begin in February, with showrunner Hana Choi returning.\n"
    "The second season was the service’s most-watched original of the year, according to figures the company published last month.",
    [], ["the company said in a press release on Monday", "according to figures the company published"])

art("s13", 0, 0.079,
    "‘Midnight Court’ star Daniel Moss to leave after season finale, network confirms planned exit",
    "Actor Daniel Moss will leave ‘Midnight Court’ after the current season, the network confirmed in a statement on Thursday, describing the departure as planned and amicable. Moss said in the same statement that he had asked to be written out to pursue a film project.\n"
    "Filming of the season’s remaining episodes continues on schedule, a network spokesperson said. The show has already been renewed for another season.",
    ["network confirms"],
    ["the network confirmed in a statement on Thursday", "Moss said in the same statement", "a network spokesperson said"],
    {"network confirms": "The headline attributes the news to the network itself, an accountable source.",
     "the network confirmed in a statement on Thursday": "An on-record statement with a date from the party responsible for the show.",
     "Moss said in the same statement": "The actor is named and quoted directly, unlike the withheld name in the fake version.",
     "a network spokesperson said": "An identified spokesperson addressing the cancellation rumor directly."},
    "The article is labeled True because the departure is confirmed on record by the network and the named actor, with a stated reason and a direct statement that filming continues, contradicting the anonymous “chaos” narrative.",
    1)

art("s14", 0, 0.126,
    "‘Midnight Court’ producers say filming continues on schedule despite cast change",
    "Producers of ‘Midnight Court’ said on Friday that filming remains on schedule following the announced departure of Daniel Moss. Executive producer Rena Caldwell told a press call that the finale script was written with the exit in mind.\n"
    "The network has renewed the series for a fifth season, according to its published schedule.",
    [], ["Executive producer Rena Caldwell told a press call", "according to its published schedule"],
    {"Executive producer Rena Caldwell told a press call": "A named producer speaking on record to press.",
     "according to its published schedule": "A reference to a published, checkable document."},
    "The article is labeled True because it quotes a named executive producer on record and cites the network’s published schedule.",
    1)

art("s15", 0, 0.094,
    "Charity confirms reality star Kira Dawson’s donation was received in full",
    "The Brightwater Children’s Fund confirmed on Tuesday that the 50,000 dollar donation reality star Kira Dawson announced on air was received and deposited in full. Executive director Tom Alvarez said the fund had issued a receipt on the day of the broadcast.\n"
    "Dawson’s representative provided a copy of the bank confirmation to reporters. The charity said it had not been contacted by any outlet before the rumor spread.",
    ["Charity confirms"],
    ["Executive director Tom Alvarez said", "provided a copy of the bank confirmation", "The charity said"],
    {"Charity confirms": "The headline attributes the confirmation to the charity itself, the party best placed to know.",
     "Executive director Tom Alvarez said": "A named official of the charity speaking on record.",
     "provided a copy of the bank confirmation": "Documentary evidence that reporters could inspect, absent from the fake story.",
     "The charity said": "The organization directly addresses the rumor on record."},
    "The article is labeled True because the charity’s named director confirms the donation on record and a bank confirmation was shown to reporters, directly refuting the anonymous accusation.",
    2)

art("s16", 1, 0.902,
    "Royal Insider: Prince Secretly Planning to Renounce Title, Palace ‘Terrified’",
    "A prince is secretly planning to renounce his title within weeks, a royal insider claims, and the palace is reportedly “terrified” of the fallout. Sources say the decision was made after a bitter family argument.\n"
    "The palace has refused to comment, which insiders describe as proof the story is true.",
    ["Royal Insider", "Secretly Planning"],
    ["a royal insider claims", "reportedly", "Sources say", "proof the story is true"])

art("s17", 0, 0.088,
    "Museum of Modern Design opens new wing after three-year renovation",
    "The Museum of Modern Design opened its new east wing on Saturday after a three-year, 60 million dollar renovation, the museum said. Director Anna Weiss cut the ribbon alongside the city’s mayor.\n"
    "The wing adds 12 galleries and will open to the public on Tuesday, according to the museum’s announcement.",
    [], ["the museum said", "according to the museum’s announcement"])

art("s18", 1, 0.796,
    "Actor’s Mysterious Hospital Dash Sparks Fears — ‘He Could Be Gone by Christmas’",
    "A beloved actor was rushed to hospital under mysterious circumstances, and a friend reportedly fears he “could be gone by Christmas.” Insiders claim his family has been called to his bedside.\n"
    "His team has not confirmed anything, but fans believe the silence speaks volumes.",
    ["Mysterious", "Sparks Fears"],
    ["under mysterious circumstances", "a friend reportedly fears", "Insiders claim", "the silence speaks volumes"])

art("s19", 0, 0.091, "City council approves new downtown bike lanes after public hearing",
    "The city council voted 7-2 on Monday to approve a 4.5 mile network of protected bike lanes downtown, according to the meeting minutes. Construction is scheduled to begin in spring, the transportation department said.\n"
    "Council member Ruth Okafor, who sponsored the measure, said the plan responds to a 2025 safety audit.",
    [], ["according to the meeting minutes", "the transportation department said", "Council member Ruth Okafor"])

art("s20", 1, 0.883, "Bombshell: Pop Star Caught Lip-Syncing Entire Tour, Crew Member Spills All",
    "A former crew member has allegedly revealed that a pop star lip-synced every show of her last tour. The anonymous crew member claims “not a single note was live.”\n"
    "Insiders say the star’s team is scrambling to contain the story. Her publicist did not respond, which sources describe as telling.",
    ["Bombshell", "Spills All"],
    ["allegedly revealed", "The anonymous crew member claims", "Insiders say", "sources describe as telling"])

art("s21", 0, 0.077, "Actress Nia Bell joins cast of historical drama, studio confirms",
    "Nia Bell has joined the cast of the historical drama ‘Salt and Iron’, studio Greyline Films confirmed on Wednesday. Bell will play a 19th-century lighthouse keeper, the studio said in its casting announcement.\n"
    "Filming begins in March in Nova Scotia, according to the production schedule released with the announcement.",
    [], ["studio Greyline Films confirmed on Wednesday", "the studio said in its casting announcement", "according to the production schedule"])

art("s22", 1, 0.842, "Shock Split! Power Couple Reportedly Done After Explosive Fight, Friends ‘Devastated’",
    "A celebrity power couple has reportedly split after an explosive fight, and friends are said to be “devastated.” A source close to the pair claims divorce papers are “days away.”\n"
    "Neither star has commented. Insiders say the couple’s recent public appearance was staged.",
    ["Shock Split", "Reportedly Done"],
    ["reportedly split", "A source close to the pair claims", "Insiders say", "was staged"])

art("s23", 0, 0.102, "Record label announces reissue of 1994 album with unreleased tracks",
    "Harbor Records will reissue the 1994 album ‘Tidewater’ on November 14 with six previously unreleased tracks, the label said on Monday. The reissue was remastered from the original tapes, according to the label’s announcement.\n"
    "Band member Elise Thorne said in the announcement that the tracks were recorded during the original sessions.",
    [], ["the label said on Monday", "according to the label’s announcement", "Band member Elise Thorne said"])

art("s24", 1, 0.811, "Doctors Stunned: Celebrity Chef’s Miracle Cure Allegedly Reverses Aging",
    "A celebrity chef has allegedly discovered a miracle cure that reverses aging, and doctors are reportedly “stunned.” Insiders claim the chef looks twenty years younger after one month.\n"
    "The chef’s team declined to share the recipe, which sources say proves it works.",
    ["Doctors Stunned", "Miracle Cure"],
    ["allegedly discovered", "reportedly “stunned.”", "Insiders claim", "sources say proves it works"])

art("s25", 0, 0.083, "Festival organizers confirm lineup for summer music weekend",
    "Organizers of the Lakeside Music Weekend confirmed the full lineup on Thursday, with 40 acts across three stages, according to the official announcement. Tickets went on sale the same day through the festival’s site.\n"
    "Festival director Marco Lindqvist said attendance is capped at 25,000 per day under the city permit.",
    [], ["confirmed the full lineup on Thursday", "according to the official announcement", "Festival director Marco Lindqvist said"])

art("s26", 1, 0.868, "You Won’t Believe What This Child Star Looks Like Now — Fans ‘Can’t Cope’",
    "A former child star has reportedly undergone a dramatic transformation, and fans “can’t cope,” according to insiders. Sources claim the star is unrecognizable.\n"
    "The star’s representatives have not responded, and the photo at the center of the story cannot be independently verified.",
    ["You Won’t Believe", "Can’t Cope"],
    ["reportedly undergone", "according to insiders", "Sources claim", "cannot be independently verified"])

art("s27", 0, 0.095, "Author wins national book prize for debut novel, judges announce",
    "Author Leon Baptiste won the National Book Prize on Tuesday for his debut novel ‘Low Country’, the prize committee announced at a ceremony in Chicago. The award carries a 50,000 dollar purse.\n"
    "Chair of judges Marianne Holt said the novel “earns every sentence.” The shortlist was published in September, according to the committee.",
    [], ["the prize committee announced", "Chair of judges Marianne Holt said", "according to the committee"])

art("s28", 1, 0.779, "Secret Memo Reveals Network Plans to Cancel Every Scripted Show, Insider Warns",
    "A secret memo allegedly reveals that a major network plans to cancel every scripted show by next year, an insider warns. Sources say executives are “in panic mode.”\n"
    "The memo has not been seen by this site, but the insider insists it is “real.”",
    ["Secret Memo", "Insider Warns"],
    ["allegedly reveals", "Sources say", "has not been seen by this site", "insists it is “real.”"])

art("s29", 0, 0.071, "Theater company announces 2027 season with four new plays",
    "The Meridian Theater Company announced its 2027 season on Monday, with four new plays and two revivals, according to the company’s press release. Artistic director Sam Ito said two of the new works come from the company’s residency program.\n"
    "Subscriptions go on sale in November, the company said.",
    [], ["according to the company’s press release", "Artistic director Sam Ito said", "the company said"])

art("s30", 1, 0.826, "Exposed: Talk Show Host Secretly Fired, Replacement Already Chosen, Sources Claim",
    "A talk show host has secretly been fired and a replacement has already been chosen, sources claim. Insiders say the host was told “minutes before taping.”\n"
    "The network denied the story, which the insiders say is exactly what they would do.",
    ["Exposed", "Secretly Fired"],
    ["sources claim", "Insiders say", "exactly what they would do"])

art("s31", 0, 0.089, "Film academy adds two new voting categories, board says",
    "The film academy’s board voted on Tuesday to add two new voting categories beginning with next year’s awards, the academy said in a statement. The categories cover casting and stunt design.\n"
    "Academy president Dana Ruiz said the change followed a two-year review, according to the statement.",
    [], ["the academy said in a statement", "Academy president Dana Ruiz said", "according to the statement"])

art("s32", 0, 0.104, "Singer’s charity concert raises 2.3 million dollars, organizers report",
    "A charity concert headlined by singer Mila Grant raised 2.3 million dollars for flood relief, organizers reported on Sunday. The figure was audited by the Red Cross chapter that received the funds, according to the organizers’ statement.\n"
    "Grant said on her verified account that the money would be distributed within a month.",
    [], ["organizers reported on Sunday", "according to the organizers’ statement", "said on her verified account"])

art("s33", 0, 0.097, "Documentary on jazz pianist to premiere at autumn festival, programmers confirm",
    "A documentary about jazz pianist Ruben Vale will premiere at the Autumn Film Festival on October 24, festival programmers confirmed on Wednesday. The film was directed by Vale’s daughter, according to the festival catalog.\n"
    "The festival said the screening will be followed by a live performance.",
    [], ["festival programmers confirmed on Wednesday", "according to the festival catalog", "The festival said"])

art("s34", 1, 0.858, "Heartbreaking: Beloved Sitcom Star Reportedly Broke and Living in Car, Friend Reveals",
    "A beloved sitcom star is reportedly broke and living in a car, a friend has revealed to this site. Insiders say the star’s fortune vanished after “bad investments.”\n"
    "The star has not commented, and sources say the silence confirms the story.",
    ["Heartbreaking", "Friend Reveals"],
    ["reportedly broke", "a friend has revealed", "Insiders say", "the silence confirms the story"])

art("s35", 0, 0.086, "Band announces farewell tour after 25 years, final show set for hometown",
    "The band Northern Line announced a farewell tour on Monday, ending with a final show in their hometown of Leeds on June 20, the band said in a statement on its official site. The tour covers 30 cities.\n"
    "Frontman Ollie Harte said the band had planned the ending for two years, according to the statement.",
    [], ["the band said in a statement on its official site", "Frontman Ollie Harte said", "according to the statement"])

art("s36", 0, 0.079, "Studio confirms release date for animated sequel, trailer to follow next month",
    "Pinewood Animation confirmed on Thursday that its animated sequel ‘Paper Kites 2’ will open on July 10, 2027, the studio said in a press release. A trailer will be released next month, according to the release.\n"
    "Director Yuki Mori returns, the studio said.",
    [], ["confirmed on Thursday", "the studio said in a press release", "according to the release"])

CONTRAST = {
    0: {"fake": "Lane secretly married at a private Las Vegas chapel; announcement “imminent”.",
        "verified": "Lane attended a friend’s wedding at a Las Vegas resort; her representative denied any marriage."},
    1: {"fake": "An unnamed star stormed off set mid-season and the show may be cancelled.",
        "verified": "Daniel Moss leaves after the finale in a planned exit; the network confirmed filming continues and the show is renewed."},
    2: {"fake": "Dawson’s on-air donation was faked and the check never cashed, per an anonymous source.",
        "verified": "The charity’s director confirmed the donation was received in full and showed a bank confirmation."},
}
PAIRS = {("s01", "s02"): 0.81, ("s01", "s09"): 0.78, ("s01", "s10"): 0.72, ("s04", "s13"): 0.77, ("s04", "s14"): 0.70, ("s06", "s15"): 0.74}
TEMPLATE = {1: "“{p}” is vague or sensational framing with no checkable attribution, a pattern the verdict model associates with fabricated stories.",
            0: "“{p}” points to a named, on-record or documentary source, the kind of attribution the verdict model associates with verified reporting."}


def locate(text, phrase, used):
    start = 0
    while True:
        i = text.find(phrase, start)
        assert i >= 0, phrase
        if all(i >= e or i + len(phrase) <= s for s, e in used):
            used.append((i, i + len(phrase)))
            return i, i + len(phrase)
        start = i + 1


def sentence_of(text, i):
    s = max(text.rfind("\n", 0, i) + 1, text.rfind(". ", 0, i) + 2 if text.rfind(". ", 0, i) >= 0 else 0)
    cands = [x for x in (text.find("\n", i), text.find(". ", i) + 1 if text.find(". ", i) >= 0 else -1) if x > 0]
    e = min(cands) if cands else len(text)
    return text[s:e].strip()


rows = []
for x in A:
    kws, used_t, used_b = [], [], []
    for n, p in enumerate(x["kt"]):
        s, e = locate(x["title"], p, used_t)
        kws.append({"phrase": p, "field": "title", "start": s, "end": e, "score": round(0.95 - 0.05 * n, 3), "sentence": x["title"]})
    for n, p in enumerate(x["kb"]):
        s, e = locate(x["body"], p, used_b)
        kws.append({"phrase": p, "field": "body", "start": s, "end": e, "score": round(0.9 - 0.06 * n, 3), "sentence": sentence_of(x["body"], s)})
    rat = dict(x["rat"]) if x["rat"] else {k["phrase"]: TEMPLATE[x["label"]].format(p=k["phrase"]) for k in kws}
    rat.pop("imminent", None)
    rat.pop("stunned", None)
    verdict = x["verdict"] or ("The article is labeled Fake because its claims rest on anonymous sources, hedges and sensational framing with nothing a reader could check."
                               if x["label"] == 1 else "The article is labeled True because its claims are attributed to named, on-record or published sources with specific dates and figures.")
    cps = []
    for (f, t), sim in PAIRS.items():
        if x["aid"] == f:
            cps.append({"article_id": t, "final_label": 0, "sim": sim, "claim_contrast": CONTRAST[x["event"]]})
        elif x["aid"] == t:
            cps.append({"article_id": f, "final_label": 1, "sim": sim, "claim_contrast": CONTRAST[x["event"]]})
    cps.sort(key=lambda c: -c["sim"])
    rows.append({"article_id": x["aid"], "dataset": "gossip", "title": x["title"], "body": x["body"], "label": x["label"], "split": "test",
                 "pred_label": x["label"], "pred_prob_fake": x["p"], "factcheck": None, "veracity_source": "model", "final_label": x["label"],
                 "keywords": kws, "rationales": rat, "verdict_rationale": verdict if x["aid"] != "s03" else "",
                 "event_id": x["event"], "counterparts": cps})

ids = [r["article_id"] for r in rows]
feed1 = ids[:18]
hist1_old, hist1_recent = ids[31:36], ids[18:31]
hist1 = [{"article_id": a, "time": BASE - 20 * DAY - (len(hist1_old) - i) * 3600} for i, a in enumerate(hist1_old)]
hist1 += [{"article_id": a, "time": BASE - 6 * DAY + i * 10 * 3600} for i, a in enumerate(hist1_recent)]
feed2 = ids[18:36]
h2_old, h2_recent = ["s05", "s07", "s08", "s12", "s11"], ["s03", "s01", "s04", "s06", "s02"]
hist2 = [{"article_id": a, "time": BASE - 30 * DAY + i * 3600} for i, a in enumerate(h2_old)]
hist2 += [{"article_id": a, "time": BASE - 5 * DAY + i * 20 * 3600} for i, a in enumerate(h2_recent)]
cache = [{"article_id": "s01", "kind": "keyword", "key": "imminent", "text": "Announcing that an official announcement is “imminent” lets the story claim future confirmation that never arrives; verified coverage reports what has already been confirmed.", "model": "sample", "prompt_version": "sample-v0"},
         {"article_id": "s03", "kind": "verdict", "key": "", "text": "The article is labeled True because the sequel is confirmed by a named studio chief at a public press event, with a release window, director and audited box-office figure.", "model": "sample", "prompt_version": "sample-v0"}]


def dump(name, items):
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


dump("articles.jsonl", rows)
dump("users.jsonl", [{"user_id": "demo_user_1", "history": hist1}, {"user_id": "demo_user_2", "history": hist2}])
dump("feeds.jsonl", [{"user_id": "demo_user_1", "feed": feed1}, {"user_id": "demo_user_2", "feed": feed2}])
dump("rationale_cache.jsonl", cache)
print(len(rows), "articles;", sum(len(r["keywords"]) for r in rows), "keywords")
