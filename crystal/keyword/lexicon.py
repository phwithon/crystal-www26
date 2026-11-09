"""데모 색칠용 단서 사전 (2026-11-03). FAKE 판정 기사에는 FAKE 목록만, TRUE 판정 기사에는 TRUE 목록만 본문에서 매칭해 칠한다.
각 항목: (정규식, 설명 템플릿). 정규식은 대소문자 무시, 단어 경계 기준. 설명의 {p}는 실제 본문 표기로 바뀐다.
LLM이 고른 단서가 아니라 사람이 정한 표이며, IG/LIME 후보(lime_words.py)를 참고해 만들었다."""

ANON = ("An unnamed voice is doing the talking here. '{p}' gives no name, job, or outlet that a reader could check, "
        "so whatever follows rests entirely on trust. Fabricated celebrity stories lean on exactly this kind of source "
        "because no one can come forward to contradict it.")
HEDGE = ("'{p}' is the writer's own warning label: it means the claim was not confirmed by the people involved. "
         "When a story's key points arrive wrapped in words like this, the article is passing along talk rather than "
         "reporting something it verified.")
LOADED = ("'{p}' is a loaded, emotional word chosen to provoke a reaction, not to describe a checkable fact. "
          "Reliable reporting usually states what happened plainly; heavy drama words like this are a common "
          "ingredient of tabloid pieces built to be shared rather than trusted.")
SECRET = ("'{p}' promises hidden, behind-the-scenes knowledge. The problem is that truly secret information has no "
          "on-record source by definition, so the story cannot show how it knows this. That gap is a classic mark of "
          "made-up gossip.")
PROFANE = ("Crude language like '{p}' is a sign the piece is written for shock and attention rather than accuracy. "
           "Careful news writing almost never needs it, while gossip pages use it to make a story feel raw and urgent.")
FEELING = ("'{p}' describes someone's private feelings. Unless the person said it themselves on the record, a writer "
           "has no way to know this — so the article is narrating an inner life it could only have imagined or heard "
           "secondhand.")
EXCL = ("'{p}' sells the story as insider access. But 'exclusive' only means no one else has it — and when the "
        "information comes from unnamed people, there is also no one else who can confirm it.")

FAKE = [
    (r"insiders?", ANON), (r"(?:a |an |one |another |multiple |several )?sources?(?: close to [\w’']+(?: \w+)?)?", ANON),
    (r"a (?:close )?(?:friend|pal)", ANON), (r"sources? (?:say|says|said|tell|tells|told)", ANON),
    (r"reportedly", HEDGE), (r"allegedly", HEDGE), (r"apparently", HEDGE), (r"supposedly", HEDGE),
    (r"rumou?r(?:s|ed)?", HEDGE), (r"speculation|speculating|speculate", HEDGE), (r"according to (?:a |an )?(?:new )?reports?", HEDGE),
    (r"may have|might have|could be", HEDGE), (r"is said to|are said to", HEDGE),
    (r"jerk", LOADED), (r"bombshell", LOADED), (r"shocking|shocked", LOADED), (r"savage", LOADED),
    (r"slam(?:s|med)?", LOADED), (r"meltdown", LOADED), (r"furious|fuming|outraged", LOADED),
    (r"freak(?:s|ed)? out|freak outs?", LOADED), (r"explosive", LOADED), (r"scandal(?:ous)?", LOADED),
    (r"bizarre", LOADED), (r"insane|crazy", LOADED), (r"drama", LOADED), (r"feud(?:ing)?", LOADED),
    (r"shade[ds]?|shady", LOADED), (r"cheat(?:ing|ed)?", LOADED), (r"nervous wreck", LOADED),
    (r"desperate(?:ly)?", LOADED), (r"wild", LOADED), (r"salacious", LOADED), (r"arrogant", LOADED),
    (r"secret(?:ly|s)?", SECRET), (r"behind closed doors", SECRET), (r"mystery", SECRET), (r"hid(?:ing|es|e)?(?: out)?", SECRET),
    (r"heartbroken|devastated|jealous(?:y)?|obsessed|terrified", FEELING),
    (r"(?:world )?exclusive(?:ly)?", EXCL),
    (r"damn|hell|shit|crap|fuck(?:ing)?|bitch|wtf|omg", PROFANE),
]

STATE = ("The claim is tied to a formal statement. '{p}' tells the reader the words came from the people involved or "
         "their official representatives, on the record — something anyone can look up, quote, and hold them to.")
NAMED = ("'{p}' names who is speaking. An identified spokesperson or representative is accountable for what they say, "
         "which is very different from an anonymous 'source' — it gives the reader a real person behind the claim.")
CONFIRM = ("'{p}' signals that the fact was confirmed by someone in a position to know, not guessed or rumored. "
           "Confirmation from the subject or an official party is the basic test genuine news passes and gossip skips.")
OFFICIAL = ("'{p}' points to an official record — the kind of document or institution (police, courts, filings) that "
            "exists independently of the article. Stories anchored to records like this can be checked line by line.")
SELF = ("The person posted or said this themselves. '{p}' means the source is a public message from the subject, "
        "which readers can see directly — the article is relaying it, not inventing it.")
INTERVIEW = ("'{p}' places the quotes in a real, recorded conversation. When a subject speaks in a named interview, "
             "their words exist on tape or in print, so the article's claims can be traced back to them.")
DATE = ("A specific date like '{p}' gives the story a concrete anchor. Real events happen at a time that can be "
        "checked against other coverage; vague timing ('recently', 'lately') is what fabricated stories prefer.")
DAY = DATE.replace("A specific date like", "A specific day like")
WIRE = ("'{p}' credits an established news organization with its own reporting standards. Citing a named outlet "
        "means the information has a traceable origin outside this article.")

TRUE = [
    (r"(?:joint |in a |a )?statement", STATE), (r"spokes(?:person|man|woman)", NAMED), (r"representative|rep\b|publicist|attorney|lawyer", NAMED),
    (r"confirm(?:ed|s|ing)?", CONFIRM), (r"announc(?:ed|es|ing|ement)", CONFIRM), (r"officially|official", CONFIRM),
    (r"police|court|judge|filed|filing|documents?|records?|autopsy|coroner|lawsuit|charged|arrested|investigation", OFFICIAL),
    (r"(?:wrote|posted|shared) (?:on|to|via) (?:Instagram|Twitter|Facebook)|Instagram post|tweeted|posted", SELF),
    (r"(?:in|during) an? (?:new |recent )?interview|told (?:People|ET|E! News|Us Weekly|Billboard|Variety|Vogue|the [A-Z]\w+)", INTERVIEW),
    (r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)(?:, (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w* \d{1,2})?", DAY),
    (r"(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\w*\.? \d{1,2}(?:, \d{4})?", DATE),
    (r"Associated Press|Reuters|AP\b|BBC|CNN|The New York Times", WIRE),
]
