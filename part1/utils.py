import hashlib
import random
import re

TYPO_PROBABILITY = 0.45
TRANSFORM_SEED = 42
NEIGHBORS = {
    'q': 'wa', 'w': 'qesa', 'e': 'wrsd', 'r': 'etdf', 't': 'ryfg',
    'y': 'tugh', 'u': 'yihj', 'i': 'uojk', 'o': 'ipkl', 'p': 'ol',
    'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc',
    'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm',
    'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb',
    'b': 'vghn', 'n': 'bhjm', 'm': 'njk',
}
PROTECTED = {'no', 'not', 'never', 'neither', 'nor', 'nothing', 'nobody',
             'nowhere', 'without', 'hardly', 'barely', 'scarcely'}


def example_transform(example):
    return {**example, 'text': example['text'].lower()}


def custom_transform(example):
    text = example['text']
    # Repeat typos per review
    digest = hashlib.sha256(text.encode('utf-8')).digest()
    rng = random.Random(int.from_bytes(digest[:8], 'big') ^ TRANSFORM_SEED)

    def typo(match):
        word = match.group()
        if len(word) < 4 or word.lower() in PROTECTED or "'" in word or '’' in word:
            return word
        if rng.random() >= TYPO_PROBABILITY:
            return word
        index = rng.randrange(1, len(word) - 1)
        replacement = rng.choice(NEIGHBORS[word[index].lower()])
        if word[index].isupper():
            replacement = replacement.upper()
        return word[:index] + replacement + word[index + 1:]

    # HTML tags and apostrophe-containing words are left intact.
    parts = re.split(r'(<[^>]*>)', text)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"[A-Za-z]+(?:['’][A-Za-z]+)*", typo, parts[i])
    return {**example, 'text': ''.join(parts)}
