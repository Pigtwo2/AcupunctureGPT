# -*- coding: utf-8 -*-
import nltk
from collections import Counter

nltk.download('punkt')


def ngrams(tokens, n):
    return list(zip(*[tokens[i:] for i in range(n)]))


def lcs(a, b):
    lengths = [[0 for _ in range(len(b) + 1)] for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    return lengths[-1][-1]


def calculate_rouge(reference, candidate, n=1):
    ref_tokens = nltk.word_tokenize(reference)
    cand_tokens = nltk.word_tokenize(candidate)
    ref_ngrams = Counter(ngrams(ref_tokens, n))
    cand_ngrams = Counter(ngrams(cand_tokens, n))
    overlap = sum((ref_ngrams & cand_ngrams).values())
    ref_count = sum(ref_ngrams.values())
    cand_count = sum(cand_ngrams.values())
    precision = overlap / cand_count if cand_count else 0
    recall = overlap / ref_count if ref_count else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    lcs_length = lcs(ref_tokens, cand_tokens)
    lcs_precision = lcs_length / cand_count if cand_count else 0
    lcs_recall = lcs_length / ref_count if ref_count else 0
    lcs_f1 = 2 * lcs_precision * lcs_recall / (lcs_precision + lcs_recall) if (lcs_precision + lcs_recall) else 0

    return {"ROUGE-N": {"precision": precision, "recall": recall, "f1": f1},
            "ROUGE-L": {"precision": lcs_precision, "recall": lcs_recall, "f1": lcs_f1}}


if __name__ == '__main__':
    sentence1 = " "  # Your reference text here.
    sentence2 = " "  # Your candidate text here."

    ChatGLM3_6B = "根据您的描述，这位58岁的女性突然出现昏倒并且不能说话的情况，可能是由于脑部疾病、心血管疾病、内分泌疾病、神经系统疾病或精神疾病等原因导致的。在这里，我作为一名专业针灸师，推荐以下穴位和操作，但请注意，这不是正式的诊断和治疗方案，具体还需结合专业医生的建议。1. 穴位推荐：- 百会穴：位于头部的顶部，头发际线附近，头维和项结合的凹陷处。刺激百会穴可以帮助改善脑部功能，缓解言语困难。- 廉泉穴：位于颈部，当喉结下方约1横指处。刺激廉泉穴可以调节神经系统，有助于恢复言语功能。2. 操作推荐：- 百会穴：使用毫针进行针刺操作，深度约10-15毫米，每次刺激约20秒，每日3-5次。- 廉泉穴：同样使用毫针进行针刺操作，深度约10-15毫米，每次刺激约20秒，每日3-5次。请注意，针灸操作需在专业医生的指导下进行，以保证安全和效果。希望这些建议能对您有所帮助。祝这位女性早日康复！ "  # GPT3_5 sentence here.
    Ours = " 该患者可能患有中风，推荐以下针灸治疗方法："  # GPT4 sentence here.
    Label = "该患者可能患有中风-中脏腑，推荐以下针灸治疗方法："  # Label sentence here."  # Label sentence here.

    rouge_scores = calculate_rouge(Ours, Label, n=1)
    print("ROUGE-N:", rouge_scores["ROUGE-N"])
    print("ROUGE-L:", rouge_scores["ROUGE-L"])
