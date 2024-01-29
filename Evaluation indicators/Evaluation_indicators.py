# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from SentenceSimilarityModel import SentenceSimilarityModel, set_seed



def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def semantic_similarity(sent1, sent2):
    vectorizer = TfidfVectorizer().fit([sent1, sent2])
    tfidf_matrix = vectorizer.transform([sent1, sent2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0]


def length_penalty(sent1, sent2, lambda_coeff):
    diff = abs(len(sent1) - len(sent2))
    return np.exp(-lambda_coeff * diff)


def evaluation_metric(sentA, sentB, Parm, alpha=0.05, beta=0.9, gamma=0.05, lambda_coeff=0.01):
    jaccard = jaccard_similarity(sentA.split(), sentB.split())
    sem_sim = semantic_similarity(sentA, sentB)
    len_pen = length_penalty(sentA, sentB, lambda_coeff)
    print("sem_sim", sem_sim)

    return alpha * jaccard + beta * Parm + gamma * len_pen


if __name__ == '__main__':
    # Given sentences
    Label = "该患者可能患有中风-中脏腑，推荐以下针灸治疗方法：主穴：水沟，百会，内关。配穴：闭证配十二井穴、合谷、太冲；脱证配关元、气海、神阙等。操作：内关用污法，水沟用强刺激，以眼球湿润为度。十二井穴用三楼针点刺出血。关元、气海用大艾炷灸，神阙用隔盐灸，不计壮数，以汗止、脉起、肢温为度。"  # Label sentence here.
    Sentence = "该患者可能患有中风，推荐以下针灸治疗方法：主穴：内关、水沟、百会配穴：中风实证，配外关、风池；中风虚证，配足三里、太溪。神昏配中脘、涌泉；风偏头面配风池、头维；风偏身配风池、曲池；风偏心配心俞、神门；风偏脾配脾俞、风池、足三里；风偏肝配肝俞、风池、足三里；风偏肺配肺俞、风池、足三里；配肾俞、心俞、肝俞、脾俞、肾经、心经。操作：急性期每日2～3次针灸，急性期针刺方法重刺、急刺、重复刺。慢痛慢刺。配穴按虚实辨证配穴。 "
    model = SentenceSimilarityModel()
    similarity_score = model(Label, Sentence).item()
    print(" similarity_score", similarity_score)
    score = evaluation_metric(Label, Sentence, similarity_score)
    print("Evaluation indicators score is ", score)
