# -*- coding: utf-8 -*-
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
def calculate_bleu_scores(reference, candidate, weights):
    reference_tokens = word_tokenize(reference)
    candidate_tokens = word_tokenize(candidate)

    return sentence_bleu([reference_tokens], candidate_tokens, weights=weights)


if __name__ == '__main__':

    Label = "This patient may have suffered a stroke - middle viscera. Recommended acupuncture methods include: Main points: Shuigou, Baihui, Neiguan. Auxiliary points: for 'closure' syndrome use the Twelve Well Points, Hegu, Taichong; for 'escape' syndrome use Guanyuan, Qihai, Shenque, etc. Method: Use the dirty method for Neiguan, strong stimulation for Shuigou, assessing effectiveness by eye moisture. Bleed the Twelve Well Points with a three-edged needle. For Guanyuan and Qihai, use large moxa cones, and for Shenque, use salt-separated moxibustion, stop when sweating ceases, pulse normalizes, and limbs warm."

    Qwen_7B_Chat = "This female patient's sudden fainting and speechlessness may be related to cerebrovascular disease. Acupuncture points like Taichong, Zusanli, Zhongwan are suggested for treatment. Disinfect the patient's skin first, then insert fine needles at these points. It's important to monitor the patient's response and avoid overstimulation. Also, maintain a balanced diet and rest. Remember, these are just suggestions; the actual treatment should be determined based on the patient's condition and doctor's advice."

    Qwen_14B_Chat = "The patient's symptoms indicate a possible stroke. Acupuncture at points like Baihui, Fengchi, Hegu, Taichong on the head, neck, hands, and feet is recommended. Start with disinfection, then use appropriate acupuncture needles and techniques. Prompt medical consultation is advised."

    ChatGLM3_32K = "Based on your description, this 58-year-old female patient likely has a stroke. Acupuncture can be effective for stroke, with recommended points including Baihui and Neiguan. Good habits and rehabilitation exercises are important. These are general recommendations; follow professional medical advice for specific treatment."

    ChatGLM3_6B = "Given your description, the 58-year-old woman's sudden collapse and inability to speak might be due to various health issues. As a professional acupuncturist, I recommend points like Baihui and Lianquan. Please note, this is not an official diagnosis or treatment plan; professional medical advice is necessary."

    Ours = "The patient likely has a stroke. Recommended acupuncture methods include: Main points: Neiguan, Shuigou, Baihui. Auxiliary points for different stroke symptoms include Wai Guan, Fengchi for actual stroke syndrome, and Zusanli, Taixi for deficiency syndrome. For confused mental state, use Zhongwan, Yongquan; for wind deviation of head and face, use Fengchi, Toutuo; for body deviation, use Fengchi, Quchi; for heart deviation, use Xinshu, Shenmen; for spleen, liver, lung deviation, use corresponding Yu points and channels. Method: Acute phase requires 2-3 acupuncture sessions daily with vigorous, rapid, and repeated needling. Chronic pain requires slow needling. Auxiliary points should be chosen based on syndrome differentiation."

    weights_options = [
        (1, 0, 0, 0),  # BLEU-1
        (0.5, 0.5, 0, 0),  # BLEU-2
        (0.33, 0.33, 0.33, 0),  # BLEU-3
        (0.25, 0.25, 0.25, 0.25)  # BLEU-4
    ]

    # Calculate and print BLEU scores for each candidate
    candidates = [Qwen_7B_Chat, Qwen_14B_Chat, ChatGLM3_32K, ChatGLM3_6B, Ours]
    for candidate in candidates:
        print(f"Candidate: {candidate}")
        for i, weights in enumerate(weights_options, 1):
            score = calculate_bleu_scores(Label, candidate, weights)
            print(f"BLEU-{i}: {score}")
        print("\n")
