from torchmetrics.text.bleu import BLEUScore
from translate import translate
import matplotlib.pyplot as plt

def calculate_bleu(target_texts, predicted_texts):
    bleu = BLEUScore()
    references = [target_texts]
    hypotheses = predicted_texts
    bleu_score = bleu(hypotheses, references)
    print('BLEU Score:', bleu_score)
    return bleu_score

def enumerate_weights():

    bleu_scores = []
    weights = []

    for i in range(100):  # Assuming you want to enumerate 10 weights
        translated_texts = []
        source_texts = []
        source, target, translated_text = translate(400 + i, f"{15:02d}")

        translated_texts.append(translated_text.strip())
        source_texts.append(target)



        bleu_score = calculate_bleu(source_texts, translated_texts)
        bleu_scores.append(bleu_score)
        weights.append(i)

    return weights, bleu_scores

weights, bleu_scores = enumerate_weights()

plt.plot(weights, bleu_scores, marker='o')
plt.xlabel('Sentences translated')
plt.ylabel('BLEU Score')
plt.title('BLEU Score vs Translations (After training inference)')
plt.grid(True)
plt.show()