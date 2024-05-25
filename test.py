from datasets import load_dataset




test1 = load_dataset(f"opus_books", f"en-fr", split='train')

test2 = load_dataset(f"Helsinki-NLP/opus-100", f"af-en", split='train')


print(test1[0])
print(test2[0])

