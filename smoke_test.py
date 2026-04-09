from transformers import pipeline

MODEL_NAME = "unitary/toxic-bert"

def main():
    clf = pipeline("text-classification", model=MODEL_NAME)

    samples = [
        "I hope you have a great day!",
        "You are disgusting and useless.",
        "Shut up.",
        "Hi"
    ]

    for text in samples:
        pred = clf(text)[0]  # list of 1 dict
        print("TEXT:", text)
        print("RAW PRED:", pred)
        print("-" * 50)

if __name__ == "__main__":
    main()