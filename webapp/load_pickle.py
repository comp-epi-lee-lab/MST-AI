import pickle

with open("/Users/santhiyatheanraj/PycharmProjects/MST_skin_tone/MODEL_IDIR/trial_0000.pckl", "rb") as f:
    data = pickle.load(f)

print(data)