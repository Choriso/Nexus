from predict import predict_topic
classes = [
    'alt.atheism',
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
    'misc.forsale',
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey',
    'sci.crypt',
    'sci.electronics',
    'sci.med',
    'sci.space',
    'soc.religion.christian',
    'talk.politics.guns',
    'talk.politics.mideast',
    'talk.politics.misc',
    'talk.religion.misc'
]
text = "Nvidia RTX 3090 is a GPU for gaming and artificial intelligence applications."
predictions = predict_topic(text, top_n=3)

for topic, prob in predictions:
    print(f"Тема: {classes[topic]}, вероятность: {prob:.2f}")
