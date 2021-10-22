import warnings
warnings.filterwarnings("ignore")

from ModelClasses import ModelClassifier

classifier = ModelClassifier()

text_one = "I love the weather today."
text_two = "The weather today is awful."

prediction = classifier.predict(text_one, text_two)

print()
print(f'Text One:  {text_one}')
print(f'Text Two:  {text_two}')
print()
print(f'Prediction: {prediction["predicted_class"]}')
print(f'Confidence: {prediction["confidence"].item()}')
print('Prediction Probabilities:')
for key in prediction["probabilities"]:
    print(f'{key}: {prediction["probabilities"][key]}')