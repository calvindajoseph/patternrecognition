import warnings
warnings.filterwarnings("ignore")

from ModelClasses import ModelClassifier

classifier = ModelClassifier()

text_one = "I am eating noodles"
text_two = "The effect of noodles"

prediction = classifier.predict(text_one, text_two)

print()
print(f'Text One:  {text_one}')
print(f'Text Two:  {text_two}')
print()
print(f'Prediction: {prediction["predicted_class"]}')
print(f'Confidence: {prediction["confidence"].item()}')
print('Prediction Probabilities:')
print(prediction["probabilities"])