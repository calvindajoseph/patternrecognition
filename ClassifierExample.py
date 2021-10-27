"""
An example how we could use the final model.
"""

# Ignore any warnings
import warnings
warnings.filterwarnings("ignore")

# Import model classifier
from ModelClasses import ModelClassifier

# Set the classifier
# May take some time, especially if loaded to CPU
classifier = ModelClassifier()

# Set dummy texts
text_one = "I love the weather today."
text_two = "The weather today is awful."

# Get prediction
prediction = classifier.predict(text_one, text_two)

# Print the results to console
print()
print(f'Text One:  {text_one}')
print(f'Text Two:  {text_two}')
print()
print(f'Prediction: {prediction["predicted_class"]}')
print(f'Confidence: {prediction["confidence"].item()}')
print('Prediction Probabilities:')
for key in prediction["probabilities"]:
    print(f'{key}: {prediction["probabilities"][key]}')