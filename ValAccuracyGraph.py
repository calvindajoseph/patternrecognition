"""
File to plot out validation accuracy as epoch progresses
"""

# Import matplotlib
import matplotlib.pyplot as plt

# Set the accuracies
accuracy_first = [0.8165,
                  0.8215,
                  0.8205,
                  0.8165,
                  0.813]

accuracy_second = [0.821,
                   0.826,
                   0.8195]

accuracy_third = [0.814,
                  0.8165,
                  0.81]

# Set the epochs
epoch_first = [1, 2, 3, 4, 5]

epoch_second = [1, 2, 3]

# Create plots with pre-defined labels.
fig, ax = plt.subplots(figsize=(5,3), dpi=320)

# Plot the lines
ax.plot(epoch_second, accuracy_second, label='lr: 2e-5')
ax.plot(epoch_second, accuracy_third, label='lr: 3e-5')
ax.plot(epoch_first, accuracy_first, label='lr: 5e-5')

# Set labels
ax.set_title("Epoch vs Validation Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")

# Set legend
legend = ax.legend(loc='lower right')

# Show the plot
plt.show()

# Save the plot
fig.savefig("evaluation/val_accuracy_graph.png")