"""
File to plot out validation accuracy as epoch progresses
"""

# Import matplotlib
import matplotlib.pyplot as plt

# Set the accuracies
val_5e = [0.8165,
          0.8215,
          0.8205,
          0.8165,
          0.813]

val_3e = [0.814,
          0.8165,
          0.81]

val_2e = [0.821,
          0.826,
          0.8195]

train_5e = [0.7698125,
            0.8845,
            0.9548125,
            0.983375,
            0.992125]

train_3e = [0.7819375,
            0.8888125,
            0.952875]

train_2e = [0.7881875,
            0.893625,
            0.9521875]

# Set the epochs
epoch_five = [1, 2, 3, 4, 5]

epoch_three = [1, 2, 3]

# Create plots with pre-defined labels.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 3), dpi=320)

# Plot the training accuracy
ax1.plot(epoch_three, train_2e, label='train lr: 2e-5')
ax1.plot(epoch_three, train_3e, label='train lr: 3e-5')
ax1.plot(epoch_five, train_5e, label='train lr: 5e-5')

# Set labels for training
ax1.set_title("Epoch vs Training Accuracy", fontsize=8)
ax1.set_xlabel("Epoch", fontsize=5)
ax1.set_ylabel("Accuracy", fontsize=5)
ax1.tick_params(axis='x', labelsize=5)
ax1.tick_params(axis='y', labelsize=5)

# Plot the validation accuracy
ax2.plot(epoch_three, val_2e, label='val lr: 2e-5')
ax2.plot(epoch_three, val_3e, label='val lr: 3e-5')
ax2.plot(epoch_five, val_5e, label='val lr: 5e-5')

# Set labels for training
ax2.set_title("Epoch vs Validation Accuracy", fontsize=8)
ax2.set_xlabel("Epoch", fontsize=5)
ax2.set_ylabel("Accuracy", fontsize=5)
ax2.tick_params(axis='x', labelsize=5)
ax2.tick_params(axis='y', labelsize=5)

# Set legend
legend = ax1.legend(loc='lower right', prop={'size': 5})
legend = ax2.legend(loc='lower right', prop={'size': 5})

# Show the plot
plt.tight_layout()
plt.show()

# Save the plot
fig.savefig("evaluation/val_accuracy_graph.png")