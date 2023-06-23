from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import FileLink
import csv

# Example evaluation metrics data
models = ['Logistic Regression', 'Decision Tree', 'SVM', 'Naive-Bayes', 'XGboost','Random Forest', 'Voting(LR+DT+RF)', 'Gradientboosting', 'Adaboosting(DT)', 'Stacking(LR+DT)']
accuracy = ['0.9988939995084443', '0.9991397773954567', '0.997929792979298', '0.9855085508550855', '0.9993699369936994','0.9996137776061234', '0.9995786664794073', '0.9989993328885924', '0.9991046662687406','0.9991748885221726']
precision = ['0.7868852459016393', '0.7207207207207207', '0.7307692307692307', '0.16129032258064516', '0.9666666666666667','0.9871794871794872', '0.9743589743589743', '0.7808219178082192', '0.7043478260869566','0.9473684210526315']
recall = ['0.4897959183673469', '0.8163265306122449', '0.5428571428571428', '0.8571428571428571', '0.8285714285714286','0.7857142857142857', '0.7755102040816326', '0.5816326530612245', '0.826530612244898','0.5510204081632653']
f1_score = ['0.6037735849056604', '0.7655502392344496', '0.6229508196721311', '0.27149321266968324', '0.8923076923076922', '0.8750000000000001', '0.8636363636363635', '0.6666666666666666', '0.7605633802816901', '0.6967741935483871']

# Combine the data into a list of lists
data = [models, accuracy, precision, recall, f1_score]

# Transpose the data so that each model's metrics are in a row
data = list(map(list, zip(*data)))

# Print the table using tabulate
headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
print(tabulate(data, headers=headers, tablefmt='orgtbl'))
# Write table to CSV file
with open('table.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
    writer.writerows(data)

# Make table downloadable
display(FileLink('table.csv'))

# Convert the metric scores to float and round off to two decimal places
accuracy = [round(float(score), 2) for score in accuracy]
precision = [round(float(score), 2) for score in precision]
recall = [round(float(score), 2) for score in recall]
f1_score = [round(float(score), 2) for score in f1_score]


# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the accuracy, precision, recall, and f1-score as bars in the first subplot
bar_width = 0.2
x = np.arange(len(models))

ax1.bar(x - 1.5*bar_width, accuracy, bar_width, label='Accuracy')
ax1.bar(x - 0.5*bar_width, precision, bar_width, label='Precision')
ax1.bar(x + 0.5*bar_width, recall, bar_width, label='Recall')
ax1.bar(x + 1.5*bar_width, f1_score, bar_width, label='F1-Score')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.set_xlabel('Models')
ax1.set_ylabel('Metric Scores')
ax1.set_title('Evaluation Metrics for Different Models')
ax1.legend()

# Plot the accuracy, precision, recall, and f1-score as lines in the second subplot
ax2.plot(models, accuracy, label='Accuracy')
ax2.plot(models, precision, label='Precision')
ax2.plot(models, recall, label='Recall')
ax2.plot(models, f1_score, label='F1-Score')
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.set_xlabel('Models')
ax2.set_ylabel('Metric Scores')
ax2.set_title('Evaluation Metrics for Different Models')
ax2.legend()

# Adjust the layout and display the figure
fig.tight_layout()
plt.show()