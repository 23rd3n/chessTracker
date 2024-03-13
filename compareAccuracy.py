import matplotlib.pyplot as plt

listGt= []
with open('GTpieces.txt', 'r') as file:
    lines = file.readlines()
    
    for line in lines:
        row = line.strip().split(' ')
        row = [int(num) for num in row]
        listGt.append(row)

listPieces = []
with open('pieces.txt', 'r') as file:
    lines = file.readlines()
    
    for line in lines:
        row = line.strip().split(' ')
        row = [int(num) for num in row]
        listPieces.append(row)

def calculate_accuracy(list1, list2):
    return sum(1 for x, y in zip(list1, list2) if x == y) / len(list1)

# Example usage
accuracyList = []
for rowGt, rowPieces in zip(listGt, listPieces):
    if (sum(rowGt) == 896 or sum(rowPieces) == 896):
        continue
    accuracy = calculate_accuracy(rowGt, rowPieces)
    accuracyList.append(accuracy)

# Calculate the mean accuracy
mean_accuracy = sum(accuracyList) / len(accuracyList)

plt.plot(accuracyList, label=f"Frame Accuracy")
plt.axhline(y=mean_accuracy, color='r', linestyle='-', label=f"Mean Accuracy: {mean_accuracy:.4f}")
plt.xlabel('Frame Index')
plt.ylabel('Accuracy')
plt.title('YoloV8n Multi Camera Accuracy')
plt.legend() # This will display the legend
plt.show()
