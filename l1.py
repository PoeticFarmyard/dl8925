import math

class LVQ:
    def winner(self, weights, sample):
        # Calculate Euclidean distance to each weight vector
        distances = []
        for w in weights:
            dist = sum((sample[i] - w[i])**2 for i in range(len(sample)))
            distances.append(dist)
        return distances.index(min(distances))  # return index of closest weight vector

    def update(self, weights, sample, J, alpha, actual):
        for i in range(len(weights[J])):
            if actual == J:
                # Move closer to sample
                weights[J][i] += alpha * (sample[i] - weights[J][i])
            else:
                # Move away from sample
                weights[J][i] -= alpha * (sample[i] - weights[J][i])

def main():
    # Input samples
    x = [[0, 0, 1, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 1],
         [0, 1, 1, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0]]
    
    # Corresponding class labels
    y = [0, 1, 0, 1, 1, 1]

    # Initialize weights using first two samples (one from each class)
    weights = [x[0][:], x[1][:]]  # deepcopy

    # Remove those samples from training set
    x = x[2:]
    y = y[2:]

    lvq = LVQ()
    alpha = 0.1
    epochs = 3

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        for i in range(len(x)):
            sample = x[i]
            label = y[i]

            # Find winner weight vector (closest)
            J = lvq.winner(weights, sample)

            # Update weights based on classification result
            lvq.update(weights, sample, J, alpha, label)

            print(f"Sample {sample} (class {label}) -> classified as {J}")
            print("Updated Weights:", weights)

if __name__ == "__main__":
    main()
