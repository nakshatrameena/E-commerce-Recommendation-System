def precision_at_k(actual, predicted, k=3):
    predicted = predicted[:k]
    return len(set(actual) & set(predicted)) / k


# Example evaluation
actual_products = [2, 3]
predicted_products = [2, 4, 3]

print("Precision@3:", precision_at_k(actual_products, predicted_products))
