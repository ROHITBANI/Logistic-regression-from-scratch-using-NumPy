import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize logistic regression parameters
        
        Parameters:
        learning_rate (float): Step size for gradient descent
        num_iterations (int): Number of iterations for training
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        """
        Compute sigmoid function
        z = w^T * x + b
        """
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Train the logistic regression model
        
        Parameters:
        X (numpy array): Training data of shape (n_samples, n_features)
        y (numpy array): Target values of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.num_iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        X (numpy array): Test data of shape (n_samples, n_features)
        
        Returns:
        numpy array: Predicted class labels
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return (y_predicted >= 0.5).astype(int)
    
    def accuracy(self, y_true, y_pred):
        """
        Compute accuracy of predictions
        
        Parameters:
        y_true (numpy array): True labels
        y_pred (numpy array): Predicted labels
        
        Returns:
        float: Accuracy score
        """
        return np.mean(y_true == y_pred)

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Split data into train and test
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create and train model
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = model.accuracy(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")