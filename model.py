import torch
import os
import torch.nn as nn
import torch.optim as optim

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, num_factors=10):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(num_users, num_factors, sparse=True)
        self.item_factors = nn.Embedding(num_items, num_factors, sparse=True)

    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)

def train_model(model, user_movie_ratings_matrix, num_epochs=10, learning_rate=0.01):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    user_movie_ratings_matrix = torch.tensor(user_movie_ratings_matrix.values, dtype=torch.float32)
    user_ids, movie_ids = user_movie_ratings_matrix.nonzero(as_tuple=True)
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(user_ids, movie_ids)
        loss = criterion(predictions, user_movie_ratings_matrix[user_ids, movie_ids])
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

def load_model(model_file, num_users, num_items, num_factors=10):
    model = MatrixFactorization(num_users, num_items, num_factors)
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
    return model

def save_model(model, model_file):
    torch.save(model.state_dict(), model_file)
