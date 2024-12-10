

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler=None, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        num_batches = 0

        for X_t, Y_t in dataloader:

            X_t = X_t.to(self.device)
            Y_t = Y_t.to(self.device)

            if len(X_t.shape) == 3:
                S, N, F = X_t.shape
            else:  
                B, S, N, F = X_t.shape # (batch_size, sequence_length, num_points, features)

            pred = self.model(X_t)
            loss = self.criterion(pred, Y_t)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        self.scheduler.step()
        return total_loss / num_batches

    def train(self, dataloader, epochs=1000):
        for epoch in range(epochs):
            avg_loss = self.train_one_epoch(dataloader)
            if self.scheduler:
                self.scheduler.step()

            print(f"Epoch {epoch}: Loss {avg_loss:.5f}")
