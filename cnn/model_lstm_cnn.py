import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

# ----------------------------
# Define the Character Set
# ----------------------------
# We reserve index 0 for the CTC blank label.
characters = "0123456789abcdefghijklmnopqrstuvwxyz"
char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}
idx_to_char = {idx + 1: char for idx, char in enumerate(characters)}
num_classes = len(characters) + 1  # +1 for the blank label (index 0)

# ----------------------------
# Dataset Definition
# ----------------------------
class CaptchaDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the preprocessed CAPTCHA images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir)
                            if os.path.isfile(os.path.join(root_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Get filename and derive label from filename (without extension)
        img_name = self.image_files[index]
        label_str = os.path.splitext(img_name)[0]
        # Convert label string to list of integer indices using the mapping dictionary.
        label = [char_to_idx[c] for c in label_str if c in char_to_idx]

        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('L')  # convert to grayscale

        # If no transform is provided, use a default transform: resize and to tensor.
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((50, 200)),  # height=50, width=200
                transforms.ToTensor(),          # scales image to [0,1]
            ])
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# ----------------------------
# Collate Function for DataLoader
# ----------------------------
def collate_fn(batch):
    """
    Prepares batches for CTCLoss.
    Returns:
        images: Tensor of shape (batch, 1, H, W)
        targets: 1D concatenated tensor of all target labels in the batch.
        target_lengths: Tensor containing the lengths of each label sequence.
    """
    images, labels = zip(*batch)
    images = torch.stack(images, 0)  # shape: (batch, C, H, W)
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    targets = torch.cat([label for label in labels])
    return images, targets, target_lengths

# ----------------------------
# CRNN Model Definition
# ----------------------------
class CRNN(nn.Module):
    def __init__(self, img_height=50, num_channels=1, num_classes=num_classes, lstm_hidden=128):
        super(CRNN, self).__init__()
        # Convolutional layers to extract features.
        self.cnn = nn.Sequential(
            # Block 1: conv -> batchnorm -> relu -> maxpool (2x2)
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: conv -> batchnorm -> relu -> maxpool (2x2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: conv -> batchnorm -> relu -> maxpool (2,1)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        # After CNN layers:
        # For input size (50,200): 
        #   Height: 50 -> 25 -> 12 -> 6
        #   Width: 200 -> 100 -> 50 -> 50   (since last pooling does not change width)
        self.conv_output_height = 6
        self.conv_output_width = 50
        self.num_filters = 128

        # Fully connected layer to reduce feature dimension.
        self.fc = nn.Linear(self.conv_output_height * self.num_filters, 64)

        # Recurrent layers: two-layer bidirectional LSTM.
        self.lstm = nn.LSTM(64, lstm_hidden, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.25)
        # Final classifier: project LSTM output to number of classes.
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        # x: (batch, channels, height, width)
        conv = self.cnn(x)  # (batch, 128, H', W') where H'=6, W'=50
        batch_size, channels, height, width = conv.size()
        # Permute to (batch, width, channels, height) so that width becomes the time dimension.
        conv = conv.permute(0, 3, 1, 2)
        conv = conv.contiguous().view(batch_size, width, channels * height)  # (batch, width, features)
        conv = self.fc(conv)  # (batch, width, 64)

        # Pass through LSTM layers (expects input shape: (batch, seq_len, input_size)).
        lstm_out, _ = self.lstm(conv)  # (batch, width, 2*lstm_hidden)
        output = self.classifier(lstm_out)  # (batch, width, num_classes)
        # Transpose to shape (seq_len, batch, num_classes) as expected by CTCLoss.
        output = output.transpose(0, 1)
        # Apply log softmax along the class dimension.
        output = nn.functional.log_softmax(output, dim=2)
        return output

# ----------------------------
# Greedy Decoder for Predictions
# ----------------------------
def decode_predictions(preds):
    """
    Decodes predictions by taking the argmax at each time step,
    collapsing repeated characters, and removing blanks (index 0).
    Args:
        preds: Tensor of shape (T, batch, num_classes)
    Returns:
        A list of predicted strings (one for each batch sample).
    """
    preds = preds.cpu().detach().numpy()
    pred_strings = []
    for i in range(preds.shape[1]):
        pred = np.argmax(preds[:, i, :], axis=1)
        decoded = []
        previous = -1
        for p in pred:
            if p != previous and p != 0:
                decoded.append(idx_to_char.get(p, ''))
            previous = p
        pred_strings.append("".join(decoded))
    return pred_strings

# ----------------------------
# Training and Validation Functions
# ----------------------------
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (images, targets, target_lengths) in enumerate(train_loader):
        images = images.to(device)
        optimizer.zero_grad()
        # Forward pass: output shape -> (T, batch, num_classes)
        output = model(images)
        T, batch_size, _ = output.size()
        # Create input lengths (all equal to T).
        input_lengths = torch.full(size=(batch_size,), fill_value=T, dtype=torch.long).to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        loss = criterion(output, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, device, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, targets, target_lengths in val_loader:
            images = images.to(device)
            output = model(images)
            T, batch_size, _ = output.size()
            input_lengths = torch.full(size=(batch_size,), fill_value=T, dtype=torch.long).to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            loss = criterion(output, targets, input_lengths, target_lengths)
            total_loss += loss.item()

            # Decode predictions and compare with ground truth.
            pred_strings = decode_predictions(output)
            start = 0
            for i, length in enumerate(target_lengths):
                gt_indices = targets[start:start+length].cpu().numpy()
                gt_str = "".join([idx_to_char.get(int(idx), '') for idx in gt_indices])
                start += length
                if pred_strings[i] == gt_str:
                    correct += 1
                total_samples += 1

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total_samples if total_samples > 0 else 0
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    return avg_loss, accuracy

# ----------------------------
# Main Training Routine
# ----------------------------
def main():
    # Hyperparameters and settings
    batch_size = 32
    epochs = 100
    learning_rate = 0.005
    data_dir = "../data/preprocessed_images"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transform: resize images to (50,200) and convert to tensor.
    transform = transforms.Compose([
        transforms.Resize((50, 200)),
        transforms.ToTensor(),
    ])

    # Load dataset and perform a train/validation split (90% train, 10% validation).
    dataset = CaptchaDataset(data_dir, transform=transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   collate_fn=collate_fn)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn)

    # Initialize model, CTC loss, and optimizer.
    model = CRNN().to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, criterion, optimizer, epoch)
        val_loss, val_accuracy = validate(model, device, val_loader, criterion)

    # Save the trained model.
    torch.save(model.state_dict(), "crnn_captcha.pth")
    print("Model saved as crnn_captcha.pth")

    # Optional: run inference on one batch from the validation set.
    model.eval()
    with torch.no_grad():
        for images, targets, target_lengths in val_loader:
            output = model(images)
            pred_strings = decode_predictions(output)
            print("Predictions on validation batch:", pred_strings)
            break

if __name__ == "__main__":
    main()
