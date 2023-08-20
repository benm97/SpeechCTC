from utils import load_data
from torch.utils.data import Dataset, DataLoader
from config import LAS_VOCAB, N_MFCC, SEQUENCE_LENGTH_MAX, MFCC_MAX_FRAME
import librosa
import numpy as np
from SpeechRecognizer import SpeechRecognitionModel
import torch.nn as nn
import torch.optim as optim
import torch
from jiwer import wer

def sequence_to_sentence(sequences):
    labels_words = []
    for label in sequences:
        sentence = []
        for i in label:
            if i == 0:
                break
            char_ = LAS_VOCAB.inverse[int(i)]
            if not char_.startswith('<'):
                sentence.append(char_)
        labels_words.append(sentence)
    sentences = ["".join(words) for words in labels_words]
    return sentences

# Defining a Dataset for training
class AudioDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        self.vocabulary = LAS_VOCAB

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]["audio_path"]
        label = self.file_paths[idx]["label"]

        # transform label to vector
        label_sequence = np.zeros(SEQUENCE_LENGTH_MAX)
        words = label.split(" ")
        i = 0
        for word in words:
            for character in word:
                label_sequence[i] = self.vocabulary[character.lower()]
                i += 1
            label_sequence[i] = self.vocabulary[" "]
            i += 1
        label_sequence[i] = self.vocabulary["<eos>"]

        # Load and process the data from the file
        data, length = self.load_data(file_path)

        return data, label_sequence, len(label)

    def load_data(self, file_path):
        data, sr = librosa.load(file_path)

        # data augmentation adding noise
        # data = data + np.random.normal(0, 1, len(data))

        if self.transform:
            data = self.transform(data)
        length = data.shape[0]
        padded_data = np.zeros((MFCC_MAX_FRAME, N_MFCC), dtype=np.float64)
        padded_data[:data.shape[0], :] = data
        return padded_data, length


def extract_mfccs(wav):
    mfccs = librosa.feature.mfcc(y=wav, n_mfcc=N_MFCC).T
    return mfccs

def load_data_to_dataloader(data_type="train", batch_size=32, transform=extract_mfccs):
    data_paths = load_data(data_type=data_type)
    data_set = AudioDataset(data_paths, transform=transform)
    dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == '__main__':
    train_dataloader = load_data_to_dataloader(data_type="train", batch_size=32,
                                               transform=extract_mfccs)
    validation_dataloader = load_data_to_dataloader(data_type="train", batch_size=32,
                                                    transform=extract_mfccs)

    # # Define hyperparameters and training params
    input_size = N_MFCC  # Example: MFCC feature size
    hidden_size = 64
    output_size = len(LAS_VOCAB)  # vocab size
    learning_rate = 0.005
    num_epochs = 200

    # Initialize the LAS model
    model = SpeechRecognitionModel((300, 39), output_size)

    # load existing model
    # model_path = 'model_epoch_49.pth'
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint)

    # Define loss and optimizer
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float('inf')  # Initialize with a high value
    model = model.float()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_wer = 0
        for encoder_inputs, labels_sequence, lengths in train_dataloader:
            optimizer.zero_grad()

            # Forward pass through the model
            outputs = model(encoder_inputs.float())

            labels = labels_sequence
            outputs_reshape = inverted_tensor = torch.transpose(outputs, 0, 1)

            loss = criterion(outputs_reshape.float(), labels, torch.ones(labels.shape[0],
                                                                         dtype=torch.long) *290,
                             torch.tensor(lengths, dtype=torch.long))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # print(f"Loss: {loss}")

            outputs_words = np.argmax(outputs.detach().numpy(), axis=2)
            ground_truth = sequence_to_sentence(labels_sequence)
            hypothesis = sequence_to_sentence(outputs_words)
            total_wer += wer(ground_truth, hypothesis)
            # print(f"Ground Truth: {ground_truth}")
            # print(f"Hypothesis: {hypothesis}")
            # print(f"wer: {wer(ground_truth, hypothesis)}")

        # Average training loss over batches
        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_wer = total_wer / len(train_dataloader)

        # Validation phase
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_val_wer = 0
            for encoder_inputs, labels_sequence, lengths in \
                    validation_dataloader:
                val_outputs = model(encoder_inputs.float())
                labels = labels_sequence
                outputs_reshape = inverted_tensor = torch.transpose(val_outputs, 0, 1)

                val_loss = criterion(outputs_reshape.float(), labels,
                                 torch.ones(labels.shape[0],
                                            dtype=torch.long) * 290,
                                 torch.tensor(lengths, dtype=torch.long))
                total_val_loss += val_loss.item()

                outputs_words = np.argmax(val_outputs.detach().numpy(), axis=2)
                ground_truth = sequence_to_sentence(labels_sequence)
                hypothesis = sequence_to_sentence(outputs_words)
                total_val_wer += wer(ground_truth, hypothesis)
                # print(f"Val Loss: {val_loss}")
                # print(f"Ground Truth: {ground_truth}")
                # print(f"Hypothesis: {hypothesis}")
                # print(f"wer: {wer(ground_truth, hypothesis)}")

            avg_val_loss = total_val_loss / len(validation_dataloader)
            avg_val_wer = total_val_wer / len(train_dataloader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Train WER: {avg_train_wer:.4f}, Val WER:"
            f" {avg_val_wer:.4f}")

        # Save the model if validation loss is lower than the previous best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
