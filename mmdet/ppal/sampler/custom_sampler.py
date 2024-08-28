import numpy as np
import torch
from torch.utils.data import Sampler

class CustomPunctaSampler(Sampler):
    def __init__(self, dataset, model, shuffle=True, confidence_threshold=0.5):
        """
        Custom sampler that selects samples where the model has low confidence in its predictions.

        Args:
            dataset (Dataset): The dataset to sample from.
            model (nn.Module): The trained model used to assess confidence.
            shuffle (bool): Whether to shuffle the data indices before sampling.
            confidence_threshold (float): Confidence threshold below which samples are considered informative.
        """
        self.dataset = dataset
        self.model = model
        self.shuffle = shuffle
        self.confidence_threshold = confidence_threshold
        self.indices = self.get_informative_indices()

    def get_informative_indices(self):
        """
        Identifies the most informative samples based on model confidence.

        Returns:
            List[int]: Indices of informative samples.
        """
        informative_indices = []
        self.model.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for idx, data in enumerate(self.dataset):
                image, _ = data
                image = image.unsqueeze(0).to(next(self.model.parameters()).device)  # Adjust for single image
                predictions = self.model(image)

                # Assuming predictions contain a 'confidence' field or similar
                confidences = predictions['scores'].cpu().numpy() if 'scores' in predictions else np.array([])
                if len(confidences) > 0 and np.max(confidences) < self.confidence_threshold:
                    informative_indices.append(idx)

        return informative_indices

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
