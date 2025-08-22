import os
import json
from PIL import Image
from torch.utils.data import Dataset
import random

class COCO_Loader(Dataset):
    def __init__(
        self,
        image_dir,
        annotations_file,
        clip_processor,
        tokenizer,
        subset_fraction=1.0,
        split="train", 
        val_split_fraction=0.1,
    ):
        self.image_dir = image_dir
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer

        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)["annotations"]

        # Shuffle and subset the data
        random.seed(42) # Add seed for reproducibility
        random.shuffle(self.annotations)
        subset_size = int(len(self.annotations) * subset_fraction)
        self.annotations = self.annotations[:subset_size]

        # Split the data into training and validation sets
        split_index = int(len(self.annotations) * (1 - val_split_fraction))
        if split == "train":
            self.annotations = self.annotations[:split_index]
            print(f"Using {len(self.annotations)} images for training.")
        elif split == "val":
            self.annotations = self.annotations[split_index:]
            print(f"Using {len(self.annotations)} images for validation.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        # Construct image path safely, handling potential missing leading zeros in image_id
        image_filename = f"{annotation['image_id']:012d}.jpg"
        image_path = os.path.join(self.image_dir, image_filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            # Return None or a placeholder to be handled in the dataloader's collate_fn if needed
            return None

        caption = annotation["caption"]

        image_processed = self.clip_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        tokenized_caption = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        input_ids = tokenized_caption["input_ids"].squeeze(0)
        attention_mask = tokenized_caption["attention_mask"].squeeze(0)

        return image_processed, input_ids, attention_mask