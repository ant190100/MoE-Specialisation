import os
import random
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class COCO_Loader(Dataset):
    def __init__(
        self,
        image_dir,
        annotations_file,
        clip_processor,
        tokenizer,
        subset_fraction=1.0,
    ):
        self.image_dir = image_dir
        self.coco = COCO(annotations_file)

        all_ids = list(sorted(self.coco.imgs.keys()))

        # --- MODIFICATION START ---
        # If a fraction less than 1.0 is specified, take a random subset
        if subset_fraction < 1.0:
            num_samples = int(len(all_ids) * subset_fraction)
            self.ids = random.sample(all_ids, num_samples)
            print(
                f"Using a random subset of {num_samples} images ({subset_fraction*100:.1f}% of total)."
            )
        else:
            self.ids = all_ids

        # 2. Store processors
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 3. On-demand data loading
        img_id = self.ids[idx]

        path = self.coco.loadImgs(img_id)[0]["file_name"]
        image = Image.open(os.path.join(self.image_dir, path)).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        caption = anns[0]["caption"]

        # 4. Apply transformations
        image_processed = self.clip_processor(
            images=image, return_tensors="pt"
        ).pixel_values
        text_tokenized = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        )

        return (
            image_processed.squeeze(0),
            text_tokenized.input_ids.squeeze(0),
            text_tokenized.attention_mask.squeeze(0),
        )
