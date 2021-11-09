import fiftyone.zoo as foz
dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits=["train", "validation", "test"],
    max_samples=25,
)