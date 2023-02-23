'''
Generate few-shot data for training
'''
import os
import random


def main():
    root_dir = "/home/xudong99/scratch/cy6cvx3ryv-1/Caltech101-NIN"
    cls_dirs = os.listdir(root_dir)
    for num_samples in [3, 5, 10, 20]:
        sample_list = []
        for cls_dir in cls_dirs:
            if "txt" in cls_dir:
                continue
            cls_path = os.path.join(root_dir, cls_dir)
            # random sample num_samples images
            sample_paths = random.sample(os.listdir(cls_path), num_samples)
            sample_paths = [
                os.path.join(cls_path, sample_path)
                for sample_path in sample_paths
            ]
            sample_list.extend(sample_paths)

        # write to file
        with open(os.path.join(root_dir, f"few_shot_train_{num_samples}.txt"),
                  "w") as f:
            f.write("\n".join(sample_list))


if __name__ == "__main__":
    main()