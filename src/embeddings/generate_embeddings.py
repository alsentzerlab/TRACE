# default
import os

# pip
import numpy as np
from tqdm import tqdm
import argparse

# ml
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from unsloth import FastSentenceTransformer




def main(args):
    # Set up model
    model = FastSentenceTransformer.from_pretrained(
        model_name = "unsloth/Qwen3-Embedding-8B",
        max_seq_length = 32000,  
        load_in_4bit = False,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # A bit more accurate, uses 2x memory,
        full_finetuning = False, # [NEW!] We have full finetuning now!
    )

    text_list = []
    # load the dataset
    dataset = load_dataset(
        "json",
        data_dir=args.input,
        split="train"
    )

    # create a data loader
    def collate_fn(batch):
        text = [ex[args.text] for ex in batch]
        labels = [ex['variables'][args.variable] for ex in batch]
        return text, labels 

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    embeddings_array = []
    labels_array = []
    cnt = 0
    for texts, labels in tqdm(dataloader):
        cnt +=1
        with torch.inference_mode():
            result = model.encode(texts, 
                                normalize_embeddings=True,
                                convert_to_numpy=True)
        embeddings_array.append(result)
        labels_array.append(labels)
        # empty the cache after 500 entries
        if cnt % 500 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        # for debugging
        # if cnt == 10:
        #     break
    embeddings_array = np.vstack(embeddings_array)
    labels_array = np.vstack(labels_array)
    np.save(f'{args.output}_{args.text}_embeddings.npy', embeddings_array)
    np.save(f'{args.output}_{args.text}_variables.npy', labels_array)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="TRACE_data/mimiciii/patients_processed_50/")
    parser.add_argument('--output', type=str, default="TRACE_data/mimiciii/patients_embeddings/processed_50")
    parser.add_argument('--text', type=str, default="original")
    parser.add_argument('--variable', type=str, default="readmission_30")

    args = parser.parse_args()
    main(args)
