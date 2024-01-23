import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse
import csv
from tqdm import tqdm
def main(args):
    tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-base-smiles2caption", model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-base-smiles2caption')
    device = f"cuda:{args['device']}"
    model.to(device)
    model.eval()


    bsz = args['batch_size']
    for candidate_type in ['train', 'valid', 'test']:
      with torch.no_grad():
        with open(f"{args['data_dir']}/data_ChatGPT-fold{args['fold']}_candidate_{candidate_type}.csv", "r") as f:
          with open(f"{args['data_dir']}/data_ChatGPT-reactant-fold{args['fold']}_{candidate_type}_processed_new.csv", "w") as g:
            print(f"{args['data_dir']}/data_ChatGPT-reactant-fold{args['fold']}_{candidate_type}_processed_new.csv")
            reader = csv.reader(f)
            data_reactant = list(reader)[1:]
            writer = csv.writer(g)
            writer.writerow(["id", "reaction", "rxn_smiles", "label", "sample id", "prod_molt5", "molt5_1", "molt5_2"])

            for idxs in tqdm(range(0, len(data_reactant), bsz)):
              rows = data_reactant[idxs:idxs+bsz]
              rows = [item[:6] for item in rows]
              batch_len = len(rows)
              reactant_list = [[row[2].split('.')[0], row[2].split('.')[1]] if row[2].count(".") == 1 else [row[2].split(".")[0], row[2].split(".")[0]] for row in rows]
              masked_list = [True if row[2].count(".") == 1 else False for row in rows]
              reactant_list = [item for j in reactant_list for item in j]
              input_ids = tokenizer(reactant_list, return_tensors="pt", padding=True).input_ids.to(device)
              outputs = tokenizer.batch_decode(model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True).to("cpu"), skip_special_tokens=True)
              outputs = [outputs[i:i+2] for i in range(0, len(outputs), 2)]
              [rows[i].extend([outputs[i][0], outputs[i][1]]) if masked_list[i] else rows[i].extend([outputs[i][0], "None"])for i in range(batch_len)]
              [writer.writerow(row) for row in rows]

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default="data/candidate_generation")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    args = args.__dict__


    main(args)