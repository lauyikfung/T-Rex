import time
from threading import Thread
import openai
import csv
import time
from tqdm import tqdm
import pubchempy as pcp
import os
import argparse

# OpenAI API key
openai.api_key = "YOUR_OPENAI_KEY" # your API key
openai.api_base =  "https://hanover-openai.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2022-12-01'


# We use the following template to generate the description of the reactants and the product
# You can use other template for generation, you can also use other examples
# EXAMPLE: Your example, in the paper, we use MolT5 examples generated by MolT5, seperated by [SEP]
# NAME: The name of the product
# REACTANT1: The first reactant
# REACTANT2: The second reactant if exists, else REACTANT1
prompt_template = """
Please delineate the structural features, functional aspects, and applicable implementations of the molecules {{ NAME }} and possible reactants {{ REACTANT1 }} and {{ REACTANT2 }} to synthesize it. You should generate the descriptions respectively as above example. These descriptions are linked by " [SEP] ", and each commences with the introduction:"The molecule is ...".
"""

# We use PubChemPy to convert SMILES to IUPAC name
def smiles_to_name(smiles):
    try:
        compound_smiles = pcp.get_compounds(smiles, 'smiles')
        cpd_id = int(str(compound_smiles[0]).split("(")[-1][:-1])
        c = pcp.Compound.from_cid(cpd_id)
        if isinstance(c.iupac_name, str):
            return c.iupac_name
        else:
            return "None"
    except:
        return "None"
    
# We use OpenAI API to generate the description of the reactants and the product
def generate_with_prompt(prompt):
  response = openai.Completion.create(
      model="gpt-3.5-turbo-0301",
      engine='text-davinci-003',
      prompt=prompt,
      temperature=0.7,
      max_tokens=512,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
  )
  return response.choices[0].text


def main(args):
    # The input file_path is a csv file with head: [id,reaction,rxn_smiles,label,sample id,prod_molt5,prod_smiles,prod_IUPAC,prod_ChatGPT]
    
    BATCH_SIZE = args['batch_size'] # parallel batch size
    # We use multi-threading to accelerate the generation process, you can adjust for your own purpose
    SLEEP_TIME = args['sleep_time'] # sleep time between each generation, you can adjust for your own purpose

    # A temporary list to store the generation results
    output_list = ["" for i in range(BATCH_SIZE)]

    # The target function for multi-threading
    def target(prompt, num, order):
        time.sleep(num * SLEEP_TIME)
        output_list[num] = ""
        outputs = ""
        try:
            outputs = generate_with_prompt(prompt).strip()
            if not isinstance(outputs, str):
                print("error type")
                outputs = "ERROR"
            elif len(outputs) < 10:
                print("error length")
                outputs = "ERROR"
        except Exception as e:
            pass
        output_list[num] = outputs

    for candidate_type in ["train", "valid", "test"]:
        file_path = f"{args['data_dir']}/data_ChatGPT-fold{args['fold']}_candidate_{candidate_type}.csv"
        target_path = f"{args['data_dir']}/data_ChatGPT-reactant-fold{args['fold']}_{candidate_type}_processed_new.csv"

        if target_path is None:
            file_cnt = 1
            target_root = file_path.split(".csv")[0]
            while os.path.exists(f"{target_root}-{file_cnt}.csv"):
                file_cnt += 1
            target_path = f"{file_path}-{file_cnt}.csv"
        with open(file_path, "r") as f:
            with open(target_path, "w") as f2:
                writer = csv.writer(f2)
                datasets = list(csv.reader(f))
                header = datasets[0]
                datasets = datasets[1:]

                header.append("Generation")
                writer.writerow(header)
                datasets = [datasets[i : i + BATCH_SIZE] for i in range(0, len(datasets), BATCH_SIZE)]

                for row in tqdm(datasets):
                    len_row = len(row)
                    replace_list = [(r[2].split(".")[0], r[2].split(".")[1], r[-2]) if r[2].count(".") > 0 else (r[2], r[2], r[-2]) for r in row]
                    input_list = [prompt_template.replace('{{ REACTANT1 }}', r1).replace('{{ REACTANT2 }}', r2).replace('{{ NAME }}', prod) for (r1, r2, prod) in replace_list]
                    thread_list = [Thread(target=target, daemon=True, name=str(i), kwargs={"prompt":input, "num":i, "order":row[i][0]}) for i, input in enumerate(input_list)]
                    [thread.start() for thread in thread_list]
                    [thread.join() for thread in thread_list]
                    [row[i].append(output_list[i]) for i in range(len_row)]
                    [writer.writerow(r) for r in row]
                    f2.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default="data/candidate_generation")
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--sleep_time', type=float, default=0.14)
    args = parser.parse_args()
    args = args.__dict__


    main(args)