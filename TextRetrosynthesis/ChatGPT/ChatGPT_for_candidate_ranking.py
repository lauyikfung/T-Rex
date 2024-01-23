import time
from threading import Thread
import openai
import csv
import time
from tqdm import tqdm
import pubchempy as pcp
import os   

# OpenAI API key
openai.api_key = "1145141919810" # your API key
openai.api_base =  "https://hanover-openai.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2022-12-01'

# We use multi-threading to accelerate the generation process, you can adjust for your own purpose
BATCH_SIZE = 20 # parallel batch size
SLEEP_TIME = 0.14 # sleep time between each generation, you can adjust for your own purpose

# We use the lat template of the following template list to generate the description of the reactants and the product
# You can use other template for generation
# NAME: The name of the product
prompt_template_list = [
"""
Describe the structure of the molecule {{ NAME }} in detail with beginning: "The molecule is". Deduce the type of reaction most possible to synthesize it and explain why. The possible reactions are:
Heteroatom alkylation and arylation, Acylation and related processes, C-C bond formation, Heterocycle formation, Protections, Deprotections, Reductions, Oxidations, Functional group interconversion (FGI) and Functional group addition (FGA)
""",
"""
Describe the structure and the possible function and application of the molecule {{ NAME }} in detail with beginning: "The molecule is". Deduce the most possible type of reaction to synthesize it and explain why. You can choose the type of the reaction among the following:
Heteroatom alkylation and arylation, Acylation and related processes, C-C bond formation, Heterocycle formation, Protections, Deprotections, Reductions, Oxidations, Functional group interconversion (FGI) and Functional group addition (FGA)
""",
"""
Please delineate the structural features, functional aspects, and applicable implementations of the molecule {{ NAME }}, commencing with the introduction:"The molecule is {{ NAME }}". Reasoning the most plausible type for synthesizing this molecule in the final step, and offer a rationale for your choice. 
"""
]
prompt_template = prompt_template_list[-1]

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

def main(file_path, target_path = None):
     # The input file_path is a csv file with head: ['',Unnamed: 0,class,id,prod_smiles,rxn_smiles,...]

    # A temporary list to store the generation results
    output_list = ["" for i in range(BATCH_SIZE)]
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
            header.append("IUPAC")
            header.append("ChatGPT_generation")
            writer.writerow(header)
            datasets = [datasets[i : i + BATCH_SIZE] for i in range(0, len(datasets), BATCH_SIZE)]
            for row in tqdm(datasets):
                len_row = len(row)
                iupac_list = [(r[4], smiles_to_name(r[4])) for r in row]
                input_list = [prompt_template.replace('{{ SMILES }}', smile).replace('{{ NAME }}', iupac) for (smile, iupac) in iupac_list]
                thread_list = [Thread(target=target, daemon=True, name=str(i), kwargs={"prompt":input, "num":i, "order":row[i][0]}) for i, input in enumerate(input_list)]
                [thread.start() for thread in thread_list]
                [thread.join() for thread in thread_list]
                [row[i].append(iupac_list[i][1]) for i in range(len_row)]
                [row[i].append(output_list[i]) for i in range(len_row)]
                [writer.writerow(r) for r in row]
                f2.flush()

if __name__ == "__main__":
    file_path = None
    target_path = None
    main(file_path, target_path)