"""
This script will perform rolling inference using the output of prepare_data.
"""
import argparse
import asyncio
import os
import json

import pandas as pd
from tqdm.asyncio import tqdm

from zeroshot_inference.utils import send_single_message, parse_llm_json, chunk_notes
from zeroshot_inference.clinical_prediction.prompts import INTIAL_SYS, INTIAL_USER, UPDATE_SYS, UPDATE_USER


MAX_WORKERS=8
NUM_ITERATIONS=3
PROCESSING_TYPES = ['original', 'removed']
# PROCESSING_TYPES = ['original', 'removed', 'removed_template', 'removed_copy', 'removed_stage1', 'removed_stage2']


async def predict_outcome_call(row, text_column, iter, variables_df):
    """
    Extract information from given text column
    """
    # formatting variable descriptions
    variable_string =  "\n\n".join(
    f"-{row.variable}:\n"
    f"--Choices:{row.choices} \n--Description: {row.description}: "
    for row in variables_df.itertuples(index=False)
    )
    schema_string = '{'
    for variable_row in variables_df.iterrows():
        schema_string += f'"{variable_row[1]['variable']}": [{variable_row[1]['choices']}],\n' 
    schema_string = schema_string[:-2]
    schema_string += '}'

    # initialize as an empty response
    response = {key: None for key, value in row['variables'].items()}
    # format prompts and run inference
    for idx, chunk in enumerate(chunk_notes(row[text_column])):
        cnt = 0
        if idx == 0:
            system_prompt = INTIAL_SYS.format(SCHEMA = schema_string, 
                                              OUTCOMES = variable_string)
            user_prompt = INTIAL_USER.format(CONDITION_LIST='\n'.join([f'- {x}' for x in list(row['variables'].keys())]),
                                             PATIENT_NOTES = chunk,
                                            SCHEMA = schema_string,
                                            OUTCOMES = variable_string)
        else:
            system_prompt = UPDATE_SYS.format(SCHEMA = schema_string, 
                                              OUTCOMES= variable_string)
            user_prompt = UPDATE_USER.format(EXISTING_JSON=response,
                                             PATIENT_NOTES = chunk,
                                             SCHEMA = schema_string,
                                             OUTCOMES = variable_string)

        while True:
            try:
                response = await send_single_message(
                    system_instructions=system_prompt,
                    user_prompt=user_prompt
                )
                # response = safe_json_parse(response)
                response = parse_llm_json(response = response,expected_fields= row['variables'].keys())
                break
            except Exception as e:
                print(f'Exception [{cnt}] for {row['pat_mrn_id']}: {e}, {response}')
                cnt += 1
                if cnt == 3:
                    break
    return response, text_column, iter
async def extract_information(row, variables_df, semaphore):
    """
    Run information extraction for each patient
    """
    # initialize output
    ret = {}
    ret['mrn'] = row['pat_mrn_id']
    ret['variables'] = row['variables']

    async def process_with_semaphore(row, text_column, iter):
        async with semaphore:
            return await predict_outcome_call(row, text_column, iter, variables_df)
    
    tasks = []
    for text_column in PROCESSING_TYPES:
        for iter in range(NUM_ITERATIONS):
            tasks.append(process_with_semaphore(row, text_column, iter)) # IE

    results = await asyncio.gather(*tasks)
    # write predictions
    for item in results:
        ret[f'variables_{item[1]}_{item[2]}'] = item[0]
    
    return ret

async def run_patient(row, variables_df, semaphore):
    results = await extract_information(row, variables_df, semaphore)
    # format variables
    ret = {"mrn": results['mrn']}
    for field in results.keys():
        if "variables" in field:
            if field == "variables":
                prefix = "true"
            else:
                prefix = '_'.join(field.split('_')[1:])
            for key, value in results[field].items():
                ret[f'{prefix}_{key}'] = value

    return pd.DataFrame([ret])

async def main(args):
    variables = pd.read_csv(args.variables).dropna()

    global_semaphore = asyncio.Semaphore(MAX_WORKERS)

    tasks = []
    for file in os.listdir(args.patient_directory):
        if 'jsonl' in file:
            with open(f'{args.patient_directory}/{file}', 'r') as f:
                for l in f:
                    item = json.loads(l)
                    task = run_patient(item.copy(), variables, global_semaphore)
                    tasks.append(task)
                    # break # debugging

    result_df = []
    for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Processing patients"):
        result = await coro
        result_df.append(result)

    result_df = pd.concat(result_df, ignore_index=True)
    result_df.to_csv(args.output, index=False)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run zero-shot infromation extraction")
    parser.add_argument('--variables', 
                        type=str,
                        required=True,
                        help="CSV containing variables to extract")
    parser.add_argument('--patient_directory', 
                        type=str,
                        default=None,
                        help="Directory for patients")
    parser.add_argument('--output', 
                        type=str,
                        required=True,
                        help="CSV file for results")
    args = parser.parse_args()
    asyncio.run(main(args))

