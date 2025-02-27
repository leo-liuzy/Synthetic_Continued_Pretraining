import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
from tqdm import tqdm

from inference.devapi import gptqa
from utils.io_utils import jload, jdump
from tasks.quality import QuALITY
from utils.io_utils import set_openai_key
import random

import numpy as np
import math
import numpy as np
from copy import deepcopy
from transformers import set_seed


def generate_entities(document_content: str, system_message: str, openai_model: str):
    prompt = f"""
    ### Document Content:
    {document_content}
    """
    can_read_entities = None
    while not can_read_entities:
        try:
            completion = gptqa(prompt, openai_model, system_message, json_format=True)
            response = json.loads(completion)
            can_read_entities = response["entities"]
        except Exception as e:
            print(f"Failed to generate entities: {str(e)}")
    return response


def generate_entity_specific_questions(
    document_content: str, entity: str, system_message: str, openai_model: str
):
    prompt = f"""
    ### Document Content:
    {document_content}
    ### Entities:
    - {entity}
    """
    completion = gptqa(prompt, openai_model, system_message)
    return completion


def generate_two_entity_relations(
    document_content: str,
    entity1: str,
    entity2: str,
    system_message: str,
    openai_model: str,
):
    prompt = f"""
    ### Document Content:
    {document_content}
    ### Entities:
    - {entity1}
    - {entity2}
    """
    completion = gptqa(prompt, openai_model, system_message)
    return completion


def generate_three_entity_relations(
    document_content: str,
    entity1: str,
    entity2: str,
    entity3: str,
    system_message: str,
    openai_model: str,
):
    prompt = f"""
    ### Document Content:
    {document_content}
    ### Entities:
    - {entity1}
    - {entity2}
    - {entity3}
    """
    completion = gptqa(prompt, openai_model, system_message)
    return completion


def generate_synthetic_data_for_document(args):
    random.seed(42)
    document_index = args.document_index
    model_name = args.generator_model_name
    set_openai_key()

    task = QuALITY("train", args.dataset)
    document = task.documents[document_index]
    print(f"Generating synthetic data for article {document.uid}")
    dir_name = f"data/dataset/raw/{args.dataset}_entigraph_{model_name}"
    if args.no_single:
        dir_name += "_no1"
    if args.no_pair:
        dir_name += "_no2"

    if args.sample_triplet_ratio is not None:
        assert not args.no_triplet
        dir_name += f"_sub3={args.sample_triplet_ratio}"
    elif args.no_triplet:
        dir_name += "_no3"

    if args.max_pair_entity_allowed:
        dir_name += f"_pairE{args.max_pair_entity_allowed}"
    if args.max_triplet_entity_allowed:
        dir_name += f"_tripletE{args.max_triplet_entity_allowed}"
        
    output_path = f"{dir_name}/{document.uid}.json"

    if os.path.exists(output_path):
        output = jload(output_path)
    else:
        output = [[]]

    # first check if entities are already generated
    if isinstance(output[0], list) and len(output[0]) > 0:
        entities = output[0]
    else:
        entities = generate_entities(
            document.content, task.openai_system_generate_entities, model_name
        )
        output[0] = entities["entities"]
        output.append(entities["summary"])
        jdump(output, output_path)
        entities = entities["entities"]
    num_generation = len(output) - 2  # 2 = entity_list + summary
    assert num_generation >= 0
    print(f"Existing #generation: {num_generation}")

    single_count = len(entities)
    # iterate over entities and generate questions
    if num_generation < single_count * (not args.no_single):
        # we don't skip it and we haven't generated for single entity
        for entity in tqdm(entities, desc=f"Generating for single-entity [{document.uid}]"):
            # if _entity_already_generated(entity, output):
            #     continue
            response = generate_entity_specific_questions(
                document.content,
                entity,
                task.openai_system_generate_entity_specific_questions,
                model_name,
            )
            if response:
                output.append(response)
            jdump(output, output_path)
    
    pair_entities = deepcopy(entities)
    if args.max_pair_entity_allowed and len(pair_entities) > args.max_pair_entity_allowed:
        
        rand_ids = np.random.choice(
            len(pair_entities), args.max_pair_entity_allowed, replace=False
        )
        pair_entities = [pair_entities[i] for i in rand_ids]
    pair_count = math.comb(len(pair_entities), 2)
    # iterate over pairs of entities and generate relations
    if num_generation < single_count * (not args.no_single) + pair_count * (not args.no_pair):
        # we don't skip it and we haven't generated for pair entity
        pair_list = []
        for i in range(len(pair_entities)):
            for j in range(i + 1, len(pair_entities)):
                pair = (pair_entities[i], pair_entities[j])
                pair_list.append(pair)
        for entity1, entity2 in tqdm(pair_list, desc=f"Generating for pair-entity [{document.uid}]"):
            # if _pair_already_generated(entity1, entity2, output):
            #     continue
            response = generate_two_entity_relations(
                document.content,
                entity1,
                entity2,
                task.openai_system_generate_two_entity_relations,
                model_name,
            )
            if response:
                output.append(response)
            jdump(output, output_path)
    
    triplet_entities = deepcopy(entities)
    if args.max_triplet_entity_allowed and len(triplet_entities) > args.max_triplet_entity_allowed:
        rand_ids = np.random.choice(
            len(triplet_entities), args.max_triplet_entity_allowed, replace=False
        )
        triplet_entities = [triplet_entities[i] for i in rand_ids]
    # iterate over triples of entities and generate relations
    triple_list = []
    triple_count = math.comb(len(triplet_entities), 3)
    if num_generation < single_count * (not args.no_single) + pair_count * (not args.no_pair) + triple_count * (not args.no_triplet):
        triple_list = []
        for i in range(len(triplet_entities)):
            for j in range(i + 1, len(triplet_entities)):
                for k in range(j + 1, len(triplet_entities)):
                    triple = (triplet_entities[i], triplet_entities[j], triplet_entities[k])
                    triple_list.append(triple)
        
        if args.sample_triplet_ratio is not None:
            assert isinstance(args.sample_triplet_ratio, float)
            n_sampled_triplets = int(args.sample_triplet_ratio * len(triple_list))
            sample_ids = np.random.choice(len(triple_list), n_sampled_triplets, replace=False)
            triple_list = [triple_list[i] for i in sample_ids]
        else:
            # I don't understand why but it's the original operation
            random.shuffle(triple_list)
        
        for entity1, entity2, entity3 in tqdm(
            triple_list, desc=f"Generating for triplet-entity [{document.uid}]"
        ):
            response = generate_three_entity_relations(
                document.content,
                entity1,
                entity2,
                entity3,
                task.openai_system_generate_three_entity_relations,
                model_name,
            )
            if response:
                output.append(response)
            jdump(output, output_path)
        


if __name__ == "__main__":
    # seq 0 264 | xargs -P 265 -I {} sh -c 'python data/entigraph.py {} > data/dataset/log/log_gpt4turbo_{}.txt 2>&1'
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("document_index", type=int)
    parser.add_argument("--dataset", type=str, default="KE-by-CP")
    parser.add_argument("--generator_model_name", type=str, default="gpt-4-turbo")
    parser.add_argument("--max_pair_entity_allowed", type=int, default=None)
    parser.add_argument("--max_triplet_entity_allowed", type=int, default=8)
    parser.add_argument("--no_single", action="store_true", default=False)
    parser.add_argument("--no_pair", action="store_true", default=False)
    parser.add_argument("--no_triplet", action="store_true", default=False)
    parser.add_argument('--sample_triplet_ratio', type=float, default=None)
    args = parser.parse_args()

    generate_synthetic_data_for_document(args)
