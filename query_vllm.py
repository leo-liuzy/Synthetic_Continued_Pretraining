import os
import json
import argparse
import requests
from tqdm import tqdm
# from reason import util
from datasets import load_dataset, load_from_disk
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm import EngineArgs, LLMEngine, RequestOutput

import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer

from knowledge_propagation.utils import io

PROJ_DIR = os.path.dirname(__file__)
EVAL_RESULT_DIR = os.path.join(PROJ_DIR, "eval_results")
RAW_DATA_DIR = os.path.join(PROJ_DIR, "data", "dataset", "raw")
CONTROLLED_RE_DATA_DIR = os.path.join(RAW_DATA_DIR, "4K_controlled_RE")


from datasets import load_dataset
from tqdm import tqdm
from knowledge_propagation.utils import vars, io, extractor
from scipy.stats import describe
from typing import List, Dict
import re
from copy import deepcopy
import pandas as pd
from glob import glob
from bespokelabs import curator
from datasets import Dataset

score_tag_extractor = extractor.tag_content_extractor("score")


class LlmAsJudge(curator.LLM):
    MAX_VAL: float = 10.0
    PROMPT: str = """
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. For this evaluation, you should primarily consider the following criteria:
accuracy: 
                Score 0: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numerical score.

[Question]
{question}

[The Start of Ground truth]
{reference}
[The End of Ground truth]

[The Start of Assistant's Answer]
{prediction}
[The End of Assistant's Answer]

Return the numerical score wrapped in <score>..</score> tag
    """.strip()

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(
            question=input["question"], prediction=input["predicted_answer"], reference=input["answer"]
        )

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        score_ = score_tag_extractor(response)
        assert len(score_) == 1
        score = score_[0].strip()
        assert score.isdigit()
        assert 0 <= float(score) <= self.MAX_VAL
        score = float(score)
        score /= self.MAX_VAL
        input["llm_accuracy-soft"] = score

        return {**input}

class LlmAsJudgeHard(curator.LLM):
    MAX_VAL: float = 1.0
    PROMPT: str = """
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. For this evaluation, you should primarily consider the following criteria:
accuracy: 
                Score 1: The answer is completely accurate and aligns perfectly with the reference.
                Score 0: Otherwise.
                Only respond with a numerical score.

[Question]
{question}

[The Start of Ground truth]
{reference}
[The End of Ground truth]

[The Start of Assistant's Answer]
{prediction}
[The End of Assistant's Answer]

Return the numerical score wrapped in <score>..</score> tag
    """.strip()

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(
            question=input["question"], prediction=input["predicted_answer"], reference=input["answer"]
        )

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        score_ = score_tag_extractor(response)
        assert len(score_) == 1
        score = score_[0].strip()
        assert score.isdigit()
        assert 0 <= float(score) <= self.MAX_VAL
        score = float(score)
        score /= self.MAX_VAL
        input["llm_accuracy-hard"] = score

        return {**input}

class LlmAsJudgeAbstention(curator.LLM):
    MAX_VAL: float = 1.0
    PROMPT: str = """
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. For this evaluation, you should primarily consider the following criteria:
                Score 1: The assistant is refusing to answer the question. e.g. "I'm sorry, I don't know."
                Score 0: Otherwise.
                Only respond with a numerical score.

[Question]
{question}

[The Start of Assistant's Response]
{prediction}
[The End of Assistant's Response]

Return the numerical score wrapped in <score>..</score> tag
    """.strip()

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(
            question=input["question"], prediction=input["predicted_answer"]
        )

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        score_ = score_tag_extractor(response)
        assert len(score_) == 1
        score = score_[0].strip()
        assert score.isdigit()
        assert 0 <= float(score) <= self.MAX_VAL
        score = float(score)
        input["llm_accuracy-abstention"] = score

        return {**input}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Query vLLM with MATH-500 examples"
    )
    parser.add_argument(
        "--model-name-or-path", type=str, required=True,
        help="Model name to query (should match the model served by vLLM)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=1,
        help="Top-p sampling parameter (nucleus sampling)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--num-samples", type=int, default=1,
        help="Number of samples to generate per example"
    )
    parser.add_argument(
        "--eval-data-name", type=str, default="controlled_RE_efficacy", choices=["all", "controlled_RE_efficacy", "controlled_RE_specificity", "mmlu_0shot_cot"],
        help="Dataset name"
    )
    parser.add_argument(
        "--test-set-choice", type=str, default="id_sample", choices=["test_id_sample", "test_ood_entity_sample", "test_ood_relation_sample", "test_ood_both_sample"],
        help="Test set choice"
    )
    parser.add_argument(
        "--llm-judge-name", type=str, default="gpt-4o-mini",
        help="LLM judge type",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Whether to overwrite the existing results"
    )
    return parser.parse_args()



def load_controlled_RE_data(file_path):
    samples = io.load_jsonlines(file_path)
    lst = []
    for s in samples:
        questions = s["questions"]
        for q in questions:
            q["text"] = s["text"]
        lst.extend(questions)
    return Dataset.from_list(lst)


def get_messages_from_problem(problem, model_name_or_path_base, dataset_name="controlled_RE_efficacy", ):
    """Extract messages from problem for vLLM API"""
    
    if dataset_name in ["controlled_RE_efficacy", "controlled_RE_specificity"]:
        return [
            {"role": "user", "content": problem}
        ]
    if dataset_name == "mmlu_0shot_cot":
        if "-Instruct" in model_name_or_path_base:
            return [
                {"role": "user", "content": f"Given the following question and four candidate answers (A, B, C and D), choose the best answer.\n\nQuestion: {problem}\n\n- For simple problems:\nDirectly provide the answer with minimal explanation.\n\n- For complex problems:\nUse this step-by-step format:\n## Step 1: [Concise description]\n[Brief explanation]\n## Step 2: [Concise description]\n[Brief explanation]\n\nRegardless of the approach, always conclude with:\nThe best answer is [the_answer_letter].\nwhere the [the_answer_letter] is one of A, B, C or D.\n\nLet's think step by step."}
            ]
        elif "-Distill" in model_name_or_path_base:
            return [
                {"role": "user", "content": f"Given the following question and four candidate answers (A, B, C and D), choose the best answer.\n\nQuestion: {problem}\n\nAlways conclude with:\nThe best answer is [the_answer_letter].\nwhere the [the_answer_letter] is one of A, B, C or D.\n\nLet's think step by step."}
            ]
        else:
            raise ValueError(f"Invalid model name: {model_name_or_path_base}")
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    


def main():
    args = parse_args()
    
    if args.eval_data_name == "all":
        eval_data_names = ["controlled_RE_efficacy", "controlled_RE_specificity", "mmlu_0shot_cot"]
    else:
        eval_data_names = [args.eval_data_name]
    model = None
    
    for eval_data_name in eval_data_names:
        model_name_or_path_base = os.path.basename(args.model_name_or_path)
        save_dir = f"{EVAL_RESULT_DIR}/{model_name_or_path_base}"
        os.makedirs(save_dir, exist_ok=True)
        output_file = f"{save_dir}/{eval_data_name}_{args.test_set_choice}_llm-judge.xlsx"
    
        if not os.path.exists(output_file) or args.overwrite:
            if model is None:
                model = LLM(args.model_name_or_path)
            sampling_params = SamplingParams(
                max_tokens=args.max_tokens, 
                top_p=args.top_p,
                temperature=args.temperature,
                n=args.num_samples,
                skip_special_tokens=False,
            )
            if eval_data_name in ["controlled_RE_efficacy", "controlled_RE_specificity"]:
                dataset = load_controlled_RE_data(f"{CONTROLLED_RE_DATA_DIR}/{args.test_set_choice}.jsonl")
            elif eval_data_name == "mmlu_0shot_cot":
                dataset = load_from_disk(f"{RAW_DATA_DIR}/sampled_mmlu")
            else:
                raise ValueError(f"Invalid dataset name: {args.dataset_name}")
            
            if eval_data_name == "controlled_RE_efficacy":
                problem_key = "alias_question"
                answer_key = "answer"
            elif eval_data_name == "controlled_RE_specificity":
                problem_key = "unalias_question"
                answer_key = "answer"
            elif eval_data_name == "mmlu_0shot_cot":
                problem_key = "question_choices_formatted"
                answer_key = "answer_letter"
            else:
                raise ValueError(f"Invalid dataset name: {eval_data_name}")
            

            all_messages = [get_messages_from_problem(d[problem_key], model_name_or_path_base=model_name_or_path_base, dataset_name=eval_data_name) for d in dataset]    
            
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            try:
                texts = tokenizer.apply_chat_template(all_messages, tokenize=False, add_generation_prompt=True)
                # import pdb; pdb.set_trace()
                outputs = model.generate(texts, sampling_params=sampling_params)
                results = []
                
                for idx, output in enumerate(outputs):
                    
                    for sample_idx, out in enumerate(output.outputs):
                        # import pdb; pdb.set_trace()
                        model_answer = out.text
                        if eval_data_name == "mmlu_0shot_cot":
                            if "-Instruct" in model_name_or_path_base:
                                model_answer = model_answer.split("The best answer is ")[-1].strip()
                            
                        if "-Distill" in model_name_or_path_base:
                            model_answer = model_answer.split("</think>")[-1].strip()
                        results.append({
                            "text": dataset[idx]["text"] if eval_data_name in ["controlled_RE_efficacy", "controlled_RE_specificity"] else None,
                            "question": dataset[idx][problem_key],
                            "eval_data_name": eval_data_name,
                            "ground_truth_answer": dataset[idx][answer_key],
                            "sample_id": sample_idx,
                            "model_response": model_answer
                        })
            except Exception as e:
                print(f"\nError processing: {e}")
            
            
            pd.DataFrame(results).to_excel(output_file)
            print(f"\nFinal results saved to {output_file}")
        
        # evaluate with llm_judge
        
        for llm_judge_type in ["hard", "abstention"]:
            df = pd.read_excel(output_file)
            if f"llm_accuracy-{llm_judge_type}" not in df.columns or args.overwrite:
                print(f"Evaluating with [{llm_judge_type}] judge")
                llm_judge_name = args.llm_judge_name
                if llm_judge_type == "hard":
                    llm_judge = LlmAsJudgeHard(
                        model_name=llm_judge_name, backend_params={"max_requests_per_minute": 30_000, "max_tokens_per_minute": 150_000_000}
                    )
                elif llm_judge_type == "abstention":
                    llm_judge = LlmAsJudgeAbstention(
                        model_name=llm_judge_name, backend_params={"max_requests_per_minute": 30_000, "max_tokens_per_minute": 150_000_000}
                    )
                else:
                    llm_judge = LlmAsJudge(
                        model_name=llm_judge_name, backend_params={"max_requests_per_minute": 30_000, "max_tokens_per_minute": 150_000_000}
                    )
                df["predicted_answer"] = df["model_response"].astype(str)
                df["answer"] = df["ground_truth_answer"].astype(str)
                scored_dataset = Dataset.from_pandas(df[:])
                scored_dataset = llm_judge(
                    scored_dataset,
                )
                # import pdb; pdb.set_trace()
                ds = scored_dataset
                if hasattr(scored_dataset, "dataset"):
                    ds = scored_dataset.dataset
                    
                scored_df = ds.to_pandas().drop(columns=['predicted_answer', 'answer',])
                scored_df["llm_judge"] = llm_judge_name
                scored_df.to_excel(output_file, index=False)


if __name__ == "__main__":
    main()