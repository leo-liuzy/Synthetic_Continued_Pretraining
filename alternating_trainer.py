#!/usr/bin/env python3
"""
Alternating Training Script for Transformers Model

This script implements a two-step gradient descent training approach:
- Step t: Causal language model loss on document text
- Step t+1: Supervised fine-tuning on propagation questions

Each dataset entry contains:
- text: The document text for causal LM training
- prop_questions: A set of propagation questions for supervised fine-tuning
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint
import wandb
from omegaconf import OmegaConf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for alternating training"""
    # Model and data
    model_name: str = "gpt2"
    dataset_path: str = "data/training_data.jsonl"
    output_dir: str = "./output"
    
    # Training parameters
    max_epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Alternating training parameters
    causal_lm_weight: float = 1.0
    sft_weight: float = 1.0
    
    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True
    dataloader_num_workers: int = 4
    
    # Wandb
    use_wandb: bool = False
    wandb_project: str = "alternating-training"
    wandb_run_name: Optional[str] = None


class AlternatingDataset(Dataset):
    """Dataset for alternating training with text and propagation questions"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load data from JSONL file"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        logger.info(f"Loaded {len(data)} examples from {data_path}")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Prepare causal LM data (document text)
        text = item['text']
        causal_lm_inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Prepare SFT data (propagation questions) - keep them separate
        prop_questions = item['prop_questions']
        if isinstance(prop_questions, str):
            prop_questions = [prop_questions]
        
        # Tokenize each question separately
        sft_inputs_list = []
        for question in prop_questions:
            sft_inputs = self.tokenizer(
                question,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            sft_inputs_list.append({
                'input_ids': sft_inputs['input_ids'].squeeze(0),
                'attention_mask': sft_inputs['attention_mask'].squeeze(0),
                'labels': sft_inputs['input_ids'].squeeze(0).clone(),
            })
        
        return {
            'causal_lm_input_ids': causal_lm_inputs['input_ids'].squeeze(0),
            'causal_lm_attention_mask': causal_lm_inputs['attention_mask'].squeeze(0),
            'causal_lm_labels': causal_lm_inputs['input_ids'].squeeze(0).clone(),
            'sft_questions': sft_inputs_list,  # List of QA pairs
        }


class AlternatingDataCollator:
    """Custom data collator for alternating training"""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch):
        """Collate batch for alternating training
        
        Returns:
            - causal_lm_*: n_batch examples for causal LM training
            - sft_*: n_batch * K examples for SFT training (flattened)
        """
        # Extract causal LM data (n_batch examples)
        causal_lm_input_ids = torch.stack([item['causal_lm_input_ids'] for item in batch])
        causal_lm_attention_mask = torch.stack([item['causal_lm_attention_mask'] for item in batch])
        causal_lm_labels = torch.stack([item['causal_lm_labels'] for item in batch])
        
        # Extract and flatten SFT data (n_batch * K examples)
        sft_input_ids_list = []
        sft_attention_mask_list = []
        sft_labels_list = []
        
        for item in batch:
            for sft_item in item['sft_questions']:
                sft_input_ids_list.append(sft_item['input_ids'])
                sft_attention_mask_list.append(sft_item['attention_mask'])
                sft_labels_list.append(sft_item['labels'])
        
        # Stack SFT data - this will work even with different numbers of questions per item
        if sft_input_ids_list:  # Check if we have any SFT data
            sft_input_ids = torch.stack(sft_input_ids_list)
            sft_attention_mask = torch.stack(sft_attention_mask_list)
            sft_labels = torch.stack(sft_labels_list)
        else:
            # Handle edge case where no SFT data exists
            sft_input_ids = torch.empty(0, causal_lm_input_ids.shape[1], dtype=causal_lm_input_ids.dtype)
            sft_attention_mask = torch.empty(0, causal_lm_attention_mask.shape[1], dtype=causal_lm_attention_mask.dtype)
            sft_labels = torch.empty(0, causal_lm_labels.shape[1], dtype=causal_lm_labels.dtype)
        
        return {
            'causal_lm_input_ids': causal_lm_input_ids,
            'causal_lm_attention_mask': causal_lm_attention_mask,
            'causal_lm_labels': causal_lm_labels,
            'sft_input_ids': sft_input_ids,
            'sft_attention_mask': sft_attention_mask,
            'sft_labels': sft_labels,
        }


class AlternatingTrainer:
    """Custom trainer for alternating between causal LM and SFT training"""
    
    def __init__(self, model, tokenizer, config: TrainingConfig, train_dataset, eval_dataset=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=config.warmup_steps
        )
        
        # Setup data collator
        self.data_collator = AlternatingDataCollator(tokenizer)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Setup wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or f"alternating-training-{int(time.time())}",
                config=config.__dict__
            )
    
    def compute_loss(self, model, inputs, step_type: str):
        """Compute loss for either causal LM or SFT step"""
        if step_type == "causal_lm":
            input_ids = inputs['causal_lm_input_ids']
            attention_mask = inputs['causal_lm_attention_mask']
            labels = inputs['causal_lm_labels']
        else:  # sft
            input_ids = inputs['sft_input_ids']
            attention_mask = inputs['sft_attention_mask']
            labels = inputs['sft_labels']
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs.loss
    
    def training_step(self, batch, step_type: str):
        """Single training step"""
        self.model.train()
        
        # Move to device
        batch = {k: v.to(self.config.device) for k, v in batch.items()}
        
        # Compute loss
        loss = self.compute_loss(self.model, batch, step_type)
        
        # Scale loss by gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return loss.item()
    
    def train_epoch(self, dataloader, epoch: int):
        """Train for one epoch with alternating steps"""
        self.model.train()
        total_causal_lm_loss = 0.0
        total_sft_loss = 0.0
        step_count = 0
        
        logger.info(f"Starting epoch {epoch}")
        
        for batch_idx, batch in enumerate(dataloader):
            # Step t: Causal LM loss on document text (n_batch examples)
            causal_lm_loss = self.training_step(batch, "causal_lm")
            total_causal_lm_loss += causal_lm_loss
            
            # Step t+1: SFT on propagation questions (n_batch * K examples)
            sft_loss = self.training_step(batch, "sft")
            total_sft_loss += sft_loss
            
            step_count += 1
            
            # Accumulate gradients and step optimizer
            if step_count % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_causal_lm_loss = total_causal_lm_loss / self.config.logging_steps
                    avg_sft_loss = total_sft_loss / self.config.logging_steps
                    avg_total_loss = avg_causal_lm_loss + avg_sft_loss
                    
                    logger.info(f"Step {self.global_step}, Causal LM Loss: {avg_causal_lm_loss:.4f}, SFT Loss: {avg_sft_loss:.4f}, Total Loss: {avg_total_loss:.4f}")
                    
                    if self.config.use_wandb:
                        wandb.log({
                            "train/causal_lm_loss": avg_causal_lm_loss,
                            "train/sft_loss": avg_sft_loss,
                            "train/total_loss": avg_total_loss,
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/global_step": self.global_step
                        })
                    
                    total_causal_lm_loss = 0.0
                    total_sft_loss = 0.0
                
                # Evaluation
                if self.eval_dataset and self.global_step % self.config.eval_steps == 0:
                    eval_loss = self.evaluate()
                    logger.info(f"Evaluation at step {self.global_step}: Eval Loss = {eval_loss:.4f}")
                    
                    if self.config.use_wandb:
                        wandb.log({
                            "eval/loss": eval_loss,
                            "eval/global_step": self.global_step
                        })
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
    
    def evaluate(self):
        """Evaluate the model"""
        self.model.eval()
        total_causal_lm_loss = 0.0
        total_sft_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.dataloader_num_workers,
                collate_fn=self.data_collator
            ):
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                
                # Evaluate on both causal LM and SFT
                causal_lm_loss = self.compute_loss(self.model, batch, "causal_lm")
                sft_loss = self.compute_loss(self.model, batch, "sft")
                
                total_causal_lm_loss += causal_lm_loss.item()
                total_sft_loss += sft_loss.item()
                num_batches += 1
        
        avg_causal_lm_loss = total_causal_lm_loss / num_batches
        avg_sft_loss = total_sft_loss / num_batches
        avg_total_loss = avg_causal_lm_loss + avg_sft_loss
        
        # Update best model
        if avg_total_loss < self.best_eval_loss:
            self.best_eval_loss = avg_total_loss
            self.save_checkpoint(is_best=True)
        
        return avg_total_loss
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_eval_loss': self.best_eval_loss,
        }, os.path.join(checkpoint_dir, 'training_state.pt'))
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        if is_best:
            best_dir = os.path.join(self.config.output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            self.model.save_pretrained(best_dir)
            self.tokenizer.save_pretrained(best_dir)
            logger.info(f"Saved best model to {best_dir}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting alternating training")
        
        # Create data loaders
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=self.data_collator
        )
        
        # Training loop
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            self.train_epoch(train_dataloader, epoch)
            
            # Final evaluation at end of epoch
            if self.eval_dataset:
                eval_loss = self.evaluate()
                logger.info(f"End of epoch {epoch}: Eval Loss = {eval_loss:.4f}")
        
        logger.info("Training completed!")
        
        # Save final model
        self.save_checkpoint()
        
        if self.config.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Alternating Training Script")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="alternating-training", help="Wandb project name")
    
    args = parser.parse_args()
    import pdb; pdb.set_trace()
    
    # Load config if provided
    if args.config:
        config = OmegaConf.load(args.config)
        config = TrainingConfig(**config)
    else:
        config = TrainingConfig(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project
        )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    logger.info(f"Loading dataset from: {config.dataset_path}")
    train_dataset = AlternatingDataset(config.dataset_path, tokenizer)
    
    # Create eval dataset (use subset of train for now)
    eval_size = min(100, len(train_dataset) // 10)
    eval_dataset = torch.utils.data.Subset(train_dataset, range(eval_size))
    
    # Initialize trainer
    trainer = AlternatingTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
