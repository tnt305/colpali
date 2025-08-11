    import os
    from dataclasses import dataclass
    from typing import Callable, Dict, List, Optional, Union
    from PIL import Image
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        PreTrainedModel,
        TrainingArguments,
    )
    from torch.utils.data import Dataset

    from colpali_engine.collators import VisualRetrieverCollator
    from colpali_engine.data.dataset import ColPaliEngineDataset
    from colpali_engine.loss.late_interaction_losses import (
        ColbertLoss,
    )
    from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
    from colpali_engine.utils.gpu_stats import print_gpu_utilization, print_summary
    from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

    #### ADD on 
    class VLSPColPaliDataset(Dataset):
        """Custom dataset cho VLSP với image support"""
        
        def __init__(self, data, query_column="query", image_column="image", 
                    pos_target_column="pos_target", neg_target_column="neg_target"):
            self.data = data
            self.query_column = query_column
            self.image_column = image_column
            self.pos_target_column = pos_target_column
            self.neg_target_column = neg_target_column
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            sample = self.data[idx]
            
            return {
                'query': sample[self.query_column],
                'image': sample[self.image_column],  # QUAN TRỌNG: Thêm image
                'pos_target': sample[self.pos_target_column] if isinstance(sample[self.pos_target_column], list) else [sample[self.pos_target_column]],
                'neg_target': sample[self.neg_target_column] if isinstance(sample[self.neg_target_column], list) else [sample[self.neg_target_column]],
            }

    @dataclass
    class ColModelTrainingConfig:
        model: Union[PreTrainedModel, PeftModel]
        processor: BaseVisualRetrieverProcessor
        train_dataset: Union[VLSPColPaliDataset, List[VLSPColPaliDataset]]
        eval_dataset: Optional[Union[VLSPColPaliDataset, Dict[str, VLSPColPaliDataset]]] = None
        tr_args: Optional[TrainingArguments] = None
        output_dir: Optional[str] = None
        max_length: int = 256
        run_eval: bool = True
        run_train: bool = True
        peft_config: Optional[LoraConfig] = None
        loss_func: Optional[Callable] = ColbertLoss()
        pretrained_peft_model_name_or_path: Optional[str] = None
        """
        Config class used for training a ColVision model.
        """

        def __post_init__(self):
            """
            Initialize the model and tokenizer if not provided
            """
            if self.output_dir is None:
                sanitized_name = str(self.model.name_or_path).replace("/", "_")
                self.output_dir = f"./models/{sanitized_name}"

            if self.tr_args is None:
                print("No training arguments provided. Using default.")
                self.tr_args = TrainingArguments(output_dir=self.output_dir)
            elif self.tr_args.output_dir is None or self.tr_args.output_dir == "trainer_output":
                self.tr_args.output_dir = self.output_dir

            if isinstance(self.tr_args.learning_rate, str):
                print("Casting learning rate to float")
                self.tr_args.learning_rate = float(self.tr_args.learning_rate)

            self.tr_args.remove_unused_columns = False

            if self.pretrained_peft_model_name_or_path is not None:
                print("Loading pretrained PEFT model")
                self.model.load_adapter(self.pretrained_peft_model_name_or_path, is_trainable=True)

            if self.peft_config is not None:
                print("Configurating PEFT model")
                if self.pretrained_peft_model_name_or_path is None:
                    self.model = get_peft_model(self.model, self.peft_config)
                    self.model.print_trainable_parameters()
                else:
                    print(f"Adapter already loaded from {self.pretrained_peft_model_name_or_path}. Not overwriting.")

        print_gpu_utilization()

    class VisualRetrieverCollatorV2:
        """Fixed collator cho ColInternVL2 với VLSP dataset"""
        
        def __init__(self, processor, max_length=512):
            self.processor = processor
            self.max_length = max_length
        
        def __call__(self, batch):
            # Separate images, queries, và documents
            images = []
            queries = []
            all_docs = []
            for example in batch:
                # Load image
                try:
                    image_path = example['image']
                    with Image.open(image_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        images.append(img.copy())
                except Exception as e:
                    print(f"Error loading image {example['image']}: {e}")
                    # Use dummy image if load fails
                    images.append(Image.new('RGB', (224, 224), color='white'))
                
                # Collect text
                queries.append(example['query'])
                all_docs.extend(example['pos_target'])
                all_docs.extend(example['neg_target'])
            
            # Process với processor
            try:
                # Process images - for documents, not queries
                image_inputs = self.processor.process_images(images)
                
                # Process text queries (chỉ text, không có ảnh)
                query_inputs = self.processor.process_queries(queries)
                
                # Process documents (text only)
                try:
                    doc_inputs = self.processor.process_docs(all_docs)
                except:
                    doc_inputs = self.processor.process_queries(all_docs)
                    
                # Tạo batch output
                batch_output = {
                    # Image inputs cho documents
                    'pixel_values': image_inputs.pixel_values,
                    
                    # Query inputs (chỉ text)
                    'query_input_ids': query_inputs.input_ids,
                    'query_attention_mask': query_inputs.attention_mask,
                    'query_pixel_values': None,  # Query không có ảnh
                    
                    # Document inputs
                    'doc_input_ids': doc_inputs.input_ids,
                    'doc_attention_mask': doc_inputs.attention_mask,
                }
                
                return batch_output
                
            except Exception as e:
                print(f"Processing error: {e}")
                raise

    class ColModelTraining:
        """
        Class that contains the training and evaluation logic for a ColVision model.
        """

        def __init__(self, config: ColModelTrainingConfig) -> None:
            self.config = config
            self.model = self.config.model
            self.current_git_hash = os.popen("git rev-parse HEAD").read().strip()
            self.train_dataset = self.config.train_dataset
            self.eval_dataset = self.config.eval_dataset
            self.collator = VisualRetrieverCollatorV2(
                processor=self.config.processor,
                max_length=self.config.max_length,
            )

        def train(self) -> None:
            trainer = ContrastiveTrainer(
                model=self.model,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                args=self.config.tr_args,
                data_collator=self.collator,
                loss_func=self.config.loss_func,
                is_vision_model=self.config.processor is not None,
            )

            trainer.args.remove_unused_columns = False

            result = trainer.train(resume_from_checkpoint=self.config.tr_args.resume_from_checkpoint)
            print_summary(result)

        def eval(self) -> None:
            raise NotImplementedError("Evaluation is not implemented yet.")

        def save(self):
            """
            Save the model with its training config, as well as the tokenizer and processor if provided.
            """
            self.model.save_pretrained(self.config.output_dir)
            self.config.processor.save_pretrained(self.config.output_dir)

            # Save git hash of the commit at beginning of training
            with open(f"{self.config.output_dir}/git_hash.txt", "w") as f:
                f.write(self.current_git_hash)
