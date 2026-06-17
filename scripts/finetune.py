from peft import LoraConfig
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from llm_finetune.data.alpaca import load_dataset
from llm_finetune.models.llama import load_llama_3_2_1b


def main() -> None:
    dataset = load_dataset()

    tokenizer, model = load_llama_3_2_1b()

    training_config = SFTConfig(
        dataloader_pin_memory=False,
        loss_type="nll",
        max_length=512,
        per_device_train_batch_size=4,
        report_to="wandb",
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"].select(range(1000)),
        processing_class=tokenizer,
        args=training_config,
        peft_config=lora_config,
    )

    trainer.train()


if __name__ == "__main__":
    main()
