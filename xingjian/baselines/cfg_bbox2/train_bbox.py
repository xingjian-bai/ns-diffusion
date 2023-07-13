import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
# from diffusers.training_utils import EMAModel
# from diffusers.utils import check_min_version, deprecate, is_wandb_available
# from diffusers.utils.import_utils import is_xformers_available
import wandb

import sys
sys.path.append("..")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="default",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="2O",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()

    

    from datetime import datetime
    current_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    args.output_dir = os.path.join(args.output_dir, current_time)

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process and args.wandb:
        wandb.init(
            project="diffusion_bbox",
            job_type="train_model",
            name=f"{args.name}-{current_time}",
            save_code=True,
        )

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler(
        num_train_timesteps = 1000,
        beta_schedule = "linear",
        prediction_type = "epsilon",
    )

    from pipeline_bbox import Denoiser, BboxPipeline
    model = Denoiser()

    # if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
    #     # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    #     def save_model_hook(models, weights, output_dir):
    #         for i, model in enumerate(models):
    #             model.save_pretrained(os.path.join(output_dir, "unet"))
    #             weights.pop()
    #     accelerator.register_save_state_pre_hook(save_model_hook)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.data == "2O":
        from dataset import RelationalDataset2O
        dataset = RelationalDataset2O()
    elif args.data == "1O":
        from dataset import RelationalDataset1O
        dataset = RelationalDataset1O()
    else:
        raise ValueError("Unknown dataset")
    print(f"data loaded, size: {len(dataset)}")


    def collate_fn(batch):
        clean_images = torch.stack([item['clean_image'] for item in batch])
        objects = torch.stack([item['objects'] for item in batch])
        relations = torch.stack([item['relations'] for item in batch])
        bboxes = torch.stack([item['bboxes'] for item in batch])

        annotated_images = [item['annotated_image'] for item in batch]

        return {
            'clean_images': clean_images,
            'objects': objects,
            'relations': relations,
            'bboxes': bboxes,
            'annotated_images': annotated_images,
        }


    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    # progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        print(f"starting epoch {epoch} \n")
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     if step % args.gradient_accumulation_steps == 0:
            #         progress_bar.update(1)
            #     continue
                
            clean_images = batch['clean_images'].to(weight_dtype)
            objects = batch['objects'].to(weight_dtype)
            relations = batch['relations'].to(weight_dtype)
            bboxes = batch['bboxes'].to(weight_dtype)

            annotated_images = batch['annotated_images']

            # print(f"type of the items retrieved: {type(clean_images)}, {type(objects)}, {type(relations)}, {type(bboxes)}, {type(annotated_images)}")

            noise = torch.randn(
                bboxes.shape, dtype=(torch.float32 if args.mixed_precision == "no" else torch.float16)
            ).to(clean_images.device)
            batch_size = bboxes.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=clean_images.device).long()

            noisy_images = noise_scheduler.add_noise(bboxes, noise, timesteps)
            target = noise

            with accelerator.accumulate(model):

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)


                model_pred = model.denoise_batch(noisy_images, objects, relations, timesteps)
                # print(f"in training")
                # print(f"bboxes are {bboxes}")
                # print(f"noise is {target}")
                # print(f"model_pred is {model_pred}")
                # print(f"noised images are {noisy_images}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # if args.use_ema:
                #     ema_unet.step(unet.parameters())
                # progress_bar.update(1)
                global_step += 1
                if args.wandb:
                    wandb.log({"train_loss": train_loss}, step=global_step)

                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        print(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "epoch": epoch, "global_step": global_step}
            # progress_bar.set_postfix(**logs)
            # print(logs)
            # if args.wandb:
            #     wandb.log(logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if epoch % args.validation_epochs == 0:
                print(f"starting validation")
                unwrap_model = accelerator.unwrap_model(model)
                pipeline = BboxPipeline(unwrap_model, noise_scheduler)

                images = []
                prompts = []
                for _ in range(8):
                    from dataset import gen_rand_scene

                    num_objects = 1
                    num_relations = 0
                    objects, relations = gen_rand_scene(num_objects, num_relations)

                    bboxes = pipeline(objects, relations)

                    # output each element join by +
                    print(f"out of the pipeline: {'+'.join([str(bbox) for bbox in bboxes])}")
                    
                    image = Image.new("RGB", (128, 128), (255, 255, 255))
                    for bbox in bboxes:
                        image = bbox.draw(image)
                    images.append(image)
                    from dataset import prompt
                    prompts.append(prompt(objects = objects, relations = relations))

                if args.wandb:
                    print(f"logging to wandb")
                    wandb.log({"validation_images": [wandb.Image(image) for image in images], "validation_prompts": prompts}, step=global_step)

                

    accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()
