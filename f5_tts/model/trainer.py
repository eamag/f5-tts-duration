from __future__ import annotations

import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from einops import rearrange

from f5_tts.model.duration import DurationPredictor
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import exists
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn


class DurationTrainer:
    def __init__(
        self,
        model: DurationPredictor,
        optimizer,
        num_warmup_steps=20000,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        sample_rate=24_000,
        accelerate_kwargs: dict = dict(),
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.target_sample_rate = sample_rate

        self.model = model
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.mel_spectrogram = MelSpec()

        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        self.max_grad_norm = max_grad_norm

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def checkpoint_path(self, step: int):
        return f"f5tts_duration_{step}.pt"

    def save_checkpoint(self, step, finetune=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.accelerator.unwrap_model(
                    self.optimizer
                ).state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                step=step,
            )

            self.accelerator.save(checkpoint, self.checkpoint_path(step))

    def load_checkpoint(self, step=0):
        if not exists(self.checkpoint_path(step)) or not os.path.exists(
            self.checkpoint_path(step)
        ):
            return 0

        checkpoint = torch.load(
            self.checkpoint_path(step), map_location="cpu", weights_only=True
        )
        self.accelerator.unwrap_model(self.model).load_state_dict(
            checkpoint["model_state_dict"]
        )
        # self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["step"]

    def train(
        self, train_dataset, epochs, max_batch_tokens, num_workers=12, save_step=1000
    ):
        dynamic_sampler = DynamicBatchSampler(
            train_dataset, max_batch_tokens=max_batch_tokens, collate_fn=collate_fn
        )
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            batch_sampler=dynamic_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        total_steps = len(train_dataloader) * epochs
        decay_steps = total_steps - self.num_warmup_steps
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=self.num_warmup_steps,
        )
        decay_scheduler = LinearLR(
            self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[self.num_warmup_steps],
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )
        start_step = self.load_checkpoint()
        global_step = start_step

        hps = {
            "epochs": epochs,
            "num_warmup_steps": self.num_warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            # "batch_size": batch_size,
        }
        self.accelerator.init_trackers("f5tts_duration", config=hps)

        for epoch in range(epochs):
            self.model.train()
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch+1}/{epochs}",
                unit="step",
                disable=not self.accelerator.is_local_main_process,
            )
            epoch_loss = 0.0

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = rearrange(batch["mel"], "b d n -> b n d")
                    mel_lengths = batch["mel_lengths"]

                    loss = self.model(
                        mel_spec, text=text_inputs, lens=mel_lengths, return_loss=True
                    )
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                self.accelerator.log(
                    {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]},
                    step=global_step,
                )

                global_step += 1
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

                if global_step % save_step == 0:
                    self.save_checkpoint(global_step)

            epoch_loss /= len(train_dataloader)
            if self.accelerator.is_local_main_process:
                self.accelerator.log(
                    {"epoch average loss": epoch_loss}, step=global_step
                )

        self.accelerator.end_training()

        # self.writer.close()
