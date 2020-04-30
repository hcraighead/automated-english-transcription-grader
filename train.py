import numpy as np
import io
import os

from transformers import AdamW, WarmupLinearSchedule
import random
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from helpers.masked_lm import get_special_tokens_mask, mask_tokens
from helpers.metrics import get_losses, compute_metrics


class Trainer:
    def __init__(self, args, grader, training_objectives, bert_tokenizer=None):
        self.args = args
        self.grader = grader
        self.training_objectives = training_objectives
        self.bert_tokenizer = bert_tokenizer

        # Sets seed
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

    def train(self, train_data, valid_data, save_file_name=None):
        """
        Trains a SpeechGraderModel for the set number of steps or epochs, while evaluating and saving models in
        accordance with the provided ags.
        """
        self.eval_data = valid_data
        tb_writer = SummaryWriter(os.path.join('runs', self.args.output_dir))
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
            len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.grader.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.grader.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.args.warmup_steps, t_total=t_total)

        self.args.logger.info("***** Running training *****")
        self.args.logger.info("  Num examples = %d", len(train_data))
        self.args.logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        self.args.logger.info("  Instantaneous batch size = %d", self.args.train_batch_size)
        self.args.logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        self.args.logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.tr_loss, self.logging_loss = 0.0, 0.0
        self.grader.zero_grad()
        self.best_valid_loss = float('inf')
        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch", disable=False)

        try:
            for _ in train_iterator:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
                for step, batch in enumerate(epoch_iterator):
                    self._train_iteration(batch, scheduler, optimizer, step, tb_writer)
                    if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                        epoch_iterator.close()
                        break
                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    train_iterator.close()
                    break

            tb_writer.close()
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        self._save_model('final')

    def test(self, test_data, prefix="test"):
        """
        Tests the model on the provided test_data.
        """
        self.eval_data = test_data
        tb_writer = SummaryWriter(os.path.join('runs', self.args.output_dir))
        results = self.evaluate(writer=tb_writer, prefix=prefix, verbose=True)
        tb_writer.close()
        return results

    def evaluate(self, writer=None, prefix="", verbose=False):
        """
        Evaluates the model on the pre-existing eval_data.
        """
        self.args.eval_batch_size = self.args.eval_batch_size
        eval_sampler = SequentialSampler(self.eval_data)
        eval_dataloader = DataLoader(self.eval_data, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        self.args.logger.info("***** Running evaluation {} *****".format(prefix))
        self.args.logger.info("  Num examples = %d", len(self.eval_data))
        self.args.logger.info("  Batch size = %d", self.args.eval_batch_size)
        nb_eval_steps = 0
        if verbose or self.args.predictions_file:
            all_score_predictions = torch.zeros(len(self.eval_data), device=self.args.device)
            all_score_targets = torch.zeros(len(self.eval_data), device=self.args.device)
        total_losses = {}
        for i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            start = i * self.args.eval_batch_size
            end = start + self.args.eval_batch_size
            if i == len(eval_dataloader):
                end = len(self.eval_data)

            self.grader.eval()
            inputs, labels = self._get_inputs_and_labels(batch, eval=True)
            predictions = self.grader(inputs)
            if verbose or self.args.predictions_file:
                all_score_predictions[start:end] = predictions['score'].detach()
                all_score_targets[start:end] = labels['score'].detach()
            for objective, loss in get_losses(self.training_objectives, predictions, labels, self.args.device).items():
                if total_losses.get(objective, None):
                    total_losses[objective] += loss.item()
                else:
                    total_losses[objective] = loss.item()
            nb_eval_steps += 1

        # Record model score predictions
        if self.args.predictions_file:
            with io.open(self.args.predictions_file, 'w') as file:
                predictions = '\n'.join([str(score) for score in all_score_predictions.cpu().tolist()])
                file.write(predictions)
                self.args.logger.info("Predictions stored at ".format(self.args.predictions_file))

        # Computes the losses per objective and records + logs them.
        total_losses = {objective: loss / nb_eval_steps for objective, loss in total_losses.items()}
        if verbose:
            compute_metrics(total_losses, all_score_predictions, all_score_targets, self.args.device)
        self.args.logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(total_losses.keys()):
            self.args.logger.info("  %s = %s", key, str(total_losses[key]))
        if verbose and writer:
            header_row = 'Predicted score | Actual score\n---|---\n'
            table_rows = ['{} | {}'.format(pred, target) for pred, target in
                          zip(all_score_predictions, all_score_targets)]
            table_rows = '\n'.join(table_rows)
            writer.add_text('test_predictions', header_row + table_rows)

        return total_losses

    def _get_inputs_and_labels(self, batch, eval=False):
        """Prepares the input and labels for a batch."""
        batch = tuple(t.to(self.args.device) for t in batch)
        if self.args.model == 'bert':
            # Tokens should only be masked during training.
            if not eval and 'mlm' in self.training_objectives:
                input_ids, mlm_mask = mask_tokens(self.bert_tokenizer, batch[0], self.args.device)
            else:
                input_ids = batch[0]
                mlm_mask = input_ids.clone()
                special_tokens_mask = torch.tensor(
                    [get_special_tokens_mask(self.bert_tokenizer, val) for val in mlm_mask.tolist()],
                    dtype=torch.bool)
                mlm_mask[special_tokens_mask] = -1
            inputs = {'input_ids': input_ids.to(self.args.device),
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]}
            labels = {
                'score': batch[3],
                'pos': batch[4],
                'deprel': batch[5],
                'native_language': batch[6],
                'mlm': mlm_mask,
            }
            return inputs, labels
        # else lstm
        input_ids = batch[0].to(self.args.device).permute(1, 0)
        lm = torch.zeros((input_ids.size(0) * 2, input_ids.size(1)), dtype=torch.long)
        lm[:len(input_ids) - 1, :] = input_ids[1:, :]
        lm[len(input_ids) + 1:, :] = input_ids[:-1, :]
        lm = torch.where(lm == 0, torch.full((lm.size(0), lm.size(1)), -1, dtype=torch.long), lm).to(self.args.device)
        labels = {
            'score': batch[2],
            'pos': batch[3].permute(1, 0),
            'deprel': batch[4].permute(1, 0),
            'native_language': batch[5].permute(1, 0),
            'lm': lm,
        }
        return input_ids, labels

    def _train_iteration(self, batch, scheduler, optimizer, step, tb_writer):
        """Performs a single training iteration for a single batch."""
        self.grader.train()
        inputs, labels = self._get_inputs_and_labels(batch)
        training_objective_predictions = self.grader(inputs)
        losses = get_losses(self.training_objectives, training_objective_predictions, labels, self.args.device)
        loss = losses['overall']
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.grader.parameters(), self.args.max_grad_norm)

        self.tr_loss += loss.item()
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            self.grader.zero_grad()
            self.global_step += 1
            train_loss = (self.tr_loss - self.logging_loss) / self.args.logging_steps

            if self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0:
                # Log metrics
                if self.args.evaluate_during_training:
                    results = self.evaluate(tb_writer)
                    for key, value in results.items():
                        tb_writer.add_scalar('eval_{}'.format(key), value, self.global_step)
                    if self.args.save_all_checkpoints:
                        self._save_model('checkpoint_{}'.format(self.global_step))

                    if self.args.save_best_on_evaluate:
                        if results['score'] >= self.best_valid_loss:
                            return
                        self.best_valid_loss = results['score']
                        self._save_model('best')

                tb_writer.add_scalar('lr', scheduler.get_lr()[0], self.global_step)
                tb_writer.add_scalar('train_loss', train_loss, self.global_step)
                for key, value in losses.items():
                    tb_writer.add_scalar('train_{}'.format(key), value, self.global_step)
                self.logging_loss = self.tr_loss

    def _save_model(self, prefix):
        """Saves model at --output_dir/prefix"""
        output_dir = os.path.join(self.args.output_dir, prefix)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.args.model == 'bert':
            self.grader.save_pretrained(output_dir)
            self.bert_tokenizer.save_pretrained(output_dir)
        else:
            torch.save(self.grader.state_dict(), os.path.join(output_dir, 'lstm.model'))
        self.args.logger.info("Saving model checkpoint to %s", output_dir)
