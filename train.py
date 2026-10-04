from MyModel import SimpleBertModel
import sys
import os
import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score,precision_score, recall_score
from transformers import get_linear_schedule_with_warmup
from data_utils import creatr_data_loader
import pandas as pd
import json
import pickle
from datetime import datetime
import math

class Instructor:
    def __init__(self,opt):
        self.opt = opt
        if opt.dataset == "twitter15":
            train_data_name = 'middleFile/twitter15_train_datas.pkl'
            val_data_name = 'middleFile/twitter15_val_datas.pkl'
            test_data_name = 'middleFile/twitter15_test_datas.pkl'
        else:
            train_data_name = 'middleFile/twitter17_train_datas.pkl'
            val_data_name = 'middleFile/twitter17_val_datas.pkl'
            test_data_name = 'middleFile/twitter17_test_datas.pkl'
        if os.path.exists(train_data_name):
            self.train_data_loader = pickle.load(open(train_data_name, 'rb'))
        else:
            train_data_loader = creatr_data_loader(opt.dataset_file['train'], 'train', opt.MAX_LEN, opt.BATCH_SIZE)
            with open(train_data_name, 'wb') as f:
                pickle.dump(train_data_loader, f)
            self.train_data_loader = train_data_loader
        if os.path.exists(val_data_name):
            self.val_data_loader = pickle.load(open(val_data_name, 'rb'))
        else:
            val_data_loader = creatr_data_loader(opt.dataset_file['dev'], 'dev', opt.MAX_LEN, opt.BATCH_SIZE)
            with open(val_data_name, 'wb') as f:
                pickle.dump(val_data_loader, f)
            self.val_data_loader = val_data_loader
        self.test_data_loader = None
        if opt.EVAL_MODE == "final_test":
            if os.path.exists(test_data_name):
                self.test_data_loader = pickle.load(open(test_data_name, 'rb'))
            else:
                test_data_loader = creatr_data_loader(
                    opt.dataset_file['test'], 'test', opt.MAX_LEN, opt.BATCH_SIZE
                )
                with open(test_data_name, 'wb') as f:
                    pickle.dump(test_data_loader, f)
                self.test_data_loader = test_data_loader

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def train_epoch(self,loss_fn, optimizer, scheduler):
        self.model.train()
        losses = []
        correct_predictions = 0
        n_total = 0
        for i_batch,sample_batched in enumerate(self.train_data_loader):
            #print(i_batch)
            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            outputs = self.model(inputs)
            targets = sample_batched['targets'].to(self.opt.device)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets).item()
            losses.append(loss.item())
            n_total += len(outputs)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        #print('train'+str(n_total))
        return correct_predictions / n_total, np.mean(losses)

    def eval_model(self,loss_fn,type):
        if type == 'dev':
            dataloader = self.val_data_loader
        else:
            if self.test_data_loader is None:
                raise RuntimeError(
                    "Test evaluation is disabled in validation_only mode."
                )
            dataloader = self.test_data_loader
        losses = []
        correct_predictions = 0
        n_total = 0
        rows = []
        self.model.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['targets'].to(self.opt.device)
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, targets)
                correct_predictions += torch.sum(preds == targets).item()
                n_total+=len(outputs)
                losses.append(loss.item())

                rows.extend(
                    zip(
                        sample_batched["review_text"],
                        sample_batched["sentiment_targets"],
                        sample_batched["targets"].numpy(),
                        preds.cpu().numpy(),
                    )
                )
        return (
            correct_predictions / n_total,
            np.mean(losses),
            self.format_eval_output(rows)
        )

    def format_eval_output(self,rows):
        tweets, targets, labels, predictions = zip(*rows)
        tweets = np.vstack(tweets)
        targets = np.vstack(targets)
        labels = np.vstack(labels)
        predictions = np.vstack(predictions)
        results_df = pd.DataFrame()
        results_df["tweet"] = tweets.reshape(-1).tolist()
        results_df["target"] = targets.reshape(-1).tolist()
        results_df["label"] = labels
        results_df["prediction"] = predictions
        return results_df


    def run(self):
        results_per_run = {}
        stage_name = "D" if self.opt.EVAL_MODE == "validation_only" else "C"

        for run_number in range(self.opt.NUM_RUNS):
            seed_num = self.opt.RANDOM_SEEDS[run_number]
            np.random.seed(seed_num)
            torch.manual_seed(seed_num)
            # np.random.seed(1)
            # torch.manual_seed(1)
            self.model = self.opt.model_class(self.opt).to(self.opt.device)
            print(f"[MODEL] GCN layers = {self.model.layers}")

            #Configure the optimizer and scheduler.
            #优化器
            optimizer = AdamW(self.model.parameters(), lr=self.opt.LEARNING_RATE)
            total_steps = len(self.train_data_loader) * self.opt.EPOCHS
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.opt.NUM_WARMUP_STEPS, num_training_steps=total_steps
            )
            #损失函数
            loss_fn = nn.CrossEntropyLoss().to(self.opt.device)
            best_val_f1 = -1.0
            best_val_epoch = 0
            best_val_acc = 0.0
            os.makedirs('./checkpoints', exist_ok=True)
            checkpoint_path = (
                f'./checkpoints/mggfn_{self.opt.dataset}_stage{stage_name}_gcn{self.opt.GCN_LAYERS}_'
                f'validation_selected_run{run_number}.pt'
            )

            for epoch in range(self.opt.EPOCHS):
                print(f"Epoch {epoch + 1}/{self.opt.EPOCHS} -- RUN {run_number}")
                print("-" * 30)
                train_acc, train_loss = self.train_epoch(loss_fn, optimizer, scheduler)
                print(f"[TRAIN] Train loss {train_loss} accuracy {train_acc}")

                val_acc, val_loss, val_detailed_results = self.eval_model(loss_fn,"dev")
                val_labels = val_detailed_results['label']
                val_predictions = val_detailed_results['prediction']
                val_macro_f1 = f1_score(
                    val_labels, val_predictions, average="macro", zero_division=0
                )
                val_macro_precision = precision_score(
                    val_labels, val_predictions, average='macro', zero_division=0
                )
                val_macro_recall = recall_score(
                    val_labels, val_predictions, average='macro', zero_division=0
                )

                print(f"[VALIDATION] Val   loss {val_loss} accuracy {val_acc}")
                print(f"[VALIDATION] MACRO F1 = {val_macro_f1:.4f}\n"
                      f"[VALIDATION] Precision = {val_macro_precision:.4f}\n"
                      f"[VALIDATION] Recall = {val_macro_recall:.4f}")

                if val_macro_f1 > best_val_f1:
                    best_val_f1 = val_macro_f1
                    best_val_epoch = epoch + 1
                    best_val_acc = val_acc
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "checkpoint_info": {
                                "stage": stage_name,
                                "run_number": run_number,
                                "seed": seed_num,
                                "epoch": best_val_epoch,
                                "selection_set": "validation",
                                "selection_metric": "macro_f1",
                                "gcn_layers": self.opt.GCN_LAYERS,
                                "validation_accuracy": best_val_acc,
                                "validation_macro_f1": best_val_f1,
                            },
                            "dataset": self.opt.dataset,
                        },
                        checkpoint_path,
                    )
                    print(
                        f"[VALIDATION] Best checkpoint saved: epoch {best_val_epoch}, "
                        f"MACRO F1 = {best_val_f1:.4f}"
                    )

            checkpoint = torch.load(
                checkpoint_path, map_location=self.opt.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"[VALIDATION] Best checkpoint loaded: epoch {best_val_epoch}, "
                f"MACRO F1 = {best_val_f1:.4f}"
            )

            results_per_run[run_number] = {
                "seed": seed_num,
                "gcn_layers": self.opt.GCN_LAYERS,
                "best_validation_epoch": best_val_epoch,
                "validation_accuracy": best_val_acc,
                "validation_macro-f1": best_val_f1,
                "checkpoint": checkpoint_path,
            }

            if self.opt.EVAL_MODE == "final_test":
                test_acc, test_loss, test_detailed_results = self.eval_model(loss_fn, 'test')
                test_labels = test_detailed_results['label']
                test_predictions = test_detailed_results['prediction']
                test_macro_f1 = f1_score(
                    test_labels, test_predictions, average="macro", zero_division=0
                )
                test_macro_precision = precision_score(
                    test_labels, test_predictions, average='macro', zero_division=0
                )
                test_macro_recall = recall_score(
                    test_labels, test_predictions, average='macro', zero_division=0
                )

                print(f"[TEST] Test  loss {test_loss} accuracy {test_acc}")
                print(f"[TEST] TEST ACC = {test_acc:.4f}\n"
                      f"[TEST] MACRO F1 = {test_macro_f1:.4f}\n"
                      f"[TEST] Precision = {test_macro_precision:.4f}\n"
                      f"[TEST] Recall = {test_macro_recall:.4f}")

                results_per_run[run_number].update({
                    "test_accuracy": test_acc,
                    "test_macro-f1": test_macro_f1,
                    "test_precision": test_macro_precision,
                    "test_recall": test_macro_recall,
                })

        os.makedirs('./result', exist_ok=True)
        per_run_result_path = (
            f'./result/results_per_run_stage{stage_name}_gcn{self.opt.GCN_LAYERS}_'
            f'validation_selected_{self.opt.dataset}.json'
        )
        with open(per_run_result_path, 'w+') as f:
            json.dump(results_per_run, f)

        resSummary = {
            "model": {
                "stage": stage_name,
                "gcn_layers": self.opt.GCN_LAYERS
            },
            "selection": {
                "dataset": "validation",
                "metric": "macro-f1"
            }
        }

        if self.opt.EVAL_MODE == "validation_only":
            validation_accuracies = [
                result["validation_accuracy"] for result in results_per_run.values()
            ]
            validation_macro_f1_scores = [
                result["validation_macro-f1"] for result in results_per_run.values()
            ]
            std_ddof = 1 if len(validation_accuracies) > 1 else 0
            resSummary["validationResult"] = {
                "acc_mean": float(np.mean(validation_accuracies)),
                "acc_std": float(np.std(validation_accuracies, ddof=std_ddof)),
                "f1_mean": float(np.mean(validation_macro_f1_scores)),
                "f1_std": float(np.std(validation_macro_f1_scores, ddof=std_ddof))
            }
        else:
            test_accuracies = [
                result["test_accuracy"] for result in results_per_run.values()
            ]
            test_macro_f1_scores = [
                result["test_macro-f1"] for result in results_per_run.values()
            ]
            std_ddof = 1 if len(test_accuracies) > 1 else 0
            resSummary["testResult"] = {
                "acc_mean": float(np.mean(test_accuracies)),
                "acc_std": float(np.std(test_accuracies, ddof=std_ddof)),
                "f1_mean": float(np.mean(test_macro_f1_scores)),
                "f1_std": float(np.std(test_macro_f1_scores, ddof=std_ddof))
            }

        curr_time = datetime.now()
        time_str = (
            f"{datetime.strftime(curr_time, '%Y-%m-%d_%H-%M-%S')}_"
            f"{self.opt.dataset}_stage{stage_name}_gcn{self.opt.GCN_LAYERS}"
        )
        with open(f'./result/{time_str}.json', 'w+') as f:
            json.dump(resSummary, f)
        print(f"[VALIDATION] Selection rule = {resSummary['selection']}")
        if self.opt.EVAL_MODE == "validation_only":
            print(f"[VALIDATION] Summary = {resSummary['validationResult']}")
            print("[TEST] Evaluation skipped (validation-only depth selection).")
        else:
            print(f"[TEST] Summary = {resSummary['testResult']}")


def main():
    model_classes = {
        "simpleBert":SimpleBertModel
    }
    dataset_files = {
        'twitter15':{
            'train':'data/twitter2015/train.tsv',
            'dev':'data/twitter2015/dev.tsv',
            'test':'data/twitter2015/test.tsv'
        },
        'twitter17': {
            'train': 'data/twitter2017/train.tsv',
            'dev': 'data/twitter2017/dev.tsv',
            'test': 'data/twitter2017/test.tsv'
        }
    }
    input_colses = {
        "simpleBert":["input_ids","attention_mask","vit_feature","transformer_mask","target_input_ids","target_attention_mask","target_mask","text_length","word_length","tran_indices","context_asp_adj_matrix","globel_input_id","globel_mask","face_input_ids","face_mask"]
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='simpleBert', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='twitter15', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--MAX_LEN', default=50, type=int)
    parser.add_argument('--BATCH_SIZE', default=32, type=int)
    parser.add_argument('--DROPOUT_PROB', default=0.1, type=float)
    parser.add_argument('--NUM_CLASSES', default=3, type=int)
    parser.add_argument('--DEVICE', default="cuda:0", type=str)
    parser.add_argument('--EPOCHS', default=20, type=int)
    parser.add_argument('--LEARNING_RATE', default=5e-5, type=float)
    parser.add_argument('--ADAMW_CORRECT_BIAS',default=True, action='store_true')
    parser.add_argument('--NUM_WARMUP_STEPS', default=0, type=int)
    parser.add_argument('--NUM_RUNS', default=1,type=int)
    parser.add_argument('--GCN_LAYERS', default=1, type=int)
    parser.add_argument(
        '--EVAL_MODE', default='final_test',
        choices=['validation_only', 'final_test'], type=str
    )
    parser.add_argument('--RANDOM_SEEDS', nargs='+', type=int, default=None)
    opt = parser.parse_args()

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    if opt.RANDOM_SEEDS is None:
        opt.RANDOM_SEEDS = [3] if opt.NUM_RUNS == 1 else list(range(opt.NUM_RUNS))
    else:
        opt.NUM_RUNS = len(opt.RANDOM_SEEDS)
    print(opt.RANDOM_SEEDS)

    opt.device = torch.device(opt.DEVICE if torch.cuda.is_available() else 'cpu')

    ins = Instructor(opt)
    ins.run()



if __name__ == '__main__':
    main()
