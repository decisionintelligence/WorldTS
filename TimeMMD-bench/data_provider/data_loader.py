import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
import json
import ast

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, llm_model=None, tokenizer=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.__read_data__()
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        # if self.args.use_closedllm==0:
        # text_name='Final_Search_'+str(self.args.text_len)
        text_name = 'fact'
        # else:
        #     print("!!!!!!!!!!!!Using output of closed source llm and Bert as encoder!!!!!!!!!!!!!!!")
        #     text_name="Final_Output"
        # Some long-horizon files only contain target, text, and date. These
        # compatibility columns are not model inputs, but Dataset_Custom
        # expects them while constructing its common interface.
        if 'prior_history_avg' not in df_raw.columns:
            df_raw['prior_history_avg'] = df_raw[self.target]
        if 'start_date' not in df_raw.columns:
            df_raw['start_date'] = df_raw['date']
        if 'end_date' not in df_raw.columns:
            df_raw['end_date'] = df_raw['date']
        df_raw = df_raw[['date'] + cols + [self.target] + ['prior_history_avg'] + ['start_date'] + ['end_date']]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.border1 = border1
        self.border2 = border2

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            df_data_prior = df_raw[['prior_history_avg']]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            data_prior = self.scaler.transform(df_data_prior.values[:, -1].reshape(-1, 1))
        else:
            data = df_data.values
            data_prior = df_data_prior.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_prior = data_prior[border1:border2]

        self.data_stamp = data_stamp
        self.date = df_raw[['date']][border1:border2].values
        self.start_date = df_raw[['start_date']][border1:border2].values
        self.end_date = df_raw[['end_date']][border1:border2].values
        self.text = df_raw[[text_name]][border1:border2].values

    def get_all_embeddings(self):
        """Pool frozen text embeddings in bounded chunks."""
        for i in range(len(self.text)):
            if pd.isnull(self.text[i][0]):
                self.text[i][0] = 'No information available'

        text_flattened = self.text.reshape(-1).tolist()
        embedding_batch_size = int(
            getattr(self.args, 'text_embedding_batch_size', 128)
        )

        pooled_chunks = []
        embedding_layer = self.llm_model.get_input_embeddings()
        pool_type = str(self.args.pool_type).lower()
        if pool_type not in {'avg', 'max', 'min'}:
            raise ValueError(f"Unsupported pool_type: {self.args.pool_type}")

        with torch.no_grad():
            for start in range(0, len(text_flattened), embedding_batch_size):
                tokenized_output = self.tokenizer(
                    text_flattened[start:start + embedding_batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                )
                input_ids = tokenized_output['input_ids'].to(
                    self.llm_model.device
                )
                attention_mask = tokenized_output['attention_mask'].to(
                    self.llm_model.device
                )
                token_embeddings = embedding_layer(input_ids)
                expanded_mask = attention_mask.unsqueeze(-1).expand_as(
                    token_embeddings
                )

                if pool_type == 'avg':
                    masked_embeddings = token_embeddings * expanded_mask
                    valid_counts = expanded_mask.sum(dim=1).clamp(min=1)
                    pooled = masked_embeddings.sum(dim=1) / valid_counts
                elif pool_type == 'max':
                    masked_embeddings = token_embeddings.masked_fill(
                        expanded_mask == 0,
                        float('-inf'),
                    )
                    pooled = masked_embeddings.max(dim=1).values
                else:
                    masked_embeddings = token_embeddings.masked_fill(
                        expanded_mask == 0,
                        float('inf'),
                    )
                    pooled = masked_embeddings.min(dim=1).values
                pooled_chunks.append(pooled.detach().float().cpu())

        self.text_embeddings = torch.cat(pooled_chunks, dim=0)
        if not bool(
            getattr(self.args, 'cache_text_embeddings_on_cpu', False)
        ):
            self.text_embeddings = self.text_embeddings.to(
                self.llm_model.device
            )

    def get_prior_y(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        r_begins = s_ends
        r_ends = r_begins + self.pred_len
        prior_y = np.array([self.data_prior[r_beg:r_end] for r_beg, r_end in zip(r_begins, r_ends)])
        return prior_y

    def get_prior_y_for_imputation(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        # r_begins = s_ends
        # r_ends = r_begins + self.pred_len
        prior_y = np.array([self.data_prior[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])
        return prior_y

    def get_text(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        text = np.array([np.array([str(self.text[i]) for i in range(s_end - self.seq_len, s_end)]) for s_end in s_ends])
        return text

    def get_text_embeddings(self, indices):

        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        bsz = len(s_begins)
        # return tensor
        text_embeddings = torch.cat([self.text_embeddings[s_end - self.seq_len: s_end] for s_end in s_ends],
                                    dim=0).view(bsz, self.seq_len, -1)
        return text_embeddings

    def get_date(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        r_begins = s_ends - self.label_len
        r_ends = r_begins + self.label_len + self.pred_len

        x_start_dates = np.array([self.start_date[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])
        x_end_dates = np.array([self.end_date[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])

        return x_start_dates, x_end_dates

    # def get_text_pred(self):
    #     return None
    def get_text_pred(self):
        prediction_path = os.path.join(self.root_path, 'stat_predictions_ecnu_rag', self.data_path.split('.')[0])
        for root, _, files in os.walk(prediction_path):
            for file in files:
                if file.endswith(".json"):
                    text_pred_json = os.path.join(root, file)
        with open(text_pred_json, 'r', encoding='utf-8') as f:
            text_pred_file = json.load(f)

        text_pred_file = {int(i): text_pred_file[i]["Prediction"] for i in text_pred_file}

        text_pred = []
        for i in range(len(text_pred_file)):
            try:
                # print(text_pred_file[i])
                if self.args.pred_len <= len(text_pred_file[i]):
                    text_pred.append(text_pred_file[i][:self.args.pred_len])
                else:
                    # 线性插值上采样
                    original = np.array(text_pred_file[i])
                    indices = np.linspace(0, len(original) - 1, self.args.pred_len)
                    upsampled = np.interp(indices, np.arange(len(original)), original)
                    text_pred.append(upsampled.tolist())
                # print(text_pred[i])

            except KeyError:
                text_pred.append([0] * self.args.pred_len)

        # text_pred = data_set.scaler.transform(np.array(text_pred[border1:border2]).reshape(-1, 1)).reshape(-1, self.args.pred_len)
        text_pred_list = []
        # for i in range(self.border1, self.border2 - self.args.seq_len - self.args.pred_len + 1):
            # print(border1,border2,i)
        for i in range(self.border1, self.border2 - self.args.seq_len - self.args.pred_len + 1):

            text_pred_list.append(self.scaler.transform(np.array(text_pred[i]).reshape(-1, 1)).reshape(1, -1))
        text_pred = torch.tensor(np.array(text_pred_list))

        return text_pred

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ChatTime(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='traffic_data.csv',
                 target='traffic_flow', scale=True, timeenc=0,
                 freq='h', seasonal_patterns=None, llm_model=None, tokenizer=None):
        """
        Dataset class for traffic flow data with historical and prediction sequences

        Args:
            args: Arguments object containing configuration
            root_path: Root directory path for data
            flag: 'train', 'val', or 'test'
            size: [seq_len, label_len, pred_len]
            features: Feature selection mode ('S' for single, 'M' for multivariate)
            data_path: Path to CSV file
            target: Target column name
            scale: Whether to apply scaling
            timeenc: Time encoding type (0 or 1)
            freq: Frequency of time series
            llm_model: Language model for text embeddings
            tokenizer: Tokenizer for text processing
        """
        self.args = args

        # Set sequence lengths
        if size is None:
            self.seq_len = 120  # Default based on your data (120 hours of history)
            self.label_len = 24  # Overlap between input and output
            self.pred_len = 24  # 24 hours prediction
        else:
            self.seq_len = size[0]
            self.label_len = self.seq_len // 2
            self.pred_len = size[2]

        # Initialize attributes
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.llm_model = llm_model
        self.tokenizer = tokenizer

        self.__read_data__()
        self.tot_len = len(self.data_x)

    def __read_data__(self):
        """Read and preprocess the traffic flow data"""
        self.scaler = StandardScaler()

        # Read CSV file
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Parse the Hist and Pred columns from string to lists
        df_raw['Hist'] = df_raw['Hist'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
        df_raw['Pred'] = df_raw['Pred'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

        # Convert date column to datetime
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])

        # Sort by date to ensure chronological order
        df_raw = df_raw.sort_values('Date').reset_index(drop=True)

        # Split data into train/val/test based on number of samples (rows)
        num_samples = len(df_raw)
        num_train = int(num_samples * 0.7)
        num_test = int(num_samples * 0.2)
        num_val = num_samples - num_train - num_test

        # Define borders for train/val/test split
        if self.set_type == 0:  # train
            start_idx = 0
            end_idx = num_train
        elif self.set_type == 1:  # val
            start_idx = num_train
            end_idx = num_train + num_val
        else:  # test
            start_idx = num_train + num_val
            end_idx = num_samples

        # Get the subset for current split
        df_subset = df_raw.iloc[start_idx:end_idx]

        # Extract sequences for each sample
        hist_sequences = []
        pred_sequences = []
        texts = []
        dates = []

        for idx, row in df_subset.iterrows():
            hist_data = row['Hist']
            pred_data = row['Pred']

            # Extract input sequence (x) from the end of Hist
            if len(hist_data) >= self.seq_len:
                x_seq = hist_data[-self.seq_len:]
            else:
                # Pad with zeros if hist_data is shorter than seq_len
                x_seq = [0.0] * (self.seq_len - len(hist_data)) + hist_data

            # Extract target sequence (y) from the beginning of Pred
            if len(pred_data) >= self.pred_len:
                y_seq = pred_data[:self.pred_len]
            else:
                # Pad with zeros if pred_data is shorter than pred_len
                y_seq = pred_data + [0.0] * (self.pred_len - len(pred_data))

            hist_sequences.append(x_seq)
            pred_sequences.append(y_seq)
            texts.append(row['Text'] if pd.notna(row['Text']) else 'No information available')
            dates.append(row['Date'])

        # Convert to numpy arrays
        hist_sequences = np.array(hist_sequences)  # Shape: (num_samples, seq_len)
        pred_sequences = np.array(pred_sequences)  # Shape: (num_samples, pred_len)

        # Apply scaling - Fit scaler on all training data first
        if self.scale:
            # Always fit scaler on the entire training portion of the dataset
            # to ensure consistency across train/val/test
            train_df = df_raw.iloc[0:num_train]
            all_train_hist = []
            all_train_pred = []

            for _, row in train_df.iterrows():
                hist_data = row['Hist']
                pred_data = row['Pred']

                if len(hist_data) >= self.seq_len:
                    x_seq = hist_data[-self.seq_len:]
                else:
                    x_seq = [0.0] * (self.seq_len - len(hist_data)) + hist_data

                if len(pred_data) >= self.pred_len:
                    y_seq = pred_data[:self.pred_len]
                else:
                    y_seq = pred_data + [0.0] * (self.pred_len - len(pred_data))

                all_train_hist.extend(x_seq)
                all_train_pred.extend(y_seq)

            all_train_data = np.array(all_train_hist + all_train_pred)
            self.scaler.fit(all_train_data.reshape(-1, 1))

            # Transform current sequences
            hist_sequences_scaled = []
            pred_sequences_scaled = []

            for i in range(len(hist_sequences)):
                hist_scaled = self.scaler.transform(hist_sequences[i].reshape(-1, 1)).flatten()
                pred_scaled = self.scaler.transform(pred_sequences[i].reshape(-1, 1)).flatten()
                hist_sequences_scaled.append(hist_scaled)
                pred_sequences_scaled.append(pred_scaled)

            hist_sequences = np.array(hist_sequences_scaled)
            pred_sequences = np.array(pred_sequences_scaled)

        # Create time stamps (dummy implementation - you may want to customize this)
        if self.timeenc == 0:
            # Manual time features based on dates
            time_stamps = []
            for date in dates:
                time_stamp = [date.month, date.day, date.weekday(), date.hour]
                time_stamps.append([time_stamp] * self.seq_len)  # Repeat for seq_len
            data_stamp = np.array(time_stamps)
        elif self.timeenc == 1:
            # Automatic time features (placeholder - implement based on your needs)
            data_stamp = np.zeros((len(dates), self.seq_len, 4))  # 4 time features

        # Store data
        self.data_x = hist_sequences.reshape(len(hist_sequences), self.seq_len, 1)  # Add feature dimension
        self.data_y = pred_sequences.reshape(len(pred_sequences), self.pred_len, 1)  # Add feature dimension
        self.data_stamp = data_stamp
        self.text = np.array([[text] for text in texts])  # Shape: (num_samples, 1)
        self.date = np.array([[date] for date in dates])  # Shape: (num_samples, 1)

        # Update tot_len to be number of samples
        self.tot_len = len(self.data_x)

        # Initialize prior data (can be customized based on requirements)
        self.data_prior = np.zeros_like(self.data_y)

    def __getitem__(self, index):
        """Get a single sample"""
        # Now index directly corresponds to sample index
        seq_x = self.data_x[index]  # Shape: (seq_len, 1)
        seq_y = self.data_y[index]  # Shape: (pred_len, 1)

        # For compatibility with original format, create seq_y with label_len + pred_len
        # by padding the beginning with the last label_len values from seq_x
        if hasattr(self, 'label_len') and self.label_len > 0:
            label_part = seq_x[-self.label_len:]  # Last label_len values from input
            seq_y_full = np.concatenate([label_part, seq_y], axis=0)  # Shape: (label_len + pred_len, 1)
        else:
            seq_y_full = seq_y

        seq_x_mark = self.data_stamp[index]  # Shape: (seq_len, num_time_features)

        # Create time marks for y (extend or truncate as needed)
        if hasattr(self, 'label_len') and self.label_len > 0:
            seq_y_mark = self.data_stamp[index][-self.label_len - self.pred_len:] if len(
                self.data_stamp[index]) >= self.label_len + self.pred_len else self.data_stamp[index]
        else:
            seq_y_mark = self.data_stamp[index][:self.pred_len] if len(self.data_stamp[index]) >= self.pred_len else \
                self.data_stamp[index]

        return seq_x, seq_y_full, seq_x_mark, seq_y_mark, index

    def __len__(self):
        """Get dataset length"""
        return self.tot_len

    def inverse_transform(self, data):
        """Inverse transform scaled data"""
        return self.scaler.inverse_transform(data)

    def get_prior_y(self, indices):
        """Get prior y values for given indices (return all zeros)"""
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        # 创建全零数组，形状为 [batch_size, pred_len]，类型为 float32
        prior_y = np.zeros(self.pred_len, dtype=np.float32)

        return prior_y

    def get_text(self, indices):
        """Get text descriptions for given indices"""
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        # Return text for each sample
        texts = []
        for idx in indices:
            texts.append([self.text[idx][0]] * self.seq_len)  # Repeat text for seq_len

        return np.array(texts)
    def get_all_embeddings(self):
        if self.llm_model is None or self.tokenizer is None:
            raise ValueError("LLM model and tokenizer must be provided for text embeddings")

        # Handle missing text
        for i in range(len(self.text)):
            if pd.isnull(self.text[i][0]) or self.text[i][0] == '':
                self.text[i][0] = 'No information available'

        text_flattened = self.text.reshape(-1).tolist()
        tokenized_output = self.tokenizer(
            text_flattened,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        self.input_ids = tokenized_output['input_ids'].to(self.llm_model.device)
        self.attn_mask = tokenized_output['attention_mask'].to(self.llm_model.device)
        text_embeddings = self.llm_model.get_input_embeddings()(self.input_ids)

        expanded_mask = self.attn_mask.unsqueeze(-1).expand_as(text_embeddings)

        # Apply pooling based on args.pool_type
        if hasattr(self.args, 'pool_type'):
            if self.args.pool_type == "avg":
                masked_emb = text_embeddings * expanded_mask
                valid_counts = expanded_mask.sum(dim=1, keepdim=True).clamp(min=1)
                pooled_emb = masked_emb.sum(dim=1) / valid_counts.squeeze(1)
                text_embeddings = pooled_emb
            elif self.args.pool_type == "max":
                masked_emb = text_embeddings.masked_fill(expanded_mask == 0, float('-inf'))
                pooled_emb, _ = masked_emb.max(dim=1)
                text_embeddings = pooled_emb
            elif self.args.pool_type == "min":
                masked_emb = text_embeddings.masked_fill(expanded_mask == 0, float('inf'))
                pooled_emb, _ = masked_emb.min(dim=1)
                text_embeddings = pooled_emb

        self.text_embeddings = text_embeddings
    def get_text_embeddings(self, indices):
        """Get text embeddings using the language model"""

        # print("text_embedding_shape: ", self.text_embeddings.shape)

        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        bsz = len(s_begins)
        #
        # text_embeddings = torch.cat([self.text_embeddings[s_end - self.seq_len: s_end] for s_end in s_ends], dim=0).view(bsz, self.seq_len, -1)
        text_embeddings = torch.cat([self.text_embeddings[s_begin].unsqueeze(0) for s_begin in s_begins], dim=0).unsqueeze(1).repeat(1, self.seq_len,1)
        print("text_embedding_shape: ", text_embeddings.shape) # 32 seq_len 768
        return text_embeddings

    def get_date(self, indices):
        """Get date information for given indices"""
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        # Return dates for each sample
        dates = [self.date[idx][0] for idx in indices]
        return np.array(dates), np.array(dates)  # Return same date as start and end


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern = '*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                   replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
            torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)
