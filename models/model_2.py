import pandas as pd
from pathlib import Path

path = Path('rossmann')
train_df = pd.read_pickle('/mnt/rossman-fastai-sample/train_clean').drop(['index', 'Date'], axis='columns')
test_df = pd.read_pickle('/mnt/rossman-fastai-sample/test_clean')

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
import numpy as np
import torch

cat_vars = [
    'Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen',
    'Promo2Weeks', 'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear',
    'State', 'Week', 'Events', 'Promo_fw', 'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw',
    'SchoolHoliday_fw', 'SchoolHoliday_bw'
]
cont_vars = [
    'CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
    'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 
    'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
    'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday'
]
target_var= 'Sales'


class ColumnFilter:
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        return X.loc[:, cat_vars + cont_vars]
        

class GroupLabelEncoder:
    def __init__(self):
        self.labeller = LabelEncoder()
    
    def fit(self, X, y):
        self.encoders = {col: None for col in X.columns if col in cat_vars}
        for col in self.encoders:
            self.encoders[col] = LabelEncoder().fit(
                X[col].fillna(value='N/A').values
            )
        return self
    
    def transform(self, X):
        X_out = []
        categorical_part = np.hstack([
            self.encoders[col].transform(X[col].fillna(value='N/A').values)[:, np.newaxis]
            for col in cat_vars
        ])
        return pd.DataFrame(categorical_part, columns=cat_vars)


class GroupNullImputer:
    def fit(self, X, y):
        return self
        
    def transform(self, X):
        return X.loc[:, cont_vars].fillna(0)


class Preprocessor:
    def __init__(self):
        self.cf = ColumnFilter()
        self.gne = GroupNullImputer()
        
    def fit(self, X, y=None):
        self.gle = GroupLabelEncoder().fit(X, y=None)
        return self
    
    def transform(self, X):
        X_out = self.cf.transform(X)
        X_out = np.hstack((self.gle.transform(X_out).values, self.gne.transform(X_out).values))
        X_out = pd.DataFrame(X_out, columns=cat_vars + cont_vars)
        return X_out


X_train_sample = Preprocessor().fit(train_df).transform(train_df)
y_train_sample = train_df[target_var]

import torch
from torch import nn
import torch.utils.data
# ^ https://discuss.pytorch.org/t/attributeerror-module-torch-utils-has-no-attribute-data/1666


class FeedforwardTabularModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.batch_size = 512
        self.base_lr, self.max_lr = 0.001, 0.003
        self.n_epochs = 20
        self.cat_vars_embedding_vector_lengths = [
            (1115, 80), (7, 4), (3, 3), (12, 6), (31, 10), (2, 2), (25, 10), (26, 10), (4, 3),
            (3, 3), (4, 3), (23, 9), (8, 4), (12, 6), (52, 15), (22, 9), (6, 4), (6, 4), (3, 3),
            (3, 3), (8, 4), (8, 4)
        ]
        self.loss_fn = torch.nn.MSELoss()
        self.score_fn = torch.nn.MSELoss()
        
        # Layer 1: embeddings.
        self.embeddings = []
        for i, (in_size, out_size) in enumerate(self.cat_vars_embedding_vector_lengths):
            emb = nn.Embedding(in_size, out_size)
            self.embeddings.append(emb)
            setattr(self, f'emb_{i}', emb)

        # Layer 1: dropout.
        self.embedding_dropout = nn.Dropout(0.04)
        
        # Layer 1: batch normalization (of the continuous variables).
        self.cont_batch_norm = nn.BatchNorm1d(16, eps=1e-05, momentum=0.1)
        
        # Layers 2 through 9: sequential feedforward model.
        self.seq_model = nn.Sequential(*[
            nn.Linear(in_features=215, out_features=1000, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(1000, eps=1e-05, momentum=0.1),
            nn.Dropout(p=0.001),
            nn.Linear(in_features=1000, out_features=500, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(500, eps=1e-05, momentum=0.1),
            nn.Dropout(p=0.01),
            nn.Linear(in_features=500, out_features=1, bias=True)
        ])
    
    
    def forward(self, x):
        # Layer 1: embeddings.
        inp_offset = 0
        embedding_subvectors = []
        for emb in self.embeddings:
            index = torch.tensor(inp_offset, dtype=torch.int64).cuda()
            inp = torch.index_select(x, dim=1, index=index).long().cuda()
            out = emb(inp)
            out = out.view(out.shape[2], out.shape[0], 1).squeeze()
            embedding_subvectors.append(out)
            inp_offset += 1
        out_cat = torch.cat(embedding_subvectors)
        out_cat = out_cat.view(out_cat.shape[::-1])
        
        # Layer 1: dropout.
        out_cat = self.embedding_dropout(out_cat)
        
        # Layer 1: batch normalization (of the continuous variables).
        out_cont = self.cont_batch_norm(x[:, inp_offset:])
        
        out = torch.cat((out_cat, out_cont), dim=1)
        
        # Layers 2 through 9: sequential feedforward model.
        out = self.seq_model(out)
            
        return out
        
        
    def fit(self, X, y):
        self.train()
        
        # TODO: set a random seed to invoke determinism.
        # Cf. GH#11278

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        # OneCycleLR with Adam.
        #
        # Implementation notes. OneCyceLR by default cycles both the learning rate /and/ the
        # momentum value.
        # Cf. https://www.kaggle.com/residentmario/one-cycle-learning-rate-schedulers
        #
        # Optimizers that don't support momentum must use a scheduler with cycle_momentum=False,
        # which disables the momentum-tuning behavior. Adam does not support momentum; it has its
        # own similar-ish thing built in.
        # Cf. https://www.kaggle.com/residentmario/keras-optimizers
        #
        # This code requires PyTorch >= 1.2 due to a bug, see GH#19003.
        optimizer = torch.optim.Adam(self.parameters(), lr=self.max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, self.max_lr,
            cycle_momentum=False,
            epochs=self.n_epochs,
            steps_per_epoch=int(np.ceil(len(X) / self.batch_size)),
        )
        batches = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y),
            batch_size=self.batch_size, shuffle=True
        )
        
        for epoch in range(self.n_epochs):
            lvs = []
            for i, (X_batch, y_batch) in enumerate(batches):
                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()
                
                y_pred = model(X_batch).squeeze()
                optimizer.step()
                scheduler.step()
                loss = self.loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                
                lv = loss.detach().cpu().numpy()
                lvs.append(lv)
                if i % 100 == 0:
                    print(f"Epoch {epoch + 1}/{self.n_epochs}; Batch {i}; Loss {lv}")
                
                loss.backward()
            print(
                f"Epoch {epoch + 1}/{self.n_epochs}; Average Loss {np.mean(lvs)}"
            )
    
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(X, dtype=torch.float32).cuda())
        return y_pred.squeeze()
    
    
    def score(self, X, y):
        y_pred = self.predict(X)
        y = torch.tensor(y, dtype=torch.float32).cuda()
        return self.score_fn(y, y_pred)

model = FeedforwardTabularModel()
model.cuda()
model.fit(X_train_sample.values, y_train_sample.values)
torch.save(model.named_parameters(), "model.pth")
