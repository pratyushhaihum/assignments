import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Combine datasets for uniform processing
train['dataset'] = 'train'
test['dataset'] = 'test'
all_data = pd.concat([train, test], sort=False)

# Handle missing values
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
all_data['Alley'] = all_data['Alley'].fillna('None')
all_data['Fence'] = all_data['Fence'].fillna('None')
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')

for col in ['GarageType','GarageFinish','GarageQual','GarageCond']:
    all_data[col] = all_data[col].fillna('None')
for col in ['GarageYrBlt','GarageArea','GarageCars']:
    all_data[col] = all_data[col].fillna(0)

for col in ['BsmtFinType1','BsmtFinType2','BsmtQual','BsmtCond','BsmtExposure']:
    all_data[col] = all_data[col].fillna('None')

bsmt_cols = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']
for col in bsmt_cols:
    all_data[col] = all_data[col].fillna(0)

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))

for col in ['MasVnrType']:
    all_data[col] = all_data[col].fillna('None')
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

# Feature Engineering
all_data['TotalSF'] = all_data['GrLivArea'] + all_data['TotalBsmtSF']
all_data['TotalBath'] = (all_data['FullBath'] + 0.5 * all_data['HalfBath'] +
                         all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath'])
all_data['AgeSinceBuilt'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['AgeSinceRemod'] = all_data['YrSold'] - all_data['YearRemodAdd']

for feat in ['GarageArea','PoolArea','Fireplaces','2ndFlrSF','OpenPorchSF','WoodDeckSF']:
    all_data[f'Has{feat}'] = (all_data[feat] > 0).astype(int)

all_data.drop(['YrSold','MoSold','Id'], axis=1, inplace=True)

# Log-transform skewed numeric features
numeric_feats = all_data.select_dtypes(include=[np.number]).columns
skewed = all_data[numeric_feats].apply(lambda x: x.skew()).abs()
skewed_feats = skewed[skewed > 0.5].index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# One-hot encoding
all_data = pd.get_dummies(all_data, drop_first=True)

# Split train and test sets back
train_df = all_data[all_data['dataset_train'] == 1].drop(['dataset_train'], axis=1)
test_df = all_data[all_data['dataset_train'] == 0].drop(['dataset_train','SalePrice'], axis=1)

X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']
