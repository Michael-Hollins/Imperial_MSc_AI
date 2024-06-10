# https://www.kaggle.com/code/shreyasajal/pytorch-forecasting-for-time-series-forecasting/notebook
import pandas as pd
import pytorch_forecasting
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import os

df_train = pd.read_csv('data/dummy_data/sales_train.csv', parse_dates=['date'])
df_shops = pd.read_csv('data/dummy_data/shops.csv')
df_items = pd.read_csv('data/dummy_data/items.csv')
df_item_categories = pd.read_csv('data/dummy_data/item_categories.csv')

# Format dates correctly
df_train['date'] = pd.to_datetime(df_train['date'], format='%d.%m.%Y')

# Item count/day --> Item count/month
df_train = df_train.groupby(["item_id","shop_id","date_block_num"]).sum(numeric_only=True).reset_index()
df_train = df_train.rename(index=str, columns = {"item_cnt_day":"item_cnt_month"})
df_train = df_train[["item_id","shop_id","date_block_num","item_cnt_month"]]

# Create the timeseries dataset
max_prediction_length = 1
max_encoder_length = 27
training_cutoff = df_train['date_block_num'].max() - max_prediction_length

training = TimeSeriesDataSet(
    df_train[lambda x: x['date_block_num'] <= training_cutoff],
    time_idx='date_block_num',
    target="item_cnt_month",
    group_ids=["shop_id", "item_id"],
    min_encoder_length=0,  
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    static_reals=["shop_id", "item_id"],
    time_varying_known_categoricals=[],  
    time_varying_known_reals=['date_block_num'],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=['date_block_num'],
    categorical_encoders={'shop_id': pytorch_forecasting.data.encoders.NaNLabelEncoder(add_nan=True),
                          'item_id':pytorch_forecasting.data.encoders.NaNLabelEncoder(add_nan=True)},
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

# Train and validation data loaders
validation = TimeSeriesDataSet.from_dataset(training, df_train, predict=True, stop_randomization=True)
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10)

## Define the TFT
# configure network and trainer
pl.seed_everything(42)
logger = TensorBoardLogger("tb_logs", name="tft_toy_model")
trainer = pl.Trainer(
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    gradient_clip_val=0.1,
    logger = logger,
    gpus=1,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    # not meaningful for finding the learning rate but otherwise very important
    learning_rate=0.03,
    hidden_size=16,  # most important hyperparameter apart from learning rate
    # number of attention heads. Set to up to 4 for large datasets
    attention_head_size=1,
    dropout=0.1,  # between 0.1 and 0.3 are good values
    hidden_continuous_size=8,  # set to <= hidden_size
    output_size=1,  # 7 quantiles by default
    loss=pytorch_forecasting.metrics.RMSE(),
    # reduce learning rate if no improvement in validation loss after x epochs
    reduce_on_plateau_patience=4,
)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# Use lightning to suggest a suitable learning rate
tuner = Tuner(trainer)

# Run learning rate finder
lr_finder = tuner.lr_find(model = tft,
                          train_dataloaders=train_dataloader, 
                          val_dataloaders=val_dataloader,
                          max_lr=0.1,
                          min_lr=1e-7)

fig = lr_finder.plot(suggest=True)
fig.show()

# Callbacks, trainer and final model
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-7, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  
logger = TensorBoardLogger("lightning_logs") 

trainer = pl.Trainer(
    max_epochs=30,
    gpus=1,
    weights_summary="top",
    gradient_clip_val=0.1,
    limit_train_batches=30,  
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=4e-6,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=1, 
    loss=pytorch_forecasting.metrics.RMSE(),
    log_interval=10,  
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# Fit the model
trainer.fit(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)