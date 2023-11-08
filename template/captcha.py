import random
import urllib.request
import time
import glob
import torch
import torch.nn as nn
from torchvision.transforms import Resize
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
from tqdm import tqdm

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from keras import layers

from zae_engine import trainer, models, measure, data_pipeline

LOOKUP = {k: v for k, v in enumerate("0123456789abcdefghijklmnopqrstuvwxyz")}
LOOKDOWN = {v: k for k, v in LOOKUP.items()}

epochs = 500
batch_size = 16
learning_rate = 1e-4


class CaptchaImgSaver:
    def __init__(self, dst: str = "./"):
        self.dst = dst
        if not os.path.exists(dst):
            os.mkdir(dst)
        self.url = "https://www.ftc.go.kr/captcha.do"

    def run(self, iter: int = 100):
        for i in tqdm(range(iter), desc="Get Captcha images from url "):
            name = str(time.time()).replace(".", "") + ".png"
            urllib.request.urlretrieve(self.url, os.path.join(self.dst, name))


class CaptchaDataset(Dataset):
    def __init__(self, x, y, _type="tuple"):
        self.x = x  # List: path of images
        self.y = y  # List: labels
        self._type = _type
        self.str2emb = lambda label: [LOOKDOWN[l] for l in label]
        self.emb2str = lambda embed: [LOOKUP[e] for e in embed]
        self.resizer = Resize((60, 250))

    def __len__(self):
        return len(self.x)

    def load_image(self, idx):
        return np.array(Image.open(self.x[idx]))

    def __getitem__(self, idx):
        x = torch.tensor(self.load_image(idx), dtype=torch.float32)
        x = torch.mean(self.resizer(torch.permute(x, [2, 0, 1])), 0, keepdim=True)
        y = torch.zeros((36, 5), dtype=torch.float32)
        for i, yy in enumerate(self.str2emb(self.y[idx])):
            y[yy, i] = 1

        if self._type == "tuple":
            return x, y
        elif self._type == "dict":
            return {"x": x, "y": y}
        else:
            raise ValueError


class CaptchaTrainer(trainer.Trainer):
    def __init__(self, model, device, mode: str, optimizer: torch.optim.Optimizer = None, scheduler=None):
        super(CaptchaTrainer, self).__init__(model, device, mode, optimizer, scheduler)

    def train_step(self, batch):
        if isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        elif isinstance(batch, tuple):
            x, y = batch
        else:
            raise TypeError
        out = self.model(x).softmax(1)
        prediction = out.argmax(1)
        loss = F.cross_entropy(out, y)
        acc = measure.accuracy(self._to_cpu(y.argmax(1)), self._to_cpu(prediction))
        return {"loss": loss, "output": out, "acc": acc}

    def test_step(self, batch):
        if isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        elif isinstance(batch, tuple):
            x, y = batch
        else:
            raise TypeError
        out = self.model(x).softmax(1)
        prediction = out.argmax(1)
        loss = F.cross_entropy(out, y)
        acc = measure.accuracy(self._to_cpu(y.argmax(1)), self._to_cpu(prediction))
        return {"loss": loss, "output": out, "acc": acc}


class CaptchaModel(models.CNNBase):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        width: int,
        kernel_size: int or tuple,
        depth: int,
        order: int,
        stride: list or tuple,
    ):
        super().__init__(
            ch_in=ch_in, ch_out=ch_out, width=width, kernel_size=kernel_size, depth=depth, order=order, stride=stride
        )
        self.pool = nn.Sequential(nn.Linear(15 * 63, 64), nn.Linear(64, 5))
        self.head = nn.Conv1d(48, 36, kernel_size=1)

    def forward(self, x):
        x = self.body(x)
        flat = x.view(x.shape[0], 48, -1)
        pool = self.pool(flat)
        return self.head(pool)


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def build_model(img_h, img_w):
    # Inputs to the model
    input_img = layers.Input(shape=(img_w, img_h, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((img_w // 4), (img_h // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(len(LOOKUP) + 1, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="ocr_model_v1")
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model


def core():
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

    # Load target dataset
    tgt_filenames = glob("Z:/dev-zae/captcha_database/labeled/*.png")
    src_filenames = glob("Z:/dev-zae/captcha_images_v2/*.png")
    tgt_labels = [os.path.splitext(os.path.split(fn)[-1])[0] for fn in tgt_filenames]
    src_labels = [os.path.splitext(os.path.split(fn)[-1])[0] for fn in src_filenames]

    # Random split train/validation/test dataset as 8:1:1 ratio
    random.shuffle(tgt_filenames)
    split_80, split_10 = int((n := len(tgt_filenames)) * 0.8), int(n * 0.2)

    x_train, x_valid = tgt_filenames[:split_80] + src_filenames, tgt_filenames[split_80:]
    y_train, y_valid = tgt_labels[:split_80] + src_labels, tgt_labels[split_80:]
    x_valid, x_test = tgt_filenames[:split_10], tgt_filenames[split_10:]
    y_valid, y_test = tgt_labels[:split_10], tgt_labels[split_10:]

    train_loader = DataLoader(dataset=CaptchaDataset(x=x_train, y=y_train), batch_size=batch_size)
    valid_loader = DataLoader(dataset=CaptchaDataset(x=x_valid, y=y_valid), batch_size=batch_size)
    test_loader = DataLoader(dataset=CaptchaDataset(x=x_test, y=y_test), batch_size=batch_size)

    # Modeling
    captcha_model = CaptchaModel(1, 36, 16, kernel_size=(3, 3), stride=[2, 2], depth=3, order=2)
    captcha_opt = torch.optim.Adam(params=captcha_model.parameters(), lr=learning_rate)
    captcha_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=captcha_opt)

    captcha_trainer = CaptchaTrainer(
        model=captcha_model, device=device, mode="train", optimizer=captcha_opt, scheduler=captcha_scheduler
    )
    t = time.time()
    captcha_trainer.run(n_epoch=epochs, loader=train_loader, valid_loader=valid_loader)
    print(time.time() - t)
    captcha_trainer.inference(test_loader)
    print(f'Accuracy @ test dataset: {captcha_trainer.log_test["acc"]}')


def core_tf():
    # Preview Dataset
    images = glob("Z:/dev-zae/captcha_database/labeled/*.png")
    images += glob("Z:/dev-zae/captcha_images_v2/*.png")
    labels = [os.path.splitext(os.path.split(fn)[-1])[0] for fn in images]

    characters = LOOKUP.values()

    # Desired image dimensions
    img_width = 200
    img_height = 50

    # Factor by which the image is going to be downsampled
    # by the convolutional blocks. We will be using two
    # convolution blocks and each block will have
    # a pooling layer which downsample the features by a factor of 2.
    # Hence total downsampling factor would be 4.
    downsample_factor = 4

    # Maximum length of any captcha in the dataset
    max_length = 5

    # Mapping characters to integers
    char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
    # Mapping integers back to original characters
    num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

    def split_data(images, labels, train_size=0.8, shuffle=True):
        # 1. Get the total size of the dataset
        size = len(images)
        # 2. Make an indices array and shuffle it, if required
        indices = np.arange(size)
        if shuffle:
            np.random.shuffle(indices)
        # 3. Get the size of training samples
        train_samples = int(size * train_size)
        # 4. Split data into training and validation sets
        x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
        x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
        return x_train, x_valid, y_train, y_valid

    # Splitting data into training and validation sets
    x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))
    x_valid, x_test, y_valid, y_test = split_data(x_valid, y_valid, train_size=0.5)

    def encode_single_sample(img_path, label):
        # 1. Read image
        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=4)
        img = tf.reduce_mean(img, axis=-1, keepdims=True)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, [img_height, img_width])
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        # 6. Map the characters in label to numbers
        label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        # 7. Return a dict as our model is expecting two inputs
        return {"image": img, "label": label}

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    validation_dataset = (
        validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = (
        test_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    _, ax = plt.subplots(4, 4, figsize=(10, 5))
    for batch in train_dataset.take(1):
        images = batch["image"]
        labels = batch["label"]
        for i in range(16):
            img = (images[i] * 255).numpy().astype("uint8")
            label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
            ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
            ax[i // 4, i % 4].set_title(label)
            ax[i // 4, i % 4].axis("off")
    plt.show()

    # Get the model
    model = build_model(img_h=img_height, img_w=img_width)
    model.summary()

    early_stopping_patience = 10
    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping],
    )

    # Get the prediction model by extracting layers till the output layer
    prediction_model = keras.models.Model(model.get_layer(name="image").input, model.get_layer(name="dense2").output)
    prediction_model.summary()

    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    #  Let's check results on some validation samples
    for batch in test_dataset.take(1):
        batch_images = batch["image"]
        batch_labels = batch["label"]

        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)

        orig_texts = []
        for label in batch_labels:
            label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
            orig_texts.append(label)

        _, ax = plt.subplots(4, 4, figsize=(15, 5))
        for i in range(len(pred_texts)):
            img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
            img = img.T
            title = f"Prediction: {pred_texts[i]}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")
    plt.show()


def pre_trained(modelpath: str):
    model = tf.saved_model.load(modelpath, options=tf.saved_model.LoadOptions(experimental_io_device="/job:localhost"))
    return model


if __name__ == "__main__":
    # cap = CaptchaImgSaver("./outputs")
    # cap.run(iter=10000)

    core()
    core_tf()
