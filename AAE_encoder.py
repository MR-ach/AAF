import tensorflow as tf
import numpy as np
import os
import pickle
from keras import layers, Model, regularizers
import random
import matplotlib.pyplot as plt

# 环境设置
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
seed = 2
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 全局参数
learning_rate = 1e-4
batch_size = 32
epochs = 50
# root_model = './model/dcae_aae/'
faults = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']


# 数据预处理函数
def preprocess_data(data):
    """归一化到[-1, 1]范围"""
    return (data.astype(np.float32) - 127.5) / 127.5


class Discriminator(tf.keras.Model):
    def __init__(self, latent_dim=128):
        super(Discriminator, self).__init__()
        self.model = tf.keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(latent_dim,)),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        return self.model(inputs)


class DCAE_AAE(Model):
    def __init__(self):
        super(DCAE_AAE, self).__init__()
        # 编码器
        self.encoder = tf.keras.Sequential([
            layers.Conv1D(64, 5, padding='same', input_shape=(3072, 1)),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.MaxPooling1D(2, padding='same'),  # 3072 -> 1536
            layers.Conv1D(32, 5, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.MaxPooling1D(2, padding='same'),  # 1536 -> 768
            layers.Conv1D(16, 5, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))
        ])
        # 解码器
        self.decoder = tf.keras.Sequential([
            layers.Dense(768 * 16),  # 恢复编码器池化前的维度
            layers.Reshape((768, 16)),
            layers.Conv1DTranspose(32, 5, strides=2, padding='same'),  # 768 -> 1536
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv1DTranspose(16, 5, strides=2, padding='same'),  # 1536 -> 3072
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv1D(1, 5, padding='same', activation='tanh')
        ])
        # 判别器
        self.discriminator = Discriminator()

    def call(self, inputs):
        latent = self.encoder(inputs)
        return self.decoder(latent)

    def compile(self, ae_optimizer, d_optimizer):
        super(DCAE_AAE, self).compile()
        self.ae_optimizer = ae_optimizer
        self.d_optimizer = d_optimizer
        self.reconstruction_loss_fn = tf.keras.losses.MeanSquaredError()
        self.adversarial_loss_fn = tf.keras.losses.BinaryCrossentropy()

    def train_step(self, data):
        real_data = data
        batch_size = tf.shape(real_data)[0]

        # 训练判别器
        latent_real = tf.random.normal(shape=(batch_size, 128))
        latent_fake = self.encoder(real_data)

        with tf.GradientTape() as tape:
            pred_real = self.discriminator(latent_real)
            pred_fake = self.discriminator(latent_fake)
            d_loss = 0.5 * (
                    self.adversarial_loss_fn(tf.ones_like(pred_real), pred_real) +
                    self.adversarial_loss_fn(tf.zeros_like(pred_fake), pred_fake)
            )
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # 训练自编码器
        with tf.GradientTape() as tape:
            reconstructions = self(real_data)
            r_loss = self.reconstruction_loss_fn(real_data, reconstructions)
            latent_fake = self.encoder(real_data)
            validity = self.discriminator(latent_fake)
            a_loss = self.adversarial_loss_fn(tf.ones_like(validity), validity)
            total_loss = r_loss + 0.1 * a_loss
        grads = tape.gradient(total_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.ae_optimizer.apply_gradients(
            zip(grads, self.encoder.trainable_variables + self.decoder.trainable_variables)
        )
        return {"d_loss": d_loss, "r_loss": r_loss, "a_loss": a_loss}


# # 创建模型检查点目录
# if not os.path.exists(root_model):
#     os.makedirs(os.path.join(root_model, 'ckpt'))

# 训练流程
for fault in faults:
    print(f"Processing fault {fault}...")
    # 数据路径
    data_path = f'./data/train/{fault}_label.pkl'
    test_path = f'./data/test/{fault}_label.pkl'
    save_train = f'./results/AAE/{fault}_train.pkl'
    save_test = f'./results/AAE/{fault}_test.pkl'

    # 加载数据
    with open(data_path, 'rb') as f:
        train_data, train_labels = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data, test_labels = pickle.load(f)

    # 数据预处理
    train_data = preprocess_data(train_data.reshape(-1, 3072, 1))
    test_data = preprocess_data(test_data.reshape(-1, 3072, 1))

    # 创建模型
    model = DCAE_AAE()
    model.compile(
        ae_optimizer=tf.keras.optimizers.Adam(learning_rate),
        d_optimizer=tf.keras.optimizers.Adam(learning_rate * 0.5)
    )

    # # 检查点回调
    # checkpoint_path = os.path.join(root_model, 'ckpt', f'{fault}_checkpoint')
    # checkpoint = tf.train.Checkpoint(
    #     encoder=model.encoder,
    #     decoder=model.decoder,
    #     discriminator=model.discriminator,
    #     ae_optimizer=model.ae_optimizer,
    #     d_optimizer=model.d_optimizer
    # )

    # # 恢复检查点
    # latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))
    # if latest:
    #     checkpoint.restore(latest)
    #     print(f"Loaded weights from {latest}")

    # 创建数据集
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(1000).batch(batch_size)

    # # 训练循环
    # best_loss = float('inf')
    # for epoch in range(epochs):
    #     print(f"Epoch {epoch + 1}/{epochs}")
    #     epoch_d_loss = []
    #     epoch_r_loss = []
    #     epoch_a_loss = []
    #
    #     for batch in train_dataset:
    #         losses = model.train_step(batch)
    #         epoch_d_loss.append(losses['d_loss'].numpy())
    #         epoch_r_loss.append(losses['r_loss'].numpy())
    #         epoch_a_loss.append(losses['a_loss'].numpy())
    #
    #     # 计算平均损失
    #     avg_d_loss = np.mean(epoch_d_loss)
    #     avg_r_loss = np.mean(epoch_r_loss)
    #     avg_a_loss = np.mean(epoch_a_loss)
    #     print(f"  d_loss: {avg_d_loss:.4f}  r_loss: {avg_r_loss:.4f}  a_loss: {avg_a_loss:.4f}")
    #     # 绘制损失曲线
    #     plt.figure(figsize=(10, 6))
    #
    #     # 绘制 d_loss 曲线
    #     plt.plot(epoch_d_loss, label='d_loss', color='blue')
    #
    #     # 绘制 r_loss 曲线
    #     plt.plot(epoch_r_loss, label='r_loss', color='green')
    #
    #     # 绘制 a_loss 曲线
    #     plt.plot(epoch_a_loss, label='a_loss', color='red')
    #
    #     # 设置图表标题和标签
    #     plt.title('Loss Curves')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #
    #     # 添加图例
    #     plt.legend()
    #
    #     # 显示图表
    #     plt.show()


        # # 保存最佳模型
        # if avg_r_loss < best_loss:
        #     best_loss = avg_r_loss
        #     checkpoint.save(file_prefix=checkpoint_path)

    # 特征提取
    train_features = model.encoder.predict(train_data, batch_size=batch_size)
    test_features = model.encoder.predict(test_data, batch_size=batch_size)

    print(train_features.shape)
    print(test_features.shape)

    # 保存特征
    os.makedirs(os.path.dirname(save_train), exist_ok=True)
    with open(save_train, 'wb') as f:
        pickle.dump(train_features, f, pickle.HIGHEST_PROTOCOL)
    with open(save_test, 'wb') as f:
        pickle.dump(test_features, f, pickle.HIGHEST_PROTOCOL)

print("Training completed!")