import torch
import torch.nn as nn
import torch.optim as optim

# Определяем генератор для GAN
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3*64*64),
            nn.Tanh()  # Возвращаем значения в диапазоне [-1, 1]
        )

    def forward(self, x):
        return self.model(x).view(-1, 3, 64, 64)

# Определяем дискриминатор для GAN
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3*64*64, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.view(-1, 3*64*64))

# Инициализация моделей генератора и дискриминатора
generator = Generator()
discriminator = Discriminator()

# Оптимизаторы и критерий потерь
optim_gen = optim.Adam(generator.parameters(), lr=0.0002)
optim_disc = optim.Adam(discriminator.parameters(), lr=0.0002)
loss_fn = nn.BCELoss()

# Тренировка GAN (упрощенная)
def train_gan(generator, discriminator, epochs=100):
    for epoch in range(epochs):
        # Тренировка дискриминатора
        real_data = torch.randn(64, 3, 64, 64)  # Заглушка для реальных данных
        fake_data = generator(torch.randn(64, 100))  # Генерация фейковых данных

        # Метки для реальных и фейковых данных
        real_labels = torch.ones(64, 1)
        fake_labels = torch.zeros(64, 1)

        # Обучение дискриминатора на реальных данных
        optim_disc.zero_grad()
        output_real = discriminator(real_data)
        loss_real = loss_fn(output_real, real_labels)
        
        # Обучение дискриминатора на фейковых данных
        output_fake = discriminator(fake_data.detach())
        loss_fake = loss_fn(output_fake, fake_labels)
        
        # Суммарный градиент для дискриминатора
        loss_disc = loss_real + loss_fake
        loss_disc.backward()
        optim_disc.step()

        # Тренировка генератора
        optim_gen.zero_grad()
        output_fake = discriminator(fake_data)
        loss_gen = loss_fn(output_fake, real_labels)  # Генератор старается обмануть дискриминатор
        loss_gen.backward()
        optim_gen.step()

        # Выводим метрики для каждого эпоха
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss D: {loss_disc.item()}, Loss G: {loss_gen.item()}")

# Запуск тренировки
train_gan(generator, discriminator, epochs=200)
