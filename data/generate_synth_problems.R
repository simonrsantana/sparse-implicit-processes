# Generate synth problems to test the IPs method
setwd("/home/simon/Desktop/BNN/problemas_sinteticos")
set.seed(123)
samples = 2000

# Heterocedastic problem

x <- runif(samples, min = -4, max = 4)

eps_2 <- rnorm(samples)
y2 <- 7 * sin(x) + 3*abs(cos(x/2))*eps_2

plot(x, y2, pch = 20, col = "blue", main = "Heteroscedastic case", ylab = "y")

data_2 <- data.frame(x = x, y = y2)

write.table(x = data_2, file = "data_2.txt", col.names = F, row.names = F)

# Bimodal problem

x_1 <- runif(samples, min = -2.0, max = 2.0)
eps_1 <- rnorm(samples)
eps_2 <- rnorm(samples)

disc <- runif(samples, min = 0, max = 1)
data_1 <- data.frame(x = x_1, y = x_1) # Hacemos el preset de "y" y luego lo sobreescribimos

data_1$y[disc < 0.5] <- 10*cos(x_1[disc < 0.5]) + eps_2[disc < 0.5]
data_1$y[disc > 0.5] <- 10*sin(x_1[disc > 0.5]) + eps_1[disc > 0.5]
plot(data_1, pch = 20, col = "blue", main = "Bi-modal problem")

write.table(x = data_1, file = "data_1.txt", col.names = F, row.names = F)
