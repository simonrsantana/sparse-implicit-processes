# Generate synth problems to test the IPs method
setwd("/home/simon/Desktop/implicit-variational-inference/implicit-processes/")
set.seed(123)
samples = 2000
error_factor = 1

# Heterocedastic problem
min_x = -4
max_x = 4

x <- runif(samples, min = min_x, max = max_x)

eps_2 <- rnorm(samples, sd = 2)
y2 <- 7 * sin(x) +10 + eps_2 * sin(x) * error_factor

data_2 <- data.frame(x = x, y = y2)

write.table(x = data_2, file = "heteroc.txt", col.names = F, row.names = F)


x_test <- seq(min_x, max_x, by = 0.1)
eps_test <- rnorm(length(x_test), sd = 2)
y_test <- 7 * sin(x_test) +10 + eps_test * sin(x_test) * error_factor

data_test <- data.frame(x = x_test, y = y_test)

write.table(x = data_test, file = "synth_data_biased_heteroc_test.txt", col.names = F, row.names = F)

plot(x, y2, pch = 20, col = rgb(red = 0, green = 0, blue = 1, alpha = 0.3), main = "Sinusoidal curve - simple case", ylab = "y")
points(x_test, y_test, pch = 4, col = rgb(red = 1, green = 0, blue = 0, alpha = 0.3))




# Bimodal problem

setwd("/home/simon/Desktop/implicit-variational-inference/implicit-processes/")


x_1 <- runif(samples, min = min_x, max = max_x)
eps_1 <- rnorm(samples)
eps_2 <- rnorm(samples)

disc <- runif(samples, min = 0, max = 1)
data_1 <- data.frame(x = x_1, y = x_1) # Hacemos el preset de "y" y luego lo sobreescribimos

data_1$y[disc < 0.5] <- 10*cos(x_1[disc < 0.5] - 0.5) + eps_2[disc < 0.5]
data_1$y[disc > 0.5] <- 10*sin(x_1[disc > 0.5] - 0.5) + eps_1[disc > 0.5]
plot(data_1, pch = 20, col = "blue", main = "Bi-modal problem")

write.table(x = data_1, file = "bim_data.txt", col.names = F, row.names = F)


x_test <- seq(min_x, max_x, by = 0.05)
eps_test <- rnorm(length(x_test), sd = 2)
eps_1 <- rnorm(length(x_test))
eps_2 <- rnorm(length(x_test))

disc <- runif(length(x_test), min = 0, max = 1)
data_test <- data.frame(x = x_test, y = x_test) # Hacemos el preset de "y" y luego lo sobreescribimos

data_test$y[disc < 0.5] <- 10*cos(x_test[disc < 0.5] - 0.5) + eps_2[disc < 0.5]
data_test$y[disc > 0.5] <- 10*sin(x_test[disc > 0.5] - 0.5) + eps_1[disc > 0.5]

plot(data_test$x, data_test$y)

write.table(x = data_test, file = "heteroc_test.txt", col.names = F, row.names = F)

