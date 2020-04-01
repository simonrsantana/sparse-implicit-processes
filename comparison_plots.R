# See the results of the truncated output of the run in "test_code.py"

setwd("/home/simon/Desktop/implicit-variational-inference/implicit-processes/synthetic_cases/")

data <- read.csv("temp_comparison.csv")
nsamples <- ncol(data) - 2

names(data) <- c("x", "y", paste0("sample_", as.character(1:nsamples)))

data$mean_estimate <- rowMeans(data[, 3:(nsamples + 2)])

data[order(data$x),]

plot(data$x, data$y, pch = 20, main = "Comparison between input and estimates") #, ylim = c(-4, 4))
#points(data$x, data$mean_estimate, pch = 4, col = "blue")
for (i in 1:nsamples){
 points(data$x, data[,i+2], pch = 20, col = rgb(red = 1, green = 0, blue = 0, alpha = 0.02))
}
  
lines(data$x, data$sample_1, lty = 1, col = "blue")


########################### INDUCING POINTS REPRESENTATIONS


setwd("/home/simon/Desktop/implicit-variational-inference/implicit-processes/synthetic_cases/")

data <- read.csv("approx_comparison.csv")
nsamples <- ncol(data) - 3

names(data) <- c("x", "z", "y", paste0("sample_", as.character(1:nsamples)))

data$mean_estimate <- rowMeans(data[, 4:(nsamples + 3)])

plot(data$x, data$y, pch = 20, main = "Comparison between input and inducing points estimates") #, ylim = c(-4, 4))
points(data$z, data$mean_estimate, pch = 4, col = "green")
#points(data$x, data$mean_estimate, pch = 4, col = "blue")
for (i in 1:nsamples){
  points(data$z, data[,i+3], pch = 20, col = rgb(red = 1, green = 0, blue = 0, alpha = 0.02))
}
data[order(data$x),]

# lines(data$z, data$sample_1, lty = 1, col = "blue")

# points(data$z, data$mean_estimate, pch = 3, col = "blue")


train_data <- read.table("../synth_data_tmp.txt")

######################### FINAL RESULTS

# See the results of the truncated output of the run in "test_code.py"

setwd("/home/simon/Desktop/implicit-variational-inference/implicit-processes/synthetic_cases/")

data <- read.csv("final_results_carlitos.csv")
nsamples <- ncol(data) - 2

names(data) <- c("x", "y", paste0("sample_", as.character(1:nsamples)))

data$mean_estimate <- rowMeans(data[, 3:(nsamples + 2)])

data[order(data$x),]

plot((train_data$V1 / 2.3), train_data$V2, pch = 20, col = rgb(red = 1, green = 1, blue = 0, alpha = 0.002), main = "Full run (carlitos)")
for (i in 1:nsamples){
  points(data$x, data[,i+2], pch = 20, col = rgb(red = 0, green = 1, blue = 0, alpha = 0.05))
}
points(data$x, data$y, pch = 20) #, ylim = c(-4, 4))
points(data$x, data$mean_estimate, pch = 17, col = "red")
lines(data$x, data$sample_1, lty = 1, col = rgb(red = 0, green = 0, blue = 1, alpha = 0.7))



