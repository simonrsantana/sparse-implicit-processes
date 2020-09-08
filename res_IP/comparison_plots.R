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



######################### FINAL RESULTS

# See the results of the truncated output of the run in "test_code.py"

setwd("/home/simon/Desktop/implicit-variational-inference/implicit-processes/synthetic_cases/")

train_data <- read.table("../synth_data_tmp.txt")

data <- read.csv("final_results_1.0.csv")
nsamples <- ncol(data) - 2

names(data) <- c("x", "y", paste0("sample_", as.character(1:nsamples)))

data$mean_estimate <- rowMeans(data[, 3:(nsamples + 2)])

data <- data[order(data$x),]

plot((train_data$V1 / 2.3), train_data$V2, pch = 20, col = rgb(red = 1, green = 1, blue = 0, alpha = 0.002), main = "Full run (alpha = 1.0)")
points(data$x, data$y, pch = 20, col = rgb(red = 0, green = 0, blue = 0, alpha = 0.5)) #, ylim = c(-4, 4))
for (i in 1:nsamples){
  points(data$x, data[,i+2], pch = 20, col = rgb(red = 0, green = 1, blue = 0, alpha = 0.05))
}
points(data$x, data$mean_estimate, pch = 17, col = "red")
lines(data$x, data$sample_1, lty = 1, col = rgb(red = 0, green = 0, blue = 1, alpha = 0.7))
points(data$x, data$y, pch = 20, col = rgb(red = 0, green = 0, blue = 0, alpha = 0.5)) #, ylim = c(-4, 4))



#################################  PRIOR FUNCTIONS SAMPLES


setwd("/home/simon/Desktop/implicit-variational-inference/implicit-processes/")

library(ggplot2)
library(reshape2)

alphas = c("0.0001", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, "1.0") 
n_alphas  = length(alphas)
data_vec = c("t_skw", "t_central", "bim", "het", "composite")

for (j in 1:length(data_vec)){
  
  data = data_vec[j]
  
  for (i in c(1, 11)){
    
    data_dir <- paste0("/home/simon/Desktop/implicit-variational-inference/synthetic_data/res_", data, "/")
    setwd(data_dir)

    data_fx_init <- read.csv(paste0(alphas[i], "/", alphas[i], "initial_prior_samples_fx.csv"))
    data_fx_final <- read.csv(paste0(alphas[i], "/", alphas[i], "final_prior_samples_fx.csv"))
    # data_z <- read.csv("synthetic_cases/prior_samples_z.csv")
    nsamples <- 20
    
    data_fx_init <- data_fx_init[, 1:(nsamples + 2) ]
    data_fx_final <- data_fx_final[, 1:(nsamples + 2) ]
    
    names_x <- paste0("sample_", c(1:nsamples), "_fx")
    # names_z <- paste0("sample_", c(1:nsamples), "_fz")
    
    names(data_fx_init) <- c("x","y", names_x)
    names(data_fx_final) <- c("x","y", names_x)
    # names(data_z) <- c("z","y", names_z)
      
    #data_x$mean_fx <- rowMeans(data[, 4:(nsamples + 3)])
    #data_z$mean_fz <- rowMeans(data[, 4:(nsamples + 3)])
    
    mdata_fx_init <- melt(data_fx_init, id.vars = c("x", "y"))
    mdata_fx_final <- melt(data_fx_final, id.vars = c("x", "y"))
    # mdata_z <- melt(data_z, id.vars = c("z", "y"))
    
    
    plots_dir <- paste0("/home/simon/Desktop/implicit-variational-inference/synthetic_data/plots/", data, "/")
    setwd(plots_dir)
    
    # Save the initial prior samples
    ggplot(mdata_fx_init, aes(x, value, col=variable, alpha = 0.9)) + 
      geom_line() + theme_bw() + theme(legend.position = "none") + 
      ggtitle("Initial prior functions sampled (BNN)") +
      xlab("x") + ylab("y") + theme(plot.title = element_text(hjust = 0.5)) 
      
    ggsave(filename = paste0(alphas[i], "_initial_fx_", data, ".png"), width=10, height=5, pointsize=12, units = "in")

    # Save the final prior samples
    ggplot(mdata_fx_final, aes(x, value, col=variable, alpha = 0.9)) + 
      geom_line() + theme_bw() + theme(legend.position = "none") + 
      ggtitle("Final prior functions sampled (BNN)") +
      xlab("x") + ylab("y") + theme(plot.title = element_text(hjust = 0.5)) 
    
    ggsave(filename = paste0(alphas[i], "_final_fx_", data, ".png"), width=10, height=5, pointsize=12, units = "in")
    
  }

}


#ggplot(mdata_z, aes(z, value, col=variable)) + 
#  geom_line() + theme(legend.position = "none")




####################### Inducing points evolution throughout epochs



setwd("/home/simon/Desktop/implicit-variational-inference/synthetic_data/res_composite/1.0/")

library(ggplot2)
library(reshape2)
require(gridExtra)
library(ggpubr)
# theme_set(theme_pubr())

# Prepare the first plot with the evolution of the induced points
data <- read.csv("IPs_split_0_composite_data.txt")
ips <- ncol(data) - 1

names(data) <- c("epoch", c(1:ips))

mdata <- melt(data, id.vars = "epoch") #, variable.name = "IP")

ips_plot <- ggplot(mdata, aes(value, epoch, col = variable, alpha = 0.95)) + 
  geom_line() + theme_bw() + theme(legend.position = "none") + 
  xlab("x") + ylab("epoch") + theme(plot.title = element_text(hjust = 0.5)) +
  xlim(-2, 2) #+ scale_y_log2()


# Prepare the second plot with the results of the algorithm
# train_data <- read.table("synth_data_tmp.txt")

data_res <- read.csv("test_results_1.0_split_0.csv")
data_res <- data_res[,1:203]
nsamples <- ncol(data_res) - 2
names(data_res) <- c("x", "y", c(1:nsamples))
data_res$mean_estimate <- rowMeans(data_res[, 3:(nsamples + 2)])
data_res <- data_res[order(data_res$x),]
mres <- melt(data_res, id.vars = c("x", "y", "mean_estimate"), variable.name = "sample")

res_plot <- ggplot(mres, aes(x,value)) + geom_point( color = "lightblue", alpha = 0.05) + theme_bw() + 
  theme(legend.position = "none") + 
  ggtitle("Test results") +
  xlab("") + ylab("y") + theme(plot.title = element_text(hjust = 0.5)) +
  geom_point(aes(x, y), color = "black") + 
  geom_line(aes(x, mean_estimate), color = "darkblue") +  xlim(-2, 2)

figure <- ggarrange(res_plot, ips_plot,
                    labels = c("A", "B"),
                    ncol = 1, nrow = 2, 
                    heights = c(2,1))

figure

setwd("/home/simon/Desktop/implicit-variational-inference/implicit-processes/figures")
ggsave("test_composite_IPs.png", width = 20, height = 13, units = "cm")




############################## EXTRA ANALISIS SOBRE VARIANZAS Y EL ERROR


library(reshape)
library(ggplot2)

alphas = c("0.0001", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, "1.0")
n_alphas  = length(alphas)

mean_var <- c()
for (i in 1:n_alphas){
  
  setwd("/home/simon/Desktop/implicit-variational-inference/synthetic_data/res_bimodal/")
  results_file <- paste0(as.character(alphas[i]), "/test_results_", as.character(alphas[i]), "_split_0.csv")
  
  results <- read.csv(results_file)
  names(results) <- c("x", "y", (1:(ncol(results) - 2 )))
  # results$mean_estimate <- rowMeans(results[, 3:ncol(results)])
  results$variance <- apply(results[, 3:ncol(results)], 1, var)
  
  mean_var[i] <- mean(results$variance)
  
  # melted <- melt( results, id.vars = c("x", "y", "mean_estimate"), variable_name = "sample")

}
