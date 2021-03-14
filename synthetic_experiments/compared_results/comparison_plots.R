maindir <- getwd()
# maindir <- c("/home/simon/Desktop/implicit-variational-inference/synthetic_data/compare_results/")

setwd(maindir)

library(ggplot2)
library(reshape2)
library(hexbin)
library(ggpubr)


alphas = c("1.0") # , "0.0001") # , "0.5", "1.0") 
n_alphas  = length(alphas)
# data_vec = c("t_skw", "t_central", "bim", "het", "composite")
# folders_to_compare <- c("vip", "fbnn_bnn") 
folders_to_compare <- c("vip", "fbnn_gp") 
ip_folders <- c("bnn", "ns")

# Get density function

get_density <- function(x, y, ...) {
  dens <- MASS::kde2d(x, y, ...)
  ix <- findInterval(x, dens$x)
  iy <- findInterval(y, dens$y)
  ii <- cbind(ix, iy)
  return(dens$z[ii])
}


# Extract the mean and stdv for the training data in case we need to renormalize
setwd(paste0(maindir, ip_folders[1], "/", alphas[1]))

moment_files <- list.files(pattern = "train_0.txt", all.files = FALSE,
                          full.names = FALSE, recursive = FALSE,
                          ignore.case = FALSE, include.dirs = FALSE, no.. = FALSE)

xmean <- read.csv(moment_files[ 1 ], header = F)[[1]]
ymean <- read.csv(moment_files[ 2 ], header = F)[[1]]
stdx <- read.csv(moment_files[ 3 ], header = F)[[1]]
stdy <- read.csv(moment_files[ 4 ], header = F)[[1]]


############################## PRIOR PLOTS ####################################

# Read the data from the ip folders 
for (j in 1:length( ip_folders )){
  
  data = ip_folders[j]
  
  for (i in c(1:length(alphas))){
    
    data_dir <- paste0(maindir, data, "/")
    setwd(data_dir)
    
    # data_fx_init <- read.csv(paste0(alphas[i], "/", alphas[i],
    #                                 "_initial_prior_samples_yx.csv"))
    
    data_fx_final <- read.csv(paste0(alphas[i], "/", alphas[i],
                                     "_final_prior_samples_yx.csv"))
    
    # data_z <- read.csv("synthetic_cases/prior_samples_z.csv")
    nsamples <- ncol(data_fx_final) - 2
    # nsamples_init <- 100
    
    # data_fx_init <- data_fx_init[, 1:(nsamples_init + 2) ]
    # data_fx_final <- data_fx_final[, 1:(nsamples + 2) ]
    
    names_x <- paste0("sample_", c(1:nsamples), "_yx")
    # names_z <- paste0("sample_", c(1:nsamples), "_fz")
    
    # names(data_fx_init) <- c("x","y", names_x)
    names(data_fx_final) <- c("x","y", names_x)
    # names(data_z) <- c("z","y", names_z)
    
    #data_x$mean_fx <- rowMeans(data[, 4:(nsamples + 3)])
    #data_z$mean_fz <- rowMeans(data[, 4:(nsamples + 3)])
    
    # mdata_fx_init <- melt(data_fx_init, id.vars = c("x", "y"))
    mdata_fx_final <- melt(data_fx_final, id.vars = c("x", "y"))
    # mdata_z <- melt(data_z, id.vars = c("z", "y"))
    
    
    # plots_dir <- paste0(maindir, "/plots/", data, "/")
    # setwd(plots_dir)
    
    # Save the initial prior samples
    # ip_plot_cont[] ggplot(mdata_fx_init, aes(x, value, col=variable, alpha = 0.9)) + 
    #   geom_line() + theme_bw() + theme(legend.position = "none") + 
    #   ggtitle("Initial prior functions sampled (BNN)") + xlim(-2,2) + # ylim(-30,30) +
    #   xlab("x") + ylab("y") + theme(plot.title = element_text(hjust = 0.5)) 
    
    # ggsave(filename = paste0(alphas[i], "_initial_fx_", data, ".png"), width=10, height=5, pointsize=12, units = "in")
    
    # Save the final prior samples
    # ggplot(mdata_fx_final, aes(x, value, col=variable, alpha = 0.9)) + 
    #   geom_line() + theme_bw() + theme(legend.position = "none") + 
    #   ggtitle("Final prior functions sampled (BNN)") + xlim(-2,2) + # ylim(-30,30) +
    #   xlab("x") + ylab("y") + theme(plot.title = element_text(hjust = 0.5)) 
    
    # ggplot(mdata_fx_init, aes(x=x, y=value)) +
    #   stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
    #   scale_fill_distiller(palette= "Spectral", direction=1) +
    #   scale_x_continuous(expand = c(0, 0)) +
    #   scale_y_continuous(expand = c(0, 0))
    
    nam <- paste("plot_ip_priors_", data, "_", alphas[ i ] , sep = "")
    
    # Create temporary plot
    # tmp_plot <- ggplot(mdata_fx_final, aes(x=x, y=value)) +
    #   stat_density_2d(aes(fill = ..level..), geom = "polygon", colour="white", bins = 7)
           
    
    mdata_fx_final$x <- mdata_fx_final$x * stdx + xmean
    
    tmp_plot <- ggplot(mdata_fx_final, aes(x=x, y=value)) +
      stat_density_2d(aes(fill = ..density..), geom = "raster" , contour = FALSE, show.legend = FALSE) +
      scale_fill_distiller(palette="Spectral", direction=-1) +
      scale_x_continuous(expand = c(0, 0), limits = c(-NA, NA)) +
      scale_y_continuous(expand = c(0, 0), limits = c(-15, 15)) + 
      scale_color_viridis_c() + 
      theme(axis.title.x = element_blank(),
            axis.title.y = element_blank()) # + 
      # theme_void()
    
    # tmp_plot

    # Plot points with color depending on density
    # mdata_fx_final$density <- get_density(mdata_fx_final$x, mdata_fx_final$value, 
    #                                       h = c(1, 1), n = 100)
    # ggplot(mdata_fx_final) + geom_point(aes(x, value, color = density)) + scale_color_viridis_c()
    
    
    # stat_density_2d(aes(fill = ..level..), geom = "polygon", colour="white",
    #                   bins = 10)
    
    # geom_hex(bins = 70) +
    # scale_fill_continuous(type = "viridis") +
    # theme_bw()
    
    # geom_density_2d()
    
    # ggsave(filename = paste0(alphas[i], "_final_fx_", data, ".png"), width=10, height=5, pointsize=12, units = "in")
    # Store the plot in a variable
    assign(nam, tmp_plot)   
    
  }
  
  setwd(maindir)
  
}


# Read the results from other methods to compare against

#######
# VIP #
#######

data_vip <- paste0(maindir, folders_to_compare[ 1 ])
setwd(data_vip)

traindata_vip <- read.csv("vip_train_data_0.txt")


prior_vip <- read.csv("vip_prior_samples_orig_0.txt")
# prior_vip <- read.csv("vip_initial_prior_samples_0.txt")
names_vip <- paste0("sample_", c(1:(ncol(prior_vip) - 2)), "_yx")

names(prior_vip) <- c("x","y", names_vip)

melt_vip_prior <- melt(prior_vip, id.vars = c("x", "y"))

melt_vip_prior$x <- melt_vip_prior$x * stdx + xmean
melt_vip_prior$y <- melt_vip_prior$y * stdy + ymean
melt_vip_prior$value <- melt_vip_prior$value * stdy + ymean


vip_prior_plot <- ggplot(melt_vip_prior, aes(x=x, y=value)) +
  stat_density_2d(aes(fill = ..density..), geom = "raster" , contour = FALSE, show.legend = FALSE) +
  scale_fill_distiller(palette="Spectral", direction=-1) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0), limits = c(-15,15)) + 
  scale_color_viridis_c() + 
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank()) # + 
  # theme_void()

# vip_prior_plot
  
# stat_density_2d(aes(fill = ..level..), geom = "polygon", colour="white", bins = 5)
# vip_prior_plot



########
# fBNN #
########

data_fbnn <- paste0(maindir, folders_to_compare[ 2 ])
setwd(data_fbnn)

prior_fbnn <- read.csv("fbnn_prior_samples_0.txt")
names_fbnn <- paste0("sample_", c(1:(ncol(prior_fbnn) - 2)), "_yx")

names(prior_fbnn) <- c("x","y", names_fbnn)

prior_fbnn$sample_464_yx <- NULL
prior_fbnn = prior_fbnn[complete.cases(prior_fbnn),]

nonumeric <- which(sapply(prior_fbnn, class) != "numeric")
for (i in 1:length(nonumeric)){
  prior_fbnn[,nonumeric[[i]]] <- as.numeric(prior_fbnn[,nonumeric[[i]]])
}

melt_fbnn_prior <- melt(prior_fbnn, id.vars = c("x", "y"))


melt_fbnn_prior$x <- melt_fbnn_prior$x * stdx + xmean
melt_fbnn_prior$y <- melt_fbnn_prior$y * stdy + ymean
melt_fbnn_prior$value <- melt_fbnn_prior$value * stdy + ymean


fbnn_prior_plot <- ggplot(melt_fbnn_prior, aes(x=x, y=value)) +
  stat_density_2d(aes(fill = ..density..), geom = "raster" , contour = FALSE, show.legend = FALSE) +
  scale_fill_distiller(palette="Spectral", direction=-1) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0), limits = c(-15, 15)) + 
  scale_color_viridis_c() + 
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank()) # + 
  # theme_void()

  
# ggplot(melt_fbnn_prior, aes(x, value, col=variable, alpha = 0.9)) + 
#   geom_line() + theme_bw() + theme(legend.position = "none") + 
#   ggtitle("Final prior functions sampled (BNN)") +
#   xlab("x") + ylab("y") + theme(plot.title = element_text(hjust = 0.5)) 

  # stat_density_2d(aes(fill = ..level..), geom = "polygon", colour="white", bins = 10)


# fbnn_prior_plot





############################## PREDICTIVE DIST SAMPLES ########################################


# Read the data from the ip folders 
for (j in 1:length( ip_folders )){
  
  data = ip_folders[j]
  
  for (i in c(1:length(alphas))){
    
    data_dir <- paste0(maindir, data, "/")
    setwd(data_dir)
    
    # data_fx_init <- read.csv(paste0(alphas[i], "/", alphas[i],
    #                                 "_initial_prior_samples_yx.csv"))
    
    
    data_fx_final <- read.csv(paste0(alphas[i], "/test_results_", alphas[i],
                                     "_split_0.csv"))
    
    # data_z <- read.csv("synthetic_cases/prior_samples_z.csv")
    nsamples <- 100
    nsamples_init <- 100
    
    # data_fx_init <- data_fx_init[, 1:(nsamples_init + 2) ]
    # data_fx_final <- data_fx_final[, 1:(nsamples + 2) ]
    
    names_x <- paste0("sample_", c(1:nsamples), "_yx")
    # names_z <- paste0("sample_", c(1:nsamples), "_fz")
    
    # names(data_fx_init) <- c("x","y", names_x)
    names(data_fx_final) <- c("x","y", names_x)
    # names(data_z) <- c("z","y", names_z)
    
    #data_x$mean_fx <- rowMeans(data[, 4:(nsamples + 3)])
    #data_z$mean_fz <- rowMeans(data[, 4:(nsamples + 3)])
    
    # mdata_fx_init <- melt(data_fx_init, id.vars = c("x", "y"))
    mdata_fx_final <- melt(data_fx_final, id.vars = c("x", "y"))
    # mdata_z <- melt(data_z, id.vars = c("z", "y"))
    
    
    # plots_dir <- paste0(maindir, "/plots/", data, "/")
    # setwd(plots_dir)
    
    # Save the initial prior samples
    # ip_plot_cont[] ggplot(mdata_fx_init, aes(x, value, col=variable, alpha = 0.9)) + 
    #   geom_line() + theme_bw() + theme(legend.position = "none") + 
    #   ggtitle("Initial prior functions sampled (BNN)") + xlim(-2,2) + # ylim(-30,30) +
    #   xlab("x") + ylab("y") + theme(plot.title = element_text(hjust = 0.5)) 
    
    # ggsave(filename = paste0(alphas[i], "_initial_fx_", data, ".png"), width=10, height=5, pointsize=12, units = "in")
    
    # Save the final prior samples
    # ggplot(mdata_fx_final, aes(x, value, col=variable, alpha = 0.9)) + 
    #   geom_line() + theme_bw() + theme(legend.position = "none") + 
    #   ggtitle("Final prior functions sampled (BNN)") + xlim(-2,2) + # ylim(-30,30) +
    #   xlab("x") + ylab("y") + theme(plot.title = element_text(hjust = 0.5)) 
    
    # ggplot(mdata_fx_init, aes(x=x, y=value)) +
    #   stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
    #   scale_fill_distiller(palette= "Spectral", direction=1) +
    #   scale_x_continuous(expand = c(0, 0)) +
    #   scale_y_continuous(expand = c(0, 0))
    
    nam <- paste("plot_ip_pred_", data, "_", alphas[ i ] , sep = "")
    
    mdata_fx_final$x <- mdata_fx_final$x * stdx + xmean
    
    # Create temporary plot
    tmp_plot <- ggplot(mdata_fx_final, aes(x=x, y=value)) +
      stat_density_2d(aes(fill = ..density..), geom = "raster" , contour = FALSE, show.legend = FALSE) +
      scale_fill_distiller(palette="Spectral", direction=-1) +
      scale_x_continuous(expand = c(0, 0), limits = c(NA, NA)) +
      scale_y_continuous(expand = c(0, 0), limits = c(-15, 15)) + 
      scale_color_viridis_c() + 
      theme(axis.title.x = element_blank(),
            axis.title.y = element_blank()) # + 
      # theme_void()
    
    # tmp_plot <- ggplot(mdata_fx_final, aes(x=x, y=value)) +
    #  stat_density_2d(aes(fill = ..density..), geom = "raster" , contour = FALSE) +
    #  scale_fill_distiller(palette="Spectral", direction=1) +
    #  scale_x_continuous(expand = c(0, 0)) +
    #  scale_y_continuous(expand = c(0, 0))
    
    
    
    # stat_density_2d(aes(fill = ..level..), geom = "polygon", colour="white",
    #                   bins = 10)
    
    # geom_hex(bins = 70) +
    # scale_fill_continuous(type = "viridis") +
    # theme_bw()
    
    # geom_density_2d()
    
    # ggsave(filename = paste0(alphas[i], "_final_fx_", data, ".png"), width=10, height=5, pointsize=12, units = "in")
    # Store the plot in a variable
    assign(nam, tmp_plot)   
    
  }
  
  setwd(maindir)
  
}


#######
# VIP #
#######

data_vip <- paste0(maindir, folders_to_compare[ 1 ])
setwd(data_vip)

# prior_vip <- read.csv("vip_prior_samples_0.txt")
pred_vip <- read.csv("vip_y_forecasted_0.txt")[,1:3]
names_vip <- paste0("sample_", c(1:(ncol(pred_vip) - 2)), "_yx")

names(pred_vip) <- c("x","y", names_vip)

melt_vip_pred <- melt(pred_vip, id.vars = c("x", "y"))

melt_vip_pred$x <- melt_vip_pred$x # * stdx + xmean
melt_vip_pred$y <- melt_vip_pred$y # * stdy + ymean
melt_vip_pred$value <- melt_vip_pred$value # * stdy + ymean


vip_pred_plot <- ggplot(melt_vip_pred, aes(x=x, y=value)) +
  stat_density_2d(aes(fill = ..density..), geom = "raster" , contour = FALSE, show.legend = FALSE) +
  scale_fill_distiller(palette="Spectral", direction=-1) +
  scale_color_viridis_c() + 
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank()) +
  scale_x_continuous(expand = c(0, 0), limits = c(NA, NA)) +
  scale_y_continuous(expand = c(0, 0), limits = c(-15, 15)) 

# +   xlim(-6, 6) + ylim(-6, 6) 
#  + theme_void()

# vip_pred_plot

# stat_density_2d(aes(fill = ..level..), geom = "polygon", colour="white", bins = 5)
# vip_pred_plot


########
# fBNN #
########

data_fbnn <- paste0(maindir, folders_to_compare[ 2 ])
setwd(data_fbnn)

pred_fbnn <- read.csv("fbnn_y_forecasted_0.txt")
names_fbnn <- paste0("sample_", c(1:(ncol(pred_fbnn) - 2)), "_yx")

names(pred_fbnn) <- c("x","y", names_fbnn)

melt_fbnn_pred <- melt(pred_fbnn, id.vars = c("x", "y"))


melt_fbnn_pred$x <- melt_fbnn_pred$x * stdx + xmean

fbnn_pred_plot <- ggplot(melt_fbnn_pred, aes(x=x, y=value)) +
  stat_density_2d(aes(fill = ..density..), geom = "raster" , contour = FALSE, show.legend = FALSE) +
  scale_fill_distiller(palette="Spectral", direction=-1) +
  scale_x_continuous(expand = c(0, 0), limits = c(NA, NA)) +
  scale_y_continuous(expand = c(0, 0), limits = c(-15, 15)) +
  scale_color_viridis_c() + 
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank()) # + 
  # theme_void()


# stat_density_2d(aes(fill = ..level..), geom = "polygon", colour="white", bins = 10)


# fbnn_pred_plot



###################################################################################
###################################################################################

#                              MAKE THE GRID PLOT 

###################################################################################
###################################################################################

gridplot <- ggarrange(plot_ip_priors_bnn_1.0, vip_prior_plot, fbnn_prior_plot,
              plot_ip_pred_bnn_1.0, vip_pred_plot, fbnn_pred_plot,
              ncol = 3, nrow = 2, 
              labels = c("prior (ours)", "prior-VIP", "prior-fBNN",
                         "pred (ours)", "pred-VIP", "pred-fBNN"))

gridplot

setwd(maindir)


ggsave(filename = "comparison_plot_fbnn_prior_gp.png", plot = gridplot,
       width=10, height=4.5, pointsize=12, units = "in")


