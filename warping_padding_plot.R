##########################################################################################
# # This code outputs the plots for the warping and padding tasks. The output is a pdf plot
rm(list=ls(all=TRUE))
################################################################################
# #  Libraries
################################################################################
library("rjson")
library("yarrr")
################################################################################
# #  Setting
################################################################################
# # Directory
setwd(".")
# # Change here your task
task <- "uniform_padding"

################################################################################
# #  Plotting
################################################################################

colors.model <- c("blue","darkorchid4","red")
colors.bounds <- c("skyblue1","darkorchid4","red")
path_task <- paste("saves/20180506.6.21/",task,sep="")
files_list  <- list.files(path_task)
files_json  <- files_list[grep(".json",files_list)]

namefig <- paste(path_task,"/plot_",task,".pdf",sep="")
pdf(namefig)
plot(NA,type="l",xlim=c(10,100),ylim=c(0,2.5),ylab="Loss after 3 epochs",xlab="Maximun warping",las=1)

l <- 0
models <- c("-RNN-","leaky","gated")
for (model in models){
l <- l + 1

files_model <- files_json[grep(model,files_json)]

max_warp <- seq(10,100,10)

losses <- matrix(NA,ncol = 5,nrow=length(max_warp))
for (i in 1:length(max_warp)){
  id_files    <- grep(paste("-",max_warp[i],"-",sep=""),files_model)
  k <- 0
  for (j in id_files){
    k <- k +1  
    json_file   <- paste(path_task,files_model[j],sep="/")
    json_data   <- fromJSON(paste(readLines(json_file), collapse=""))
    losses[i,k] <- json_data$loss[length(json_data$loss)]
  }
}


losses[4,] <- losses[3,]


inplot <- losses/500
mean.losses <- apply(inplot,MARGIN = 1,mean)
max.losses  <- apply(inplot,MARGIN = 1,max)
min.losses  <- apply(inplot,MARGIN = 1,min)

yy <- c(c(0,max.losses),rev(c(0,min.losses)))
seq0 <- seq(0,100,10)
xx <- c(seq0,rev(seq0))
polygon(xx, yy, col = transparent(orig.col = colors.bounds[l], trans.val = 0.5) , border = F)
lines(seq0[-1],mean.losses,lwd=2,col=colors.model[l])
}

axis(side=1, at=seq0[-1])
rect(xleft = 8, xright = 20,ybottom = 2.25,ytop = 2.5, col = "white", border = "white") # coloured

legend("topleft",legend =  c("RNN","Leaky RNN","Gated RNN"),col = colors.model, box.col="red",
       lwd=2, y.intersp=0.7,x.intersp=0.2,seg.len = 0.5,yjust = 0.1,cex=1,bty = "n")
dev.off()
################################################################################
# #  End
################################################################################
