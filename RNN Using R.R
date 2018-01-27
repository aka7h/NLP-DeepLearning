
#### RNN using R

library(tm)
library(readr)
library(hash)
library(ramify)#for clip

##read the text file
files <- read_file('kafka.txt')
files

#converting the text file to independent string
c <- array((strsplit(files,"")))
cfile <- c[[1]]
unique.char <- sort(unique(c[[1]]))
vocab.length <- length(unique.char)
cat("Total characters are ",length(c[[1]])," and unique char are ",length(unique.char))

#Set index to each character
ix_to_char <- data.frame(unique.char,1:vocab.length)
ix_to_char
## Create a hash 

# ix_2_char <- hash(key=1:vocab.length,values=unique.char)
# 
char_to_ix <- hash(unique.char,1:vocab.length)
char_to_ix

ix_2_char <- hash(1:vocab.length,unique.char)
ix_2_char

#create one hot encoding
vector_char_of_a <- matrix(0,vocab.length,1)
vector_char_of_a[char_to_ix[['a']]] <- 1
vector_char_of_a




#### Parameters for RNN

hidden_size <- 100
seq_length <- 25
learning_rate <- 1e-1

#matrix(8000 values,rows,col)

## input to hidden
Wxh <- matrix(rnorm(hidden_size * vocab.length), hidden_size, vocab.length) * 0.01

## hidden to hidden
Whh <- matrix(rnorm(hidden_size * hidden_size), hidden_size, hidden_size) * 0.01

## hidden to output
Why <- matrix(rnorm(vocab.length*hidden_size),vocab.length,hidden_size) * 0.01

#bias for hidden state
bh <- matrix(0,hidden_size,1)

#bias for output
by <- matrix(0,vocab.length,1)

dim(Wxh)
dim(Whh)
dim(Why)
dim(bh)
dim(by)

weight <- list(Wxh=Wxh,Whh=Whh,Why=Why,bh=bh,by=by)


##Forward Pass

# hs= input*Wxh + last_value_of_the_hidden_state*Whh+bh
# ys = hs*Why+by
# ps = normalized(ys)


#### Simple example of Forward pass (a) Forward Propogation
# loss <- 0
# input_length <- length(inputs)  #Length of the window
# xs <- matrix(0,vocab.length,input_length) #
# hs <- matrix(0,hidden_size,input_length+1)
# hs[,1] <- hprev
# ys <- xs
# ps <- ys
# 
# for(i in 1:input_length){
#   xs[inputs[i],i] <- 1
#   hs[,(i+1)] <- tanh((Wxh %*% xs[,i])+(Whh %*% hs[,(i-1+1)])+bh)
#   ys[,i]<- (Why %*% hs[,(i+1)])+by
#   ps[,i] <- exp(ys[,i])/sum(exp(ys[,i]))
#   loss <- loss+(-1.00)*log(ps[target[i],i])
# }
# 
# inputs <- c(23,4,3,1,2,4,23,44,25,53,12,76) #1:12
# target<- c(4,3,1,2,4,23,44,25,53,12,76,22) #2:12+1
# hprev <- matrix(0,hidden_size,1) #weights of previous hiddens
# 
# 
# #### Simple example of Backward Pass (a) Back Propogation
# 
# dWxh <- matrix(0, hidden_size, vocab.length) 
# dWhh <- matrix(0, hidden_size, hidden_size) 
# dWhy <- matrix(0,vocab.length,hidden_size)
# dbh <- bh; 
# dby <- by;
# dhnext <- 0
# 
# 
# for(i in input_length:1){
#   dy <- ps[,i]
#   dy[target[i]] = dy[target[i]] - 1
#   dWhy <- dWhy + (dy %*% t(hs[,i]))
#   dby <- dby+ dy
#   
#   dh  <- t(Why) %*% dy + dhnext
#   dhraw <- (1-hs[,i]*hs[,i])*dh
#   dbh <- dbh +dhraw
#   dWxh <- dWxh + (dhraw %*% t(xs[,i]))
#   dWhh <- dWhh + (dhraw %*% t(hs[,(i-1+1)]))
#   dhnext <- t(Whh)%*%dhraw
# }

#derivative
# dWxh <- matrix(0,hidden_size,vocab.length)
# dWhh <- matrix(0,hidden_size,hidden_size)
# dWhy <- matrix(0,vocab.length,hidden_size)
# dbh <- matrix(0,hidden_size,1)
# dby <- matrix(0,vocab.length,1)


"lossFun" <- function(input, target,hprev,w){
  loss <- 0
  input_length <- length(input)
  
  
  ## Forward Pass or Feed forward
  xs <- matrix(0,vocab.length,input_length)
  hs <- matrix(0,hidden_size,input_length+1)
  hs[,1] <- hprev
  ys <- xs
  ps <- ys
  
  for(i in 1:input_length){
    xs[input[i],i] <- 1
    hs[,(i+1)] <- tanh((w$Wxh %*% xs[,i])+(w$Whh %*% hs[,(i-1+1)])+w$bh)
    ys[,i] <- (w$Why %*% hs[,(i+1)])+w$by
    ps[,i]<- exp(ys[,i])/sum(exp(ys[,i]))
    loss <- loss + (-1.0)*log(ps[target[i], i])
  }
  
  ## Backward Pass or Back Prop
  
  dWxh <- matrix(0,hidden_size,vocab.length)
  dWhh <- matrix(0,hidden_size,hidden_size)
  dWhy <- matrix(0,vocab.length,hidden_size)
  dbh <- matrix(0,hidden_size,1)
  dby <- matrix(0,vocab.length,1)
  
  dhnext <- 0
  
  for(i in input_length:1){
    dy <- ps[,i]
    dy[target[i]] = dy[target[i]]-1
    dWhy <- dWhy + (dy %*% t(hs[,i]))
    dby <- dby+dy
    dh <- t(w$Why) %*% dy+dhnext
    dhraw <- (1-hs[,i]*hs[,i])*dh
    dbh <- dbh+dhraw
    dWxh <- dWxh +(dhraw %*% t(xs[,i]))
    dWhh <- dWhh +(dhraw %*% t(hs[,(i-1+1)]))
    dhnext <- t(w$Whh)%*%dhraw
  }
  dW_b <- sapply(list(dWxh = dWxh, dWhh = dWhh, dWhy = dWhy,dbh = dbh, dby = dby),function(x) clip(x,-5,5))
  return(list(dWeight = dW_b,loss=loss,hprev=hs[,input_length+1]))
  
}

# dr <- lossFun(inputs,target,hprev,we)


### Sample sentense for model
"sampled" <- function(h,seed_x,n,w){
  x <- matrix(0,vocab.length,1)
  x[seed_x,] <- 1
  ixes <- NULL
  for(i in 1:n){
    h = tanh(w$Wxh%*%x+w$Whh%*%h+w$bh)
    y = (w$Why %*% h)+w$by
    p <- exp(y)/sum(exp(y))
    ix  <- sample(x = 1:vocab.length, size = 1, p = drop(p), replace = TRUE)
    x <- matrix(0,vocab.length,1)
    x[ix,] <- 1
    ixes <- c(ixes,ix)
  }
  return(ixes)
}
# hprev <- matrix(0,hidden_size,1)
#   
# char <- sampled(hprev,char_to_ix$a,200)
# cat(paste(as.list(ix_2_char)[char], collapse = ""))


### Smooth Loss
# smooth_loss <- smooth_loss*0.999+loss*0.01

## Optimization using AdaGrad
mWxh <- matrix(0, hidden_size, vocab.length)
mWhh <- matrix(0, hidden_size, hidden_size) 
mWhy <- matrix(0,vocab.length,hidden_size)
mbh <- bh 
mby <- by

memWeight <- list(mWxh=mWxh,mWhh=mWhh,mWhy=mWhy,mbh=mbh,mby=mby)
# dev <- list(dWxh=dWxh,dWhh=dWhh,dWhy=dWhy,dbh=dbh,dby=dby)
# we <- list(Wxh=Wxh,Whh=Whh,Why=Why,bh=bh,by=by)

n <- 1
p <- 1

smooth_loss <- -log(1.0/vocab.length)*seq_length

while(n<=1000*100){
  if(p+seq_length+1 >= length(data) || n==1){
    hprev <- matrix(0,hidden_size,1) #reset rnn memory
    p=1
  }
  inputs <- apply(as.matrix(cfile[p:(p+seq_length-1)]),1,function(x)grep(x,ix_to_char[,1],fixed = T))
  target <- apply(as.matrix(cfile[p+1:(p+seq_length)]),1,function(x)grep(x,ix_to_char[,1],fixed = T))
  
  der <- lossFun(inputs,target,hprev,weight)
  smooth_loss <- smooth_loss*0.999+der$loss*0.001
  hprev <- der$hprev
  
  if(n%%1000==0){
    cat('\n------------------------------------------------------')
    cat('\niteration: ',n,' loss:\t',smooth_loss,'\n')
    cat('\n------------------------------------------------------\n')
    char <- sampled(hprev,inputs[1],200,weight)
    cat(paste(as.list(ix_2_char)[char], collapse = ""))
    cat('\n------------------------------------------------------')
    cat('\n------------------------------------------------------')
    
  }
  
  'Weight_Update' <- mapply(FUN=function(we,dev,mem){
    mem <- mem+dev*dev
    we <- we - (learning_rate*dev)/sqrt(mem +1.0/1e-8)
    return(list(we=we,mem=mem))
  },we = weight,dev=der$dWeight,mem=memWeight,SIMPLIFY = FALSE)
  
  weight <- lapply(Weight_Update,function(x) x$we)
  memWeight <- lapply(Weight_Update,function(x) x$mem)

  
  p<- p+seq_length
  n = n+1
}
