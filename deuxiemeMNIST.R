library(rpart)
library(rpart.plot)
library(MASS)
library(FactoMineR)
library(mlr)
library(tensorflow)


datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = FALSE)
dim(mnist$train$images)
train=mnist$train$images
Class.train=as.factor(mnist$train$labels)
test=mnist$test$images
Class.test=as.factor(mnist$test$labels)


set.seed(72)

######Arbre de decision classiques sans analyse factorielle:
fit <- rpart(Class.train ~ .,data=as.data.frame(train))
prp(fit,extra=1)
resNormal=predict(fit, as.data.frame(test), type="class")



#####arbre de decision apres PCA on garde 100% de variance 
a=PCA(train)
train3=as.matrix(train[,-5])%*%as.matrix(a$var$coord)
plot(train3, col = Class.train)

test3=as.matrix(test[,-5])%*%as.matrix(a$var$coord)
plot(test3, col = Class.test)

fit <- rpart(Class.train ~ .,data=as.data.frame(train3)) ####on  entrain le cart sur le train
prp(fit,extra=1)
resPCA=res=predict(fit, as.data.frame(test3), type="class")


####arbre de decision post LDA

a=lda(Species ~., data=train) #on fait la LDA sur train
plot(a, col = as.numeric(train[ ,5])) #on regarde la tete des axes factoriels

train2=as.matrix(train[,-5])%*%as.matrix(a$scaling) ####on projete sur le nouvel espace de dim train via les axes trouv??s via train
plot(train2, col = as.numeric(train[ ,5]))

test2=as.matrix(test[,-5])%*%as.matrix(a$scaling) ###On projete sur le nouvel espace de dim test via les axes trouv??s via train
plot(test2, col = as.numeric(test[ ,5]))


fit <- rpart(train$Species ~ .,data=as.data.frame(train2))  ###on fit sur train
prp(fit,extra=1)
resLDA=res=predict(fit, as.data.frame(test2), type="class") ##on predit




#####Comparaison resultats:
table(test[,5],resNormal) #Decision tree tout  seul
"resNormal
setosa versicolor virginica
setosa        263          0         0
versicolor      0        260         4
virginica       0          7       266"

table(test[,5],resPCA) #Decission tree post PCA
"            resPCA
setosa versicolor virginica
setosa        263          0         0
versicolor      0        260         4
virginica       0          9       264"
table(test[,5],resLDA) #Decision tree post LDA
"            resLDA
setosa versicolor virginica
setosa        263          0         0
versicolor      0        253        11
virginica       0          0       273"


