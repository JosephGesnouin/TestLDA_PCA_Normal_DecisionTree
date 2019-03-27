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
a=FactoMineR::PCA(rbind(as.data.frame(train),as.data.frame(test)))
train3=as.matrix(train)%*%as.matrix(a$var$coord)
plot(train3, col = Class.train)

test3=as.matrix(test)%*%as.matrix(a$var$coord)
plot(test3, col = Class.test)

fit <- rpart(Class.train ~ .,data=as.data.frame(train3)) ####on  entrain le cart sur le train
prp(fit,extra=1)
resPCA=predict(fit, as.data.frame(test3), type="class")


####arbre de decision post LDA

a=MASS::lda(Class.train ~., data=as.data.frame(jitter(train)))#on fait la LDA sur train
plot(a, col = as.numeric(train)) #on regarde la tete des axes factoriels

train2=as.matrix(train)%*%as.matrix(a$scaling) ####on projete sur le nouvel espace de dim train via les axes trouv??s via train
plot(train2, col = as.numeric(train))

test2=as.matrix(test)%*%as.matrix(a$scaling) ###On projete sur le nouvel espace de dim test via les axes trouv??s via train
plot(test2, col = as.numeric(test))


fit <- rpart(Class.train ~ .,data=as.data.frame(train2))  ###on fit sur train
prp(fit,extra=1)
resLDA=predict(fit, as.data.frame(test2), type="class") ##on predit




#####Comparaison resultats:

#Decision tree tout  seul
sum(diag(table(Class.test,resNormal)))/(dim(table(Class.test,resNormal))[1]*dim(table(Class.test,resNormal))[2]) 
#61.9

#Decission tree post PCA
sum(diag(table(Class.test,resPCA)))/(dim(table(Class.test,resPCA))[1]*dim(table(Class.test,resPCA))[2]) 
#57.08


#Decision tree post LDA
sum(diag(table(Class.test,resLDA)))/(dim(table(Class.test,resLDA))[1]*dim(table(Class.test,resLDA))[2])
#77.47




