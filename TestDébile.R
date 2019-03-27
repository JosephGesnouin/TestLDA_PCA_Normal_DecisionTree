library(rpart)
library(rpart.plot)
library(MASS)
library(FactoMineR)


set.seed(72)

data(iris)
colnames(iris)
dim(iris)

library(mlr)
iris=iris[rep(seq_len(nrow(iris)), 20), ]
for(i in 1:4){
  iris[,i]=jitter(iris[,i])
}



x=sample(nrow(iris))
train=iris[x[1:2200],]
test=iris[x[2201:3000],]


######Arbre de d??cision classiques:
fit <- rpart(Species ~ .,data=train)
prp(fit,extra=1)
resNormal=predict(fit, test[,-5], type="class")
table(test[,5],resNormal)


#####arbre de d??cision apr??s PCA on garde 100% de variance
a=PCA(train[,-5])
train3=as.matrix(train[,-5])%*%as.matrix(a$var$coord)
plot(train3, col = as.numeric(train[ ,5]))

test3=as.matrix(test[,-5])%*%as.matrix(a$var$coord)
plot(test3, col = as.numeric(test[ ,5]))

fit <- rpart(train$Species ~ .,data=as.data.frame(train3))
prp(fit,extra=1)
resPCA=res=predict(fit, as.data.frame(test3), type="class")
table(test[,5],resPCA)



####arbre de d??cision post LDA

a=lda(Species ~., data=train)
plot(a, col = as.numeric(train[ ,5]))

lda.pred <- predict(a)$class
table(train[,5],lda.pred)


train2=as.matrix(train[,-5])%*%as.matrix(a$scaling)
plot(train2, col = as.numeric(train[ ,5]))

test2=as.matrix(test[,-5])%*%as.matrix(a$scaling)
plot(test2, col = as.numeric(test[ ,5]))


fit <- rpart(train$Species ~ .,data=as.data.frame(train2))
prp(fit,extra=1)
resLDA=res=predict(fit, as.data.frame(test2), type="class")
table(test[,5],resLDA)



#####Comparaison resultats:
table(test[,5],resNormal)
table(test[,5],resPCA)
table(test[,5],resLDA)



