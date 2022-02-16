

#           ************* IVA-510- Dönem Projesi ***************




#                   **** DOMAIN ve IS PROBLEMI BILGISI ****

# Çalismamiz bir banka sektöründe gerçeklestirilecektir. Bu sebeple bir banka, elindeki verilerden yola çikarak müsterilerinin aldigi kredilerinin geri ödeme performansini ölçmek istiyor. Bu çiktiya göre temerrüte düsme ihtimalleri, hangi müsteri profillerine kredi verilmesi yada verilmemesi gerekiyor gibi sorularin cevaplanmasi istenmektedir.
 


#                 *****  Veri seti hikayesi ve Çalisma Alani  *****

# train ve test veri setlerini belirlerken ögrenci numaranizi seed olarak kullaniniz.

#Karar agaci, regresyon ve benzerlik(?) modellerini kullanarak model tahminlerini yapiniz. Uc  modelin de performanslarini karsilastiriniz. Bu veri setinde:

#  Hedef degisken: default payment (Y)





#Is Problemi : 

#Banka müsterilerinin temerrüte düsme riskini tahmin etmek istenmektedir. 


#Veri Seti hikayesi:

#Veri setimizde yaklasik


data_16 <- read.csv("C:/Users/ahmet/Desktop/VeriAnaligi/Dersler/IVA-510_IsletmelericinVeriAnalitigi/DonemProjesi/data_16.csv")



df<-data_16


# Libraries
library(ggplot2)
library(entropy)
library(dplyr)
library(tidyr)
library(caret)
library(xgboost)
library(pastecs)
library(rpart)
library(rpart.plot)
library(MASS)
library(outliers)
library(randomForest)
library(class)
library(rpart)
library(readr)
library(caTools)
library(dplyr)
library(party)
library(partykit)
library(rpart.plot)
library(DMwR)
library(DMwR2)
library(FNN)
library(tree)
library(ISLR)
library(rpart.plot)
library(parameters)

#                **** EDA (Exploratory Data Analysis)****

head(df)
summary(df)
stat.desc(df)
is.na(df)  #False
colnames(df)
str(df)
# 23400 obs. of  32 variables (Num - Int)

--
#Hedef degiskenimiz 0'a homojen mi heterojen mi oldugunu görüyoruz.
table(df$Y)

#  0     1 
#18240  5160 

--
#Proportion'i bakarak yüzde ne kadar homojen oldugu görebiliyoruz.

prop.table(table(df$Y))

#     0          1 
# 0.7794872 0.2205128 

--

# Burada Hedef degiskenin entropisini hesapladik. Entropi 0'a ne kadar yakinsa o kadar saf oldugunu görüyoruz.
df_entropy <- entropy(table(df$Y), unit="log2")

# 0.7611028    0- saf / 1 - Heterojen



#   **********Feature Selection************


##Ortalama degerleri de görmek istedigim için yas ve verilen kredi miktarinin ortalamasini yeni deger olarak ekliyorum.

df$mean_age = as.numeric(df$X5>35)
df$mean_credit = as.numeric(df$X1>11700)
df$total_bill = df$X12+df$X13+df$X14+df$X15+df$X16+df$X17
df$total_amount_pay = df$X18+df$X19+df$X20+df$X21+df$X22+df$X23
df$irregular_payment = as.numeric(df$X6+df$X7+df$X8+df$X9+df$X10+df$X11<=1)
df$difference = as.numeric(df$total_amount_pay-df$X1 < 0)





#Numerik degiskenlerden olusacak bir veri seti olusturuyoruz

df1=df[c('Y', 'difference', 'irregular_payment', 'total_amount_pay','total_bill','mean_credit','mean_age','X1','X2','X3','X4','X5')]

str(df1)

#integer olan degerlern hepsini numeric olarak degistiriyorum.
df1<- lapply(df1[2:12], as.numeric)

anyNA(df1)
#FALSE


                       ###### MODEL OLUSTURULMASI ######


# Train ve Test setlerinin olusturulmasi

  #Her denemede ayni sonucu elde edebilmek için sabit bir seed degeri atiyoruz


set.seed(202167009)

  # train indeks ön tanimli argümani olusturarak her seferinde ayirma islemi yapmaya gerek kalmiyor. Train ve test setleri 75'e 25 olarak ayrildi.

train_indeks <- createDataPartition(df1$Y,
                                    p=.75, list = FALSE, times = 2 )
  
train <- df1[trainind,]
test <- df1[-trainind,]

train_x <- train %>% dplyr::select(-Y)
train_y <- train$Y

test_x <- test %>% dplyr::select(-Y)
test_y <- test$Y



#               *********   Logistic Regresyon Modeli  *********

model_glm <- glm(Y
                 ~.,
                 data=train, 
                 family = binomial)
summary(model_glm)


train$pred <- predict(model_glm, newdata=train, type="response")
test$pred <- predict(model_glm, newdata=test, type="response")


summary(train$pred)
#     Min.  1st Qu.   Median     Mean     3rd Qu.     Max. 
#0.0000001 0.1346931 0.1589006 0.2225321 0.2021653 0.6599302

summary(test$pred)

#  Min.   1st Qu.  Median  Mean   3rd Qu.   Max. 
#0.0000  0.1337  0.1576  0.2185  0.1973  0.6621 


train$class <- as.numeric(train$pred>mean(train$pred))


table_mat <- table(train$Y, train$class)
rownames(table_mat) <- paste("Actual", rownames(table_mat), sep = ":")
colnames(table_mat) <- paste("Predicted", colnames(table_mat), sep = ":")
print(table_mat)

#           Predicted:0    Predicted:1
#Actual:0        8505        2923
#Actual:1        1336        1935

#test veri setinin acc'sini hesaplayalim:
acc_test <- sum(diag(table_mat)/sum(table_mat))

#0.7102524



#              ****************      KNN     ******************

# Model

knn_train <- train
knn_test <- test

#knn_train<- lapply(train[1:12], as.numeric)
#knn_test<- lapply(test[1:12], as.numeric)

#modeli 4 komsu ve 3 komsu üzerinden fiteyip çiktilarina bakiyorum. KNN'de komsu sayisi arttirkça modeli anlamini yiritdigi için Tuning modeline ihtiyaç duymadan sadece 3 ve 4.ncü komsu degerleri üzerinden kontrol sagliyorum

knn_fit4 <- knn(train = knn_train, test = knn_test, cl=train_y, k = 4)

summary(knn_fit4)
# 0    1 
#8225  476


knn_fit3 <- knn(train = knn_train, test = knn_test, cl=train_y, k = 3)

summary(knn_fit3)

# 0    1 
#7494 1207


#test veri setinin acc'sini hesaplayalim:

knn_tb3 <- table(knn_test$Y, knn_fit3)
acc_knn3 <- sum(diag(knn_tb)/sum(knn_tb))
#0.7347431
#model fit 3'ün acc'si %73 olarak belirlendi


knn_tb4 <- table(knn_test$Y, knn_fit4)
acc_knn4 <- sum(diag(knn_tb4)/sum(knn_tb4))
#0.7697966
#model fit 4'ün acc'si %77 olarak belirlendi



#              **************   Decision Tree  **************

#Model olusturulmasi

tree_pred <- tree(Y~., data=train, method = "class")
summary(tree_pred)

#Number of terminal nodes:  3 
#Residual mean deviance:  0.1498 = 2201 / 14700 
#Distribution of residuals:
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#-0.4852 -0.1180 -0.1180  0.0000 -0.1180  0.8820 

# Agacin görsellestirilmesi

plot(tree_pred)
text(tree_pred, pretty = 0)



#rpart ile modelleme

tree_pred_rpart <- rpart(Y~., data=train, method="class")

summary(tree_pred_rpart)

# Karmasiklik parametresine karsilik olusan degerlerin gösterimi
plotcp(tree_pred_rpart)


min_cp <- tree_pred_rpart$cptable[which.min(tree_pred_rpart$cptable[, "xerror"]), "CP"]
#En iyi CP degerini bulmus olduk.
#0.01

prp(tree_pred_rpart, type=1)
rpart.plot(tree_pred_rpart)


#*** train veri seti için Accuracy hesaplama
predict(tree_pred_rpart, train_x, type = "class")

tb <- table(predict(tree_pred_rpart, train_x, type = "class"), train_y)

confusionMatrix(tb, positive = "0")
?confusionMatrix

#Accuracy : 0.7829          
#95% CI : (0.7762, 0.7896)
#No Information Rate : 0.7775          
#P-Value [Acc > NIR] : 0.05714  

#***Test veri seti için Accuracy hesaplama
tb_test <- table(predict(tree_pred_rpart, test_x, type = "class"), test_y)
confusionMatrix(tb_test, positive = "0")

#Accuracy : 0.7877         
#95% CI : (0.779, 0.7963)
#No Information Rate : 0.7829         
#P-Value [Acc > NIR] : 0.1402 


# Github

library(usethis)

use_git_config(user.name = "ahmetcanyalcin", user.email = "ahmetcanyalcin@gmail.com")
 