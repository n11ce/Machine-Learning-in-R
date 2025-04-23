## N11CE R programlama Makine Öğrenmesi

data <- read.csv("kalp.csv") # veri yolu

# kullandığımız kütüphaneler
library(Amelia)
library(caret)
library(ggplot2)
library(class)
library(randomForest)
library(e1071)
library(nnet)
library(rpart)
library(dplyr)
library(kernlab)
library(glmnet)


#is.na metodu ile veri setinde eksik parça olup olmadığını kontrol ediyoruz.
sum(is.na(data))
# biraz eksik veri yaratalım
set.seed(100)
data_yeni <- data
data_yeni[sample(1:nrow(data),8),"trestbps"] <- NA ## null değer atadık
data_yeni[sample(1:nrow(data),20),"thalach"] <- NA ## null değer atadık
data_yeni[sample(1:nrow(data),5),"chol"] <- NA ## null değer atadık
sum(is.na(data_yeni))
## toplam 33 tane eksik verimiz var

## veri setinde en çok geçen değeri bulan fonkisyon
bul <- function(veri){
  ta <- names(sort(table(veri), decreasing = TRUE)[1]) # sort ile sıralayıp decreasing ile azalana göre sıralayıp 1. değeri alıyoruz
  return (ta)
}

## eksik verileri o sütünda geçen en fazla değer ile doldurma


eksik <- colSums(is.na(data_yeni)) # eksik sütünları bulduk
eksik_sutunlar <- names(eksik[eksik > 0]) ## eksik sütün isimlerini kaydettik

i <- 1
while(i<=length(eksik_sutunlar)){
  satir <- eksik_sutunlar[i]
  data_yeni[is.na(data_yeni[, satir]), satir] <- as.numeric(bul(data_yeni[, satir]))
  #her eksik değeri o sütünün en çok geçen değeri ile doldurdük
  i <- i +1 
}

sum(is.na(data_yeni)) # kontrol ettik


# yaş ve trestbps kalp hastalığı grafiği
ggplot(data_yeni,aes(x=age,
                     y=trestbps,
                     color=target)) +
  geom_point() + 
  labs(title="yaş / trestbps bağlı hastalık",
       x="yaş",
       y="trestbps")

# Kan Basıncı mmHg / serum kolestoral kalp hastalığı grafiği
ggplot(data_yeni,aes(x=thalach,
                     y=chol  ,
                     color=target)) +
  geom_point() + 
  labs(title="Kan Basıncı mmHg / serum kolestoral  bağlı hastalık",
       x="Kan Basıncı mmHg",
       y="serum kolestoral")

## boxplot ile aykırı verileri tespit edip fonksiyon ile düzenliyoruz

boxplot(data_yeni)

duzenleyici <- function(veri) {
  # Veri çerçevesinin bir kopyasını oluşturma
  veri_kopya <- veri
  
  ilk_ceyrek <- quantile(veri_kopya, 0.25)
  son_ceyrek <- quantile(veri_kopya, 0.75)
  sapma <- son_ceyrek - ilk_ceyrek
  
  # Alt ve üst sınırları hesaplama
  alt_sinir <- ilk_ceyrek - 1.5 * sapma
  ust_sinir <- son_ceyrek + 1.5 * sapma
  
  # Aykırı değerleri düzenleme
  veri_kopya[veri_kopya < alt_sinir] <- alt_sinir
  veri_kopya[veri_kopya > ust_sinir] <- ust_sinir
  
  # return yapıyoruz
  return(veri_kopya)
}

# Fonksiyonu kullanarak veriyi düzenleme

data_yeni$chol <- duzenleyici(data_yeni$chol)
data_yeni$trestbps <- duzenleyici(data_yeni$trestbps)
data_yeni$thalach <- duzenleyici(data_yeni$thalach)
data_yeni$oldpeak <- duzenleyici(data_yeni$oldpeak)
data_yeni$slope <- duzenleyici(data_yeni$oldpeak)
data_yeni$ca <- duzenleyici(data_yeni$ca)
data_yeni$thal <- duzenleyici(data_yeni$thal)

## boxplot ile tekrardan kontrol ediyoruz ve artık aykırı verimiz yok daha stabil çaılşabiliriz
boxplot(data_yeni)


## feature extraction işlemi kandaki toplam basıncı bulmak

data_yeni$toplambasinc <- data_yeni$trestbps + data_yeni$chol
data_yeni$toplambasinc <- duzenleyici(data_yeni$toplambasinc)

###################################################################
## kan basıncının kalp hastalığına etkisi grafiği
ggplot(data_yeni, aes(x = age, y = toplambasinc, color = target)) +
  geom_point() +
  labs(title = "yaş/kan basıncı'nın kalp hastalığı üstündeki etkisi", x = "Yaş", y = "Toplam kan basıncı") +
  theme_minimal()

head(data_yeni) ## feature işlemimiz en son sütüna geldi bunu değiştirip targeti oraya alacağız.

data_yeni <- data_yeni %>%
  select(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,toplambasinc,target)

head(data_yeni) ## target en son sütüna alındı.

##########################################
# sınıflandırıcı işlemleri

#####################################
## sonuçlar için fonksiyon

dogruluk <- function(matrix){
  # Doğruluk  hesapla
  dogru <- sum(diag(matrix)) / sum(matrix)
  
  # Özgüllük hesapla
  TN <- matrix["0", "0"]
  FP <- matrix["0", "1"]
  ozellik <- TN / (TN + FP)
  
  # Duyarlılık hesapla
  TP <- matrix["1", "1"]
  FN <- matrix["1", "0"]
  duyar <- TP / (TP + FN)
  
  # F1-Skor hesapla
  kesin <- TP / (TP + FP)
  f1_score <- 2 * (kesin * duyar) / (kesin + duyar)
  
  # Sonuçları yazdır
  print(matrix)
  cat("----------------\n")
  cat("Doğruluk:", dogru, "\n")
  cat("Özgüllük:", ozellik, "\n")
  cat("Duyarlılık:", duyar, "\n")
  cat("F1-Skor:", f1_score, "\n")
  
  return(f1_score*100)
}



############## datamızı ayırdık
data_yeni$target <- as.factor(data_yeni$target)
set.seed(255)
dataindex <- createDataPartition(data_yeni$target,p=0.7,list = FALSE)
traindata <- data_yeni[dataindex, ] # train datamız
selectdata <- data_yeni[-dataindex, ]

tempdata <- createDataPartition(selectdata$target,p=0.5,list= FALSE)
testdata <- selectdata[tempdata, ]
validata <- selectdata[-tempdata, ]

ctrl <- trainControl(method = "cv", number = 5) ## çoklu kontrol kondları 5 çapraz kontrol 

#########  KNN MODELİ

# Hiperparametre kombinasyonlarını belirle
hyper_params <- expand.grid(
  k = seq(1, 99, by = 2)  
)


# hiperparametre ayarlamaları
model <- train(target ~ ., data = traindata, method = "knn", tuneGrid = hyper_params)
a <- model$bestTune$k
model1 <- knn(train=traindata[,1:14],test=testdata[,1:14],cl=traindata$target,k=a)
conf_matrix <- table(Referance=testdata$target, Prediction=model1) ##conficion matrix
knn_f1 <- dogruluk(conf_matrix)


###### knn validation

### validation 5 kademeli
folds <- createFolds(validata$target,k=5)
cv <- lapply(folds, function(x){
  training_fold <- validata[-x, ]
  test_fold <- validata[x, ]
  
  model <- train(target ~ ., data = training_fold, method = "knn", tuneGrid = hyper_params,
                 trControl = ctrl) ## 5 çoklu doğrulama
  #en iyi k değeri
  a <- model$bestTune$k
  
  model1 <- knn(train=training_fold[,1:14],test=test_fold[,1:14],cl=training_fold$target,k=a)  
  # conf matrix
  conf_matrix <- table(Referans = test_fold$target, Prediction=model1)
  # değerleri yazdır
  karar_f1 <- dogruluk(conf_matrix)
  return(karar_f1)
})
warnings()
## sonuçlar
knnvali_f1 <- (cv$Fold1+cv$Fold2+cv$Fold3+cv$Fold4+cv$Fold5) / length(cv)


##########################################################
### karar ağacı


hyper_params <- expand.grid(
  cp = seq(0.01, 0.1, by = 0.01)
)

# Hiperparametre optimizasyonu
agac_hyper <- train(
  target ~ .,
  data = traindata,
  method = "rpart",
  trControl = ctrl, ## 5 çoklu doğrulama
  tuneGrid = hyper_params
)

#en iyi modeli bulma
best_tune <- rpart(target ~ ., data = traindata, cp = agac_hyper$bestTune$cp, method = "class")


# modelimizi test ediyoruz
tahminler <- predict(best_tune, newdata = testdata, type="class")
# conf matrix
conf_matrix <- table(Referans = testdata$target, Tahmin = tahminler)
# değerleri yazdır
karar_f1 <- dogruluk(conf_matrix)

### validation 5 kademeli
folds <- createFolds(validata$target,k=5)
cv <- lapply(folds, function(x){
  training_fold <- validata[-x, ]
  test_fold <- validata[x, ]
  best_tune <- rpart(target ~ ., data = training_fold, cp = agac_hyper$bestTune$cp, method = "class")
  # modelimizi test ediyoruz
  tahminler <- predict(best_tune, newdata = test_fold, type="class")
  # conf matrix
  conf_matrix <- table(Referans = test_fold$target, Tahmin = tahminler)
  # değerleri yazdır
  karar_f1 <- dogruluk(conf_matrix)
  return(karar_f1)
})

# değerleri yazdır
kararvali_f1 <- (cv$Fold1+cv$Fold2+cv$Fold3+cv$Fold4+cv$Fold5) / length(cv)

##########################################################
### Naive Bayes Modeli
hyper_params <- data.frame(laplace = seq(0, 1, by = 0.1),
                           usekernel = seq(0, 1, by = 0.1),
                           adjust = seq(0, 1, by = 0.1))

#modelimizi oluşturduk

model_n <- train(
  target ~ .,
  data = traindata,
  method = "naive_bayes",
  trControl = ctrl,
  tuneGrid = hyper_params
)

# en iyi modeli kullanma
best_tune <- naiveBayes(target ~ ., data = traindata, laplace = model_n$bestTune$laplace,
                usekernel = model_n$bestTune$usekernel,adjust = model_n$bestTune$adjust)

#test yaptık
tahminler <- predict(best_tune, newdata = testdata)
# conf matrix oluşturduk
conf_matrix <- table(Referans = testdata$target, Tahmin = tahminler)
#sonuçları gösterdik
naive_f1 <- dogruluk(conf_matrix)

## validation işlemleri
folds <- createFolds(validata$target,k=5)
cv <- lapply(folds, function(x){
  training_fold <- validata[-x, ]
  test_fold <- validata[x, ]
  
  best_tune <- naiveBayes(target ~ ., data = training_fold, laplace = model_n$bestTune$laplace,
                  usekernel = model_n$bestTune$usekernel,adjust = model_n$bestTune$adjust)  # modelimizi test ediyoruz
  tahminler <- predict(best_tune, newdata = test_fold)
  # conf matrix
  conf_matrix <- table(Referans = test_fold$target, Tahmin = tahminler)
  # değerleri yazdır
  karar_f1 <- dogruluk(conf_matrix)
  return(karar_f1)
})

# değerleri yazdır
naivevali_f1 <- (cv$Fold1+cv$Fold2+cv$Fold3+cv$Fold4+cv$Fold5) / length(cv)

##########################################################
### Rastgele Orman modeli
tune <- expand.grid(
  mtry = c(2, 4, 6, 8, 10))

# Modeli oluşturuyoruz
o_model <- train(target ~ ., data = traindata, method = "rf",
                 trControl = ctrl, tuneGrid = tune)
# en iyi modeli kullanalım
best_tune <- randomForest(target ~ ., data = traindata, mtry = o_model$bestTune$mtry)

# verimizi test ediyoruz
tahminler <- predict(best_tune, newdata = testdata)

# conf matrix hesaplama
conf_matrix <- table(Referans = testdata$target, Tahmin = tahminler)

#sonuçları gösterdik
rastgele_f1 <- dogruluk(conf_matrix)
#
## validation işlemleri
folds <- createFolds(validata$target,k=5)
cv <- lapply(folds, function(x){
  training_fold <- validata[-x, ]
  test_fold <- validata[x, ]
  
  best_tune <- randomForest(target ~ ., data = training_fold, mtry = o_model$bestTune$mtry)
  
  tahminler <- predict(best_tune, newdata = test_fold)
  # conf matrix
  conf_matrix <- table(Referans = test_fold$target, Tahmin = tahminler)
  # değerleri yazdır
  karar_f1 <- dogruluk(conf_matrix)
  return(karar_f1)
})

# değerleri yazdır
rastgelevali_f1 <- (cv$Fold1+cv$Fold2+cv$Fold3+cv$Fold4+cv$Fold5) / length(cv)

##########################################################
### svm modeli


tune <- data.frame(
  C = c(0.1, 1, 10),        
  sigma = c(0.01, 0.1, 1)
)

model <- train(target ~ ., data = traindata, method = "svmRadial", trControl = ctrl, tuneGrid = tune)

# en iyi modeli kullanalım
best_tune <- svm(target ~ ., data = traindata, kernel = "radial", cost = model$bestTune$C, gamma = model$bestTune$sigma)

## tahmin yapıyoruz
tahminler <- predict(best_tune, newdata = testdata)

# conf matrix hesaplama
conf_matrix <- table(Referans = testdata$target, Tahmin = tahminler)

# sonuçlar
svm_f1 <- dogruluk(conf_matrix)
#
## validation işlemleri
folds <- createFolds(validata$target,k=5)
cv <- lapply(folds, function(x){
  training_fold <- validata[-x, ]
  test_fold <- validata[x, ]
  
  best_tune <- svm(target ~ ., data = training_fold, kernel = "radial", cost = model$bestTune$C, gamma = model$bestTune$sigma)
  
  tahminler <- predict(best_tune, newdata = test_fold)
  # conf matrix
  conf_matrix <- table(Referans = test_fold$target, Tahmin = tahminler)
  # değerleri yazdır
  karar_f1 <- dogruluk(conf_matrix)
  return(karar_f1)
})

# değerleri yazdır
svmvali_f1 <- (cv$Fold1+cv$Fold2+cv$Fold3+cv$Fold4+cv$Fold5) / length(cv)


##################################################################
## normalize data ile sınıflandırma

## normalizasyon

normalize <- function(x){
  return ((x - min(x)) / (max(x) - min(x)))
}
normalizedata <- data_yeni

normalizedata[, 1:14] <- lapply(normalizedata[, 1:14],normalize)



##########################################################


##########################################################

set.seed(255)
dataindex <- createDataPartition(normalizedata$target,p=0.7,list = FALSE)
traindata <- normalizedata[dataindex, ] # train datamız
selectdata <- normalizedata[-dataindex, ]

tempdata <- createDataPartition(selectdata$target,p=0.5,list= FALSE)
testdata <- selectdata[tempdata, ]
validata <- selectdata[-tempdata, ]

ctrl <- trainControl(method = "cv", number = 5) ## çoklu kontrol kondları 5 çapraz kontrol 

#########  KNN MODELİ

# Hiperparametre kombinasyonlarını belirle
hyper_params <- expand.grid(
  k = seq(1, 99, by = 2)  
)


# hiperparametre ayarlamaları
model <- train(target ~ ., data = traindata, method = "knn", tuneGrid = hyper_params)
a <- model$bestTune$k
model1 <- knn(train=traindata[,1:14],test=testdata[,1:14],cl=traindata$target,k=a)
conf_matrix <- table(Referance=testdata$target, Prediction=model1) ##conficion matrix
knn_f1_n <- dogruluk(conf_matrix)


###### knn validation

### validation 5 kademeli
folds <- createFolds(validata$target,k=5)
cv <- lapply(folds, function(x){
  training_fold <- validata[-x, ]
  test_fold <- validata[x, ]
  
  model <- train(target ~ ., data = training_fold, method = "knn", tuneGrid = hyper_params,
                 trControl = ctrl) ## 5 çoklu doğrulama
  #en iyi k değeri
  a <- model$bestTune$k
  
  model1 <- knn(train=training_fold[,1:14],test=test_fold[,1:14],cl=training_fold$target,k=a)  
  # conf matrix
  conf_matrix <- table(Referans = test_fold$target, Prediction=model1)
  # değerleri yazdır
  karar_f1 <- dogruluk(conf_matrix)
  return(karar_f1)
})
warnings()
## sonuçlar
knnvali_f1_n <- (cv$Fold1+cv$Fold2+cv$Fold3+cv$Fold4+cv$Fold5) / length(cv)


##########################################################
### karar ağacı


hyper_params <- expand.grid(
  cp = seq(0.01, 0.1, by = 0.01)
)

# Hiperparametre optimizasyonu
agac_hyper <- train(
  target ~ .,
  data = traindata,
  method = "rpart",
  trControl = ctrl, ## 5 çoklu doğrulama
  tuneGrid = hyper_params
)

#en iyi modeli bulma
best_tune <- rpart(target ~ ., data = traindata, cp = agac_hyper$bestTune$cp, method = "class")


# modelimizi test ediyoruz
tahminler <- predict(best_tune, newdata = testdata, type="class")
# conf matrix
conf_matrix <- table(Referans = testdata$target, Tahmin = tahminler)
# değerleri yazdır
karar_f1_n <- dogruluk(conf_matrix)

### validation 5 kademeli
folds <- createFolds(validata$target,k=5)
cv <- lapply(folds, function(x){
  training_fold <- validata[-x, ]
  test_fold <- validata[x, ]
  best_tune <- rpart(target ~ ., data = training_fold, cp = agac_hyper$bestTune$cp, method = "class")
  # modelimizi test ediyoruz
  tahminler <- predict(best_tune, newdata = test_fold, type="class")
  # conf matrix
  conf_matrix <- table(Referans = test_fold$target, Tahmin = tahminler)
  # değerleri yazdır
  karar_f1 <- dogruluk(conf_matrix)
  return(karar_f1)
})

# değerleri yazdır
kararvali_f1_n <- (cv$Fold1+cv$Fold2+cv$Fold3+cv$Fold4+cv$Fold5) / length(cv)

##########################################################
### Naive Bayes Modeli
hyper_params <- data.frame(laplace = seq(0, 1, by = 0.1),
                           usekernel = seq(0, 1, by = 0.1),
                           adjust = seq(0, 1, by = 0.1))

#modelimizi oluşturduk

model_n <- train(
  target ~ .,
  data = traindata,
  method = "naive_bayes",
  trControl = ctrl,
  tuneGrid = hyper_params
)

# en iyi modeli kullanma
best_tune <- naiveBayes(target ~ ., data = traindata, laplace = model_n$bestTune$laplace,
                        usekernel = model_n$bestTune$usekernel,adjust = model_n$bestTune$adjust)

#test yaptık
tahminler <- predict(best_tune, newdata = testdata)
# conf matrix oluşturduk
conf_matrix <- table(Referans = testdata$target, Tahmin = tahminler)
#sonuçları gösterdik
naive_f1_n <- dogruluk(conf_matrix)

## validation işlemleri
folds <- createFolds(validata$target,k=5)
cv <- lapply(folds, function(x){
  training_fold <- validata[-x, ]
  test_fold <- validata[x, ]
  
  best_tune <- naiveBayes(target ~ ., data = training_fold, laplace = model_n$bestTune$laplace,
                          usekernel = model_n$bestTune$usekernel,adjust = model_n$bestTune$adjust)  # modelimizi test ediyoruz
  tahminler <- predict(best_tune, newdata = test_fold)
  # conf matrix
  conf_matrix <- table(Referans = test_fold$target, Tahmin = tahminler)
  # değerleri yazdır
  karar_f1 <- dogruluk(conf_matrix)
  return(karar_f1)
})

# değerleri yazdır
naivevali_f1_n <- (cv$Fold1+cv$Fold2+cv$Fold3+cv$Fold4+cv$Fold5) / length(cv)

##########################################################
### Rastgele Orman modeli
tune <- expand.grid(
  mtry = c(2, 4, 6, 8, 10))

# Modeli oluşturuyoruz
o_model <- train(target ~ ., data = traindata, method = "rf",
                 trControl = ctrl, tuneGrid = tune)
# en iyi modeli kullanalım
best_tune <- randomForest(target ~ ., data = traindata, mtry = o_model$bestTune$mtry)

# verimizi test ediyoruz
tahminler <- predict(best_tune, newdata = testdata)

# conf matrix hesaplama
conf_matrix <- table(Referans = testdata$target, Tahmin = tahminler)

#sonuçları gösterdik
rastgele_f1_n <- dogruluk(conf_matrix)
#
## validation işlemleri
folds <- createFolds(validata$target,k=5)
cv <- lapply(folds, function(x){
  training_fold <- validata[-x, ]
  test_fold <- validata[x, ]
  
  best_tune <- randomForest(target ~ ., data = training_fold, mtry = o_model$bestTune$mtry)
  
  tahminler <- predict(best_tune, newdata = test_fold)
  # conf matrix
  conf_matrix <- table(Referans = test_fold$target, Tahmin = tahminler)
  # değerleri yazdır
  karar_f1 <- dogruluk(conf_matrix)
  return(karar_f1)
})

# değerleri yazdır
rastgelevali_f1_n <- (cv$Fold1+cv$Fold2+cv$Fold3+cv$Fold4+cv$Fold5) / length(cv)

##########################################################
### svm modeli


tune <- data.frame(
  C = c(0.1, 1, 10),        
  sigma = c(0.01, 0.1, 1)
)

model <- train(target ~ ., data = traindata, method = "svmRadial", trControl = ctrl, tuneGrid = tune)

# en iyi modeli kullanalım
best_tune <- svm(target ~ ., data = traindata, kernel = "radial", cost = model$bestTune$C, gamma = model$bestTune$sigma)

## tahmin yapıyoruz
tahminler <- predict(best_tune, newdata = testdata)

# conf matrix hesaplama
conf_matrix <- table(Referans = testdata$target, Tahmin = tahminler)

# sonuçlar
svm_f1_n <- dogruluk(conf_matrix)
#
## validation işlemleri
folds <- createFolds(validata$target,k=5)
cv <- lapply(folds, function(x){
  training_fold <- validata[-x, ]
  test_fold <- validata[x, ]
  
  best_tune <- svm(target ~ ., data = training_fold, kernel = "radial", cost = model$bestTune$C, gamma = model$bestTune$sigma)
  
  tahminler <- predict(best_tune, newdata = test_fold)
  # conf matrix
  conf_matrix <- table(Referans = test_fold$target, Tahmin = tahminler)
  # değerleri yazdır
  karar_f1 <- dogruluk(conf_matrix)
  return(karar_f1)
})

# değerleri yazdır
svmvali_f1_n <- (cv$Fold1+cv$Fold2+cv$Fold3+cv$Fold4+cv$Fold5) / length(cv)


##################################################################
## normalize data ile sınıflandırma

## normalizasyon

normalize <- function(x){
  return ((x - min(x)) / (max(x) - min(x)))
}
normalizedata <- data_yeni

normalizedata[, 1:14] <- lapply(normalizedata[, 1:14],normalize)




###################################
##################################
# sonuçların hepsini görsellştirmek için data frame olarak kaydediyoruz
sonuclar <- data.frame(
  Model = c("KKN","Karar Ağacı","Naive Bayes","Rastgele Orman","SVM"),
  Normalize = c(knn_f1_n, karar_f1_n, naive_f1, rastgele_f1_n, svm_f1_n),
  No_Normalize = c(knn_f1, karar_f1, naive_f1_n, rastgele_f1, svm_f1),
  Normalize_Validation = c(knnvali_f1_n, kararvali_f1_n, naivevali_f1_n, rastgelevali_f1_n, svmvali_f1_n),
  No_Normalize_Validation = c(knnvali_f1, kararvali_f1, naivevali_f1, rastgelevali_f1, svmvali_f1)
)

# Verileri hepsini görsellikte sorun çıkarmasınlar diye topluyoruz
results_long <- tidyr::gather(sonuclar, key = "Metric", value = "Value", -Model)

# görselleştirme
ggplot(results_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", 
           position = position_dodge(), 
           color = "black") +
  labs(title = "Normalize ve Normalize Edilmemiş Tüm Sonuçlar",
       x = "Model",
       y = "F1 Skoru") +
  scale_fill_manual(values = c("Normalize" = "green", "No_Normalize" = "brown",
                               "Normalize_Validation" = "blue", "No_Normalize_Validation" = "pink")) +
  theme_minimal()


print(sonuclar)