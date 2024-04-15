# SkinGuardian
 A web application designed to predict and classify the type of skin cancer using dermatoscopic images uploaded by users.

## 1. Project description
### Context and goals
Skin disease poses a major global health challenge, affecting a vast number of people. It often proves difficult for individuals to visually recognize the signs of skin conditions. Early detection and treatment are critical, as they substantially lower the risk of morbidity and mortality related to these diseases.

**SkinGuardian** is a web application that utilizes **convolutional neural networks** to accurately classify skin diseases. This deep learning model is trained using the HAM10000 dataset, which includes 10,015 dermatoscopic images covering a variety of diagnostic categories. This application enables users to upload images of their skin for analysis, providing reliable disease identification and aiding users in understanding their skin health and promoting timely medical consultations and interventions.

### Dataset information
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 <br>
**HAM10000** is a dataset containing 10,015 dermatoscopic images that fall into the following seven diagnostic categories:
- Actinic Keratosis
- Basal Cell Carcinoma
- Benign Keratosis
- Dermatofibroma
- Melanoma
- Melanocytic Nevi
- Vascular Lesions
  
<img src="/images/dataset image.png"><br>

 ## 2. Project Stages
 0. Data Loading
 1. Data Preprocessing
 2. Data Exploration
 3. Model Training and Testing
 4. Web Application Development

### Model architecture
<img src="/images/model architecture.png" height="600"><br>

### Metrics
<img src="/images/metrics.png"><br>
<img src="/images/metrics curves.png"><br>

### Web application
Home page:
<img src="/images/website home.png" height="300"><br>

Results page:<br>
<img src="/images/website results 1.png" height="500"><br><br>
<img src="/images/website results 2.png" height="436"><br>

The web application has been trained exclusively on a dataset that encompasses seven distinct types of skin conditions. Should a user upload an image that doesn't correspond to any of these recognized categories, (if the model's confidence level falls below a 75% threshold), the output will indicate that no disease has been detected. It suggests that the skin might be healthy or possibly affected by a condition not included in the model's training set.
