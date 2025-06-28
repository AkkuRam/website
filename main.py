import streamlit as st
import pandas as pd

st.title("Akhilesh Ramesh")

st.write("""
I'm Akhilesh Ramesh, a student primarily coding in Rust and Python
focusing on projects related to system utils and data science. 
This portfolio showcases my skills, projects, and contact information.
""")

# About Section
st.header("About Me")

st.image("images/about_me.jpg", use_container_width=True)

st.write("""
I come from Germany, where I speak the languages: English, German & Kannada.
I have completed my bachelor's degree in Data Science and Artificial Intelligence at Maastricht University. 
Currently, I am studying the Data Science for Decision Making masters at Maastricht University.
In my free time, I enjoy programming and playing chess. 
The main languages I code in are Python and Rust, however I have experience coding in other languages as well (i.e. Java, MATLAB, R, Golang).
""")

# Skills Section
st.header("Skills")
st.write("""
- **Programming Languages**: Python, Rust, Java, MATLAB, R, Golang, SQL
- **Topics of interest**: Computer Vision, Machine Learning, Databases, Algorithms for Big Data, Data Fusion
""")

# Projects Section
st.header("Projects")
st.subheader("Disk Analyzer (Individual)")

st.image("images/disk_analyzer.png", use_container_width=True)

st.write("""
This project was done in python, where the main objective was to display system utils. Essentially a system tool
similar to a task manager, that depicts different statistics (i.e. CPU usage, network speed, device storage, etc). Here the python 
library used for this was "Rich". 
         
KEY NOTE: CPU Usage & Network Speed are not static, these are live updates
""")
st.markdown("[Github: Disk Analyzer](https://github.com/AkkuRam/disk-analyzer)")

st.subheader("Plastic Type Prediction (Group)")
st.write("""
Objective: Fuse near-infrared spectroscopy signal measurements with categorical descriptors on a dataset obtained from household plastics
in a group of 4 people
         
#### Dataset description
- Size: 373 rows, 998 columns
- Range of signals: ~700nm to ~1000nm
- Numerical columns: "spectrum_k", "sample_raw_k", "wr_raw_k" are 331 columns each
- Categorical columns: Color & Transparency
- Target variable: 1: PET, 2: HDPE, 3: PVC, 4: LDPE, 5: PP, 6: PS, 7: Other, 8: Unknown

The column "spectrum_k" are the columns of interest for us, since it is obtained by (spectrum_k = sample_raw_k / wr_raw_k), 
essentially dividing the raw signal by the white reference. Hence our total columns are 331 + 2 categorical columns to predict the target 
variable consisting of 7 different plastic types. We discarded the 8th plastic type, since it was unknown and it was only 10 samples.
             
Contributions:
- Preprocessing: Oversampling -> Baseline Correction -> Normalization -> Savitzy Golay
- High-level fusion: Bayesian Consensus
- Mid-level fusion: PCA + Models (XGBoost, AdaBoost, Multilayer Perceptron (MLP), Random Forest (RF))     
""")



st.write("""
#### Preprocessing
        
Oversampling was perfoming, this there was a class imbalance in the target variable. BorderlineSMOTE was used to balance 
the class imbalance. It uses the basic implementation of SMOTE and improves the problem, where the synthetic samples from
SMOTE are too similar to the existing minority samples. Hence, BorderlineSMOTE generates new samples that are near the borderline between 
the minority and majority class majority classes. It is easier view a visualization from the imbalanced-learn docs, where the 3 different 
colors/classes are better oversampled with BorderlineSMOTE.
""")

col1, col2 = st.columns(2)
with col1:
    st.image("images/smote.png", use_container_width=True)
with col2:
    st.image("images/borderlinesmote.png", use_container_width=True)

st.write("""
Now with BorderlineSMOTE, the left image below represents the imbalanced classes, where the right image represents the oversampled classes.
""")

col1, col2 = st.columns(2)
with col1:
    st.image("images/class_imb.png", use_container_width=True)
with col2:
    st.image("images/class_bal.png", use_container_width=True)

st.write("""
These images represent the preprocessing steps of the signal. For 7 plastic types there are many samples, hence there are multiple signal lines.
In addition, once the signals are preprocessed, if for each plastic type the samples are averaged, it should represent a distinct signal for each 
plastic type, which makes it identifiable which plastic type is which signal. 

Referring to the images below, the first image from the left are the signals after being oversampled. There is clear background noise,
therefore baseline correction is applied to isolate the peaks and flatten out the noise by using a quadratic polynomial (image 2). Hereafter, normalization is applied to bring them in a standard scale (image 3). Then the final step is to apply a Savitzky-Golay filter,
which mainly smoothens out the signal (image 4). 
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.image("images/signal_oversampling.png", use_container_width=True)
with col2:
    st.image("images/signal_baseline_correction.png", use_container_width=True)
with col3:
    st.image("images/signal_normalized.png", use_container_width=True)
with col4:
    st.image("images\signal_sav_golay.png", use_container_width=True)

st.write("""
#### Mid-level fusion
         
- Extracting features using PCA + Categorical features

From the previous preprocessing steps, now the data is ready to use for the models. PCA is 
applied to reduce the dimensionality, since we are dealing with ~300 columns, where the PCA plot of 2 components below
depicts a reasonable separation of the 7 classes. Moreover, from literature a threshold of keeping 90\% of the variance 
was used which selected top 5 components in our case. The elbow method was not used, since this chose < 5 components, and
the results (i.e. accuracy, precision, recall, f1score) were suboptimal. 
""")

col1, col2 = st.columns(2)
with col1:
    st.image("images/pca.png", use_container_width=True)
with col2:
    st.image("images/pca_thresholded.png", use_container_width=True)

st.write("""
With the following selections for mid-level fusion, it lead to the following results for the below metrics. Overall, the best performing models
in terms of accuracy was XGBoost (78%), but in terms of other metrics, but XGBoost and RF were stable. 
""")

data = {
    "Models": ["XGBoost", "AdaBoost", "MLP", "RF"],
    "Accuracy": [0.78, 0.51, 0.60, 0.71],
    "Precision": [0.77, 0.55, 0.53, 0.74],
    "Recall": [0.77, 0.53, 0.56, 0.73],
    "F1-score": [0.77, 0.49, 0.53, 0.72]
}
df = pd.DataFrame(data)

st.table(df)

st.write("""
#### High-level fusion

Unlike mid-level fusion, high-level fusion focuses on combining the outputs of multiple models, there is no
feature extraction in this stage. Hence, for high-level, all the preprocessing steps are used on the raw data, but without applying PCA.
The formula for bayesian consensus is shown below. 

$$
p(h_g | e) = \\frac{p(e | h_g) p(h_g)}{ \sum_g p(e | h_g) p(h_g)}
$$

- $p(e|h_g)$ is the likelihood estimate of the conditional probability that evidence e is observed
given that hypothesis $h_g$ is true
- $p(h_g)$ is the prior probability that hypothesis $h_g$ is true

For our case we calculate for each class (1 vs Rest), therefore this is a $2 \\times 2$ confusion matrix,
hence our prior probabilities are initialized to $[0.5, 0.5]$, essentially random. The models used for combining the outputs 
are RF, Decision Tree, XGBoost, Logistic Regression, SVM, KNN. The main crux of bayesian consensus is that, in each iteration
it gives out two probabilities, essentially for the label "yes" and "no". Since for 1 vs Rest, this becomes binary, so after iterating 
for each class, we get as an output (two probabilites), where we choose the higher probability and this is accuracy 
of predicting that class. 

Key Takeaway: The choice of the yes/no labels which is done by us, heavily impacts the results/accuracy. The predicted labels, will
be given by the model, the actual/true label we define, hence this definition by us impacts the result (i.e. bias).     
""")

st.write("""
#### Future work
- Could have tried out many more fusion techniques to compare (i.e. Low-level, Federating learning), as well as 
a few more within Mid/Low-level fusion
- Try another technique, which is averaging out all samples for each plastic type to create 7 distinct signals, then using cosine similarity 
to compare the true signal of each plastic type with our 7 distinct signals. 
""")

st.markdown("[Github: Spectroscopy](https://github.com/AkkuRam/df-project)")

st.subheader("Neural Network (Individual)")
st.write("""
TODO!!!!!
""")
st.markdown("[View Project](https://github.com/yourusername/project-two)")

# Contact Section
st.header("Contact")

st.write("📧 Email: [reachakhil10@gmail.com](mailto:your.email@example.com)")
st.write("🔗 LinkedIn: [www.linkedin.com/in/akhilesh-ramesh](https://linkedin.com/in/yourusername)")
st.write("🐙 Github: [https://github.com/Akkuram](https://github.com/Akkuram)")

# Footer
st.write("---")
st.write("© 2025 Akhilesh Ramesh | Built with Streamlit")
