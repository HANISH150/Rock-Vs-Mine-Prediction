#importing streamlit library
import streamlit as st

#importing pandas library
import pandas as pd

#importing sklearn packages for dividing training and testing data 
from sklearn.model_selection import train_test_split

#importing accuracy_score method to determine the accuracy of our models
from sklearn.metrics import accuracy_score

#-----------------------Preprocessing Data and Defining the Required Models-------------------------

#Reading the input as csv file for preparing our models
sonar_data = pd.read_csv("SonarData.csv",header=None)
#st.dataframe(sonar_data)

# Split the data into features and labels
X = sonar_data.iloc[:, :-1].values
y = sonar_data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#                      ----Defining our Machine Learning Models----

#    ----Logistic Regression Algorithm
#importing LogisticRegression Model
from sklearn.linear_model import LogisticRegression

#defining Logistic Regression Model
LRModel = LogisticRegression() 

#Training the Logistic Regression model
LRModel.fit(X_train,y_train) 

#   ----K-Nearest-Neighbors ALGORITHM (KNN)
#importing KNN model
from sklearn.neighbors import KNeighborsClassifier

#defining the KNN Model
KnnModel = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

#Training the KNN Model
KnnModel.fit(X_train,y_train) 

#   ----Support Vector Machine ALGORITHM (SVM)
#importing the SVM model
from sklearn.svm import SVC

#defining SVM model
svm = SVC(kernel='rbf', C=1, gamma=1)

#training SVM model
svm.fit(X_train, y_train) 

#-----------------------END OF CODE for Preprocessing Data and Defining the Required Models-------------------------

#-----CODE for WRITING A FUNCTION TO MAKE PREDICTIONS ON NEW SAMPLE DATA THAT HAS BEEN UPLOADED BY THE USER-----
def Predict_New_Class(fresh_sample):
    st.warning("RESULTS FOR GIVEN DATA")
    #transposing the data frame
    fresh_sample = fresh_sample.transpose()

    #Converting the data frame into list of lists 
    fresh_sample_list=[]

    for column in fresh_sample.columns:
        fresh_sample_list.append(fresh_sample[column].to_list())
    

    #Logistic Regression prediction accuracy
    y_pred_LR = LRModel.predict(X_test)
    accuracy_LR = accuracy_score(y_test,y_pred_LR)

    #KNN test prediction accuracy
    y_pred_knn = KnnModel.predict(X_test)
    accuracy_knn = accuracy_score(y_test,y_pred_knn)

    #SVM test prediction accuracy
    y_pred_svm = svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test,y_pred_svm)
    #st.write("KNN : " , accuracy_knn)
    #st.write("SVM : " , accuracy_svm)

    #Predicting the class of new sample
    for index in range(0,len(fresh_sample_list)):
        #making prediction using Logistic Regression Model
        prediction_of_LR = LRModel.predict([fresh_sample_list[index]])

        #making prediction using KNN Model
        prediction_of_knn = KnnModel.predict([fresh_sample_list[index]])

        #making prediction using SVM Model
        prediction_of_svm = svm.predict([fresh_sample_list[index]])

        #variable to store count of Number of models that predicted the Sample as a Rock
        Rock_count = 0
        #variable to store count of Number of models that predicted the Sample as a Mine
        Mine_count = 0

        #Logic to count the number of predictions for rock and mine
        #if the prediction is 'R' then the count of rock is increased else if predictionn is 'M' then the count of mine is increased
        if(prediction_of_LR=='R'):
            Rock_count+=1
        else:
            Mine_count+=1

        if(prediction_of_knn=='R'):
            Rock_count+=1
        else:
            Mine_count+=1

        if(prediction_of_svm=='R'):
            Rock_count+=1
        else:
            Mine_count+=1
        
        #variable to store the final answer based on the Three models
        final_prediction=""

        #if RockCount is more than Mine count then the given sample is Rock else it is Mine
        if(Rock_count>Mine_count):
            final_prediction = "  The given sample is a Rock!!"
        else:
            final_prediction = "  The given sample is a Mine!!"
        
        #IF accuracy if all 3 models is good then only we are going to display the final result
        if(accuracy_knn>0.8 and accuracy_svm>0.8 and accuracy_LR>0.8):
            text = "Sample " + str(index) + " :"
            st.write(text , final_prediction)
        else :
            st.write("Unable to Predict due to Poor Accuracy of Model!!")


#-----END OF CODE for WRITING A FUNCTION TO MAKE PREDICTIONS ON NEW SAMPLE DATA THAT HAS BEEN UPLOADED BY THE USER-----

#*********Designinng the InterFace*************

#code to remove waterMarks from webpage
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

#code to remove link icon for title
st.markdown("""
        <style>
        .css-15zrgzn {display: none}
        .css-eczf16 {display: none}
        .css-jn99sy {display: none}
        </style>
        """, unsafe_allow_html=True)


#title for our WebPage
st.title("Rock Vs Mine Detection using Machine Learning")
st.markdown("<style>h1{font-size: 2rem;}</style>", unsafe_allow_html=True)

#code for uploading a csv file to make prediction
uploaded_file = st.file_uploader("Upload Your Input For Prediction as a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the Excel file into a pandas DataFrame
    uploaded_file_as_dataFrame = pd.read_csv(uploaded_file,header=None)

    # Display the DataFrame in Streamlit
    st.dataframe(uploaded_file_as_dataFrame)

    #calling the predictive function by sending the dataFrame as arguments
    Predict_New_Class(uploaded_file_as_dataFrame)
