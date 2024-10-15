import streamlit as st
import numpy as np
import pickle


model = pickle.load(open("C:/Users/srika/PGA36/Projects/Capstone project 2/Trained_Model.sav",'rb'))


def Dropout_Prediction(input_data):

    input_data_as_numpy_array = np.asarray(input_data,dtype=np.float64)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
        return 'Dropout'
    elif (prediction[0]==1):
        return 'Enrolled'
    else:
        return 'Graduate'

def main():
    st.title("Student Dropout Prediction")

    Course = st.text_input("Course")
    Tuition_fees_up_to_date = st.text_input("Tuition fees up to date")
    No_of_Subjects_Passed_in_First_Sem = st.text_input("No of Subjects Passed in First Sem")
    No_of_Subjects_Passed_in_Second_Sem = st.text_input("No of Subjects Passed in Second Sem")
    First_Sem_Grade = st.text_input("First Sem Grade")
    Second_Sem_Grade = st.text_input("Second Sem Grade")
    First_Sem_Projects_Grade = st.text_input("First Sem Projects Grade")



    diagnosis=''


    if st.button('Dropout Prediction'):
        diagnosis=Dropout_Prediction([Course, Tuition_fees_up_to_date, No_of_Subjects_Passed_in_First_Sem, No_of_Subjects_Passed_in_Second_Sem, First_Sem_Grade, Second_Sem_Grade, First_Sem_Projects_Grade])
        st.success(diagnosis)
    


if __name__=='__main__':
    main()



