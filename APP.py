import streamlit as st
import pickle
model = pickle.load(open('RF_price_predicting_model.pkl','rb'))

def main():
    string = "Car Price Predictor"
    st.set_page_config(page_title=string, page_icon="🚗") 
    st.title("🚗 Car Price Predictor 🚗")
    st.image(
            "https://tse3.mm.bing.net/th?id=OIP.g65fbVofHlXcFNqdwz5T_gHaEo&pid=Api&P=0&w=290&h=181",
            width=400, # Manually Adjust the width of the image as per requirement
        )
    st.write('')
    st.write('')
    years = st.number_input('Year of Purchased',1990, 2021, step=1, key ='year')
    Years_old = 2021-years

    Present_Price = st.number_input('Present ex-showroom price of the car ?  (In ₹lakhs)', 0.00, 50.00, step=0.5, key ='present_price')

    Kms_Driven = st.number_input('Millege ?', 0.00, 500000.00, step=500.00, key ='drived')

    Owner = st.radio("Previous number of owners ?", (0, 1, 3), key='owner')

    Fuel_Type_Petrol = st.selectbox('Mode of fuel ?',('Petrol','Diesel', 'CNG'), key='fuel')
    if(Fuel_Type_Petrol=='Petrol'):
        Fuel_Type_Petrol=1
        Fuel_Type_Diesel=0
    elif(Fuel_Type_Petrol=='Diesel'):
        Fuel_Type_Petrol=0
        Fuel_Type_Diesel=1
    else:
        Fuel_Type_Petrol=0
        Fuel_Type_Diesel=0

    Seller_Type_Individual = st.selectbox('Dealer / Individual', ('Dealer','Individual'), key='Dealer')
    if(Seller_Type_Individual=='Individual'):
        Seller_Type_Individual=1
    else:
        Seller_Type_Individual=0	

    Transmission_Mannual = st.selectbox('Transmission Type ', ('Manual','Automatic'), key='manual')
    if(Transmission_Mannual=='Mannual'):
        Transmission_Mannual=1
    else:
        Transmission_Mannual=0

    if st.button("Estimate Price", key='predict'):
        try:
            Model = model  #get_model()
            prediction = Model.predict([[Present_Price, Kms_Driven, Owner, Years_old, Fuel_Type_Diesel, Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Mannual]])
            output = round(prediction[0],2)
            if output<0:
                st.warning("You can't sell this car !!")
            else:
                st.success("You can sell the car for {} L  👍".format(output))
                st.success("Thankyou for visiting us".format(output))
                
        except:
            st.warning(" Something went wrong\nTry again")

if __name__ == '__main__':
    main()