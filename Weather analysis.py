#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ==========================================
# WEATHER DATA ANALYSIS & PREDICTION SYSTEM
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# ==========================================
# CREATE WEATHER DATASET
# ==========================================

data = {
    'Day': list(range(1, 31)),

    'Temperature': [
        24,25,26,27,28,29,30,31,30,29,
        28,27,26,25,24,23,22,21,20,19,
        20,21,22,23,24,25,26,27,28,29
    ],

    'Humidity': [
        65,63,60,59,58,57,55,54,56,58,
        60,61,62,64,65,67,68,70,72,73,
        71,69,67,66,64,63,62,60,59,58
    ],

    'Rainfall': [
        2,0,1,0,0,0,0,0,1,2,
        3,2,1,0,0,4,5,6,7,8,
        6,5,4,3,2,1,0,0,0,0
    ],

    'Pollution': [
        110,108,105,102,101,100,99,97,98,100,
        104,106,108,110,112,115,118,120,125,130,
        126,122,118,115,112,110,108,106,104,102
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Remove Missing Values
df.dropna(inplace=True)

# Moving Average
df['Moving_Average'] = df['Temperature'].rolling(3).mean()

# ==========================================
# TRAIN LINEAR REGRESSION MODEL
# ==========================================

X = df[['Humidity', 'Rainfall', 'Pollution']]
y = df['Temperature']

model = LinearRegression()

model.fit(X, y)

# ==========================================
# MENU SYSTEM
# ==========================================

while True:

    print("\n===================================")
    print(" WEATHER ANALYSIS SYSTEM")
    print("===================================")

    print("1. Display Weather Data")
    print("2. Display Insights / Statistics")
    print("3. Detect Anomalies")
    print("4. Visualize Weather Parameters")
    print("5. Train Linear Regression Model")
    print("6. Predict Future Temperature")
    print("7. Interactive Prediction via User Input")
    print("8. Exit")

    choice = input("\nEnter Your Choice: ")

    # ==========================================
    # DISPLAY WEATHER DATA
    # ==========================================

    if choice == '1':

        print("\n===== WEATHER DATA =====")
        print(df)

    # ==========================================
    # DISPLAY INSIGHTS OR STATISTICS
    # ==========================================

    elif choice == '2':

        print("\n===== WEATHER INSIGHTS =====")

        print("\nAverage Temperature:",
              round(df['Temperature'].mean(),2))

        print("Maximum Temperature:",
              df['Temperature'].max())

        print("Minimum Temperature:",
              df['Temperature'].min())

        print("Average Humidity:",
              round(df['Humidity'].mean(),2))

        print("Total Rainfall:",
              df['Rainfall'].sum())

        print("\n===== STATISTICAL SUMMARY =====")
        print(df.describe())

    # ==========================================
    # DETECT ANOMALIES
    # ==========================================

    elif choice == '3':

        print("\n===== ANOMALY DETECTION =====")

        average_temp = df['Temperature'].mean()

        found = False

        for temp in df['Temperature']:

            if temp > average_temp + 5:

                print("High Temperature Anomaly:", temp)
                found = True

        if found == False:

            print("No Major Anomalies Found")

    # ==========================================
    # VISUALIZE WEATHER PARAMETERS
    # ==========================================

    elif choice == '4':

        while True:

            print("\n===== VISUALIZATION MENU =====")

            print("1. Temperature Graph")
            print("2. Humidity Graph")
            print("3. Rainfall Graph")
            print("4. Pollution Graph")
            print("5. Boxplot")
            print("6. Heatmap")
            print("7. Moving Average Graph")
            print("8. Back")

            graph_choice = input("Enter Choice: ")

            # Temperature Graph
            if graph_choice == '1':

                plt.figure(figsize=(10,5))

                plt.plot(df['Day'],
                         df['Temperature'],
                         marker='o')

                plt.title('Temperature Trend')
                plt.xlabel('Day')
                plt.ylabel('Temperature')

                plt.grid(True)

                plt.show()

            # Humidity Graph
            elif graph_choice == '2':

                plt.figure(figsize=(10,5))

                plt.plot(df['Day'],
                         df['Humidity'],
                         color='green')

                plt.title('Humidity Trend')
                plt.xlabel('Day')
                plt.ylabel('Humidity')

                plt.grid(True)

                plt.show()

            # Rainfall Graph
            elif graph_choice == '3':

                plt.figure(figsize=(10,5))

                plt.bar(df['Day'],
                        df['Rainfall'])

                plt.title('Rainfall Graph')
                plt.xlabel('Day')
                plt.ylabel('Rainfall')

                plt.show()

            # Pollution Graph
            elif graph_choice == '4':

                plt.figure(figsize=(10,5))

                plt.plot(df['Day'],
                         df['Pollution'],
                         color='red')

                plt.title('Pollution Trend')
                plt.xlabel('Day')
                plt.ylabel('Pollution')

                plt.grid(True)

                plt.show()

            # Boxplot
            elif graph_choice == '5':

                plt.figure(figsize=(8,5))

                sns.boxplot(
                    data=df[['Temperature',
                             'Humidity',
                             'Rainfall',
                             'Pollution']]
                )

                plt.title('Weather Data Boxplot')

                plt.show()

            # Heatmap
            elif graph_choice == '6':

                plt.figure(figsize=(8,6))

                sns.heatmap(
                    df[['Temperature',
                        'Humidity',
                        'Rainfall',
                        'Pollution']].corr(),
                    annot=True
                )

                plt.title('Correlation Heatmap')

                plt.show()

            # Moving Average Graph
            elif graph_choice == '7':

                plt.figure(figsize=(10,5))

                plt.plot(df['Day'],
                         df['Temperature'],
                         label='Temperature')

                plt.plot(df['Day'],
                         df['Moving_Average'],
                         label='Moving Average',
                         linewidth=3)

                plt.title('Moving Average Temperature')

                plt.xlabel('Day')
                plt.ylabel('Temperature')

                plt.legend()

                plt.grid(True)

                plt.show()

            # Back
            elif graph_choice == '8':

                break

            else:

                print("Invalid Choice")

    # ==========================================
    # TRAIN LINEAR REGRESSION MODEL
    # ==========================================

    elif choice == '5':

        print("\n===== LINEAR REGRESSION MODEL =====")

        print("Model Trained Successfully!")

        print("\nModel uses:")

        print("- Humidity")
        print("- Rainfall")
        print("- Pollution")

        print("to predict Temperature.")

    # ==========================================
    # PREDICT FUTURE TEMPERATURE
    # ==========================================

    elif choice == '6':

        print("\n===== FUTURE TEMPERATURE PREDICTION =====")

        latest_data = [[
            df['Humidity'].iloc[-1],
            df['Rainfall'].iloc[-1],
            df['Pollution'].iloc[-1]
        ]]

        prediction = model.predict(latest_data)

        print("Predicted Next Day Temperature:",
              round(prediction[0],2), "°C")

    # ==========================================
    # INTERACTIVE PREDICTION
    # ==========================================

    elif choice == '7':

        print("\n===== INTERACTIVE PREDICTION =====")

        humidity = float(input("Enter Humidity: "))
        rainfall = float(input("Enter Rainfall: "))
        pollution = float(input("Enter Pollution Level: "))

        user_data = [[humidity, rainfall, pollution]]

        prediction = model.predict(user_data)

        print("\nPredicted Temperature:",
              round(prediction[0],2), "°C")

    # ==========================================
    # EXIT
    # ==========================================

    elif choice == '8':

        print("\nProject Completed Successfully")
        break

    # ==========================================
    # INVALID OPTION
    # ==========================================

    else:

        print("\nInvalid Choice! Please Try Again.")

