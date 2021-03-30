import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from datetime import datetime

import streamlit as st

from model import predictor, outback_df

st.set_page_config(page_title='Outback Predictor',
                    layout='centered',
                    page_icon=':car:')

MAE = 1159.81 # Mean Absolute Error in USD

st.write('''# Subaru Outback price predictor''')
options = st.sidebar.beta_expander("Estimate the value of an Outback", expanded=True)
with options:
    miles_km_col, CAD_USD_col = st.beta_columns(2)

    with miles_km_col:
        in_miles = st.checkbox('Use miles')

    with CAD_USD_col:
        use_USD = st.checkbox('US Dollars')

    miles = st.number_input('Mileage')
    if in_miles:
        pass
    else:
        miles = miles * 0.6213712

    year = st.number_input('Year', min_value=1990, max_value=datetime.now().year, step=1, format='%d')
    if year < 1994:
        model = 'legacy'
    else:
        model = 'outback'

    paint_color = st.selectbox('Color',
                                    ('black',
                                    'blue',
                                    'brown',
                                    'green',
                                    'grey',
                                    'purple',
                                    'red',
                                    'silver',
                                    'white',
                                    'yellow',
                                    'custom',
                                    'other'))

    condition = st.selectbox('Condition',
                                    ('like new',
                                    'excellent',
                                    'good',
                                    'fair',
                                    'no idea'))

    cylinders = st.selectbox('Cylinders',
                                    ('6 cylinders',
                                    '4 cylinders',
                                    'unknown'))

    transmission = st.selectbox('Transmission',
                                    ('automatic',
                                    'manual',
                                    'no idea'))


    title_status = st.selectbox('Title status',
                                    ('clean',
                                    'rebuilt',
                                    'unknown'))


    predict = st.button('Predict!')

    input_data = pd.DataFrame({'year': [year],
                      'model' : [model],
                      'condition':[condition],
                      'cylinders':[cylinders],
                      'fuel' : ['gas'],
                      'miles':[miles],
                      'title_status':[title_status],
                      'transmission':[transmission],
                      'paint_color':[paint_color]})

    predicted_value = predictor.predict(input_data)
    print(predicted_value)

if not predict:
    st.write('## USD vs. Miles')
    sns.set_theme(style='darkgrid')
    fig = sns.relplot(data=outback_df, x='miles', y='USD', hue='year')
    st.pyplot(fig)
    fig = sns.jointplot(data=outback_df, x='miles', y='USD', kind='reg', hue_norm='miles',  truncate=True)
    st.pyplot(fig)

    st.write('## USD vs Year')
    fig = sns.relplot(data=outback_df, x='year', y='USD', hue='miles')
    st.pyplot(fig)

    st.write(' ## Year vs Miles')
    fig = sns.relplot(data=outback_df, x='year', y='miles', hue='USD')
    st.pyplot(fig)

    st.write('## USD vs Condition')
    fig = sns.relplot(data=outback_df[outback_df.condition != 'unknown'], x='miles', y='USD',   hue='condition')
    st.pyplot(fig)

if predict:
    if use_USD:
        st.write(f'''The predicted value for this vehicle is **${round(predicted_value[0])}.00 USD**.''')
        st.write(f'The predicted value range for a vehicle like this is between ${round((predicted_value[0]) + MAE)}.00 USD and ${round((predicted_value[0]) + MAE)}.00 USD')

        st.write(f'''The prediction takes into account cars as old as {round(outback_df.year.min())} that are being sold for parts, as well as cars as new as {round(outback_df.year.max())}. As a result there is quite a bit of variance in the results. Specifically, the "mean absolute error" is ${round(MAE)}.00 USD, which means any prediction estimate could be off by plus/minus that amount. This number is of course much more significant for older cars, whose value is a few thousands.''')

        st.write(f'For this vehicle, the error is **{round(MAE/predicted_value[0]*100)}% of the predicted value**. The lower the error percentage is, the more confident you can be about the precise sell value of the vehicle.')

    else:
        st.write(f'The vehicle is probably worth around **${round(predicted_value[0] * 1.26)}.00 CAD**.')
        st.write(f'The predicted value range for a vehicle like this is between ${round((predicted_value[0]* 1.26) - (MAE * 1.26))}.00 CAD and ${round((predicted_value[0]* 1.26) + (MAE * 1.26))}.00 CAD')

        st.write(f'''The prediction takes into account cars as old as {round(outback_df.year.min())} that are being sold for parts, as well as cars as new as {round(outback_df.year.max())} that are basically brand new. As a result there is quite a bit of variance in the results. Specifically, the "mean absolute error" is ${round(1159.81* 1.26)}.00 CAD, which means any prediction estimate could be off by plus/minus that amount. This number is of course much more significant for older cars, whose value is a few thousands.''')

        st.write(f'For this vehicle, the error is **{round(MAE/predicted_value[0]*100)}% of the predicted value**. The lower the error percentage is, the more confident you can be about the estimated sell value of the vehicle.')

        st.button('Go back to graphs')