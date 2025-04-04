import pandas as pd

# Loading of data
df = pd.read_csv("/content/hotel_bookings.csv")

# Data Cleaning

df.drop('company', inplace=True, axis=1)

df.drop('arrival_date_week_number', inplace=True, axis=1)

df['children'] = df['children'].fillna(df['children'].median())

df['agent'] = df['agent'].fillna(-1)

df.fillna({'country': 'Unknown'}, inplace=True)

df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])

df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'] + '-' + df['arrival_date_day_of_month'].astype(str), errors='coerce')
df['adr'] = df['adr'].astype(float)

df.drop_duplicates(inplace=True)

filter = (df.children == 0) & (df.adults == 0) & (df.babies == 0)
indexguest = df[(df['children'] == 0) & (df['adults'] == 0) & (df['babies'] == 0)].index

df.drop(indexguest, inplace=True)

df.reset_index(drop=True, inplace=True)

df = df[df['lead_time'] >= 0]

df = df[df['adr'] >= 0]

# Saving cleaned data
df.to_csv("data/cleaned_hotel_bookings.csv", index=False)
print("Cleaned data saved to data/cleaned_hotel_bookings.csv")