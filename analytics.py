import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Loading cleaned data
df1 = pd.read_csv("data/cleaned_hotel_bookings.csv")

# Feature Engineering

df1['revenue'] = df1['adr'] * (df1['stays_in_weekend_nights'] + df1['stays_in_week_nights'])
df1['arrival_date'] = pd.to_datetime(df1[['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']].astype(str).agg('-'.join, axis=1), errors='coerce')

# Analytics Calculation
monthly_revenue = df1.groupby(df1['arrival_date'].dt.to_period('M'))['revenue'].sum()
monthly_revenue_str_keys = {str(period): value for period, value in monthly_revenue.items()}
cancellation_rate = df1['is_canceled'].mean() * 100
country_counts = df1['country'].value_counts().head(10).to_dict()
lead_time_stats = df1['lead_time'].describe().to_dict()
geo_distribution = df1["country"].value_counts()
popular_hotel_type = df1["hotel"].value_counts().idxmax()

analytics = {
    "monthly_revenue": monthly_revenue_str_keys,
    "cancellation_rate": cancellation_rate,
    "geo_distribution": geo_distribution.to_dict(),
    "lead_time_stats": lead_time_stats,
    "top_countries": country_counts,
    "popular_hotel_type": popular_hotel_type,
}

# Saving the analysis

with open("data/analytics.json", "w") as f:
    json.dump(analytics, f)
print("Analytics saved to data/analytics.json")

# Visualizations
# Revenue Trends
plt.figure(figsize=(10, 5))
plt.plot(list(monthly_revenue_str_keys.keys()), list(monthly_revenue_str_keys.values()), marker='o', linestyle='--')
plt.xticks(rotation=45)
plt.title("Revenue Trends Over Time")
plt.xlabel("Month-Year")
plt.ylabel("Total Revenue")
plt.grid(True)
plt.show()

# Cancellation Rate
plt.figure(figsize=(6, 6))
labels = ["Cancelled", "Not Cancelled"]
sizes = [df1["is_canceled"].sum(), len(df1) - df1["is_canceled"].sum()]
plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["red", "cyan"], startangle=90)
plt.title("Booking Cancellation Rate")
plt.show()

# Geographical Distribution
plt.figure(figsize=(12, 6))
top_countries = df1["country"].value_counts().head(10)
sns.barplot(x=top_countries.index, y=top_countries.values, palette="coolwarm")
plt.title("Top 10 Countries by Booking Volume")
plt.xlabel("Country")
plt.ylabel("Number of Bookings")
plt.xticks(rotation=45)
plt.show()

# Lead Time Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df1["lead_time"], bins=30, kde=True)
plt.title('Booking Lead Time Distribution')
plt.xlabel('Days before Booking')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Popular Hotel Type
plt.figure(figsize=(5, 5))
sns.barplot(x=df1["hotel"].value_counts().index, y=df1["hotel"].value_counts().values, palette="coolwarm")
plt.title("Most Popular Hotel Type")
plt.xlabel("Hotel Type")
plt.ylabel("Number of Bookings")
plt.show()