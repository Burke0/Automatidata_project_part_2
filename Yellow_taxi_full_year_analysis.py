# Import packages and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
# Load dataset into dataframe
df = pd.read_csv('2017_Yellow_Taxi_Trip_Data.csv')

# Assess size, shape, and makeuup of data set
print(df.head(10))
print(df.shape)
print(df.info())
print(df.describe())

# Convert date columns to Datetime
df['tpep_pickup_datetime']=pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime']=pd.to_datetime(df['tpep_dropoff_datetime'])

# Create box plot of trip_distance
plt.figure(figsize=(7,2))
plt.title('Trip Distance')
sns.boxplot(data=None, x=df['trip_distance'], fliersize=1)
plt.show()

# Create histogram of trip distance
plt.figure(figsize=(10,5))
sns.histplot(df['trip_distance'], bins=range(0, 26, 1))
plt.title('Trip distance Histogram')
plt.show()
# note: The majority of trips were journeys of less than two miles. The number of trips falls away steeply as the distance traveled increases beyond two miles.

# Create box plot of total_amount
plt.figure(figsize=(7,2))
plt.title('Total Amount')
sns.boxplot(x=df['total_amount'], fliersize=1)
plt.show()

# Create histogram of total_amount
plt.figure(figsize=(12,6))
ax = sns.histplot(df['total_amount'], bins=range(-10,101,5))
ax.set_xticks(range(-10, 101, 5))
ax.set_xticklabels(range(-10,101,5))
plt.title('Total Amount Histogram')
plt.show()
# note: The total cost of each trip also has a distribution that skews right, with most costs falling in the $5-15 range.

# Create box plot of tip_amount
plt.figure(figsize=(7,2))
plt.title('Tip Amount')
sns.boxplot(x=df['tip_amount'], fliersize=1)
plt.show()

# Create histogram of tip_amount
plt.figure(figsize=(12,6))
ax = sns.histplot(df['tip_amount'], bins=range(0,21,1))
ax.set_xticks(range(0,21,2))
plt.title('Tip Amount Histogram')
plt.show()
# note: The distribution for tip amount is right-skewed, with nearly all the tips in the $0-3 range.

# Create histogram of tip_amount by vendor
plt.figure(figsize=(12,7))
ax = sns.histplot(data = df, x='tip_amount', bins=range(0,21,1),
                  hue = 'VendorID',
                  multiple = 'stack',
                  palette='pastel')
ax.set_xticks(range(0,21,1))
ax.set_xticklabels(range(0,21,1))
plt.title('Tip amount by vendor histogram')
plt.show()
# Separating the tip amount by vendor reveals that there are no noticeable aberrations in the distribution of tips between the two vendors in the dataset. 
# Vendor two has a slightly higher share of the rides, and this proportion is approximately maintained for all tip amounts.

# Create histogram of tip_amount by vendor for tips > $10
tips_over_ten=df[df['tip_amount'] > 10]
plt.figure(figsize=(12,7))
ax = sns.histplot(data=tips_over_ten, x='tip_amount', bins=range(10,21,1),
                  hue='VendorID',
                  multiple='stack',
                  palette='pastel')
ax.set_xticks(range(10,21,1))
ax.set_xticklabels(range(10,21,1))
plt.title('Tip amount by vendor histogram')
plt.show()
# note: The proportions are maintained even at these higher tip amounts, with the exception being at highest extremity, 
# but this is not noteworthy due to the low sample size at these tip amounts.

# Examine unique values in the passenger_count column 
print(df['passenger_count'].value_counts())
# note: Nearly two thirds of the rides were single occupancy, though there were still nearly 700 rides with as many as six passengers. 
# Also, there are 33 rides with an occupancy count of zero, which doesn't make sense. 
# These would likely be dropped unless a reasonable explanation can be found for them.

# Calculate mean tips by passenger count
invalid_tip_rows = df[df['tip_amount'].isna()]
print(invalid_tip_rows)
mean_tip_amount = df['tip_amount'].mean()
df['tip_amount'].fillna(mean_tip_amount, inplace=True)
mean_tips_by_passenger_count = df.groupby('passenger_count')['tip_amount'].mean()

print(mean_tips_by_passenger_count)

# Create bar plot for mean tips by passenger count
data = df[df['passenger_count'] != 0]

mean_tips = data.groupby('passenger_count')['tip_amount'].mean()
global_mean_tip = data['tip_amount'].mean()
mean_tips.plot(kind='bar', figsize=(10, 6))
plt.axhline(global_mean_tip, color='red', linestyle='--', label="Global Mean")
plt.xlabel('Passenger Count')
plt.ylabel('Mean Tips')
plt.title('Mean Tips by Passenger Count')
plt.legend(labels=["Global Mean", "Tip Amount"])
plt.show()

# Create a month column
df['month'] = df['tpep_pickup_datetime'].dt.month_name()

# Create a day column
df['day'] = df['tpep_pickup_datetime'].dt.day_name()

# Get total number of rides for each month
monthly_rides = df['month'].value_counts()

# Reorder the monthly ride list so months go in order
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December']

monthly_rides = monthly_rides.reindex(index=month_order)
print(monthly_rides)

# Create a bar plot of total rides per month
plt.figure(figsize=(12,7))
ax = sns.barplot(x=monthly_rides.index, y=monthly_rides)
ax.set_xticklabels(month_order)
plt.title('Ride Count by Month', fontsize=16)
plt.show()
# note: Monthly rides are fairly consistent, with notable dips in the summer months of July, August, and September, and also in February.

# Repeat the above process, this time for rides by day
daily_rides = df['day'].value_counts()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_rides = daily_rides.reindex(index=day_order)
print(daily_rides)

# Create bar plot for ride count by day
plt.figure(figsize=(12,7))
ax = sns.barplot(x=daily_rides.index, y=daily_rides)
ax.set_xticklabels(day_order)
ax.set_ylabel('Count')
plt.title('Ride Count by Day', fontsize=16)
plt.show()
# note: Suprisingly, Wednesday through Saturday had the highest number of daily rides, while Sunday and Monday had the least.


# Repeat the process, this time for total revenue by day

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
total_amount_day = df.groupby('day').sum(numeric_only=True)[['total_amount']]
total_amount_day = total_amount_day.reindex(index=day_order)
print(total_amount_day)

# Create bar plot of total revenue by day
plt.figure(figsize=(12,7))
ax = sns.barplot(x=total_amount_day.index, y=total_amount_day['total_amount'])
ax.set_xticklabels(day_order)
ax.set_ylabel('Revenue (USD)')
plt.title('Total revenue by day', fontsize=16)
plt.show()
# note: Thursday had the highest gross revenue of all days, and Sunday and Monday had the least. 
# Interestingly, although Saturday had only 35 fewer rides than Thursday, its gross revenue was ~$6,000 less than Thursday's—more than a 10% drop.

# Repeat the process, this time for total revenue by month
total_amount_month = df.groupby('month').sum(numeric_only=True)[['total_amount']]
total_amount_month = total_amount_month.reindex(index=month_order)
print(total_amount_month)

# Create a bar plot of total revenue by month
plt.figure(figsize=(12,7))
ax = sns.barplot(x=total_amount_month.index, y=total_amount_month['total_amount'])
plt.title('Total revenue by month', fontsize=16)
plt.show()
# note: Monthly revenue generally follows the pattern of monthly rides, with noticeable dips in the summer months of July, August, and September, and also one in February.

# Get number of unique drop-off location IDs
df['DOLocationID'].nunique()

# Calculate the mean trip distance for each drop-off location
distance_by_dropoff = df.groupby('DOLocationID').mean(numeric_only=True)[['trip_distance']]

# Sort the results in descending order by mean trip distance
distance_by_dropoff = distance_by_dropoff.sort_values(by='trip_distance')
distance_by_dropoff 

# Create a bar plot of mean trip distances by drop-off location in ascending order by distance
plt.figure(figsize=(14,6))
ax = sns.barplot(x=distance_by_dropoff.index, 
                 y=distance_by_dropoff['trip_distance'],
                 order=distance_by_dropoff.index)
ax.set_xticklabels([])
ax.set_xticks([])
plt.title('Mean trip distance by drop-off location', fontsize=16)
plt.show()
# note: This plot presents a characteristic curve related to the cumulative density function of a normal distribution. 
# In other words, it indicates that the drop-off points are relatively evenly distributed over the terrain. 
# This is good to know, because geographic coordinates were not included in this dataset, so there was no obvious way to test for the distibution of locations.

# Check if all drop-off locations are consecutively numbered
print(df['DOLocationID'].max() - len(set(df['DOLocationID'])))
# Exemplar note: There are 49 numbers that do not represent a drop-off location.

# To eliminate the spaces in the historgram that these missing numbers would create, sort the unique drop-off location values, then convert them to strings. This will make the histplot function display all bars directly next to each other.
# Create histogram of rides by drop of location
plt.figure(figsize=(16,4))
# DOLocationID column is numeric, so sort in ascending order
sorted_dropoffs = df['DOLocationID'].sort_values()
# Convert to string
sorted_dropoffs = sorted_dropoffs.astype('str')
sns.histplot(sorted_dropoffs, bins=range(0, df['DOLocationID'].max()+1, 1))
plt.xticks([])
plt.xlabel('Drop-off locations')
plt.title('Histogram of rides by drop-off location', fontsize=16)
plt.show()
# note: Notice that out of the 200+ drop-off locations, a disproportionate number of locations receive the majority of the traffic, while all the rest get relatively few trips. 
# It's likely that these high-traffic locations are near popular tourist attractions like the Empire State Building or Times Square, airports, and train and bus terminals. 
# However, it would be helpful to know the location that each ID corresponds with. Unfortunately, this is not in the data.

######### Key Takeaways #########
# The highest distribution of trip distances are below 5 miles, but there are outliers all the way out to 35 miles. There are no missing values.
# There are several trips that have a trip distance of "0.0." What might those trips be? Will they impact our model? 
# The data includes dropoff and pickup times. We can use that information to derive a trip duration for each line of data. This would likely be something that will help the client with their model. 
 
# df['trip_duration'] = (df['tpep_dropoff_datetime']-df['tpep_pickup_datetime'])