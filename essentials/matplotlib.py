# Function : Purpose
# plt.plot() Line chart
# plt.bar() Bar chart
# plt.scatter() Scatter plot
# plt.hist() Histogram
# plt.xlabel() X-axis label
# plt.ylabel() Y-axis label
# plt.title() Chart title
# plt.show() Display chart

# Seaborn makes visualising Pandas DataFrames much easier
# hue allows you to distinguish data points using different colours based on another variable.
# bins -> number of intervals
# kde=True -> adds a smooth density curve to help visualise the distribution

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


house_size = [500, 700, 900, 1100, 1300]
house_price = [300000, 400000, 500000, 600000, 750000]

plt.scatter(house_size, house_price)
plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price ($)")
plt.title("House Size vs House Price")
plt.show()

prices = [300, 350, 400, 450, 450, 500, 550, 600, 700, 900]
plt.hist(prices, bins=5)
plt.xlabel("House Price ($ thousands)")
plt.ylabel("Frequency")
plt.title("Distribution of House Prices")
plt.show()


data = {
    "house_size": [500, 700, 900, 1100, 1300],
    "bedrooms": [1, 2, 2, 3, 4],
    "house_price": [300000, 400000, 500000, 600000, 750000]
}

df = pd.DataFrame(data)

# Scatter Plot
sns.scatterplot(
    data=df,
    x="house_size",
    y="house_price",
    hue="bedrooms"
)

plt.title("House Size VS House Price")
plt.show()

# Histogram
sns.histplot(
    data=df,
    x="house_price",
    bins=5,
    kde=True
)

plt.title("House Price Distribution")
plt.show()

# Box Plot
sns.boxplot(
    data=df,
    y="house_price"
)

plt.title("House Price Box Plot")
plt.show()

# Heatmap -- important
correlation = df.corr()
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)
plt.title("Feature Correlation")
plt.show()
