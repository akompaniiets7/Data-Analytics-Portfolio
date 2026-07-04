import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from operator import attrgetter

# ======================================================
# Load and prepare data
# ======================================================

df = pd.read_csv("retail_sales_dataset.csv")

# Generate synthetic customer IDs
np.random.seed(42)
customer_pool = [f"CUST{i:03d}" for i in range(1, 101)]
df["Customer ID"] = np.random.choice(customer_pool, size=len(df))

# Convert date column
df["Date"] = pd.to_datetime(df["Date"])

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(df.head())

print("\nDataset statistics:")
print(df.describe())

# ======================================================
# RFM Analysis
# ======================================================

snapshot_date = df["Date"].max() + pd.Timedelta(days=1)

rfm = df.groupby("Customer ID").agg({
    "Date": lambda x: (snapshot_date - x.max()).days,
    "Transaction ID": "count",
    "Total Amount": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

rfm["R_Score"] = pd.qcut(
    rfm["Recency"],
    5,
    labels=[5, 4, 3, 2, 1]
)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"],
    5,
    labels=[1, 2, 3, 4, 5]
)

rfm["RFM_Segment"] = (
    rfm["R_Score"].astype(str)
    + rfm["F_Score"].astype(str)
    + rfm["M_Score"].astype(str)
)

print("\n" + "=" * 60)
print("RFM SEGMENTATION")
print("=" * 60)
print(rfm.head())

print("\nTop customers by Monetary value:")
print(rfm.sort_values("Monetary", ascending=False).head())

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=rfm,
    x="Frequency",
    y="Monetary",
    size="Recency"
)

plt.title("RFM Segmentation")
plt.tight_layout()
plt.show()

# ======================================================
# Cohort Analysis
# ======================================================

df["InvoiceMonth"] = df["Date"].dt.to_period("M")

df["CohortMonth"] = (
    df.groupby("Customer ID")["Date"]
    .transform("min")
    .dt.to_period("M")
)

df["CohortIndex"] = (
    (df["InvoiceMonth"] - df["CohortMonth"])
    .apply(attrgetter("n"))
)

cohort_data = (
    df.groupby(["CohortMonth", "CohortIndex"])["Customer ID"]
      .nunique()
      .reset_index()
)

cohort_counts = cohort_data.pivot(
    index="CohortMonth",
    columns="CohortIndex",
    values="Customer ID"
)

cohort_sizes = cohort_counts.iloc[:, 0]

retention = cohort_counts.divide(
    cohort_sizes,
    axis=0
)

print("\n" + "=" * 60)
print("COHORT RETENTION")
print("=" * 60)
print(retention)

plt.figure(figsize=(12, 8))

sns.heatmap(
    retention,
    annot=True,
    fmt=".0%",
    cmap="Blues"
)

plt.title("Cohort Retention Analysis")
plt.tight_layout()
plt.show()

# ======================================================
# Lean Analytics Metrics
# ======================================================

active_users = df["Customer ID"].nunique()

average_revenue = df["Total Amount"].mean()

repeat_customers = (
    df.groupby("Customer ID")
      .size()
      .gt(1)
      .sum()
)

print("\n" + "=" * 60)
print("LEAN ANALYTICS METRICS")
print("=" * 60)

print(f"Active Users      : {active_users}")
print(f"Average Revenue   : {average_revenue:.2f}")
print(f"Repeat Customers  : {repeat_customers}")

# ======================================================
# Sales Analysis
# ======================================================

sales_by_category = (
    df.groupby("Product Category")["Total Amount"]
      .sum()
      .sort_values()
)

plt.figure(figsize=(10, 6))

sales_by_category.plot(kind="bar")

plt.title("Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Revenue")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Gender",
    y="Total Amount"
)

plt.title("Revenue Distribution by Gender")

plt.tight_layout()
plt.show()