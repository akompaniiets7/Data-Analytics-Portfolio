import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

# ==========================================================
# Marketing Campaign Performance Analysis
# ==========================================================

DATA_PATH = "control_group.csv"
FORECAST_DAYS = 3

# ==========================================================
# Load and preprocess data
# ==========================================================

df = pd.read_csv(DATA_PATH, sep=";")

df.columns = df.columns.str.strip()

df.rename(columns={
    "Campaign Name": "campaign",
    "Date": "date",
    "Spend [USD]": "spend",
    "# of Impressions": "impressions",
    "Reach": "reach",
    "# of Website Clicks": "clicks",
    "# of Searches": "searches",
    "# of View Content": "views",
    "# of Add to Cart": "cart",
    "# of Purchase": "purchase"
}, inplace=True)

df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df.fillna(0, inplace=True)

# ==========================================================
# KPI calculation
# ==========================================================

df["CR"] = df["purchase"] / df["clicks"]
df["CR"] = df["CR"].replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=["CR"])

campaign_metrics = df.groupby("campaign").agg({
    "clicks": "sum",
    "impressions": "sum",
    "purchase": "sum",
    "spend": "sum"
})

campaign_metrics["CTR"] = (
    campaign_metrics["clicks"] /
    campaign_metrics["impressions"]
)

campaign_metrics["CR"] = (
    campaign_metrics["purchase"] /
    campaign_metrics["clicks"]
)

campaign_metrics["ROI"] = (
    campaign_metrics["purchase"] -
    campaign_metrics["spend"]
) / campaign_metrics["spend"]

print("\n=== Aggregated Campaign Metrics ===")
print(campaign_metrics)

print("\n=== KPI Summary ===")
print(campaign_metrics[["CTR", "CR", "ROI"]])

# ==========================================================
# A/B testing
# ==========================================================

control_group = df[
    df["campaign"] == "Control Campaign"
]["CR"].dropna()

test_group = df[
    df["campaign"] == "Test Campaign"
]["CR"].dropna()

t_statistic, p_value = stats.ttest_ind(
    control_group,
    test_group,
    equal_var=False
)

print("\n=== A/B Test ===")
print(f"T-statistic : {t_statistic:.4f}")
print(f"P-value     : {p_value:.6f}")

# ==========================================================
# Funnel analysis
# ==========================================================

funnel = df.groupby("campaign")[
    ["clicks", "views", "cart", "purchase"]
].sum()

stages = ["clicks", "views", "cart", "purchase"]

y = np.arange(len(stages))
bar_width = 0.35

control = funnel.loc["Control Campaign"]
test = funnel.loc["Test Campaign"]

plt.figure(figsize=(8, 5))

plt.barh(
    y - bar_width / 2,
    control[stages],
    height=bar_width,
    label="Control"
)

plt.barh(
    y + bar_width / 2,
    test[stages],
    height=bar_width,
    label="Test"
)

plt.yticks(y, stages)
plt.gca().invert_yaxis()

plt.title("Marketing Funnel Comparison")
plt.xlabel("Number of Users")
plt.legend()
plt.tight_layout()

plt.show()

# ==========================================================
# Purchase dynamics
# ==========================================================

daily = df.groupby("date").sum(numeric_only=True)
daily = daily.asfreq("D")

plt.figure(figsize=(9, 4))

daily["purchase"].plot()

plt.title("Daily Purchases")
plt.xlabel("Date")
plt.ylabel("Purchases")
plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================================================
# ARIMA forecasting
# ==========================================================

time_series = daily["purchase"]

model = ARIMA(time_series, order=(1, 1, 1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=FORECAST_DAYS)

print(f"\n=== {FORECAST_DAYS}-Day Forecast ===")
print(forecast)

plt.figure(figsize=(9, 4))

time_series.plot(label="Actual")
forecast.plot(label="Forecast")

plt.title("Purchase Forecast Using ARIMA")
plt.xlabel("Date")
plt.ylabel("Purchases")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()