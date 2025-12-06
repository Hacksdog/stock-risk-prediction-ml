import joblib
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay

MODEL_PATH = "knn_model.pkl"
TICKER = "GOOGL"

model = joblib.load(MODEL_PATH)

end = datetime.now().date()
start = end - timedelta(days=90)
df = yf.download(TICKER, start=start.isoformat(), end=end.isoformat())

df = df.rename(columns={
    "Open": f"Open_{TICKER}",
    "High": f"High_{TICKER}",
    "Low": f"Low_{TICKER}",
    "Close": f"Close_{TICKER}",
    "Adj Close": f"AdjClose_{TICKER}",
    "Volume": f"Volume_{TICKER}"
})

close_col = f"Close_{TICKER}"
df["Z_score"] = (df[close_col] - df[close_col].mean()) / df[close_col].std()

row = df.iloc[-1]

X_new = pd.DataFrame([[
    row[f"Close_{TICKER}"],
    row[f"High_{TICKER}"],
    row[f"Low_{TICKER}"],
    row[f"Open_{TICKER}"],
    row[f"Volume_{TICKER}"],
    row["Z_score"]
]], columns=[
    f"Close_{TICKER}", f"High_{TICKER}", f"Low_{TICKER}",
    f"Open_{TICKER}", f"Volume_{TICKER}", "Z_score"
])

prediction = model.predict(X_new)[0]

mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

target_date = (row.name + BDay(1)).date()

print(mapping[int(prediction)])
print(target_date)
