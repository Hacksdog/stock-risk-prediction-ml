import joblib
import pandas as pd

model = joblib.load("knn_model.pkl")

new_data = [[
    145.20,   # Close_GOOGL  # so here you have to put your own data by your own . cause this is not a automated model 
    147.00,   # High_GOOGL   #also try other saved model 
    143.80,   # Low_GOOGL
    144.50,   # Open_GOOGL
    32000000, # Volume_GOOGL
    0.52      # Z_score
]]

X_new = pd.DataFrame(new_data, columns=[
    "Close_GOOGL","High_GOOGL","Low_GOOGL",
    "Open_GOOGL","Volume_GOOGL","Z_score"
])

prediction = model.predict(X_new)[0]

mapping = {0:"Low Risk",1:"Medium Risk",2:"High Risk"}
print(mapping[int(prediction)])




