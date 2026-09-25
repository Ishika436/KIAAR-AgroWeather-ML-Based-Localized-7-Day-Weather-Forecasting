KIAAR AgroWeather - Localized 7-Day Temperature Forecast
Code by Ishika Vyas 
----------------------------------------------------------

Improved pipeline:
1. ECMWF IFS forecast from Open-Meteo Previous Runs API
2. Automatic ERA5-Land historical reference temperature
3. Residual/bias correction: reference temperature - ECMWF_temp
4. Compare:
      - Linear Regression
      - Random Forest
      - Gradient Boosting
      - AdaBoost
      - XGBoost
5. Chronological train/validation/test split
6. Model selection + correction gate are decided ONLY on validation data
7. Test data is kept untouched for final evaluation
8. Separate correction model for each lead day (1..7)
9. Extra time/weather/lag features
10. Final production model is retrained on train + validation data
11. Live 7-day ECMWF forecast is corrected only for lead days
    where validation proved that correction helps.

IMPORTANT:

This version automatically downloads an independent
ERA5-Land reanalysis reference through the Open-Meteo API.

ERA5-Land is NOT a physical KIAAR station measurement.
It is a gridded reanalysis/reference dataset.


STEPS TO RUN THE PROGRAM:

Install:
    pip install pandas numpy requests scikit-learn xgboost

Run:
    python kiaar_best_forecast.py

Force retraining:
    python kiaar_best_forecast.py --retrain

For another location, change LATITUDE/LONGITUDE and REFERENCE_MODEL
