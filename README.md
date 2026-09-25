
KIAAR AgroWeather - Localized 7-Day Temperature Forecast
Code by Ishika Vyas COMPS TYB 47
----------------------------------------------------------

**Improved pipeline:**
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

**IMPORTANT:**

This version automatically downloads an independent
ERA5-Land reanalysis reference through the Open-Meteo API.

ERA5-Land is NOT a physical KIAAR station measurement.
It is a gridded reanalysis/reference dataset.

**STEPS TO RUN THE PROGRAM:**

Install:
    pip install pandas numpy requests scikit-learn xgboost

Run:
    python model.py

Force retraining:
    python model.py --retrain

For another location, change LATITUDE/LONGITUDE and REFERENCE_MODEL.



----------------------------------------------------------------------------
**LIVE FORECAST OUTPUT**:

The program generates:

Current temperature,
Next 7 days of temperature forecast,
Daily minimum temperature,
Daily maximum temperature,
Daily mean temperature,
ML-corrected final temperature,
Selected correction model for each forecast day,
Correction status for each lead day.


Example output format:
<img width="1355" height="822" alt="image" src="https://github.com/user-attachments/assets/77f47213-f765-42a7-bc2e-4766f2d1a700" />

<img width="1312" height="842" alt="image" src="https://github.com/user-attachments/assets/26a6f670-8b30-4462-9a95-9343bc678042" />

