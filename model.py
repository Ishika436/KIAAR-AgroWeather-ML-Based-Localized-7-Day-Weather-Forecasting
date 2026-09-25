"""
KIAAR AgroWeather - Localized 7-Day Temperature Forecast
Code by Ishika Vyas COMPS TYB 47
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

For another location, change LATITUDE/LONGITUDE and REFERENCE_MODEL.

"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# ================================================================
# CONFIGURATION
# ================================================================

LATITUDE = 16.6489
LONGITUDE = 75.0481

SITE_NAME = "KIAAR Agro Farms"

# Put your REAL local observations here.
# No manual observation CSV is required.
# Historical reference observations are downloaded automatically
# from Open-Meteo ERA5-Land.
REFERENCE_MODEL = "era5_land"

NWP_MODEL = "ecmwf_ifs025"

HISTORY_START = "2024-01-01"
MAX_LEAD = 7

MODEL_FILE = "kiaar_best_models.json"
META_FILE = "kiaar_forecast_meta.json"
FORECAST_FILE = "kiaar_7day_forecast.csv"
EVALUATION_FILE = "kiaar_model_comparison.csv"

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ================================================================
# MAIN PIPELINE
# ================================================================

class KiaarAgroWeather:

    def __init__(self):
        self.weather_vars = list(WEATHER_VARS)
        self.feature_cols = []
        self.selected_models = {}
        self.use_correction = {}
        self.best_params = {}
        self.correction_bounds = {}

    # ------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------

    def get_with_retry(self, url, params, timeout=90, retries=5):
        """Request with retry/backoff to reduce temporary API failures."""
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=timeout,
                )

                if response.status_code == 200:
                    return response

                reason = ""
                try:
                    reason = response.json().get("reason", "")
                except Exception:
                    pass

                logger.warning(
                    "HTTP %s %s (attempt %s/%s)",
                    response.status_code,
                    reason,
                    attempt,
                    retries,
                )

                # 400 usually means the request itself is invalid.
                if response.status_code == 400:
                    return None

            except requests.RequestException as exc:
                logger.warning(
                    "Request error: %s (attempt %s/%s)",
                    exc,
                    attempt,
                    retries,
                )

            time.sleep(min(2 * attempt, 10))

        return None

    # ------------------------------------------------------------
    # TIME FEATURES
    # ------------------------------------------------------------

    @staticmethod
    def add_calendar_features(df):
        df = df.copy()

        t = pd.to_datetime(df["time"])

        df["hour"] = t.dt.hour
        df["dayofweek"] = t.dt.dayofweek
        df["month"] = t.dt.month
        df["dayofyear"] = t.dt.dayofyear

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["doy_sin"] = np.sin(
            2 * np.pi * df["dayofyear"] / 365.25
        )
        df["doy_cos"] = np.cos(
            2 * np.pi * df["dayofyear"] / 365.25
        )

        return df

    # ------------------------------------------------------------
    # AUTOMATIC HISTORICAL REFERENCE DATA
    # ------------------------------------------------------------

    def fetch_reference_observations(self):
        """
        Automatically download historical ERA5-Land temperature.

        This removes the need for observations.csv.

        IMPORTANT:
        ERA5-Land is a reanalysis/reference dataset, not a physical
        KIAAR weather-station measurement. It is used as an independent
        historical reference for bias correction.
        """

        end_date = (
            datetime.now().date()
            - timedelta(days=6)
        )

        start_date = datetime.strptime(
            HISTORY_START,
            "%Y-%m-%d",
        ).date()

        logger.info(
            "Downloading automatic historical reference data "
            "(ERA5-Land): %s to %s",
            start_date,
            end_date,
        )

        # Keep the reference dataset focused on variables needed
        # for training and live lag features.
        reference_vars = [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
            "wind_speed_10m",
            "cloud_cover",
            "shortwave_radiation",
        ]

        archive_url = (
            "https://archive-api.open-meteo.com/v1/archive"
        )

        chunks = []
        current = start_date

        while current <= end_date:

            # Smaller chunks reduce API response size and make
            # temporary failures easier to recover from.
            chunk_end = min(
                current + timedelta(days=89),
                end_date,
            )

            logger.info(
                "Reference data: %s to %s",
                current,
                chunk_end,
            )

            params = {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "start_date": current.strftime("%Y-%m-%d"),
                "end_date": chunk_end.strftime("%Y-%m-%d"),
                "hourly": ",".join(reference_vars),
                "models": REFERENCE_MODEL,
                "timezone": "auto",
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
                "precipitation_unit": "mm",
            }

            response = self.get_with_retry(
                archive_url,
                params,
                timeout=90,
                retries=5,
            )

            if response is None:
                logger.error(
                    "Could not download reference block "
                    "%s to %s",
                    current,
                    chunk_end,
                )
            else:
                try:
                    payload = response.json()

                    if "hourly" not in payload:
                        logger.error(
                            "Reference API returned no hourly data."
                        )
                    else:
                        chunk = pd.DataFrame(
                            payload["hourly"]
                        )

                        if not chunk.empty:
                            chunks.append(chunk)
                            logger.info(
                                "Received %s reference rows.",
                                len(chunk),
                            )

                except Exception as exc:
                    logger.error(
                        "Reference data parsing failed: %s",
                        exc,
                    )

            time.sleep(1.5)
            current = chunk_end + timedelta(days=1)

        if not chunks:
            raise RuntimeError(
                "No historical reference data could be downloaded."
            )

        obs = pd.concat(
            chunks,
            ignore_index=True,
        )

        obs["time"] = pd.to_datetime(
            obs["time"]
        )

        # Remove duplicate timestamps caused by overlapping blocks.
        obs = (
            obs.sort_values("time")
            .drop_duplicates("time")
            .reset_index(drop=True)
        )

        required = [
            "time",
            "temperature_2m",
        ]

        missing = [
            col
            for col in required
            if col not in obs.columns
        ]

        if missing:
            raise RuntimeError(
                "Reference API is missing required columns: "
                + str(missing)
            )

        logger.info(
            "Automatic reference data loaded: %s rows",
            len(obs),
        )

        logger.info(
            "Reference period: %s to %s",
            obs["time"].min(),
            obs["time"].max(),
        )

        return obs

    # ------------------------------------------------------------
    # HISTORICAL ECMWF FORECASTS
    # ------------------------------------------------------------

    @staticmethod
    def strip_model_suffix(mapping, model_name):
        suffix = f"_{model_name}"

        return {
            (
                key[:-len(suffix)]
                if key.endswith(suffix)
                else key
            ): value
            for key, value in mapping.items()
        }

    def fetch_history(self):
        """
        Download ECMWF Previous Runs data.

        The returned temperature_2m is FORECAST data only.
        It is NOT used as truth.
        Real observations are merged later.
        """

        end_date = (
            datetime.now().date()
            - timedelta(days=8)
        )

        start_date = datetime.strptime(
            HISTORY_START,
            "%Y-%m-%d",
        ).date()

        logger.info(
            "Fetching ECMWF history: %s to %s",
            start_date,
            end_date,
        )

        requested = ["temperature_2m"]

        for variable in self.weather_vars:
            for lead in range(1, MAX_LEAD + 1):
                requested.append(
                    f"{variable}_previous_day{lead}"
                )

        chunks = []

        current = start_date

        while current <= end_date:

            chunk_end = min(
                current + timedelta(days=89),
                end_date,
            )

            logger.info(
                "Downloading %s to %s",
                current,
                chunk_end,
            )

            params = {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "start_date": current.strftime("%Y-%m-%d"),
                "end_date": chunk_end.strftime("%Y-%m-%d"),
                "hourly": ",".join(requested),
                "models": NWP_MODEL,
                "timezone": "auto",
            }

            response = self.get_with_retry(
                PREVIOUS_RUNS_URL,
                params,
            )

            if response is not None:
                try:
                    payload = response.json()

                    if "hourly" not in payload:
                        logger.warning(
                            "No hourly section in response."
                        )
                    else:
                        hourly = self.strip_model_suffix(
                            payload["hourly"],
                            NWP_MODEL,
                        )

                        chunk = pd.DataFrame(hourly)

                        if not chunk.empty:
                            chunks.append(chunk)
                            logger.info(
                                "Received %s rows",
                                len(chunk),
                            )

                except Exception as exc:
                    logger.warning(
                        "Could not parse response: %s",
                        exc,
                    )

            time.sleep(1.5)
            current = chunk_end + timedelta(days=1)

        if not chunks:
            raise RuntimeError(
                "No historical ECMWF data was downloaded."
            )

        df = pd.concat(
            chunks,
            ignore_index=True,
        )

        df["time"] = pd.to_datetime(df["time"])

        df = (
            df.sort_values("time")
            .drop_duplicates("time")
            .reset_index(drop=True)
        )

        return df

    # ------------------------------------------------------------
    # TRAINING DATA
    # ------------------------------------------------------------

    def build_training_data(self, forecast_df, obs_df):
        """
        Build one row per VALID TIME per lead day.

        Target:
            residual = observation - ECMWF forecast

        This is the key bias-correction target.
        """

        frames = []

        available_vars = [
            v for v in self.weather_vars
            if f"{v}_previous_day1" in forecast_df.columns
        ]

        if "temperature_2m" not in available_vars:
            raise ValueError(
                "ECMWF temperature forecast is missing."
            )

        self.weather_vars = available_vars

        for lead in range(1, MAX_LEAD + 1):

            row = pd.DataFrame({
                "time": forecast_df["time"],
                "lead_days": lead,
            })

            for variable in self.weather_vars:
                source_col = (
                    f"{variable}_previous_day{lead}"
                )

                row[f"fc_{variable}"] = (
                    forecast_df[source_col]
                    if source_col in forecast_df.columns
                    else np.nan
                )

            # Merge REAL observation at the same valid time.
            row = row.merge(
                obs_df[
                    ["time", "temperature_2m"]
                ].rename(
                    columns={
                        "temperature_2m":
                        "actual_temp"
                    }
                ),
                on="time",
                how="inner",
            )

            frames.append(row)

        data = pd.concat(
            frames,
            ignore_index=True,
        )

        data = self.add_calendar_features(data)

        # --------------------------------------------------------
        # Additional physically useful features
        # --------------------------------------------------------

        data["temp_humidity_interaction"] = (
            data["fc_temperature_2m"]
            * data["fc_relative_humidity_2m"]
            if "fc_relative_humidity_2m" in data
            else 0.0
        )

        if "fc_temperature_2m" in data:
            data["temp_squared"] = (
                data["fc_temperature_2m"] ** 2
            )

        if "fc_wind_speed_10m" in data:
            data["wind_squared"] = (
                data["fc_wind_speed_10m"] ** 2
            )

        # --------------------------------------------------------
        # IMPORTANT:
        # Do not use historical observation lags as ML input here.
        # They are not reliably available at live forecast time
        # (ERA5-Land is delayed/reanalysis data). Using them in
        # training but not having the same information live creates
        # train/live feature mismatch.
        # --------------------------------------------------------

        # --------------------------------------------------------
        # Residual target
        # --------------------------------------------------------

        data["target_residual"] = (
            data["actual_temp"]
            - data["fc_temperature_2m"]
        )

        data = data.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # Remove rows without required values.
        # We keep only features that actually exist.
        self.feature_cols = [
            "lead_days",
            "fc_temperature_2m",
            "fc_relative_humidity_2m",
            "fc_precipitation",
            "fc_surface_pressure",
            "fc_wind_speed_10m",
            "fc_cloud_cover",
            "fc_shortwave_radiation",
            "hour_sin",
            "hour_cos",
            "doy_sin",
            "doy_cos",
            "dayofweek",
            "temp_humidity_interaction",
            "temp_squared",
            "wind_squared",
        ]

        self.feature_cols = [
            col
            for col in self.feature_cols
            if col in data.columns
        ]

        data = data.dropna(
            subset=(
                self.feature_cols
                + [
                    "actual_temp",
                    "fc_temperature_2m",
                    "target_residual",
                ]
            )
        )

        data = (
            data.sort_values("time")
            .reset_index(drop=True)
        )

        logger.info(
            "Training rows: %s",
            len(data),
        )

        logger.info(
            "Features: %s",
            len(self.feature_cols),
        )

        if len(data) < 5000:
            logger.warning(
                "Training data is relatively small. "
                "More observations will usually improve reliability."
            )

        return data

    # ------------------------------------------------------------
    # MODELS
    # ------------------------------------------------------------

    def make_models(self):
        """
        Multiple regression models.

        We do NOT assume XGBoost is automatically the best.
        Validation data chooses the model separately for each lead day.
        """

        models = {

            "Ridge": make_pipeline(
                StandardScaler(),
                Ridge(alpha=10.0),
            ),

            "RandomForest": RandomForestRegressor(
                n_estimators=500,
                max_depth=12,
                min_samples_leaf=3,
                max_features=0.8,
                random_state=42,
                n_jobs=-1,
            ),

            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=10,
                loss="huber",
                random_state=42,
            ),

            "AdaBoost": AdaBoostRegressor(
                n_estimators=400,
                learning_rate=0.03,
                loss="square",
                random_state=42,
            ),

    "XGBoost": XGBRegressor(
                n_estimators=1500,
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=8,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=1.5,
                objective="reg:pseudohubererror",
                eval_metric="mae",
                random_state=42,
                n_jobs=-1,
            ),
        }

        return models

    # ------------------------------------------------------------
    # SPLIT
    # ------------------------------------------------------------

    def chronological_split(self, data):
        """
        Time split:
            68% train
            12% validation
            20% final test

        No random shuffle.
        """

        unique_times = np.sort(
            data["time"].unique()
        )

        n = len(unique_times)

        train_end = int(n * 0.68)
        val_end = int(n * 0.80)

        train_cut = unique_times[train_end]
        val_cut = unique_times[val_end]

        train = data[
            data["time"] < train_cut
        ].copy()

        validation = data[
            (data["time"] >= train_cut)
            & (data["time"] < val_cut)
        ].copy()

        test = data[
            data["time"] >= val_cut
        ].copy()

        logger.info(
            "TRAIN: %s -> %s rows",
            train["time"].min(),
            len(train),
        )

        logger.info(
            "VALIDATION: %s -> %s rows",
            validation["time"].min(),
            len(validation),
        )

        logger.info(
            "TEST: %s -> %s rows",
            test["time"].min(),
            len(test),
        )

        return train, validation, test

    # ------------------------------------------------------------
    # BASELINES
    # ------------------------------------------------------------

    @staticmethod
    def mae(y_true, y_pred):
        return float(
            mean_absolute_error(y_true, y_pred)
        )

    @staticmethod
    def rmse(y_true, y_pred):
        return float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        )

    def persistence_prediction(
        self,
        validation,
        obs_df,
    ):
        """
        Persistence baseline:
        temperature predicted from recent observed temperature.

        This is intentionally simple and is only used as a benchmark.
        """

        obs = obs_df[
            ["time", "temperature_2m"]
        ].copy()

        obs = obs.sort_values("time")

        # Previous 24-hour observation.
        obs["persistence_24h"] = (
            obs["temperature_2m"].shift(24)
        )

        result = validation.merge(
            obs[
                ["time", "persistence_24h"]
            ],
            on="time",
            how="left",
        )

        return result["persistence_24h"].to_numpy()

    def climatology_prediction(
        self,
        train,
        target,
    ):
        """
        Seasonal/hourly climatology computed only from TRAIN.
        """

        means = (
            train
            .groupby(["month", "hour"])["actual_temp"]
            .mean()
            .rename("clim")
            .reset_index()
        )

        result = target.merge(
            means,
            on=["month", "hour"],
            how="left",
        )

        return result["clim"].to_numpy()

    # ------------------------------------------------------------
    # MODEL SELECTION
    # ------------------------------------------------------------

    def select_models(
        self,
        train,
        validation,
        obs_df,
    ):
        """
        Select model + correction gate using VALIDATION only.

        TEST is not touched here.
        """

        models = self.make_models()

        all_rows = []

        for lead in range(1, MAX_LEAD + 1):

            tr = train[
                train["lead_days"] == lead
            ].copy()

            va = validation[
                validation["lead_days"] == lead
            ].copy()

            if tr.empty or va.empty:
                logger.warning(
                    "Skipping lead day %s: insufficient data.",
                    lead,
                )
                continue

            X_train = tr[self.feature_cols]
            y_train = tr["target_residual"]

            X_val = va[self.feature_cols]
            y_val = va["target_residual"]

            raw_val = va["fc_temperature_2m"]

            raw_mae = self.mae(
                va["actual_temp"],
                raw_val,
            )

            raw_rmse = self.rmse(
                va["actual_temp"],
                raw_val,
            )

            clim_pred = self.climatology_prediction(
                train,
                va,
            )

            clim_mae = self.mae(
                va["actual_temp"],
                clim_pred,
            )

            persistence_pred = self.persistence_prediction(
                va,
                obs_df,
            )

            persistence_mae = (
                self.mae(
                    va["actual_temp"],
                    persistence_pred,
                )
                if not np.isnan(persistence_pred).all()
                else np.nan
            )

            best_name = None
            best_model = None
            best_ml_mae = np.inf

            for name, model in models.items():

                logger.info(
                    "Training %s for lead day %s...",
                    name,
                    lead,
                )

                model.fit(
                    X_train,
                    y_train,
                )

                correction = model.predict(X_val)

                # Prevent extreme residual corrections during validation.
                residuals = y_train.to_numpy()
                if len(residuals) >= 20:
                    low, high = np.quantile(
                        residuals,
                        [0.01, 0.99],
                    )
                    correction = np.clip(
                        correction,
                        low,
                        high,
                    )

                ml_prediction = (
                    raw_val.to_numpy()
                    + correction
                )

                ml_mae = self.mae(
                    va["actual_temp"],
                    ml_prediction,
                )

                ml_rmse = self.rmse(
                    va["actual_temp"],
                    ml_prediction,
                )

                all_rows.append({
                    "split": "validation",
                    "lead_day": lead,
                    "model": name,
                    "raw_MAE": raw_mae,
                    "ML_MAE": ml_mae,
                    "ML_RMSE": ml_rmse,
                    "climatology_MAE": clim_mae,
                    "persistence_MAE": persistence_mae,
                })

                if ml_mae < best_ml_mae:
                    best_ml_mae = ml_mae
                    best_name = name
                    best_model = model

            # ----------------------------------------------------
            # Correction gate
            #
            # Apply correction only if the selected ML model
            # beats the raw ECMWF forecast on VALIDATION.
            # ----------------------------------------------------

            use_correction = (
                best_ml_mae < raw_mae
            )

            self.selected_models[lead] = best_name
            self.use_correction[lead] = use_correction

            # Robust correction limits learned from VALIDATION only.
            # This prevents an extreme live correction from overwhelming
            # the original ECMWF forecast.
            selected_correction = (
                best_model.predict(X_val)
                if best_model is not None
                else np.zeros(len(X_val))
            )

            low = float(np.percentile(selected_correction, 5))
            high = float(np.percentile(selected_correction, 95))

            # Keep the correction bounded to a practical temperature
            # adjustment while still allowing the model to correct bias.
            low = max(low, -3.0)
            high = min(high, 3.0)

            if low > high:
                low, high = -3.0, 3.0

            self.correction_bounds[lead] = [low, high]

            # Learn a robust correction range from TRAIN only.
            # This is a safety guard for live predictions.
            residuals = tr["target_residual"].dropna().to_numpy()
            if len(residuals) >= 20:
                low, high = np.quantile(
                    residuals,
                    [0.01, 0.99],
                )
                self.correction_bounds[lead] = [
                    float(low),
                    float(high),
                ]

            logger.info(
                "Lead %s -> selected=%s | raw MAE=%.3f | "
                "ML MAE=%.3f | correction=%s",
                lead,
                best_name,
                raw_mae,
                best_ml_mae,
                use_correction,
            )

        comparison = pd.DataFrame(all_rows)

        return comparison

    # ------------------------------------------------------------
    # FINAL TEST
    # ------------------------------------------------------------

    def evaluate_test(
        self,
        train,
        validation,
        test,
    ):
        """
        Final unbiased evaluation.

        Model choices and correction gates were already frozen
        using validation data.

        Test is used only once here.
        """

        results = []

        combined_train = pd.concat(
            [train, validation],
            ignore_index=True,
        )

        for lead in range(1, MAX_LEAD + 1):

            if lead not in self.selected_models:
                continue

            tr = combined_train[
                combined_train["lead_days"] == lead
            ].copy()

            te = test[
                test["lead_days"] == lead
            ].copy()

            if tr.empty or te.empty:
                continue

            model_name = self.selected_models[lead]

            # Fresh model for final test.
            model = self.make_models()[
                model_name
            ]

            model.fit(
                tr[self.feature_cols],
                tr["target_residual"],
            )

            correction = model.predict(
                te[self.feature_cols]
            )

            bounds = self.correction_bounds.get(lead)
            if bounds is not None:
                correction = np.clip(
                    correction,
                    bounds[0],
                    bounds[1],
                )

            raw_pred = (
                te["fc_temperature_2m"]
                .to_numpy()
            )

            ml_pred = (
                raw_pred + correction
            )

            # The gate was already selected on validation.
            if self.use_correction[lead]:
                final_pred = ml_pred
            else:
                final_pred = raw_pred

            raw_mae = self.mae(
                te["actual_temp"],
                raw_pred,
            )

            raw_rmse = self.rmse(
                te["actual_temp"],
                raw_pred,
            )

            ml_mae = self.mae(
                te["actual_temp"],
                ml_pred,
            )

            ml_rmse = self.rmse(
                te["actual_temp"],
                ml_pred,
            )

            final_mae = self.mae(
                te["actual_temp"],
                final_pred,
            )

            final_rmse = self.rmse(
                te["actual_temp"],
                final_pred,
            )

            results.append({
                "lead_day": lead,
                "selected_model": model_name,
                "correction_enabled": self.use_correction[lead],
                "raw_MAE": raw_mae,
                "ML_MAE": ml_mae,
                "final_MAE": final_mae,
                "raw_RMSE": raw_rmse,
                "ML_RMSE": ml_rmse,
                "final_RMSE": final_rmse,
                "gain_percent": (
                    100
                    * (raw_mae - final_mae)
                    / raw_mae
                    if raw_mae > 0
                    else 0
                ),
            })

        report = pd.DataFrame(results)

        print("\n" + "=" * 90)
        print("FINAL TEST RESULTS - UNSEEN DATA")
        print("=" * 90)

        if not report.empty:
            display_report = report.copy()

            numeric_cols = display_report.select_dtypes(
                include=np.number
            ).columns

            display_report[numeric_cols] = (
                display_report[numeric_cols]
                .round(3)
            )

            print(
                display_report.to_string(
                    index=False
                )
            )

        print("=" * 90)

        return report

    # ------------------------------------------------------------
    # SAVE PRODUCTION MODELS
    # ------------------------------------------------------------

    def train_production_models(
        self,
        train,
        validation,
    ):
        """
        Retrain the selected model for each lead day
        using TRAIN + VALIDATION.

        TEST remains untouched.
        """

        production_data = pd.concat(
            [train, validation],
            ignore_index=True,
        )

        saved_models = {}

        for lead in range(1, MAX_LEAD + 1):

            if lead not in self.selected_models:
                continue

            subset = production_data[
                production_data["lead_days"] == lead
            ]

            if subset.empty:
                continue

            model_name = self.selected_models[lead]

            model = self.make_models()[
                model_name
            ]

            model.fit(
                subset[self.feature_cols],
                subset["target_residual"],
            )

            saved_models[str(lead)] = model

        # --------------------------------------------------------
        # XGBoost / sklearn models are not all serialized in the
        # same way, so use joblib for production model objects.
        # --------------------------------------------------------

        import joblib

        joblib.dump(
            saved_models,
            MODEL_FILE.replace(
                ".json",
                ".joblib",
            ),
        )

        meta = {
            "site_name": SITE_NAME,
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "nwp_model": NWP_MODEL,
            "reference_model": REFERENCE_MODEL,
            "trained_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "weather_vars": self.weather_vars,
            "feature_cols": self.feature_cols,
            "selected_models": {
                str(k): v
                for k, v in self.selected_models.items()
            },
            "use_correction": {
                str(k): bool(v)
                for k, v in self.use_correction.items()
            },
            "correction_bounds": {
                str(k): [float(v[0]), float(v[1])]
                for k, v in self.correction_bounds.items()
            },
        }

        with open(
            META_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                meta,
                file,
                indent=2,
            )

        logger.info(
            "Saved production models to %s",
            MODEL_FILE.replace(
                ".json",
                ".joblib",
            ),
        )

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

    def load_production(self):
        import joblib

        model_file = MODEL_FILE.replace(
            ".json",
            ".joblib",
        )

        if not (
            os.path.exists(model_file)
            and os.path.exists(META_FILE)
        ):
            return None

        with open(
            META_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            meta = json.load(file)

        if meta.get("reference_model", REFERENCE_MODEL) != REFERENCE_MODEL:
            raise ValueError(
                "Saved model uses a different reference dataset. "
                "Run with --retrain."
            )

        self.weather_vars = meta["weather_vars"]
        self.feature_cols = meta["feature_cols"]
        self.selected_models = {
            int(k): v
            for k, v in meta["selected_models"].items()
        }
        self.use_correction = {
            int(k): bool(v)
            for k, v in meta["use_correction"].items()
        }
        self.correction_bounds = {
            int(k): [float(v[0]), float(v[1])]
            for k, v in meta.get("correction_bounds", {}).items()
        }

        models = joblib.load(model_file)

        logger.info(
            "Loaded production models trained at %s",
            meta["trained_at"],
        )

        return models

    # ------------------------------------------------------------
    # LIVE FORECAST
    # ------------------------------------------------------------

    def live_forecast(self, models):
        logger.info(
            "Fetching live ECMWF forecast..."
        )

        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": ",".join(
                self.weather_vars
            ),
            "current": (
                "temperature_2m,"
                "relative_humidity_2m"
            ),
            "models": NWP_MODEL,
            "forecast_days": 8,
            "timezone": "auto",
        }

        response = self.get_with_retry(
            FORECAST_URL,
            params,
            timeout=60,
        )

        if response is None:
            raise RuntimeError(
                "Live forecast download failed."
            )

        payload = response.json()

        hourly = pd.DataFrame(
            self.strip_model_suffix(
                payload["hourly"],
                NWP_MODEL,
            )
        )

        current = self.strip_model_suffix(
            payload.get("current", {}),
            NWP_MODEL,
        )

        hourly["time"] = pd.to_datetime(
            hourly["time"]
        )

        now = (
            pd.to_datetime(current["time"])
            if "time" in current
            else pd.Timestamp.now()
        )

        future = hourly[
            hourly["time"] > now.floor("h")
        ].copy()

        future = future[
            future["temperature_2m"].notna()
        ].copy()

        future["hours_ahead"] = (
            future["time"] - now
        ).dt.total_seconds() / 3600

        future["lead_days"] = np.clip(
            np.ceil(
                future["hours_ahead"] / 24
            ),
            1,
            MAX_LEAD,
        ).astype(int)

        # Keep exactly the next 7 calendar days:
        # tomorrow through 7 days ahead.
        # Today's current temperature is shown separately.
        current_date = now.date()
        last_forecast_date = (
            current_date + timedelta(days=MAX_LEAD)
        )

        future = future[
            (future["time"].dt.date > current_date)
            & (future["time"].dt.date <= last_forecast_date)
        ].copy()

        for variable in self.weather_vars:
            future[f"fc_{variable}"] = (
                future[variable]
            )

        future = self.add_calendar_features(
            future
        )

        future["temp_humidity_interaction"] = (
            future["fc_temperature_2m"]
            * future["fc_relative_humidity_2m"]
            if "fc_relative_humidity_2m"
            in future.columns
            else 0.0
        )

        future["temp_squared"] = (
            future["fc_temperature_2m"] ** 2
        )

        future["wind_squared"] = (
            future["fc_wind_speed_10m"] ** 2
            if "fc_wind_speed_10m" in future.columns
            else 0.0
        )

        # No observation-lag features are used in the production
        # model, because the same live information is not reliably
        # available from the delayed ERA5-Land reference dataset.

        # Missing live lag values are filled using
        # training-time feature medians if needed.
        X = future[
            self.feature_cols
        ].copy()

        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        X = X.fillna(
            X.median(
                numeric_only=True
            )
        )

        # --------------------------------------------------------
        # Apply the selected model separately for each lead day.
        # --------------------------------------------------------

        future["nwp_temp"] = (
            future["temperature_2m"]
        )

        future["ml_temp"] = (
            future["nwp_temp"]
        )

        future["correction"] = 0.0

        for lead in range(1, MAX_LEAD + 1):

            mask = (
                future["lead_days"]
                == lead
            )

            if not mask.any():
                continue

            if not self.use_correction.get(
                lead,
                False,
            ):
                continue

            model = models.get(
                str(lead)
            )

            if model is None:
                continue

            correction = model.predict(
                X.loc[mask]
            )

            # Safety bound learned from historical training residuals.
            bounds = self.correction_bounds.get(lead)
            if bounds is not None:
                correction = np.clip(
                    correction,
                    bounds[0],
                    bounds[1],
                )

            future.loc[
                mask,
                "correction"
            ] = correction

            future.loc[
                mask,
                "ml_temp"
            ] = (
                future.loc[
                    mask,
                    "nwp_temp"
                ].to_numpy()
                + correction
            )

        # --------------------------------------------------------
        # Daily summary
        # --------------------------------------------------------

        daily = (
            future.groupby(
                future["time"].dt.date
            )
            .agg(
                nwp_min=(
                    "nwp_temp",
                    "min",
                ),
                nwp_max=(
                    "nwp_temp",
                    "max",
                ),
                ml_min=(
                    "ml_temp",
                    "min",
                ),
                ml_max=(
                    "ml_temp",
                    "max",
                ),
                ml_mean=(
                    "ml_temp",
                    "mean",
                ),
                final_temperature=(
                    "ml_temp",
                    "max",
                ),
            )
            .round(1)
        )

        print("\n" + "=" * 78)
        print(
            f"{SITE_NAME.upper()} - LOCALIZED 7-DAY FORECAST"
        )
        print("=" * 78)

        print(
            f"As of: {now.strftime('%Y-%m-%d %H:%M')}"
        )

        # Current temperature is taken directly from the live ECMWF
        # current-temperature field and is displayed separately from
        # the 7 full forecast days.
        current_temp = current.get("temperature_2m")
        current_temp = (
            float(current_temp)
            if current_temp is not None
            and pd.notna(current_temp)
            else float("nan")
        )

        print("\nCurrent temperature:")
        if np.isfinite(current_temp):
            print(f"{current_temp:.1f} °C")
        else:
            print("Unavailable")

        print("\nNext 7-day temperature outlook (°C):")
        print(
            "Final temperature = ML-corrected daily maximum; "
            "Mean is the ML-corrected daily average."
        )

        display_daily = daily[[
            "ml_min",
            "ml_max",
            "ml_mean",
            "final_temperature",
        ]].copy()

        display_daily.columns = [
            "min_temperature",
            "max_temperature",
            "mean_temperature",
            "final_temperature",
        ]

        print(display_daily.to_string())

        print(
            "\nSelected correction models:"
        )

        for lead in range(1, MAX_LEAD + 1):
            print(
                f"Day {lead}: "
                f"{self.selected_models.get(lead, 'N/A')} | "
                f"Correction: "
                f"{self.use_correction.get(lead, False)}"
            )

        print("=" * 78)

        output = future[
            [
                "time",
                "lead_days",
                "nwp_temp",
                "correction",
                "ml_temp",
            ]
        ].copy()

        # Save the 7-day daily summary as a separate, easy-to-use file.
        daily_output = daily.reset_index().rename(
            columns={"time": "date"}
        )
        daily_output.to_csv(
            "kiaar_7day_daily_summary.csv",
            index=False,
        )

        output[
            [
                "nwp_temp",
                "correction",
                "ml_temp",
            ]
        ] = output[
            [
                "nwp_temp",
                "correction",
                "ml_temp",
            ]
        ].round(2)

        output.to_csv(
            FORECAST_FILE,
            index=False,
        )

        logger.info(
            "Saved forecast to %s",
            FORECAST_FILE,
        )

        return output


# ================================================================
# TRAINING WORKFLOW
# ================================================================

def train_pipeline():

    pipeline = KiaarAgroWeather()

    logger.info(
        "STEP 1/7 - Downloading automatic historical reference data"
    )

    observations = (
        pipeline.fetch_reference_observations()
    )

    logger.info(
        "STEP 2/7 - Downloading historical ECMWF forecasts"
    )

    forecast_history = (
        pipeline.fetch_history()
    )

    logger.info(
        "STEP 3/7 - Building residual training dataset"
    )

    data = pipeline.build_training_data(
        forecast_history,
        observations,
    )

    logger.info(
        "STEP 4/7 - Chronological train/validation/test split"
    )

    train, validation, test = (
        pipeline.chronological_split(data)
    )

    logger.info(
        "STEP 5/7 - Comparing ML models using validation only"
    )

    validation_comparison = (
        pipeline.select_models(
            train,
            validation,
            observations,
        )
    )

    if not validation_comparison.empty:
        validation_comparison.to_csv(
            EVALUATION_FILE,
            index=False,
        )

    logger.info(
        "STEP 6/7 - Final evaluation on untouched test data"
    )

    test_report = pipeline.evaluate_test(
        train,
        validation,
        test,
    )

    # Append test summary to a separate file.
    test_report.to_csv(
        "kiaar_final_test_results.csv",
        index=False,
    )

    logger.info(
        "STEP 7/7 - Retraining selected models on train + validation"
    )

    pipeline.train_production_models(
        train,
        validation,
    )

    return pipeline


# ================================================================
# MAIN
# ================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train the complete model comparison pipeline.",
    )

    args = parser.parse_args()

    pipeline = KiaarAgroWeather()

    model_file = MODEL_FILE.replace(
        ".json",
        ".joblib",
    )

    if (
        args.retrain
        or not os.path.exists(model_file)
        or not os.path.exists(META_FILE)
    ):

        pipeline = train_pipeline()

        models = pipeline.load_production()

    else:

        logger.info(
            "Loading existing production models."
        )

        models = pipeline.load_production()

        if models is None:
            pipeline = train_pipeline()
            models = pipeline.load_production()

    pipeline.live_forecast(
        models
    )


if __name__ == "__main__":
    main()
