"""A deliberately broken rugby workload model.

Audit input, not source. Every leak the native checks look for is in here on
purpose, so the test suite can assert that each one fires and each one lands on
the right line. Nothing in this file is ever imported or executed: `ruck audit`
parses it with `ast`.

Do not copy this. `demo/expected/audit.md` is ruck's report on this file.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

GPS_COLUMNS = ["total_distance", "high_speed_running", "player_load", "session_rpe"]


def load_panel(path):
    panel = pd.read_csv(path, parse_dates=["date"])
    panel = panel.sort_values(["player_id", "date"])
    # SV002: unrestricted dropna over a panel with athletes and a label.
    panel = panel.dropna()
    return panel


def build_features(panel, injury_date):
    features = pd.DataFrame(index=panel.index)

    # LA003: backfill pulls the next recorded session onto a missing one.
    panel["total_distance"] = panel["total_distance"].bfill()
    # LA004: interpolation fills from both sides.
    panel["player_load"] = panel["player_load"].interpolate()

    # LA007 + AC006: the window ends on the session it is a feature for.
    features["acute_load"] = panel["total_distance"].rolling(7).sum()
    # LA002: a centred window reaches into the future.
    features["smoothed_load"] = panel["player_load"].rolling(14, center=True).mean()
    # LA001: a negative shift is the future, on a feature.
    features["next_session_load"] = panel["session_rpe"].shift(-1)
    # LA005: differencing against a later row.
    features["load_delta"] = panel["high_speed_running"].diff(-3)

    # AC001 + AC004: coupled ratio, no floor on the denominator.
    acute = panel["total_distance"].rolling(7).sum()
    chronic = panel["total_distance"].rolling(28).sum() / 4
    features["acwr"] = acute / chronic

    # AC003: EWMA smoothing removes the window edge, not the coupling.
    ewma_acute = panel["total_distance"].ewm(span=7).mean()
    ewma_chronic = panel["total_distance"].ewm(span=28).mean()
    features["ewma_acwr"] = ewma_acute / ewma_chronic

    # AC005: the sweet-spot bands.
    features["acwr_zone"] = pd.cut(features["acwr"], bins=[0, 0.8, 1.3, 1.5, 99])

    # LB003: the window covers the outcome column too.
    features["recent_history"] = panel[["total_distance", "injured"]].rolling(28).mean()
    # LB002: a feature computed from the outcome.
    features["risk_score"] = panel["injured"] * 0.9 + np.random.rand(len(panel)) * 0.1
    # LB005: defined by the outcome it predicts.
    features["days_since_injury"] = (panel["date"] - panel["injury_date"]).dt.days

    # LB004: the window is inclusive of the injury day.
    window = panel[panel["date"] <= injury_date]
    features["window_rows"] = len(window)
    return features


def select_cohort(panel):
    # SV001: the population is chosen by the outcome.
    fit_players = panel[panel["injured"] == 0]
    # SV003: a cohort chosen on a season total.
    regulars = fit_players[fit_players["games_played"] > 12]
    return regulars


def fit_model(features, panel):
    # LB001: the outcome is still in the feature matrix.
    X = features[["acute_load", "acwr", "injured"]]
    y = panel["injured"]

    # LA006: the scaler learns its statistics from the test rows too.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # CV002: train_test_split shuffles by default.
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    # CV001: shuffling a time-ordered panel.
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    # CV003: the default K-fold, ungrouped.
    scores = cross_val_score(model, X_scaled, y, cv=5)

    # CV005: time-ordered but no embargo, with 28-day windows in the features.
    ordered = TimeSeriesSplit(n_splits=5)
    return model, scores, folds, ordered, X_test, y_test


def main():
    panel = load_panel("data/gps_panel.csv")
    cohort = select_cohort(panel)
    features = build_features(cohort, injury_date=pd.Timestamp("2024-03-01"))
    return fit_model(features, cohort)
