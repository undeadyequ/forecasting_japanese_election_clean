import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, BayesianRidge, ARDRegression
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, \
    ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from boostedLinearRegression import blr as BLRModel
import logging
import argparse
import time
import json


pd.options.display.float_format = '{:.2f}'.format


# ── data helpers ───────────────────────────────────────────────────────────────

def random_select_data(data, s, data_condition="N2012"):
    csvdata = pd.read_csv(data, delimiter=",")
    if "U2009" in data_condition:
        csvdata = csvdata[csvdata['Year'] < 2010]
    elif "N2009_2012" in data_condition:
        csvdata = csvdata[(csvdata['Year'] != 2009) & (csvdata['Year'] != 2012)]
    elif "N2012" in data_condition:
        csvdata = csvdata[csvdata['Year'] != 2012]

    np.random.seed(seed=s)
    csvdata = csvdata.reindex(np.random.permutation(csvdata.index))

    # Split train/test dataset
    data_train = csvdata.iloc[int(len(csvdata) / 4):, :]
    data_test = csvdata.iloc[:int(len(csvdata) / 4), :]
    x_train = data_train.drop(['Year', 'LDP_votes', 'LDP_seats'], axis=1)
    y_train = data_train['LDP_seats']
    x_test = data_test.drop(['Year', 'LDP_votes', 'LDP_seats'], axis=1)
    y_test = data_test['LDP_seats']
    return x_train, y_train, x_test, y_test


def _blr_select_data(data):
    """Data selection for BLR: always uses N2012 condition (verbatim from pred_blr.py)."""
    csvdata = pd.read_csv(data, delimiter=",")
    csvdata = csvdata[csvdata['Year'] != 2012]
    x = csvdata.drop(['Year', 'LDP_votes', 'LDP_seats'], axis=1)
    y = csvdata['LDP_seats']
    return x, y


# ── model training ─────────────────────────────────────────────────────────────

def train_clf(method, seeds_num, csv_data, data_condition):
    train_maes = []
    test_maes = []
    train_rmses = []
    test_rmses = []

    feature_importances = []
    r_sqrs = []
    estimators = []

    y_preds = []
    y_preds_index = []

    x_tests = []
    y_tests = []

    for s in range(2, seeds_num+2):
        x_train, y_train, x_test, y_test = random_select_data(csv_data, s, data_condition)

        # Instantiate a fresh classifier for each seed so estimators[i] is a distinct object
        if method == "lr":
            besthyper = {'fit_intercept': True, 'normalize': False}
            clf = LinearRegression(**besthyper)

        elif method == "dt":
            besthyper = {'splitter': 'random', 'random_state': 0, 'max_depth': 3, 'criterion': 'mse'}
            clf = DecisionTreeRegressor(**besthyper)

        elif method == "svr":
            best_hyper = {'kernel': 'poly'}
            clf = SVR(**best_hyper)

        elif method == "rf":
            best_hyper = {
                'criterion': 'mae',
                'max_depth': 4,
                'max_features': 'auto',
                'min_samples_leaf': 2,
                'min_samples_split': 2,
                'n_estimators': 125,
                'n_jobs': 1,
                'random_state': 0}
            clf = RandomForestRegressor(**best_hyper)

        elif method == "bag_lr":
            best_hyper = {
                'base_estimator': LinearRegression(),
                'bootstrap': True,
                'max_features': 3,
                'max_samples': 12,
                'n_estimators': 2000,
                'n_jobs': 1,
                'random_state': 0}
            clf = BaggingRegressor(**best_hyper)

        elif method == "bag_dt":
            best_hyper = {
                'base_estimator': DecisionTreeRegressor(),
                'bootstrap': True,
                'max_features': 3,
                'max_samples': 12,
                'n_estimators': 2000,
                'n_jobs': 1,
                'random_state': 0}
            clf = BaggingRegressor(**best_hyper)

        elif method == "adaboost_lr":
            besthyper = {
                'base_estimator': LinearRegression(),
                'learning_rate': 1.75,
                'loss': 'linear',
                'n_estimators': 10,
                'random_state': 0
            }
            clf = AdaBoostRegressor(**besthyper)

        elif method == "adaboost_dt":
            besthyper = {
                'base_estimator': DecisionTreeRegressor(max_depth=4),
                'learning_rate': 1.75,
                'loss': 'linear',
                'n_estimators': 10,
                'random_state': 0
            }
            clf = AdaBoostRegressor(**besthyper)

        elif method == "gradient_boost":
            besthyper = {
                'criterion': 'mae',
                'learning_rate': 0.2,
                'loss': 'lad',
                'max_depth': 3,
                'max_features': 'auto',
                'min_samples_leaf': 3,
                'min_samples_split': 2,
                'n_estimators': 125,
                'random_state': 0
            }
            clf = GradientBoostingRegressor(**besthyper)

        elif method == "xgb":
            besthyper = {
                'booster': 'dart',
                'eta': 0.05,
                'eval_metric': 'mae',
                'gamma': 0,
                'lambda': 0.5,
                'max_depth': 5,
                'max_leaf_nodes': 1,
                'n_estimators': 100,
                'n_jobs': 1,
                'random_state': 0,
                'seed': 0,
                'verbosity': 0}
            clf = XGBRegressor(**besthyper)
        else:
            clf = None
            print("Undefined method")

        clf.fit(x_train, y_train)
        # Evaluation of train and test
        y_train_pred = clf.predict(x_train)
        y_train_pred_index = x_train.index
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

        y_test_pred = clf.predict(x_test)
        y_test_pred_index = x_test.index
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Get prediction of LDP seats with index (both train and test)
        y_train_pred_cp = y_train_pred.copy()
        y_test_pred_cp = y_test_pred.copy()
        y_pred_index = y_train_pred_index.union(y_test_pred_index, sort=False)
        y_preds_index.append(y_pred_index)
        y_pred = np.hstack((y_train_pred_cp, y_test_pred_cp))

        # append result of different seeds to a list.
        train_maes.append(round(train_mae, 2))
        train_rmses.append(round(train_rmse, 2))
        test_maes.append(round(test_mae, 2))
        test_rmses.append(round(test_rmse, 2))
        y_preds.append(y_pred)

        estimators.append(clf)
        r_sqr = clf.score(x_test, y_test)
        r_sqrs.append(r_sqr)

        x_tests.append(x_test)
        y_tests.append(y_test)

    # Best hyper: find champion shuffle
    champion_number = np.argmin(test_maes)
    champ_estimator = estimators[champion_number]

    # find champion prediction
    y_pred_champ = [round(x, 2) for x in y_preds[champion_number]]
    y_pred_index_y_pred_champ = y_preds_index[champion_number]
    y_pred_champ = pd.DataFrame(y_pred_champ, index=y_pred_index_y_pred_champ)

    # Got Permutation Feature Importance
    feats_important_dicts, feats_import_array = show_permutation_importance(
        champ_estimator, x_tests[champion_number], y_tests[champion_number])
    if method == "rf":
        feature_importance = champ_estimator.feature_importances_
        feature_importances.append(feature_importance)
        logging.info("ft importance: {}".format(feature_importances))

    return train_maes, train_rmses, test_maes, test_rmses, y_pred_champ, feats_important_dicts, feats_import_array


def run_blr(csv_data):
    """Reproduce pred_blr.py results with identical splitting logic (always N2012 condition).

    Uses train_test_split(test_size=4) then further splits 4 points into 3 test + 1 val.
    Seeds range(100) to match original pred_blr.py output exactly.
    """
    learning_rate, max_iter, early_stopping_count = 0.1, 1000, 20

    x, y = _blr_select_data(csv_data)
    y_arr = y.values.reshape(-1, 1)
    x_arr = x.values

    mae_train_list, mae_test_list = [], []
    rmse_train_list, rmse_test_list = [], []

    for r in range(100):
        x_train, x_val_test, y_train, y_val_test = train_test_split(
            x_arr, y_arr, test_size=4, random_state=r)
        x_val, x_test, y_val, y_test = train_test_split(
            x_val_test, y_val_test, test_size=3, random_state=r)

        model = BLRModel(learning_rate, max_iter, early_stopping_count)
        model.fit(x_train, y_train, x_val, y_val)

        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        mae_train_list.append(mean_absolute_error(y_train, y_train_pred))
        mae_test_list.append(mean_absolute_error(y_test, y_test_pred))
        rmse_train_list.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        rmse_test_list.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

    return (np.mean(mae_train_list), np.std(mae_train_list),
            np.mean(rmse_train_list), np.std(rmse_train_list),
            np.mean(mae_test_list), np.std(mae_test_list),
            np.mean(rmse_test_list), np.std(rmse_test_list))


# ── helpers ────────────────────────────────────────────────────────────────────

def vis_y_pred_and_true(y_preds, y_true):
    for method in y_preds.keys():
        x = np.arange(y_preds[method].shape[0])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("sample")
        ax.set_ylabel("ldp_seats")
        ax.plot(x, y_preds[method], label=method)
        ax.plot(x, y_true, label="gd")
    plt.legend()
    plt.savefig("pred_true_diff.png")


def show_permutation_importance(model, x_test, y_test):
    feat_import_dict = {}
    r = permutation_importance(model, x_test, y_test,
                               n_repeats=30,
                               random_state=0)
    logging.info("Feature Importance")
    for i in r.importances_mean.argsort()[::-1]:
        feat_import_dict[x_test.columns[i]] = [round(r.importances_mean[i], 3),
                                               round(r.importances_std[i], 3)]
    return feat_import_dict, r.importances


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(filename="log_correct1.txt", filemode="a",
                        format="%(asctime)s - %(message)s", level=logging.INFO)

    csv_data = "data/data_election_2020_correct2012.csv"
    os.makedirs("intermediary_data", exist_ok=True)

    seed_num = 100

    # All conditions run automatically; no manual condition switching needed.
    condition_configs = {
        "N2012":      ["lr", "dt", "bag_lr", "bag_dt", "gradient_boost"],
        "U2009":      ["lr", "gradient_boost"],
        "N2009_2012": ["lr", "gradient_boost"],
    }

    RESULT_COLS = [
        "train_mae", "train_mae_std",
        "train_rmse", "train_rmse_std",
        "test_mae", "test_mae_std",
        "test_rmse", "test_rmse_std",
        "imp_Days_mean", "imp_Days_std",
        "imp_GDP_mean", "imp_GDP_std",
        "imp_approval_mean", "imp_approval_std",
    ]

    for data_condtion, methods in condition_configs.items():
        print(f"\n=== Condition: {data_condtion} ===")

        # Filtered dataset for attaching champion predictions (N2012 only)
        csvdata = pd.read_csv(csv_data, delimiter=",")
        if "U2009" in data_condtion:
            csvdata = csvdata[csvdata['Year'] < 2010]
        elif "N2009_2012" in data_condtion:
            csvdata = csvdata[(csvdata['Year'] != 2009) & (csvdata['Year'] != 2012)]
        elif "N2012" in data_condtion:
            csvdata = csvdata[csvdata['Year'] != 2012]

        result = []
        feats_arrays = {}
        row_labels = list(methods)

        for method in methods:
            st = time.time()
            train_maes, train_rmses, test_maes, test_rmses, y_pred_champ, \
            feats_important_dicts, feats_import_array = \
                train_clf(method, seed_num, csv_data, data_condition=(data_condtion))
            end = time.time()

            cls_res = [
                np.mean(train_maes), np.std(train_maes),
                np.mean(train_rmses), np.std(train_rmses),
                np.mean(test_maes), np.std(test_maes),
                np.mean(test_rmses), np.std(test_rmses),
                feats_important_dicts["DAYS"][0],
                feats_important_dicts["DAYS"][1],
                feats_important_dicts["GDP"][0],
                feats_important_dicts["GDP"][1],
                feats_important_dicts["PM_approval"][0],
                feats_important_dicts["PM_approval"][1],
            ]
            result.append(cls_res)
            print(f"  {method}: {end - st:.1f}s")

            if data_condtion == "N2012":
                csvdata[method] = y_pred_champ
                feats_arrays[method] = feats_import_array.tolist()

        # Add BLR (gradient_lr) for N2012 only — uses its own splitting strategy
        if data_condtion == "N2012":
            print("  gradient_lr (BLR): running 100 seeds...")
            st = time.time()
            blr_stats = run_blr(csv_data)
            end = time.time()
            # gradient_lr has no feature importance — fill with NaN
            blr_res = list(blr_stats) + [float('nan')] * 6
            result.append(blr_res)
            row_labels.append("gradient_lr")
            print(f"  gradient_lr: {end - st:.1f}s")

        # Save model results
        pd_data = pd.DataFrame(
            np.array(result, dtype=float),
            index=row_labels,
            columns=RESULT_COLS,
        )
        pd_data.to_csv(f"intermediary_data/model_results_{data_condtion}.csv")
        print(f"  Saved intermediary_data/model_results_{data_condtion}.csv")

        # Save predictions and feature importance (N2012 only)
        if data_condtion == "N2012":
            csvdata.to_csv("intermediary_data/predictions_N2012.csv")
            with open("intermediary_data/feature_importance_N2012.json", "w") as f:
                json.dump(feats_arrays, f, indent=4)
            print("  Saved intermediary_data/predictions_N2012.csv")
            print("  Saved intermediary_data/feature_importance_N2012.json")

    print("\nDone. All results saved to intermediary_data/")
