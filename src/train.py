import configparser
import os
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import sys
import traceback

from logger import Logger

SHOW_LOG = True


class MultiModel():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        # self.config.read(os.path.join(os.getcwd(), "config.ini"))
        self.config.read("C:/Users/ada/Maga/MLE/lab1/src/config.ini")
        # path = '/'.join((os.path.abspath("config.ini").replace('\\', '/')).split('/')[:-1])
        #
        # self.config.read(os.path.join(path, 'config.ini'))
        # print(self.config)
        # print(os.path.join(os.getcwd(), "config.ini"))
        self.X_train = pd.read_csv(
            self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.y_train = pd.read_csv(
            self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.X_test = pd.read_csv(
            self.config["SPLIT_DATA"]["X_test"], index_col=0)
        self.y_test = pd.read_csv(
            self.config["SPLIT_DATA"]["y_test"], index_col=0)
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        self.project_path = os.path.join(os.getcwd(), "experiments")
        self.lasso = os.path.join(self.project_path, "lasso.sav")
        self.rand_forest_path = os.path.join(
            self.project_path, "rand_forest.sav")
        self.log.info("MultiModel is ready")

    def log_reg(self, predict=False) -> bool:
        reg = Lasso()
        try:
            reg.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = reg.predict(self.X_test)
            print("R2 score for Lasso = ", r2_score(self.y_test, y_pred))
        params = {'path': self.lasso}
        return self.save_model(reg, self.lasso , "LASSO", params)


    def rand_forest(self, n_trees=100,
                    criterion = "squared_error",
                    predict=False) -> bool:
        reg = RandomForestRegressor(
            n_estimators=n_trees,
            # criterion=criterion
        )
        try:
            reg.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = reg.predict(self.X_test)
            print("R2 score for RF = ", r2_score(self.y_test, y_pred))
        params = {'n_estimators': n_trees,
                  'criterion': criterion,
                  'path': self.rand_forest_path}
        return self.save_model(reg, self.rand_forest_path, "RAND_FOREST", params)

    def save_model(self, reg, path: str, name: str, params: dict) -> bool:
        self.config[name] = params
        os.remove('C:/Users/ada/Maga/MLE/lab1/src/config.ini')
        with open('C:/Users/ada/Maga/MLE/lab1/src/config.ini', 'w') as configfile:
            self.config.write(configfile)
        pickle.dump(reg, open(path, 'wb'))
        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


if __name__ == "__main__":
    multi_model = MultiModel()
    multi_model.log_reg(predict=True)
    multi_model.rand_forest(predict=True)
