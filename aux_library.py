import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def xg_boost_predictor(data, target,
                       x_train, x_eval, y_train, y_eval,
                       plot_title, depth=2, lr=0.01, forests=100,
                       bytree=0.8, subsample=0.7, a=0, l=1):

    """

     Trains and evaluates a model on a train and test set. Then retrains the model on a given dataset.

    :param data: The dataset to be used at the end to train the final model.
    :param target: The target of the above dataset.
    :param x_train: Training dataset.
    :param x_eval: Evaluation dataset used during the training process.
    :param y_train:  Training dataset's target.
    :param y_eval: Evaluation dataset target.
    :param plot_title: Title for the plot.
    :param depth: Tree depth.
    :param lr:  Learning rate.
    :param forests: Number of estimators.
    :param bytree: Columns tree.
    :param subsample: Percentage of data per tree.
    :param a: L1 regularization param.
    :param l: L2 regularization param.
    :return: A trained model on the specified dataset and the rmse of each observation in a list.
    """


    model = xgb.XGBRegressor(objective='reg:squarederror',
                             max_depth=depth,
                             learning_rate=lr,
                             n_estimators=forests,
                             n_jobs=-1,
                             colsample_bytree=bytree,
                             subsample=subsample,
                             reg_alpha =a,
                             reg_lambda=l,
                             colsample_bylevel=1

                             )

    eval_set = [(x_train, y_train), (x_eval, y_eval)]
    eval_metric = ['rmse']

    model.fit(x_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=False, early_stopping_rounds=10)
    print(' The feature importances for this model are: \n {}'.format(
        list(sorted(zip(data.columns,model.feature_importances_), key=lambda x: x[1], reverse=True))))

    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
    ax.legend()
    plt.ylabel('rmse')
    plt.title('XGBoost rmse ' + plot_title)
    plt.show()

    y_pred = model.predict(x_eval)
    rmse = mean_squared_error(y_eval, y_pred, squared= False)
    print(' RMSE for the test set : {}'.format(rmse))

    return model.fit(data, target), rmse


def make_predictions(model, dataframe, fill_column):

    """
    Makes predictions for multiple rows by predicting row by row and using the result to forward fill a parameter in
    the next row.

    :param model: The trained model that will be used for the predictions.
    :param dataframe: A dataframe for predictions.
    :param fill_column: The column that needs to be filled with the result of the previous predicgion.
    :return: Returns the predictions for the given dataset.
    """
    predictions = []
    for row in range(dataframe.shape[0]):
        prediction = model.predict(dataframe.loc[row:row])

        predictions.append(prediction[0])

        # fill the suborders_of_previous_week of the next row with the predictions
        if row < dataframe.shape[0] - 1:
            dataframe[fill_column].loc[row + 1] = prediction[0]

    return predictions
