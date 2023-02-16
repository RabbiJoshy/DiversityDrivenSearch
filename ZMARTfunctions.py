import pandas as pd
import numpy as np
import xgboost
import scipy.spatial
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None  # default='warn'

def IO(df):
    inputs = []
    outputs = []
    for column in df.columns:
        if len(df[column].unique()) > 1:
            if column[:2] == 'in':
                #print(column)
                inputs.append(column)
            if column[:3] == 'out':
                #print(column)
                outputs.append(column)

    return inputs, outputs
def Scale(df, inputs, outputs):
    scaler = MinMaxScaler()
    df_scaled = df[inputs + outputs]
    df_scaled[list(df_scaled.columns)] = scaler.fit_transform(df_scaled[list(df_scaled.columns)])

    return df_scaled
def find_first_sample(df_scaled, inputs, n=1):
    first_sample_indices = []
    for input in inputs:
        input_unique_options = df_scaled[input].unique()
        for option in input_unique_options:
            unique_df = df_scaled[df_scaled[input] == option]
            if first_sample_indices == []:
                unique_df_no_duplicates = unique_df
            else:
                unique_df_no_duplicates = unique_df
                for index in first_sample_indices:
                    if index in unique_df.index:
                        unique_df_no_duplicates = unique_df_no_duplicates.drop(index)
            unique_df_sample = unique_df_no_duplicates.sample(n=n)
            unique_indices = list(unique_df_sample.index)
            for index in unique_indices:
                first_sample_indices.append(index)

    return first_sample_indices
def get_model_dict(computed_df, inputs, outputs):
    model_dict = dict()
    for output in outputs:
        model = xgboost.XGBRegressor()
        model.fit(computed_df[inputs], computed_df[output])
        model_dict[output] = model
        # y_pred = model.predict(df_scaled[inputs])
        # y_true = df_scaled[output]
        # rmse = np.sqrt(MSE(y_true, y_pred))
        # mae = np.sqrt(MAE(y_true, y_pred))
        # print(str(output), " RMSE : % f" % (rmse))
        # print(str(output), " MAE : % f" % (mae))
        # print(model.feature_importances_)
        # feature_imp_dict[output] = model.feature_importances_
    return model_dict
def get_pred_df2(df_scaled, inputs, outputs, model_dict, computed_df):

    pred_df = df_scaled.copy()
    # pred_df.index = df_scaled.index
    for output in outputs:
        model = model_dict[str(output)]
        pred_df['predicted_' + str(output)] = model.predict(df_scaled[inputs])
        pred_df['predicted_' + str(output)].loc[computed_df.index] = computed_df[output]
    pred_df['computed'] = 0 * len(pred_df)
    pred_df['computed'].loc[computed_df.index] = 1
    #TODO = add confidence column based on accuracy - 100 being computed
    # pred_df['score'] = (pred_df[outputs] * output_weights).sum(1)

    return pred_df


def add_PCA(df, outputs, components = 3, prediction = False):

    pca_df = df.copy()
    pca = PCA(n_components = components)
    if prediction == True:
        outputs = ['predicted_' + output for output in outputs]
    PCA_array = pca.fit_transform(pca_df[outputs])

    for i in range(components):
        if prediction == True:
            pca_df['PCA' + str(i+1) + '_Prediction'] = PCA_array[:, i]
        else:
            pca_df['PCA' + str(i + 1)] = PCA_array[:, i]

    return pca_df, pca
def addPCAerror(pred_df, components):
    pred_df['pcaerror'] = pred_df.apply(
        lambda row: np.sum([(row['PCA' + str(i)] - row['PCA' + str(i) +'_Prediction'])**2
                            for i in range(1,components+1)])**(1/components), axis =1 )

    return pred_df
def get_global_accuracy(computed_df, df_scaled, inputs, outputs, metric = 'MAE', verbose = False):
    accuracy_dict = dict()
    print('using training points:', len(computed_df))
    for output in outputs:
        model = xgboost.XGBRegressor()
        model.fit(computed_df[inputs], computed_df[output])
        test_df = df_scaled[~df_scaled.index.isin(computed_df.index)]

        y_pred = model.predict(test_df[inputs])
        y_true = test_df[output]
        if metric == 'MAE':
            accuracy = MAE(y_true, y_pred)
        else:
            accuracy = np.sqrt(MSE(y_true, y_pred))
        if verbose:
            print("{0} - {1} : {2}".format(output[4:], metric, accuracy))
        # print(model.feature_importances_)
        accuracy_dict[output] = accuracy
    print('avg. acc:', np.mean(list(accuracy_dict.values())))

    return accuracy_dict
# def predict_position(computed_df_reduced, inputs):
#     """Predict position in the space of *arg: computed_reduced_df* - which is the computed df so far
#     - used for search version 4"""
#     feature_imp_dict = dict()
#     position_model_dict = dict()
#     for dim in ['PCA1', 'PCA2']:
#         model = xgboost.XGBRegressor()
#         model.fit(computed_df_reduced[inputs], computed_df_reduced[dim])
#         position_model_dict[dim] = model
#         feature_imp_dict[dim] = dict(zip(inputs, model.feature_importances_))
#
#     return position_model_dict, feature_imp_dict
def Range_Specifications(df, specification_dict):

    old = len(df)

    for output in specification_dict.keys():
        df = df[df[output] <= specification_dict[output]['<']]


    for output in specification_dict.keys():
        df = df[df[output] >= specification_dict[output]['>']]

    print('within range:', old,'->', len(df))

    return df
# def iteration0(df_scaled, inputs, outputs):
#     first_sample_indices = find_first_sample(df_scaled, inputs, n=1)
#     first_sample = df_scaled.loc[first_sample_indices]
#     computed_df = first_sample.copy()
#     reduced_df, _ = add_PCA(computed_df, outputs)
#     _, feature_imp_dict = predict_position(reduced_df, inputs)
#
#     useless_inputs = [k for k,v in feature_imp_dict['PCA1'].items() if v == 0]
#     df_scaled_inputs_simplified = df_scaled.drop(useless_inputs, axis = 1)
#     inputs_simplified = [input for input in inputs if input not in useless_inputs]
#
#     first_sample_indices = find_first_sample(df_scaled_inputs_simplified, inputs_simplified, n=1)
#     first_sample = df_scaled_inputs_simplified.loc[first_sample_indices]
#
#
#     reduced_df_scaled, _ = add_PCA(df_scaled.copy(), outputs)
#     print(first_sample_indices)
#     reduced_df_scaled['first_sample'] = 0
#     reduced_df_scaled['first_sample'].loc[first_sample_indices] = 1
#     # sns.scatterplot(reduced_df_scaled, x = 'PCA1', y= 'PCA2', hue = 'first_sample', size = 'first_sample', sizes = (20,2), palette = ["tab:gray", "tab:blue"])
#     # plt.title('First Sample' + str(len(first_sample_indices)) + '/' + str(len(df_scaled.copy())))
#     # plt.show()
#     # plt.clf()
#
#     return df_scaled_inputs_simplified, inputs_simplified, first_sample
def first_iteration(df_scaled, inputs, r =0.25, components = 3, γ=1.2):
    first_sample_indices = find_first_sample(df_scaled, inputs, n=1)
    first_sample = df_scaled.loc[first_sample_indices]
    iteration = 1

    return first_sample, r, iteration, components, γ
def find_variants_to_remove2(pred_df, computed_indices, components = 3 , r=0.05):
    def get_ckdtreeND(df, components=3):
        points = np.array([np.array(df['PCA' + str(i + 1) + '_Prediction']) for i in range(components)]).T
        ckdtree = scipy.spatial.cKDTree(points)
        return ckdtree

    pred_ckd_tree = get_ckdtreeND(pred_df.copy(), components)

    to_remove = []
    for index in computed_indices:
        to_remove.extend(
            pred_ckd_tree.query_ball_point([pred_df['PCA' + str(i+1) + '_Prediction'][index] for i in range(components)], r=r))
    to_remove_unique_idcs = list(set(to_remove))

    pred_df_2 = pred_df.drop(to_remove_unique_idcs)
    pred_ckd_tree_2 = get_ckdtreeND(pred_df_2, components)
    pairs = pred_ckd_tree_2.query_pairs(r=0)
    to_remove2 = [pair[1] for pair in list(pairs)]
    to_remove_unique_idcs.extend(list(set(to_remove2)))
    to_remove_unique_idcs.extend(computed_indices)
    to_remove_unique_idcs = list(set(to_remove_unique_idcs))

    pred_df['overlap'] = 0
    pred_df['overlap'].loc[to_remove_unique_idcs] = 1

    return pred_df
def find_next_batch_7(pred_df, max_variants):
    candidate_df = pred_df.copy()
    candidate_df = candidate_df[candidate_df['computed'] == 0]
    to_compute_df = candidate_df.sample(n = min(len(candidate_df), max_variants))
    to_compute_indices = list(to_compute_df.index)

    return to_compute_indices, to_compute_df
