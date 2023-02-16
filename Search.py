# project = 

import pathlib
absolute = str(pathlib.Path(__file__).parent.resolve()) + '/curiositydrivenalgos'
from curiositydrivenalgos.ZMARTfunctions import *
from curiositydrivenalgos.ZMARTplots import *
from sklearn.cluster import DBSCAN

def initialise(project):
    csvfilepath = str(absolute) + '/data/clean/' + project
    range_specification_filepath = str(absolute) + '/data/output_range_dictionaries/' + project
    raw_df = pd.read_csv(csvfilepath) #, sep=';')
    # if len(raw_df.columns) < 2:
    #     raw_df = pd.read_csv(filepath, sep=',')
    inputs, outputs = IO(raw_df)
    df_scaled = Scale(raw_df, inputs, outputs)
    # specification_dict = {output: {['predicted_' + output for output in outputs][0]: 1.1}, '>': {}}
    specification_dict = pd.read_pickle(range_specification_filepath)

    return df_scaled, inputs, outputs, specification_dict

def iterate(iteration, computed_df, inputs, outputs, df_scaled, components = 3, r = 0.05, γ =1.2, plots = 5, show_all_variants= False, comparison = True):
    """available plots: show_overlap, sanity, final_sanity, final_scores, view_clusters_realspace, solutions_in_predicted_space, compare_predictions_to_true"""

    computed_indices = list(computed_df.index)
    # position_model_dict, feature_imp_dict = predict_position(reduced_df, inputs)
    model_dict = get_model_dict(computed_df, inputs, outputs)
    pred_df = get_pred_df2(df_scaled, inputs, outputs, model_dict, computed_df)
    pred_df, _ = add_PCA(pred_df, outputs, components, prediction = True)
    pred_df, _ = add_PCA(pred_df, outputs, components)
    pred_df = addPCAerror(pred_df, components)
    if iteration % (plots*2) == 0:
        compare_predictions(pred_df, iteration, components)
        db = DBSCAN(eps=0.12, min_samples=1).fit(pred_df[outputs])
        pred_df['cluster'] = db.labels_
        cluster3D(pred_df, iteration, components)

    uncomputed = len(pred_df) - len(computed_indices)
    pred_df = find_variants_to_remove2(pred_df, computed_indices, components, r)
    print('points outside', r, ': ', uncomputed, '->', len(pred_df[pred_df['overlap'] == 0]))

    candidate_df = Range_Specifications(pred_df[pred_df['overlap'] == 0], specification_dict)
    if len(candidate_df) == 0:
        r /=  γ

    idcs_to_add, to_compute_df = find_next_batch_7(candidate_df, round(len(df_scaled)/60))

    computed_df = df_scaled.loc[computed_indices + idcs_to_add]
    if iteration % plots == 0:
        display_solutions(iteration, df_scaled, computed_df, outputs, components, specification_dict,
                          comparison=True,
                          show_all_variants=False)

    return computed_df, pred_df, r

# df_scaled_inputs_simplified, inputs_simplified, first_sample = iteration0(df_scaled, inputs, outputs)
# first_sample = df_scaled_inputs_simplified.sample(n = len(first_sample))
# computed_df = first_sample.copy()
# inputs = inputs_simplified
# df_scaled = df_scaled_inputs_simplified
df_scaled, inputs, outputs, specification_dict = initialise(project)
computed_df, r, iteration, components, γ = first_iteration(df_scaled, inputs,
                                                        r = 0.1,
                                                        components = 3,
                                                        γ= 1.2,
                                                           )
# computed_df, pred_df, r = iterate(iteration, computed_df, inputs, outputs, df_scaled, r=r, components= components,γ= γ)
while len(computed_df) < len(df_scaled)/4: #number of variants is less than 1/3 of all
    if iteration > 40:
        break
    print(len(computed_df))
    iteration += 1
    print('Iteration {}:'.format(iteration))
    computed_df, pred_df, r = iterate(iteration, computed_df, inputs, outputs, df_scaled, r=r, components=components, γ=γ)
_, pca = display_solutions(iteration, df_scaled, computed_df, outputs, components, specification_dict, comparison = True, show_all_variants= False)
compare_predictions(pred_df, iteration, components)

def paracoord():
    import plotly.express as px
    def paracoordorder(df_scaled, outputs):

        ouco = df_scaled[outputs].corr()
        next = ouco.unstack().sort_values(kind="quicksort").index[-(len(ouco)+1)][0]
        order = [next]
        for i in range(len(ouco.columns) - 1):
            previous = next
            highest = 1
            next = ouco.columns[ouco[next].argsort()[::-1][highest]]
            # print(ouco.loc[previous, next])
            while next == previous:
                highest += 1
                next = ouco.columns[ouco[next].argsort()[::-1][highest]]
            ouco = ouco.drop([previous], axis=0)
            ouco = ouco.drop([previous], axis=1)

            order.append(next)

        return order
    orderedcolumns = paracoordorder(df_scaled, outputs)
    # uniquedict = {column: len(df_scaled[column].unique()) for column in df_scaled[outputs].columns}
    # uniquedict = {column: np.std(df_scaled[column]) for column in df_scaled[outputs].columns}
    # paracoordcolumns = dict(sorted(uniquedict.items(), key=lambda item: item[1]))
    # orderedcolumns = list(paracoordcolumns.keys())

    fig = px.parallel_coordinates(computed_df[orderedcolumns], labels = {output: output[4:10] for output in orderedcolumns},
                                 color_continuous_scale=px.colors.diverging.Tealrose,
                                  title = 'ZMART - Zoektocht (met) Meer Afwisseling (en een) Rendement Toename'
                                 )
    fig.show()

    fig = px.parallel_coordinates(df_scaled[orderedcolumns], labels = {output: output[4:15] for output in orderedcolumns},
                                 color_continuous_scale=px.colors.diverging.Tealrose,
                                  title = 'all'
                                 )
    fig.show()

    fig = px.parallel_coordinates(df_scaled.sample(n= len(computed_df))[orderedcolumns], labels = {output: output[4:15] for output in orderedcolumns},
                                 color_continuous_scale=px.colors.diverging.Tealrose,
                                 title = 'Random')
    fig.show()
    return
paracoord()


#TODO - Learn when to stop, learn what the value of r should be, and the iteration sample size, also make it so that the most important fixed column changes when new values arent being added


#TODO - prioritise a region of space i.e is it possible to warp space accoridng to which outputs are considered more interesting?
#TODO end iterations when output accuracy reaches a threshold? - ask projects
#TODO - create a plot of the PCA space but coloured by what the outputs/inputs are
#TODO plot that shows the rest of points defined only by those that meet specifications