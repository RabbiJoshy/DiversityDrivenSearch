# from mpl_toolkits import mplot3d
import seaborn as sns
import matplotlib.pyplot as plt
from curiositydrivenalgos.ZMARTfunctions import * #add_PCA numpy

def final_space3D(outputs, iteration, df_scaled, computed_df, comparison_df, components, comparison = False, show_all_variants = False):
    reduced_all, pca = add_PCA(df_scaled, outputs, components)
    reduced_all['algorithm'] = 'None'
    reduced_all['algorithm'].loc[computed_df.index] = 'ZMART'

    reduced_all['algorithm'].loc[comparison_df.index] = 'Random'
    both_idcs = list(set(comparison_df.index).intersection(computed_df.index))
    reduced_all['algorithm'].loc[both_idcs] = 'Both'

    # #PLOTS:
    # palette = {"ZMART": "tab:red", "Random": "tab:red", "None": "tab:gray", 'Both': "tab:red"}
    # sizes = {"ZMART": 20, "Random": 20, "None": 1, 'Both': 20}



    algorithm = 'ZMART'
    if comparison == True:
        algorithm = 'Random'
    Display = reduced_all[reduced_all['algorithm'] == algorithm]
    Rest = reduced_all[reduced_all['algorithm'] != algorithm]
    displaypoints = [(Display, 'o', 10)]

    if show_all_variants == True:
        displaypoints.append((Rest,'^', 1))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in displaypoints:
    # Creating dataset
        z = np.array(i[0]['PCA1'])
        x = np.array(i[0]['PCA2'])
        y = np.array(i[0]['PCA3'])

        # Creating figure
        ax.scatter(x, y, z, marker= i[1], s = i[2])
        ax.set_xlabel('pca 1', fontweight='bold')
        ax.set_ylabel('pca 2', fontweight='bold')
        ax.set_zlabel('pca 3', fontweight='bold')
    plt.title(algorithm + ': ' + str(iteration))
    plt.show()

    return reduced_all, pca

def compare_predictions(pred_df, iteration, components):
    # Prediction shown vs Actual in same space
    x = pred_df['PCA1']
    y = pred_df['PCA2']
    x_pred = pred_df['PCA1_Prediction']
    y_pred = pred_df['PCA2_Prediction']
    if components == 3:
        z = pred_df['PCA3']
        z_pred = pred_df['PCA3_Prediction']

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # for i in [('True', 'o', 3),  ('Pred', '^', 1)]:
            # Creating figure
        ax.scatter(x, y, z, marker='o', c= 'tab:orange', s = 2)
        ax.scatter(x_pred, y_pred, z_pred, marker='o', c = 'tab:blue', s =2)

        ax.set_xlabel('pca 1', fontweight='bold')
        ax.set_ylabel('pca 2', fontweight='bold')
        ax.set_zlabel('pca 3', fontweight='bold')


    else:
        sns.scatterplot(x= x, y=y, s=2)
        sns.scatterplot(x= x_pred, y= y_pred, s=3)

    plt.title('Predictions shown against true space, iteration ' + str(iteration) +'comps: ' + str(components))
    # plt.text(x = x.max(), y = y.min(), z = z.max(), s = 'dad')
    ax.text(x = x.max(), y = y.min() * 1.5, z = 1.5 * z.max(), s = 'error: ' + str(round(pred_df['pcaerror'].mean(), 4)), fontsize = 'large')

    plt.show()
    # plt.clf()
    return

def cluster3D(pred_df, iteration, components):
    # Prediction shown vs Actual in same space
    x = pred_df['PCA1']
    y = pred_df['PCA2']
    x_pred = pred_df['PCA1_Prediction']
    y_pred = pred_df['PCA2_Prediction']
    if components == 3:
        z = pred_df['PCA3']
        z_pred = pred_df['PCA3_Prediction']

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # for i in [('True', 'o', 3),  ('Pred', '^', 1)]:
            # Creating figure
        # ax.scatter(x, y, z, marker='o', c= 'tab:orange', s = 2)
        ax.scatter(x_pred, y_pred, z_pred, marker='o', c = pred_df['cluster'], s =2)

        ax.set_xlabel('pca 1', fontweight='bold')
        ax.set_ylabel('pca 2', fontweight='bold')
        ax.set_zlabel('pca 3', fontweight='bold')


    else:
        sns.scatterplot(x= x, y=y, s=2)
        sns.scatterplot(x= x_pred, y= y_pred, s=3)

    plt.title('Predictions shown against true space, iteration ' + str(iteration) +'comps: ' + str(components))
    # plt.text(x = x.max(), y = y.min(), z = z.max(), s = 'dad')
    ax.text(x = x.max(), y = y.min() * 1.5, z = 1.5 * z.max(), s = 'error: ' + str(round(pred_df['pcaerror'].mean(), 4)), fontsize = 'large')

    plt.show()
    # plt.clf()
    return

def generate_comparison2(df_scaled, samples, outputs, components, specification_dict):

    comparison_df = Range_Specifications(df_scaled.copy(), specification_dict)
    comparison_df = df_scaled.sample(n = samples)
    comparison_df['algorithm'] = 'random'
    comparison_reduced, _ = add_PCA(comparison_df, outputs, components)

    return comparison_reduced

# fig = plt.figure(figsize=(10, 10))
# ax = plt.axes(projection='3d')
# colors = {0: 'tab:blue', 1:'tab:orange'}
# sizes = {0: 5, 1:50}
# # ax.scatter3D(pred_df['PCA1'], pred_df['PCA2'], pred_df['PCA3'], c = pred_df['overlap'].map(colors), s = pred_df['computed'].map(sizes))
# ax.scatter3D(pred_df['PCA1_Prediction'], pred_df['PCA2_Prediction'], pred_df['PCA3_Prediction'], c= pred_df['overlap'].map(colors), s = pred_df['computed'].map(sizes))
# plt.show()

def final_space2D(outputs, iteration, df_scaled, computed_df, comparison_df, comparison = False):
    reduced_all, pca = add_PCA(df_scaled, outputs)
    reduced_all['algorithm'] = 'None'
    reduced_all['algorithm'].loc[computed_df.index] = 'ZMART'
    if comparison:
        reduced_all['algorithm'].loc[comparison_df.index] = 'Random'
        both_idcs = list(set(comparison_df.index).intersection(computed_df.index))
        reduced_all['algorithm'].loc[both_idcs] = 'Both'

    #PLOTS:
    palette = {"ZMART": "tab:red", "Random": "tab:red", "None": "tab:gray", 'Both': "tab:red"}
    sizes = {"ZMART": 20, "Random": 20, "None": 1, 'Both': 20}


    if comparison:
        fig, axes = plt.subplots(1, 2, constrained_layout=True)
        fig.suptitle('Solutions in Real Space, iteration ' + str(iteration) + ' : ' + str(len(computed_df)) + '/' + str(len(df_scaled)))
        sns.scatterplot(data=reduced_all[reduced_all['algorithm'] != 'Random'],
                        x='PCA1', y='PCA2', hue='algorithm', size = 'algorithm', sizes = sizes,
                        palette= palette, ax=axes[0], alpha = 0.6)
        axes[0].get_legend().remove()
        axes[0].set_title('ZMART')
        sns.scatterplot(data=reduced_all[reduced_all['algorithm'] != 'ZMART'],
                        x='PCA1', y='PCA2', hue='algorithm', size='algorithm', sizes=sizes, alpha = 0.6,
                        palette=palette, ax=axes[1])
        axes[1].get_legend().remove()
        axes[1].set_title('Random')
        axes[1].get_yaxis().set_visible(False)
        axes[1].set(ylabel=None)
        axes[1].tick_params(left=False)

    else:
        sns.scatterplot(data=reduced_all, x='PCA1', y='PCA2', hue='algorithm', size='algorithm', sizes= sizes, palette= palette)

    # plt.text(0.9 * reduced_all['PCA1'].min(), 0.9 * reduced_all['PCA2'].max(), str(len(computed_df)) + '/' + str(len(df_scaled)),
    #              horizontalalignment='left', size='large', color='black', weight='semibold')

    plt.show()
    return reduced_all, pca

def display_solutions(iteration, df_scaled, computed_df, outputs, components, specification_dict, comparison = False, show_all_variants= False):
    comparison_df = generate_comparison2(df_scaled, len(computed_df), outputs, components, specification_dict)
    if components == 2:
        final_space, pca = final_space2D(outputs, iteration, df_scaled, computed_df, comparison_df)
    else:
        final_space, pca = final_space3D(outputs, iteration, df_scaled, computed_df, comparison_df, components,
                                         comparison = False, show_all_variants=show_all_variants)
        if comparison == True:
            fsc, _ = final_space3D(outputs, iteration, df_scaled, computed_df, comparison_df, components,
                                           comparison=True, show_all_variants=show_all_variants)

    return final_space, pca