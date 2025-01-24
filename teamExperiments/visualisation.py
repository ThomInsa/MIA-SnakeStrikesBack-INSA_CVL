from data.teamExperiments.compareDistributions import *

# Graphic conventions for our figures
palette_default = sns.color_palette("crest_r", n_colors=NUMBER_OF_COLUMNS_DATASET)
palette_PublicDS = sns.color_palette("viridis_r", n_colors=NUMBER_OF_COLUMNS_DATASET)
palette_SyntheticDS = sns.color_palette("magma_r", n_colors=NUMBER_OF_COLUMNS_DATASET)
palette_TrainingDS = sns.color_palette("cividis_r", n_colors=NUMBER_OF_COLUMNS_DATASET)
palette_PrivateDS = sns.color_palette("afmhot", n_colors=NUMBER_OF_COLUMNS_DATASET)
palette_Divergence = sns.color_palette("mako", n_colors=NUMBER_OF_COLUMNS_DATASET)

palette_doubleHistogram = sns.cubehelix_palette(start=.5, rot=-.5, n_colors=2, reverse=True)

figureLength = 40
figureWidth = 24
titleFontSize = 30
xlabelFontSize = 20
ylabelFontSize = 20
legendFontSize = 25

def plotDistributionByDay(dataframe, palette = palette_default, title = 'default_title', saveFile = False, savePath = ""):
    myLabels = []

    plt.figure(figsize=(figureLength, figureWidth))
    for j in range(NUMBER_OF_COLUMNS_DATASET):
        sns.kdeplot(
            dataframe[j],
            color = palette[j],
            fill=True,
            alpha=.5,
        )
        myLabels.append(j)

    plt.title(title, fontsize = titleFontSize)
    plt.xlabel('Intervals', fontsize = ylabelFontSize)
    plt.ylabel('Amount', fontsize = xlabelFontSize)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.legend(
        labels=myLabels,
        loc='upper right',
        fontsize=legendFontSize,
    )

    if saveFile:
        plt.savefig(savePath + '.png')

    # Show the plot
    plt.show()

def plotDoubleHistogram(dict1, dict2, xlabel = "Column Index", ylabel = "JS Divergence", palette = palette_doubleHistogram, title = 'default_title', df1label = "df1label", df2label = "df2label",saveFile = False, savePath = ""):
    data = []
    for col, value in dict1.items():
        data.append({"Column": col, "JS_Divergence": value, "Dataset": df1label})
    for col, value in dict2.items():
        data.append({"Column": col, "JS_Divergence": value, "Dataset": df2label})

    js_df = pd.DataFrame(data)

    sns.barplot(x="Column", y="JS_Divergence", hue="Dataset", data=js_df, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title="Dataset")
    plt.show()

    if saveFile:
        plt.savefig(savePath + '.png')

def plotSimpleDivergenceHistogram(targetSet, evaluationSet, xlabel ="Column Index", ylabel ="JS Divergence", palette = palette_default, title ='default_title', saveFile = False, savePath =""):
    JS = getJensenShannonDivergences(targetSet, evaluationSet)
    data = []
    for col, value in JS.items():
        data.append({"Column": col, "JS_Divergence": value})
    js_df = pd.DataFrame(data)

    plt.figure(figsize=(figureLength, figureWidth))
    sns.barplot(data=js_df, palette=palette, x="Column", y="JS_Divergence", hue="Column")
    plt.title(title, fontsize = titleFontSize)
    plt.xlabel(xlabel, fontsize = xlabelFontSize)
    plt.ylabel(ylabel, fontsize = ylabelFontSize)
    plt.legend(title="Dataset")

    if saveFile:
        plt.savefig(savePath + '.png')

    plt.show()

