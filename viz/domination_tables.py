import pandas as pd

loss = "twophase_linear_loss"
alpha=0.6
entropy_factor=0.01

relclass_col = "t=0.001"
edsc_col = "t=2.5"
ects_col = "sup=0.05"

def load(file, column, name):
    accuracy = pd.read_csv(file.format("accuracy"), sep=' ').set_index("Dataset")[column] # accuracy is scaled 0-1
    accuracy.name = name + "_accuracy"
    earliness = pd.read_csv(file.format("earliness"), sep=' ').set_index("Dataset")[column] * 0.01 # earliness is scaled 1-100
    earliness.name = name + "_earliness"
    return pd.concat([accuracy,earliness],axis=1)


def load_approaches(alpha=0.6,relclass_col="t=0.001",edsc_col="t=2.5",ects_col="sup=0.05"):
    mori = load("data/morietal2017/mori-{}-sr2-cf2.csv","a={}".format(alpha), "mori")
    relclass = load("data/morietal2017/relclass-{}-gaussian-quadratic-set.csv",relclass_col, "relclass")
    edsc = load("data/morietal2017/edsc-{}.csv",edsc_col, "edsc")
    ects = load("data/morietal2017/ects-{}-strict-method.csv",ects_col, "ects")

    csvfile = "data/sota_comparison/runs.csv"
    df = pd.read_csv(csvfile)
    ours = df.loc[df.earliness_factor == alpha].set_index("dataset")[["accuracy","earliness"]]
    ours = ours.rename(columns={"accuracy":"ours_accuracy","earliness":"ours_earliness"})


    #csvfile = "data/{loss}/a{alpha}e{entropy_factor}.csv".format(loss=loss, alpha=alpha, entropy_factor=entropy_factor)
    #accuracy = pd.read_csv(csvfile, index_col=0)["phase2_accuracy"]
    #accuracy.name = "ours_accuracy"
    #earliness = pd.read_csv(csvfile, index_col=0)["phase2_earliness"]
    #earliness.name = "ours_earliness"

    return pd.concat([mori,relclass,edsc,ects, ours], axis=1, join="inner")

def parse_domination_score(dataframe, compare="mori"):

    score_earliness = dataframe["ours_earliness"] <= dataframe[compare+"_earliness"]
    score_accuracy = dataframe["ours_accuracy"] >= dataframe[compare+"_accuracy"]

    score = pd.concat([score_earliness,score_accuracy],axis=1).sum(1)

    won = (score==2).sum()
    draw = (score==1).sum()
    lost = (score==0).sum()

    return r"\textbf{"+str(won)+"}"+" / {draw} / {lost}".format(draw=draw, lost=lost)


approaches = ["mori","edsc","relclass","ects"]

for alpha in [0.6, 0.7, 0.8, 0.9]: # , 0.7, 0.8, 0.8

    dataframe = load_approaches(alpha, relclass_col, edsc_col, ects_col)

    line = list()

    for compare in approaches:
        line.append(parse_domination_score(dataframe=dataframe, compare=compare))

    print( r"${}$ & ".format(alpha) + " & ".join(line) + r' \\')