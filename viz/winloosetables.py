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


def choose_params(alpha):
    if alpha == 0.6:
        relclass_col="t=0.001"
        edsc_col="t=2.5"
        ects_col="sup=0.1"
    elif alpha == 0.7:
        relclass_col = "t=0.1"
        ects_col="sup=0.2"
        edsc_col="t=3"
    elif alpha == 0.8:
        relclass_col = "t=0.5"
        ects_col="sup=0.4"
        edsc_col = "t=3.5"
    elif alpha == 0.9:
        relclass_col = "t=0.9"
        ects_col="sup=0.8"
        edsc_col="t=3.5"

    return relclass_col, ects_col, edsc_col

def load_approaches(alpha=0.6,relclass_col="t=0.001",edsc_col="t=2.5",ects_col="sup=0.05", csvfile = "data/sota_comparison/runs.csv"):

    relclass_col, ects_col, edsc_col = choose_params(alpha)

    mori = load("data/morietal2017/mori-{}-sr2-cf2.csv","a={}".format(alpha), "mori")
    relclass = load("data/morietal2017/relclass-{}-gaussian-quadratic-set.csv",relclass_col, "relclass")
    edsc = load("data/morietal2017/edsc-{}.csv",edsc_col, "edsc")
    ects = load("data/morietal2017/ects-{}-strict-method.csv",ects_col, "ects")

    df = pd.read_csv(csvfile)
    ours = df.loc[df.earliness_factor == alpha].set_index("dataset")[["accuracy","earliness"]]
    ours = ours.rename(columns={"accuracy":"ours_accuracy","earliness":"ours_earliness"})


    #csvfile = "data/{loss}/a{alpha}e{entropy_factor}.csv".format(loss=loss, alpha=alpha, entropy_factor=entropy_factor)
    #accuracy = pd.read_csv(csvfile, index_col=0)["phase2_accuracy"]
    #accuracy.name = "ours_accuracy"
    #earliness = pd.read_csv(csvfile, index_col=0)["phase2_earliness"]
    #earliness.name = "ours_earliness"

    return pd.concat([mori,relclass,edsc,ects, ours], axis=1, join="inner")

def parse_domination_score(dataframe, compare="mori", alpha=None):

    score_earliness = dataframe["ours_earliness"] < dataframe[compare+"_earliness"]
    score_accuracy = dataframe["ours_accuracy"] > dataframe[compare+"_accuracy"]

    score = pd.concat([score_earliness,score_accuracy],axis=1).sum(1)

    won = (score==2).sum()
    draw = (score==1).sum()
    lost = (score==0).sum()

    return r"\textbf{"+str(won)+"}"+" / {draw} / {lost}".format(draw=draw, lost=lost)

def parse_winloose_score(dataframe, compare="mori", alpha=None, mode="score"):

    def calc_loss(accuracy,earliness):
        return alpha * accuracy + (1 - alpha) * (earliness)

    def textbf(v):
        return r"\textbf{" + str(v) + "}"

    if mode=="score":
        ours = calc_loss(dataframe["ours_accuracy"], 1-dataframe["ours_earliness"])
        other = calc_loss(dataframe[compare+"_accuracy"], 1-dataframe[compare+"_earliness"])
    elif mode=="accuracy":
        ours = dataframe["ours_accuracy"]
        other = dataframe[compare+"_accuracy"]
    elif mode=="earliness":
        ours = dataframe["ours_earliness"]
        other = dataframe[compare + "_earliness"]

    won = (ours > other).sum()
    draw = (ours == other).sum()
    lost = (ours < other).sum()

    if won > lost:
        won = textbf(won)
    elif lost > won:
        lost = textbf(lost)

    if draw > 0:
        return r"{won} / {draw} / {lost}".format(won=won, draw=draw, lost=lost)
    if draw == 0:
        return r"{won} / {lost}".format(won = won, lost=lost)


def parse_winloose_score_accuracy_only(dataframe, compare="mori", alpha=None, mode="score"):

    def calc_loss(accuracy,earliness):
        return alpha * accuracy + (1 - alpha) * (earliness)

    def textbf(v):
        return r"\textbf{" + str(v) + "}"

    if mode=="score":
        ours = calc_loss(dataframe["ours_accuracy"], 1-dataframe["ours_earliness"])
        other = calc_loss(dataframe[compare+"_accuracy"], 1-dataframe[compare+"_earliness"])
    elif mode=="accuracy":
        ours = dataframe["ours_accuracy"]
        other = dataframe[compare+"_accuracy"]
    elif mode=="earliness":
        ours = dataframe["ours_earliness"]
        other = dataframe[compare + "_earliness"]

    won = (ours > other).sum()
    draw = (ours == other).sum()
    lost = (ours < other).sum()

    if won > lost:
        won = textbf(won)
    elif lost > won:
        lost = textbf(lost)

    if draw > 0:
        return r"{won} / {draw} / {lost}".format(won=won, draw=draw, lost=lost)
    if draw == 0:
        return r"{won} / {lost}".format(won = won, lost=lost)


def parse_sota(mode="score", runscsv = "data/sota_comparison/runs.csv", comparepath="data/morietal2017"):

    approaches = ["mori", "relclass", "edsc", "ects"] # "e0",

    outstring = list()
    outstring.append(r"& " + " & ".join(approaches) + r" \\")
    outstring.append("".join(["\cmidrule(lr){" + str(i) + "-" + str(i) + "}" for i in range(1, len(approaches) + 1)]))

    for alpha in [0.6, 0.7, 0.8, 0.9]: # , 0.7, 0.8, 0.8

        relclass_col, ects_col, edsc_col = choose_params(alpha)

        mori = load(comparepath+"/mori-{}-sr2-cf2.csv", "a={}".format(alpha), "mori")
        relclass = load(comparepath+"/relclass-{}-gaussian-quadratic-set.csv", relclass_col, "relclass")
        edsc = load(comparepath+"/edsc-{}.csv", edsc_col, "edsc")
        ects = load(comparepath+"/ects-{}-strict-method.csv", ects_col, "ects")

        # ours regularized based on beta = 0.01 and epsilon=5/T
        def load_ours(ptsepsilon, entropy_factor):
            df = pd.read_csv(runscsv)
            df = df.loc[df.ptsepsilon == ptsepsilon]
            df = df.loc[df.entropy_factor == entropy_factor]
            ours = df.loc[df.earliness_factor == alpha].set_index("dataset")[["accuracy", "earliness"]].sort_values(by="accuracy")
            return ours.groupby(ours.index).mean() #ours[~ours.index.duplicated(keep='first')]

        ours_unregularized = load_ours(ptsepsilon=0, entropy_factor=0)
        ours_unregularized = ours_unregularized.rename(columns={"accuracy": "unreg_accuracy", "earliness": "unreg_earliness"})

        ours = load_ours(ptsepsilon=0, entropy_factor=0)
        ours = ours.rename(columns={"accuracy": "ours_accuracy", "earliness": "ours_earliness"})

        # ours regularized beta = 0 and epsilon = 0
        #df = pd.read_csv("data/entropy_pts/runs.csv")
        #e0 = select(df, entropy_factor=0)
        #e0 = e0.rename(columns={"accuracy": "e0_accuracy", "earliness": "e0_earliness"})

        #e0,
        dataframe = pd.concat([mori, relclass, edsc, ects, ours, ours_unregularized], axis=1, join="inner")

        line = list()

        for compare in approaches:
            line.append(parse_winloose_score(dataframe=dataframe, compare=compare, alpha=alpha, mode=mode))

        outstring.append( r"${}$ & ".format(alpha) + " & ".join(line) + r' \\')

    return "\n".join(outstring)

def parse_domination_more_betas():

    approaches = ["mori", "edsc", "relclass", "ects"]

    for entropy_factor in [0.1, 0.01, 0]:
        print()
        print("entropy factor {}".format(entropy_factor))
        print()
        for alpha in [0.6, 0.7, 0.8, 0.9]:  # , 0.7, 0.8, 0.8

            mori = load("data/morietal2017/mori-{}-sr2-cf2.csv", "a={}".format(alpha), "mori")
            relclass = load("data/morietal2017/relclass-{}-gaussian-quadratic-set.csv", relclass_col, "relclass")
            edsc = load("data/morietal2017/edsc-{}.csv", edsc_col, "edsc")
            ects = load("data/morietal2017/ects-{}-strict-method.csv", ects_col, "ects")

            df = pd.read_csv("data/entropy_pts/runs.csv")
            df = df.loc[df.entropy_factor == entropy_factor]


            ours = df.loc[df.earliness_factor == alpha].set_index("dataset")[["accuracy", "earliness"]].sort_values(by="accuracy")
            ours = ours.rename(columns={"accuracy": "ours_accuracy", "earliness": "ours_earliness"})

            ours = ours[~ours.index.duplicated(keep='first')]

            dataframe = pd.concat([mori, relclass, edsc, ects, ours], axis=1, join="inner")

            line = list()

            for compare in approaches:
                line.append(parse_winloose_score(dataframe=dataframe, compare=compare, alpha=alpha))

            print(r"$\alpha={}$ & ".format(alpha) + " & ".join(line) + r' \\')

def select(df, entropy_factor, ptsepsilon=20):
    df = df.loc[df.entropy_factor == entropy_factor]

    ours = df.loc[df.earliness_factor == alpha].set_index("dataset")[["accuracy", "earliness"]].sort_values(
        by="accuracy")

    return ours[~ours.index.duplicated(keep='first')]


def parse_domination_inter_betas():

    approaches = ["e001", "e01"]

    for alpha in [0.6, 0.7, 0.8, 0.9]:  # , 0.7, 0.8, 0.8

        df = pd.read_csv("data/entropy_pts/runs.csv")

        e001 = select(df,entropy_factor=0.01)
        e001 = e001.rename(columns={"accuracy": "e001_accuracy", "earliness": "e001_earliness"})

        e01 = select(df,entropy_factor=0.1)
        e01 = e01.rename(columns={"accuracy": "e01_accuracy", "earliness": "e01_earliness"})

        e0 = select(df, entropy_factor=0)
        e0 = e0.rename(columns={"accuracy": "ours_accuracy", "earliness": "ours_earliness"})

        dataframe = pd.concat([e0, e001, e01], axis=1, join="inner")

        line = list()

        for compare in approaches:
            line.append(parse_winloose_score(dataframe=dataframe, compare=compare, alpha=alpha))

        print(r"${}$ & ".format(alpha) + " & ".join(line) + r' \\')

def parse_accuracy_all():

    UCR = pd.read_csv("data/UCR_Datasets/singleTrainTest.csv", index_col=0)

    approaches = list(UCR.columns)
    new_colnames = [name + "_accuracy" for name in approaches]
    UCR = UCR.rename(columns=dict(zip(approaches, new_colnames)))

    print(r"& " + " & ".join(approaches) + r" \\")
    print("".join(["\cmidrule(lr){" + str(i) + "-" + str(i) + "}" for i in range(1, len(approaches) + 1)]))

    #"e0", "mori", "edsc", "relclass", "ects",
    #approaches = list(UCR_Datasets.columns)

    for alpha in [0.6, 0.7, 0.8, 0.9]:  # , 0.7, 0.8, 0.8

        mori = load("data/morietal2017/mori-{}-sr2-cf2.csv", "a={}".format(alpha), "mori")
        relclass = load("data/morietal2017/relclass-{}-gaussian-quadratic-set.csv", relclass_col, "relclass")
        edsc = load("data/morietal2017/edsc-{}.csv", edsc_col, "edsc")
        ects = load("data/morietal2017/ects-{}-strict-method.csv", ects_col, "ects")

        # ours regularized based on beta = 0.01 and epsilon=5/T
        df = pd.read_csv("data/sota_comparison/runs.csv")
        ours = df.loc[df.earliness_factor == alpha].set_index("dataset")[["accuracy", "earliness"]].sort_values(
            by="accuracy")
        ours = ours[~ours.index.duplicated(keep='first')]
        ours = ours.rename(columns={"accuracy": "ours_accuracy", "earliness": "ours_earliness"})

        # ours regularized beta = 0 and epsilon = 0
        df = pd.read_csv("data/entropy_pts/runs.csv")
        e0 = select(df, entropy_factor=0)
        e0 = e0.rename(columns={"accuracy": "e0_accuracy", "earliness": "e0_earliness"})

        # UCR_Datasets dataset (no)

        # UCR_Datasets dataset (no)

        dataframe = pd.concat([e0, mori, relclass, edsc, ects, ours, UCR], axis=1, join="inner")

        line=list()

        for compare in approaches:
            line.append(parse_winloose_score_accuracy_only(dataframe=dataframe, compare=compare, alpha=alpha))

        print(r"${}$ & ".format(alpha) + " & ".join(line) + r' \\')

if __name__=="__main__":

    print("SOTA score")
    table = parse_sota(mode="score", comparepath="../data/UCR_Datasets")

    #print("SOTA accuracy")
    #parse_sota(mode="accuracy")

    #print("SOTA earliness")
    #parse_sota(mode="earliness")

    #print("BETAS")
    #parse_domination_more_betas()

    #print("INTER BETAS")
    #parse_domination_inter_betas()

    #parse_accuracy_all()
