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


def load_approaches(alpha=0.6,relclass_col="t=0.001",edsc_col="t=2.5",ects_col="sup=0.05", csvfile = "data/sota_comparison/runs.csv"):
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

def parse_domination_score(dataframe, compare="mori"):

    score_earliness = dataframe["ours_earliness"] < dataframe[compare+"_earliness"]
    score_accuracy = dataframe["ours_accuracy"] > dataframe[compare+"_accuracy"]

    score = pd.concat([score_earliness,score_accuracy],axis=1).sum(1)

    won = (score==2).sum()
    draw = (score==1).sum()
    lost = (score==0).sum()

    return r"\textbf{"+str(won)+"}"+" / {draw} / {lost}".format(draw=draw, lost=lost)

def parse_winloose_score(dataframe, compare="mori", alpha=None):

    def calc_loss(accuracy,earliness):
        return alpha * accuracy + (1 - alpha) * (1-earliness)

    def textbf(v):
        return r"\textbf{" + str(v) + "}"

    ours = calc_loss(dataframe["ours_accuracy"],dataframe["ours_earliness"])
    other = calc_loss(dataframe[compare+"_accuracy"],dataframe[compare+"_earliness"])

    won = (ours > other).sum()
    draw = (ours == other).sum()
    lost = (ours < other).sum()

    if won > lost:
        won = textbf(won)
    elif lost > won:
        lost = textbf(lost)
    else:
        draw = textbf(draw)

    if draw > 0:
        return r"{won} / {draw} / {lost}".format(won=won, draw=draw, lost=lost)
    if draw == 0:
        return r"{won} / {lost}".format(won = won, lost=lost)


def parse_winloose_score_accuracy_only(dataframe, compare="mori", alpha=None):

    def calc_loss(accuracy,earliness):
        return alpha * accuracy + (1 - alpha) * (1-earliness)

    def textbf(v):
        return r"\textbf{" + str(v) + "}"

    ours = dataframe["ours_accuracy"]
    other = dataframe[compare+"_accuracy"]

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


def parse_sota():

    for alpha in [0.6, 0.7, 0.8, 0.9]: # , 0.7, 0.8, 0.8

        mori = load("data/morietal2017/mori-{}-sr2-cf2.csv", "a={}".format(alpha), "mori")
        relclass = load("data/morietal2017/relclass-{}-gaussian-quadratic-set.csv", relclass_col, "relclass")
        edsc = load("data/morietal2017/edsc-{}.csv", edsc_col, "edsc")
        ects = load("data/morietal2017/ects-{}-strict-method.csv", ects_col, "ects")

        # ours regularized based on beta = 0.01 and epsilon=5/T
        df = pd.read_csv("data/sota_comparison/runs.csv")
        ours = df.loc[df.earliness_factor == alpha].set_index("dataset")[["accuracy", "earliness"]]
        ours = ours.rename(columns={"accuracy": "ours_accuracy", "earliness": "ours_earliness"})

        # ours regularized beta = 0 and epsilon = 0
        df = pd.read_csv("data/entropy_pts/runs.csv")
        e0 = select(df, entropy_factor=0)
        e0 = e0.rename(columns={"accuracy": "e0_accuracy", "earliness": "e0_earliness"})

        dataframe = pd.concat([e0, mori, relclass, edsc, ects, ours], axis=1, join="inner")

        line = list()

        for compare in ["e0", "mori", "edsc", "relclass", "ects"]:
            line.append(parse_winloose_score(dataframe=dataframe, compare=compare, alpha=alpha))

        print( r"${}$ & ".format(alpha) + " & ".join(line) + r' \\')

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

def select(df, entropy_factor):
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

    UCR = pd.read_csv("data/UCR/singleTrainTest.csv", index_col=0)

    approaches = list(UCR.columns)
    new_colnames = [name + "_accuracy" for name in approaches]
    UCR = UCR.rename(columns=dict(zip(approaches, new_colnames)))

    print(r"& " + " & ".join(approaches) + r" \\")
    print("".join(["\cmidrule(lr){" + str(i) + "-" + str(i) + "}" for i in range(1, len(approaches) + 1)]))

    #"e0", "mori", "edsc", "relclass", "ects",
    #approaches = list(UCR.columns)

    for alpha in [0.6, 0.7, 0.8, 0.9]:  # , 0.7, 0.8, 0.8

        mori = load("data/morietal2017/mori-{}-sr2-cf2.csv", "a={}".format(alpha), "mori")
        relclass = load("data/morietal2017/relclass-{}-gaussian-quadratic-set.csv", relclass_col, "relclass")
        edsc = load("data/morietal2017/edsc-{}.csv", edsc_col, "edsc")
        ects = load("data/morietal2017/ects-{}-strict-method.csv", ects_col, "ects")

        # ours regularized based on beta = 0.01 and epsilon=5/T
        df = pd.read_csv("data/sota_comparison/runs.csv")
        ours = df.loc[df.earliness_factor == alpha].set_index("dataset")[["accuracy", "earliness"]]
        ours = ours.rename(columns={"accuracy": "ours_accuracy", "earliness": "ours_earliness"})

        # ours regularized beta = 0 and epsilon = 0
        df = pd.read_csv("data/entropy_pts/runs.csv")
        e0 = select(df, entropy_factor=0)
        e0 = e0.rename(columns={"accuracy": "e0_accuracy", "earliness": "e0_earliness"})

        # UCR dataset (no)

        # UCR dataset (no)

        dataframe = pd.concat([e0, mori, relclass, edsc, ects, ours, UCR], axis=1, join="inner")

        line=list()

        for compare in approaches:
            line.append(parse_winloose_score_accuracy_only(dataframe=dataframe, compare=compare, alpha=alpha))

        print(r"${}$ & ".format(alpha) + " & ".join(line) + r' \\')

if __name__=="__main__":

    print("SOTA")
    parse_sota()

    print("BETAS")
    parse_domination_more_betas()

    print("INTER BETAS")
    parse_domination_inter_betas()

    parse_accuracy_all()
