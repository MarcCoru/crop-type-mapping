import sys
sys.path.append("..")
import unittest
from viz.winloosetables import parse_sota
import os
import hashlib
import filecmp
import glob
from viz.viz import plot_scatter, qualitative_figure
thisdir = os.path.dirname(os.path.realpath(__file__))
import shutil

class TestExperiments(unittest.TestCase):

    def test_winlosstable(self):
        table = parse_sota(mode="score",runscsv = "../data/runs_conv1d.csv", comparepath="../data/UCR_Datasets")
        table = table.replace(" ", "")

        tablehash = hashlib.sha1(table.encode()).hexdigest()


        """ reference table from the paper
        &mori&relclass&edsc&ects\\
        \cmidrule(lr){1-1}\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}
        $0.6$&7/\textbf{38}&\textbf{31}/14&\textbf{34}/8&\textbf{40}/5\\
        $0.7$&3/\textbf{42}&\textbf{28}/17&\textbf{28}/14&\textbf{35}/10\\
        $0.8$&5/\textbf{40}&\textbf{23}/22&\textbf{30}/12&\textbf{34}/11\\
        $0.9$&12/\textbf{33}&19/\textbf{26}&\textbf{33}/9&\textbf{26}/19\\
        """
        tablehash_reference = '080233f973beb725880a8ae1dd8cce155c87e655'

        self.assertEqual(tablehash, tablehash_reference)

    def test_scatter(self):
        "tests/data/paper"

        paper_csv_directory = "tests/data/paper/mori"
        computed_csv_directory = "../viz/csv/mori"

        if os.path.exists(computed_csv_directory):
            print("deleting "+computed_csv_directory)
            #shutil.rmtree(computed_csv_directory)

        csvfile = "../data/runs_conv1d.csv"
        metafile = "../data/UCR_Datasets/DataSummary.csv"
        mori_accuracy = "../data/UCR_Datasets/mori-accuracy-sr2-cf2.csv"
        mori_earliness = "../data/UCR_Datasets/mori-earliness-sr2-cf2.csv"
        relclass_accuracy = "../data/UCR_Datasets/relclass-accuracy-gaussian-quadratic-set.csv"
        relclass_earliness = "../data/UCR_Datasets/relclass-earliness-gaussian-quadratic-set.csv"

        plot_scatter(csvfile, metafile, mori_accuracy, mori_earliness, relclass_accuracy, relclass_earliness)
        #qualitative_figure(csvfile,compare_accuracy=mori_accuracy,compare_earliness=mori_earliness)

        paper_csv_files = glob.glob(paper_csv_directory+"/*.csv")

        for f in paper_csv_files:
            computed_file = f.replace(paper_csv_directory, computed_csv_directory)
            self.assertTrue(os.path.exists(computed_file),"Expected computed file {} not found".format(computed_file))
            self.assertTrue(filecmp.cmp(computed_file, f), "Computed file {} not identical with reference {}".format(computed_file, f))



if __name__ == '__main__':
    unittest.main()