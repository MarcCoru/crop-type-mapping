#!/usr/bin/env bash

convert -delay 200 -loop 0 plots/accuracy*.png plots/accuracy.gif
convert -delay 200 -loop 0 plots/earliness*.png plots/earliness.gif
convert -delay 200 -loop 0 plots/phase1vs2_accuracy*.png plots/phase1vs2_accuracy.gif
convert -delay 200 -loop 0 $(ls -tr plots/accuracy_vs_earliness*.png) plots/accuracy_vs_earliness.gif