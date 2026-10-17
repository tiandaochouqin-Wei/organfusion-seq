#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch the UCR benchmark of Stage 3P: every two-class, equal-length, complete dataset of the
UCR 2018 archive (Dau et al., 2019).

Selection rule (fixed before any result was seen, so the benchmark is not hand-picked):
  number of classes == 2 in the archive's DataSummary.csv          -> 42 datasets
  minus the two with missing values (DodgerLoopGame, DodgerLoopWeekend) -> 40 datasets
No two-class dataset has variable length.

Source: https://www.timeseriesclassification.com/aeon-toolkit/<Name>.zip (the archive's mirror;
the four datasets of the earlier runs came from the same place). Only <Name>_TRAIN.txt and
<Name>_TEST.txt are extracted, each dataset into its own directory.

Usage: python download_ucr_binary.py [target_dir]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, sys, io, time, zipfile, urllib.request

BINARY_UCR = [
    "BeetleFly", "BirdChicken", "Chinatown", "Coffee", "Computers", "DistalPhalanxOutlineCorrect",
    "ECG200", "ECGFiveDays", "Earthquakes", "FordA", "FordB", "FreezerRegularTrain",
    "FreezerSmallTrain", "GunPoint", "GunPointAgeSpan", "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung", "Ham", "HandOutlines", "Herring", "HouseTwenty", "ItalyPowerDemand",
    "Lightning2", "MiddlePhalanxOutlineCorrect", "MoteStrain", "PhalangesOutlinesCorrect",
    "PowerCons", "ProximalPhalanxOutlineCorrect", "SemgHandGenderCh2", "ShapeletSim",
    "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "Strawberry", "ToeSegmentation1",
    "ToeSegmentation2", "TwoLeadECG", "Wafer", "Wine", "WormsTwoClass", "Yoga",
]
URL = "https://www.timeseriesclassification.com/aeon-toolkit/{}.zip"


def main(target):
    os.makedirs(target, exist_ok=True)
    total = 0
    for name in BINARY_UCR:
        ddir = os.path.join(target, name)
        want = [f"{name}_TRAIN.txt", f"{name}_TEST.txt"]
        if all(os.path.isfile(os.path.join(ddir, w)) for w in want):
            print(f"  have   {name}")
            continue
        req = urllib.request.Request(URL.format(name), headers={"User-Agent": "python-urllib"})
        for attempt in range(3):
            try:
                blob = urllib.request.urlopen(req, timeout=120).read()
                break
            except Exception as e:                       # transient network errors: retry
                print(f"  retry  {name}: {type(e).__name__}: {e}")
                time.sleep(3 * (attempt + 1))
        else:
            print(f"  FAILED {name}")
            continue
        total += len(blob)
        os.makedirs(ddir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            members = {os.path.basename(m): m for m in z.namelist()}
            for w in want:                               # fixed target names: no path from the zip
                if w not in members:
                    print(f"  !! {name}: {w} not in the archive ({sorted(members)[:6]})")
                    continue
                with open(os.path.join(ddir, w), "wb") as f:
                    f.write(z.read(members[w]))
        print(f"  got    {name:32s} {len(blob)/1e6:6.1f} MB")
    print(f"downloaded {total/1e6:.1f} MB; "
          f"{sum(os.path.isfile(os.path.join(target, n, n + '_TRAIN.txt')) for n in BINARY_UCR)}"
          f"/{len(BINARY_UCR)} datasets present in {target}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_DIR, "ucr"))
