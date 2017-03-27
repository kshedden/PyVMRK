#!/usr/bin/python

"""
Process one VMRK file into a set of summary statistics, optionally
filtering outliers.

The work is done in two steps.  First run process_vmrk to obtain a
condensed form of the VMRK data, then run summarize_vmrk to obtain the
summary statistics.
"""

import csv
import numpy as np
import os
import logging
from collections import OrderedDict


class smry(object):
    """
    Holds all information about a set of trials
    """
    def __init__(self):
        self.RT = []  # Response times
        self.NT = []  # Number of responses within a trial


def process_trial(qu, data):
    """
    Insert summary values obtained from qu into data.

    Parameters
    ----------
    qu :
        Data from a single trial.
    data :
        All data grouped by trial/response type
    """

    # Trial must start with S99 (fixation cross), then have a stimulus
    # and response.  Return early if this is not the case.
    if len(qu) < 3 or qu[0][0] != 99:
        return

    # ky is the stimulus and response type
    ky = (qu[1][0], qu[2][0])

    # Response time for first response, multiplication by 2 is a scale
    # conversion.
    rt = 2 * (qu[2][1] - qu[1][1])
    data[ky].RT.append(rt)
    data[ky].NT.append(len(qu))


def summarize_vmrk(filename, data):
    """
    Create summary statistics from a VMRK file.

    The VMRK file must be processed with process_vmrk before running
    this function.
    """

    results = OrderedDict()

    results["sid"] = filename.split(".")[0]

    all_trials = []
    correct_trials = []
    error_trials = []
    for k, v in data.items():
        all_trials.extend(v.RT)
        if k[1] in (5, 6):
            correct_trials.extend(v.RT)
        else:
            error_trials.extend(v.RT)

    results["fcn"] = len(correct_trials)
    results["fen"] = len(error_trials)
    results["facc"] = 100 * results["fcn"] / len(all_trials)

    # All trial summaries
    results["frtm"] = np.mean(all_trials)
    results["frtsd"] = np.std(all_trials)

    # All correct trial summaries
    results["frtmc"] = np.mean(correct_trials)
    results["frtsdc"] = np.std(correct_trials)

    # All error trial summaries
    results["frtme"] = np.mean(error_trials)
    results["frtsde"] = np.std(error_trials)

    # Congruent correct trials
    v = data[(1, 5)].RT + data[(1, 6)].RT + data[(3, 5)].RT + data[(3, 6)].RT
    results["fccn"] = len(v)
    results["fcrtmc"] = np.mean(v)
    results["fcrtsdc"] = np.std(v)

    # Congruent error trials
    v = data[(1, 15)].RT + data[(1, 16)].RT + data[(2, 15)].RT + data[(2, 16)].RT
    results["fcen"] = len(v)
    results["fcrtme"] = np.mean(v)
    results["fcrtsde"] = np.std(v)

    # Congruent accuracy
    results["fcacc"] = 100 * results["fccn"] / (results["fccn"] + results["fcen"])

    # Incongruent correct trials
    v = data[(2, 5)].RT + data[(2, 6)].RT + data[(4, 5)].RT + data[(4, 6)].RT
    results["ficn"] = len(v)
    results["firtmc"] = np.mean(v)
    results["firtsdc"] = np.std(v)

    # Incongruent error trials
    v = data[(2, 15)].RT + data[(2, 16)].RT + data[(4, 15)].RT + data[(4, 16)].RT
    results["fien"] = len(v)
    results["firtme"] = np.mean(v)
    results["firtsde"] = np.std(v)

    # Incongruent accuracy
    results["fiacc"] = 100 * results["ficn"] / (results["ficn"] + results["fien"])

    # Anticipatory responses
    results["fan"] = np.sum(np.asarray(all_trials) < 150)
    results["faen"] = np.sum(np.asarray(error_trials) < 150)

    # Trials with extra responses
    results["fscn"] = 0
    for _, v in data.items():
        results["fscn"] += sum(np.asarray(v.NT) > 3)

    return results


def process_vmrk(filename, save_reduced=False, log=False):
    """
    Process the VMRK format file with name filename.

    Parameters
    ----------
    save_reduced : boolean
        If save_reduced is True, and the input file has name xyz.vmrk,
        then a file called xyz_reduced.vmrk is created that contains
        only the non-practice experiment records.

    Returns
    -------
    data : dictionary
        data[(x, y)] contains all response times with stimulus type x and
        response type y.
    """

    if log:
        logging.basicConfig(filename=filename + ".log",
                            level=logging.DEBUG)

    fid = open(filename)
    rdr = csv.reader(fid)

    # Assume that we start in practice mode
    mode = "practice"
    qu = []
    data = {}
    for j in 1, 2, 3, 4:
        for k in 5, 6, 15, 16:
            data[(j, k)] = smry()

    # Open file for reduced output
    q = os.path.splitext(filename)
    outname = q[0] + "_reduced" + q[1]
    out = open(outname, "w")

    for line in rdr:

        # Only process "mark" lines
        if len(line) == 0 or not line[0].startswith("Mk"):
            if log:
                logging.info("Skipping row: %s" % line)
            continue

        # Lines have format Mk###=type, where type=comment, stimulus
        f0 = line[0].split("=")
        fl = f0[1].lower()
        if fl == "comment":
            # Switch to (or remain in) "experiment" mode
            if "experiment" in line[1].lower():
                mode = "experiment"
            continue
        elif fl == "stimulus":
            pass
        else:
            # Not sure what else exists, log it and move on
            if log:
                logging.info("Skipping row: %s" % line)
            continue

        if mode != "experiment":
            continue

        out.write(",".join(line) + "\n")

        # Get the type code, e.g. if S16 then n=16
        f1 = line[1].replace(" ", "")
        n = int(f1[1:])
        qu.append([n, int(line[2])])

        if n == 99:
            process_trial(qu[0:-1], data)
            qu = [qu[-1]]

    # Final trial may not have been processed
    process_trial(qu, data)

    out.close()

    return data


filename = "000041.vmrk"
data = process_vmrk(filename, log=True)
results = summarize_vmrk(filename, data)
