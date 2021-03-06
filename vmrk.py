#!/usr/bin/python

"""
Process one VMRK file into a set of summary statistics.
"""

import csv
import numpy as np
import os
import logging
from collections import OrderedDict
import argparse

# Outliers are defined to fall outside the interval (low, high)
low = 150
high = 3000

# Degrees of freedom adjustment for standard deviations, should generally be 1
ddof = 1


class Code(object):
    """
    A representation of a stimulus/response code
    """
    def __init__(self, side, congruent, correct):
        self.side = side
        self.congruent = congruent
        self.correct = correct

    def __str__(self):
        return "side=%s congruent=%r correct=%r" % (
            self.side, self.congruent, self.correct)

    def fromSRCodes(a, b):
        if a in (1, 2):
            side = "left"
        else:
            side = "right"
        if a in (1, 3):
            congruent = True
        else:
            congruent = False
        if b in (5, 6):
            correct = True
        else:
            correct = False

        return Code(side, congruent, correct)


class Trial(object):
    """
    A representation of a trial.

    A trial is represented by a mark,  stimulus/response code, and a timestamp.
    """
    def __init__(self, mark, srcode, time):
        self.mark = mark
        self.srcode = srcode
        self.time = time

    def __str__(self):
        return "mark=%s srcode=%s time=%d" % (
                self.mark, self.srcode, self.time)


class Block(object):
    """
    A representation of a block of trials
    """
    def __init__(self):
        self.Mark = []    # Mark label
        self.Code = []    # Stimulus/response codes
        self.Rtim = []    # Response times
        self.Ntri = []    # Number of responses within a trial
        self.Outl = []    # Outlier trial indicator

    def check_outliers(self, low, high):
        """
        Identify outlier trials from the block.

        Outlier trials are replace with None.
        """

        for i, rt in enumerate(self.Rtim):
            if rt < low or rt > high:
                self.Outl[i] = True

    def query(self, side=None, congruent=None, correct=None, lastcorrect=None,
              exclude_outliers=True, exclude_first=False):
        """
        Return all response times with a given stimulus/response status.

        Any of the query parameters that is None is ignored.
        """
        rtm = []
        marks = []
        for j, (o, m, c, x) in enumerate(
                  zip(self.Outl, self.Mark, self.Code, self.Rtim)):
            if exclude_outliers and o:
                continue
            if exclude_first and j == 0:
                continue
            if side is not None and c.side != side:
                continue
            if congruent is not None and c.congruent != congruent:
                continue
            if correct is not None and c.correct != correct:
                continue
            if lastcorrect is not None and j > 0:
                if (self.Code[j-1] is None or
                   self.Code[j-1].correct != lastcorrect):
                    continue
            rtm.append(x)
            marks.append(m)

        return rtm, marks

    def __str__(self):
        print(self.RT)


# Need to customize this to handle ddof=1 and n=1 (want 0 not NaN in
# this case).
def std(x):
    if len(x) <= ddof:
        return np.nan
    return np.std(x, ddof=ddof)


def process_trial(trial, block):
    """
    Insert summary values obtained from qu into block.

    Parameters
    ----------
    trial :
        Data from a single trial.
    block :
        All data grouped by trial/response type
    """

    # A trial must start with S99 (fixation cross), then have a
    # stimulus and response.  Return early if this is not the case.
    if len(trial) < 3 or trial[0].srcode != 99:
        sr = trial[0].srcode if len(trial) > 0 else ""
        logging.info(
            "Skipping trial: length=%d (expected 3), code=%s (expected 99)"
            % (len(trial), str(sr)))
        return

    # ky is the stimulus and response type
    code = Code.fromSRCodes(trial[1].srcode, trial[2].srcode)

    # Response time for first response, multiplication by 2 is a scale
    # conversion.
    rt = 2 * (trial[2].time - trial[1].time)
    block.Mark.append(trial[0].mark)
    block.Code.append(code)
    block.Rtim.append(rt)
    block.Ntri.append(len(trial))
    block.Outl.append(False)  # will be set later


def collapse_blocks(data):
    """
    Collapse a list of blocks into a single block.
    """
    blk = Block()
    for block in data:
        blk.Mark.extend(block.Mark)
        blk.Code.extend(block.Code)
        blk.Rtim.extend(block.Rtim)
        blk.Ntri.extend(block.Ntri)
        blk.Outl.extend(block.Outl)
    return blk


def summarize_vmrk(filename, data):
    """
    Create summary statistics from a VMRK file.

    The VMRK file must be processed with the process_vmrk method
    before running this function.
    """

    cdata = collapse_blocks(data)

    # Count the number of trials per block
    bc = [len(b.Outl) - np.sum(b.Outl) for b in data]
    logging.info("Trials per block: " + str(bc))

    results = OrderedDict()
    results["sid"] = filename.split(".")[0]

    all_trials = [x for o, x in zip(cdata.Outl, cdata.Rtim) if not o]
    correct_trials = [x for o, k, x in zip(cdata.Outl, cdata.Code, cdata.Rtim)
                      if not o and k.correct]
    error_trials = [x for o, k, x in zip(cdata.Outl, cdata.Code, cdata.Rtim)
                    if not o and not k.correct]

    # All/error trials without outlier filtering
    all_trials_all = [x for x in cdata.Rtim]
    error_trials_all = [x for k, x in zip(cdata.Code, cdata.Rtim)
                        if not k.correct]

    logging.info("Found %d trials" % len(all_trials_all))
    logging.info("Found %d error trials" % len(error_trials_all))
    logging.info("Found %d after removing outliers" % len(all_trials))
    logging.info("Found %d error trials after removing outliers",
                 len(error_trials))

    results["fcn"] = len(correct_trials)
    results["fen"] = len(error_trials)
    results["facc"] = 100 * results["fcn"] / len(all_trials)

    # All trial summaries
    results["frtm"] = np.mean(all_trials)
    results["frtsd"] = std(all_trials)

    # All correct trial summaries
    results["frtmc"] = np.mean(correct_trials)
    results["frtsdc"] = std(correct_trials)

    # All error trial summaries
    results["frtme"] = np.mean(error_trials)
    results["frtsde"] = std(error_trials)

    # Congruent correct trials
    v, _ = cdata.query(correct=True, congruent=True)
    results["fccn"] = len(v)
    results["fcrtmc"] = np.mean(v)
    results["fcrtsdc"] = std(v)

    # Congruent error trials
    v, _ = cdata.query(correct=False, congruent=True)
    results["fcen"] = len(v)
    results["fcrtme"] = np.mean(v)
    results["fcrtsde"] = std(v)

    # Congruent accuracy
    tot = results["fccn"] + results["fcen"]
    results["fcacc"] = 100 * results["fccn"] / tot

    # Incongruent correct trials
    v, _ = cdata.query(correct=True, congruent=False)
    results["ficn"] = len(v)
    results["firtmc"] = np.mean(v)
    results["firtsdc"] = std(v)

    # Incongruent error trials
    v, _ = cdata.query(correct=False, congruent=False)
    results["fien"] = len(v)
    results["firtme"] = np.mean(v)
    results["firtsde"] = std(v)

    # Incongruent accuracy
    tot = results["ficn"] + results["fien"]
    results["fiacc"] = 100 * results["ficn"] / tot

    # Post correct correct trials
    # (don't count first trial of each block)
    u = [b.query(correct=True, lastcorrect=True, exclude_outliers=True,
         exclude_first=True) for b in data]
    v = [x[0] for x in u]
    results["fpccn"] = sum([len(x) for x in v])

    # Post correct error trials
    # (don't count first trial of each block)
    u = [b.query(correct=False, lastcorrect=True, exclude_outliers=True,
         exclude_first=True) for b in data]
    v = [x[0] for x in u]
    results["fpcen"] = sum([len(x) for x in v])
    if results["fpcen"] > 0:
        results["fpcertm"] = sum([sum(x) for x in v]) / results["fpcen"]
    else:
        results["fpcertm"] = 0.

    # Post error correct trials
    # (don't count first trial of each block)
    u = [b.query(correct=True, lastcorrect=False, exclude_outliers=True,
                 exclude_first=True) for b in data]
    v = [x[0] for x in u]
    results["fpecn"] = sum([len(x) for x in v])
    if results["fpecn"] > 0:
        results["fpecrtm"] = sum([sum(x) for x in v]) / results["fpecn"]
    else:
        results["fpecrtm"] = 0.

    # Post error error trials
    # (don't count first trial of each block)
    u = [b.query(correct=False, lastcorrect=False, exclude_outliers=True,
         exclude_first=True) for b in data]
    v = [x[0] for x in u]
    results["fpeen"] = sum([len(x) for x in v])
    if results["fpeen"] > 0:
        results["fpeertm"] = sum([sum(x) for x in v]) / results["fpeen"]
    else:
        results["fpeertm"] = 0.

    # Post error any trials
    # (don't count first trial of each block)
    u = [b.query(lastcorrect=False, exclude_outliers=True, exclude_first=True)
         for b in data]
    v = [x[0] for x in u]
    results["fpexn"] = sum([len(x) for x in v])
    if results["fpexn"] > 0:
        results["fpexrtm"] = sum([sum(x) for x in v]) / results["fpexn"]
    else:
        results["fpexrtm"] = 0.

    # Post any error trials
    # (don't count first trial of each block)
    u = [b.query(correct=False, exclude_outliers=True, exclude_first=True)
         for b in data]
    v = [x[0] for x in u]
    results["fpxen"] = sum([len(x) for x in v])
    if results["fpxen"] > 0:
        results["fpxertm"] = sum([sum(x) for x in v]) / results["fpxen"]
    else:
        results["fpxertm"] = 0.

    # Post correct accuracy
    tot = results["fpccn"] + results["fpcen"]
    results["faccpc"] = results["fpccn"] / tot

    # Post error accuracy
    tot = results["fpecn"] + results["fpeen"]
    results["faccpe"] = results["fpecn"] / tot

    # Post error slowing
    results["fpes"] = results["fpeertm"] - results["fpecrtm"]
    results["fpes2"] = results["fpcertm"] - results["fpecrtm"]
    results["fpes3"] = results["fpxertm"] - results["fpexrtm"]

    # Anticipatory responses, calculate from all data, prior to removing
    # outliers
    results["fan"] = np.sum(np.asarray(all_trials_all) < 150)
    results["faen"] = np.sum(np.asarray(error_trials_all) < 150)

    # Trials with extra responses
    u = np.asarray([x for x in cdata.Ntri if x is not None])
    results["fscn"] = sum(u > 3)

    return results


def process_vmrk(filename):
    """
    Process the VMRK format file with name filename.

    Parameters
    ----------
    filename :
        Name of a vmrk format file.

    Returns
    -------
    data : list of Blocks
        data[j] contains the data for block j, with outliers removed.
    """

    fid = open(filename)
    logging.info("Starting file: %s" % filename)
    rdr = csv.reader(fid)

    # Keep track of which block we are in
    blocknum = 0
    dblock = 0

    # Assume that we start in practice mode
    mode = "practice"
    if nopractice:
        mode = "experiment"
    n99 = 0
    n144 = False
    trials, data = [], []
    block = Block()

    for line_num, line in enumerate(rdr):

        # Only process "mark" lines
        if len(line) == 0 or not line[0].lower().startswith("mk"):
            logging.info("Skipping row %d [A]: %s" % (line_num + 1, line))
            continue

        # Lines have format Mk###=type, where type=comment, stimulus
        f0 = line[0].split("=")
        mark = f0[0]
        fl = f0[1].lower()
        if fl == "comment":
            continue
        elif fl != "stimulus":
            # Not sure what else exists, log it and move on
            logging.info("Skipping row %d [B]: %s" % (line_num + 1, line))
            continue

        # Get the type code, e.g. if S16 then n=16
        f1 = line[1].replace(" ", "")
        stimcode = int(f1[1:])

        # Check for the end of the practice session
        if mode == "practice":
            if stimcode == 99:
                n99 += 1
            if n99 == 3 and (stimcode == 144 or stimcode == 181):
                n144 = True
            if n99 == 3 and n144 and stimcode == 255:
                mode = "experiment"
                logging.info(
                    "Starting experiment mode on row %d" % (line_num + 2))
                continue

        if mode == "practice":
            continue

        trials.append(Trial(mark, stimcode, int(line[2])))

        # Handle end of block markers
        if stimcode in (144, 255):
            if dblock > 0:
                process_trial(trials[0:-1], block)
                trials = [trials[-1]]
                blocknum += 1
                dblock = 0
                block.check_outliers(low, high)
                data.append(block)
                block = Block()
            continue
        dblock += 1

        if stimcode == 99:
            process_trial(trials[0:-1], block)
            trials = [trials[-1]]

    # Final trial may not have been processed
    process_trial(trials, block)

    return data


if __name__ == "__main__":

    logging.basicConfig(filename="vmrk.log", level=logging.DEBUG, filemode='w')

    parser = argparse.ArgumentParser()
    parser.add_argument(
         '--nopractice',
         help='Use all trials without attempting to exclude practice trials',
         action='store_const', const=True)
    args = parser.parse_args()

    nopractice = False
    if args.nopractice:
        nopractice = True

    # Get all the vmrk files from the current directory.
    files = os.listdir()
    files = [f for f in files if f.lower().endswith(".vmrk")]

    out = open("results.csv", "w")

    results = []
    for i, fname in enumerate(files):

        # Process one file
        data = process_vmrk(fname)
        result = summarize_vmrk(fname, data)

        if i == 0:
            # Write header on first iteration only.
            wtr = csv.writer(out)
            header = [k for k in result]
            wtr.writerow(header)

        # Write the results for the current file.
        wtr.writerow([result[k] for k in result])

    out.close()
